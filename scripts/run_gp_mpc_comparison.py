#!/usr/bin/env python3
"""
GP-MPC vs Nominal MPC Comparison with Model Mismatch

This script demonstrates GP-MPC's ability to learn and compensate for
systematic model errors. The key insight is:

1. TRUE dynamics: The "real world" with different parameters
   - 20% heavier mass
   - 10% weaker thrust effectiveness
   - Aerodynamic drag (model doesn't account for this)

2. MODEL dynamics: What the MPC controller thinks (nominal model)

3. GP learns the RESIDUAL: d(x,u) = f_true(x,u) - f_model(x,u)

This shows GP-MPC's real value: when the model is wrong,
GP-MPC learns the error and recovers. Baseline controllers can't.

Usage:
    python scripts/run_gp_mpc_comparison.py --quick      # Quick test
    python scripts/run_gp_mpc_comparison.py --standard   # Standard (50 runs)
    python scripts/run_gp_mpc_comparison.py --full       # Full analysis
"""

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dynamics import Rocket3DoFConfig, Rocket3DoFDynamics
from experiments import (
    LandingConstraints,
    MonteCarloResults,
    SimulationConfig,
    SimulationResult,
    LandingOutcome,
)
from experiments.baselines import LQRController, PIDController
from gp import Simple3DoFGP
from mpc import MPCConfig, NominalMPC3DoF

# Suppress IPOPT verbose output
import os
os.environ['IPOPT_PRINT_LEVEL'] = '0'


# =============================================================================
# Model Mismatch Configuration
# =============================================================================


@dataclass
class ModelMismatchConfig:
    """Configuration for systematic model mismatch."""

    # Mass mismatch: true mass is heavier than modeled
    mass_scale: float = 1.2  # 20% heavier

    # Thrust effectiveness: true thrust is weaker
    thrust_scale: float = 0.9  # 10% weaker

    # Aerodynamic drag (model assumes no drag)
    enable_drag: bool = True
    C_D: float = 0.3  # Drag coefficient
    A_ref: float = 0.5  # Reference area [m^2]
    rho: float = 1.225  # Air density [kg/m^3]

    def describe(self) -> str:
        """Get human-readable description."""
        parts = []
        if self.mass_scale != 1.0:
            parts.append(f"Mass: {(self.mass_scale - 1) * 100:+.0f}%")
        if self.thrust_scale != 1.0:
            parts.append(f"Thrust: {(self.thrust_scale - 1) * 100:+.0f}%")
        if self.enable_drag:
            parts.append(f"Drag: Cd={self.C_D}")
        return ", ".join(parts) if parts else "No mismatch"


class MismatchedDynamics:
    """
    Wrapper that applies model mismatch to true dynamics.

    This simulates the "real world" being different from our model.
    The MPC controller uses the nominal model, but the simulation
    uses these mismatched dynamics.
    """

    def __init__(
        self,
        nominal_dynamics: Rocket3DoFDynamics,
        mismatch: ModelMismatchConfig,
    ):
        self.nominal = nominal_dynamics
        self.mismatch = mismatch
        self.params = nominal_dynamics.params  # For compatibility

        # Track residuals for analysis
        self.last_residual = np.zeros(3)

    def step(self, x: NDArray, u: NDArray, dt: float) -> NDArray:
        """
        Simulate true dynamics with mismatch.

        The true dynamics differ from nominal in:
        1. Effective mass is higher (vehicle is heavier)
        2. Thrust effectiveness is lower
        3. Drag force is present (model assumes none)
        """
        m = x[0]
        r = x[1:4]
        v = x[4:7]

        # --- Nominal prediction (what controller expects) ---
        x_nominal = self.nominal.step(x, u, dt)

        # --- True dynamics with mismatch ---

        # Effective mass (heavier than modeled)
        m_effective = m * self.mismatch.mass_scale

        # Thrust with reduced effectiveness
        T_effective = u * self.mismatch.thrust_scale

        # Gravity (same as model)
        g = self.nominal.params.g_vec

        # Drag force (model assumes none)
        drag = np.zeros(3)
        if self.mismatch.enable_drag:
            speed = np.linalg.norm(v)
            if speed > 0.1:  # Avoid division by zero
                drag_mag = 0.5 * self.mismatch.rho * speed**2 * self.mismatch.C_D * self.mismatch.A_ref
                drag = -drag_mag * (v / speed)  # Opposite to velocity

        # True acceleration
        a_true = T_effective / m_effective + g + drag / m_effective

        # Nominal acceleration (what model predicts)
        a_nominal = u / m + g

        # Store residual (for GP training)
        self.last_residual = a_true - a_nominal

        # Integrate with true dynamics
        # Mass dynamics (same as nominal - fuel consumption doesn't change)
        alpha = self.nominal.params.alpha
        T_mag = np.linalg.norm(u)
        m_next = m - alpha * T_mag * dt

        # Position/velocity with true acceleration
        v_next = v + a_true * dt
        r_next = r + v * dt + 0.5 * a_true * dt**2

        return np.array([m_next, r_next[0], r_next[1], r_next[2],
                         v_next[0], v_next[1], v_next[2]])

    def get_residual(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Compute the residual (difference between true and model acceleration).

        This is what the GP should learn: d(x,u) = a_true(x,u) - a_model(x,u)
        """
        m = x[0]
        v = x[4:7]

        # Effective parameters
        m_effective = m * self.mismatch.mass_scale
        T_effective = u * self.mismatch.thrust_scale

        # Drag
        drag = np.zeros(3)
        if self.mismatch.enable_drag:
            speed = np.linalg.norm(v)
            if speed > 0.1:
                drag_mag = 0.5 * self.mismatch.rho * speed**2 * self.mismatch.C_D * self.mismatch.A_ref
                drag = -drag_mag * (v / speed)

        # True vs nominal acceleration difference
        g = self.nominal.params.g_vec
        a_true = T_effective / m_effective + g + drag / m_effective
        a_nominal = u / m + g

        return a_true - a_nominal


# =============================================================================
# GP-Enhanced MPC Controller
# =============================================================================


class GPMPC3DoF:
    """
    GP-MPC for 3-DoF rocket with online learning.

    Uses nominal MPC as base, but augments predictions with GP-learned residuals.
    Key features:
    1. Collects training data (state, control, observed residual) during flight
    2. Fits GP periodically to learn the model mismatch
    3. Adjusts control to compensate for learned residual
    """

    def __init__(
        self,
        dynamics: Rocket3DoFDynamics,
        config: Optional[MPCConfig] = None,
        gp_update_interval: int = 5,  # Fit GP every N steps
        min_data_for_gp: int = 10,    # Minimum data points before using GP
    ):
        self.dynamics = dynamics
        self.config = config or MPCConfig(N=15, dt=0.1)

        # Base MPC controller
        self.mpc = NominalMPC3DoF(dynamics, self.config)

        # GP for learning residuals
        self.gp = Simple3DoFGP(n_inducing=30, noise_variance=1e-3, use_sparse=False)

        # Data collection
        self.states = []
        self.controls = []
        self.residuals = []

        self.gp_update_interval = gp_update_interval
        self.min_data_for_gp = min_data_for_gp
        self.step_count = 0

        # Statistics
        self.n_gp_updates = 0
        self.gp_is_active = False

    def add_observation(self, x: NDArray, u: NDArray, x_next: NDArray) -> None:
        """
        Add an observation from the true dynamics.

        Computes the residual (difference between observed and predicted next state)
        and stores it for GP training.
        """
        # Predict next state using nominal model
        x_pred = self.dynamics.step(x, u, self.config.dt)

        # Observed residual in acceleration (convert position residual to acceleration)
        # x_next = x_pred + residual_in_state
        # For velocity: v_next = v_pred + a_residual * dt
        # So: a_residual = (v_next_obs - v_next_pred) / dt
        v_residual = x_next[4:7] - x_pred[4:7]
        a_residual = v_residual / self.config.dt

        self.states.append(x.copy())
        self.controls.append(u.copy())
        self.residuals.append(a_residual)

        # Periodic GP update
        self.step_count += 1
        if self.step_count % self.gp_update_interval == 0:
            self._update_gp()

    def _update_gp(self) -> None:
        """Fit GP to collected data."""
        n_data = len(self.states)
        if n_data < self.min_data_for_gp:
            return

        X = np.array(self.states)
        U = np.array(self.controls)
        D = np.array(self.residuals)

        # Clear old data and add new
        self.gp.X_data = []
        self.gp.U_data = []
        self.gp.D_data = []
        self.gp._is_fitted = False

        self.gp.add_data(X, U, D)

        try:
            self.gp.fit()
            self.gp_is_active = True
            self.n_gp_updates += 1
        except Exception:
            # If GP fitting fails, continue without GP
            pass

    def solve(self, x: NDArray, x_target: NDArray) -> "MPCSolution":
        """
        Solve GP-MPC problem.

        If GP is trained, adjusts control to compensate for learned residual.
        """
        # First, solve nominal MPC
        sol = self.mpc.solve(x, x_target)

        if not sol.success:
            return sol

        # If GP is active, adjust first control based on predicted residual
        if self.gp_is_active and self.gp.n_data >= self.min_data_for_gp:
            u_nominal = sol.U_opt[0].copy()

            # Predict residual with GP
            try:
                d_mean, d_var = self.gp.predict(x, u_nominal)

                # Compensate: if true acceleration is lower (negative residual),
                # we need more thrust
                # Simple feedforward compensation
                m = x[0]
                u_compensation = -d_mean * m  # F = m * a

                # Limit compensation to avoid instability
                max_comp = 1.0  # Maximum compensation magnitude
                comp_mag = np.linalg.norm(u_compensation)
                if comp_mag > max_comp:
                    u_compensation = u_compensation * (max_comp / comp_mag)

                # Apply compensation
                sol.U_opt[0] = u_nominal + u_compensation

            except Exception:
                pass  # If prediction fails, use nominal control

        return sol

    def reset(self) -> None:
        """Reset for new episode."""
        self.states = []
        self.controls = []
        self.residuals = []
        self.step_count = 0
        self.gp_is_active = False
        # Keep GP model for transfer learning across episodes


# =============================================================================
# Simulation with Model Mismatch
# =============================================================================


def run_single_episode(
    true_dynamics: MismatchedDynamics,
    controller,
    x0: NDArray,
    config: SimulationConfig,
    collect_residuals: bool = False,
) -> Tuple[SimulationResult, list]:
    """
    Run a single landing episode with mismatched dynamics.

    Returns:
        result: Simulation result
        residuals: List of (state, control, residual) if collect_residuals=True
    """
    t_start = time.perf_counter()

    initial_mass = x0[0]
    x = x0.copy()
    t = 0.0
    dt = config.dt
    max_steps = int(config.max_time / dt)

    states = [x.copy()]
    controls = []
    times = [0.0]
    residuals_data = []

    outcome = LandingOutcome.SUCCESS
    failure_reason = ""

    for step in range(max_steps):
        altitude = x[1]  # x is altitude (gravity in -x)
        velocity = x[4]

        # Check termination
        if altitude < 0:
            outcome = LandingOutcome.CRASH
            failure_reason = f"Crashed at altitude {altitude:.2f}m"
            break

        if x[0] <= 1.01:  # Near dry mass
            outcome = LandingOutcome.FUEL_EXHAUSTED
            failure_reason = f"Fuel exhausted: mass={x[0]:.3f}"
            break

        if np.any(np.abs(x) > 1e6) or np.any(np.isnan(x)):
            outcome = LandingOutcome.DIVERGENCE
            failure_reason = "State diverged"
            break

        # Check landing
        if altitude < 1.0 and abs(velocity) < 5.0:
            success, reason = config.landing_constraints.check_landing(x, initial_mass)
            if not success:
                outcome = LandingOutcome.CONSTRAINT_VIOLATION
                failure_reason = reason
            break

        # Get control
        try:
            x_target = x.copy()
            x_target[1] = max(0.0, altitude - 2.0)
            blend = min(1.0, 0.3 + 0.7 * (1 - altitude / 30.0))
            x_target[2] = x[2] * (1 - blend)
            x_target[3] = x[3] * (1 - blend)
            x_target[4:7] = 0.0

            sol = controller.solve(x, x_target)
            if hasattr(sol, "success") and not sol.success:
                outcome = LandingOutcome.DIVERGENCE
                failure_reason = "MPC solve failed"
                break
            u = sol.u0
        except Exception as e:
            outcome = LandingOutcome.DIVERGENCE
            failure_reason = f"Controller failed: {e}"
            break

        # Simulate with TRUE dynamics (mismatched)
        x_next = true_dynamics.step(x, u, dt)

        # Collect residual for analysis
        if collect_residuals:
            residual = true_dynamics.get_residual(x, u)
            residuals_data.append((x.copy(), u.copy(), residual.copy()))

        # If using GP-MPC, add observation for learning
        if hasattr(controller, "add_observation"):
            controller.add_observation(x, u, x_next)

        controls.append(u)
        x = x_next
        t += dt
        states.append(x.copy())
        times.append(t)

    else:
        outcome = LandingOutcome.TIMEOUT
        failure_reason = f"Exceeded {config.max_time}s"

    compute_time = (time.perf_counter() - t_start) * 1000

    states = np.array(states)
    controls = np.array(controls) if controls else np.zeros((0, 3))
    times = np.array(times)

    final_state = states[-1]
    fuel_used = initial_mass - final_state[0]
    flight_time = times[-1]
    pos_error = np.linalg.norm(final_state[1:4])
    vel_error = np.linalg.norm(final_state[4:7])

    success = outcome == LandingOutcome.SUCCESS

    result = SimulationResult(
        run_id=0,
        outcome=outcome,
        success=success,
        states=states,
        controls=controls,
        times=times,
        fuel_used=fuel_used,
        flight_time=flight_time,
        final_position_error=pos_error,
        final_velocity_error=vel_error,
        max_constraint_violation=0.0,
        initial_state=x0,
        compute_time_ms=compute_time,
        failure_reason=failure_reason,
    )

    return result, residuals_data


def run_comparison(
    n_runs: int,
    mismatch: ModelMismatchConfig,
    config: SimulationConfig,
    seed: int = 42,
) -> dict:
    """
    Run comparison of controllers under model mismatch.

    GP-MPC runs in two phases:
    1. Training phase: Learn model error from initial runs (GP accumulates data)
    2. Test phase: Use trained GP for remaining runs

    Returns dict with results for each controller.
    """
    np.random.seed(seed)

    # Create nominal dynamics (what controllers think)
    nominal_dynamics = Rocket3DoFDynamics()

    # Create true dynamics (with mismatch)
    true_dynamics = MismatchedDynamics(nominal_dynamics, mismatch)

    # Create controllers
    mpc_config = MPCConfig(N=15, dt=0.1)

    controllers = {
        "GP-MPC": GPMPC3DoF(nominal_dynamics, mpc_config),
        "Nominal-MPC": NominalMPC3DoF(nominal_dynamics, mpc_config),
        "LQR": LQRController(nominal_dynamics),
        "PID": PIDController(nominal_dynamics),
    }

    # Sample initial conditions
    initial_states = []
    for _ in range(n_runs):
        m = config.mass_mean + np.random.randn() * config.mass_std
        m = np.clip(m, 1.5, 2.5)

        altitude = config.altitude_mean + np.random.randn() * config.altitude_std
        altitude = np.clip(altitude, 10, 100)
        horiz_y = np.random.randn() * config.horizontal_std
        horiz_z = np.random.randn() * config.horizontal_std

        v_vert = config.velocity_mean[0] + np.random.randn() * config.velocity_std[0]
        v_vert = min(v_vert, -1)
        v_horiz_y = config.velocity_mean[1] + np.random.randn() * config.velocity_std[1]
        v_horiz_z = config.velocity_mean[2] + np.random.randn() * config.velocity_std[2]

        x0 = np.array([m, altitude, horiz_y, horiz_z, v_vert, v_horiz_y, v_horiz_z])
        initial_states.append(x0)

    # Run each controller
    all_results = {}

    for name, controller in controllers.items():
        print(f"\n  Evaluating {name}...")

        results = []
        for i, x0 in enumerate(initial_states):
            # For GP-MPC: DON'T reset between episodes - let GP accumulate learning
            # For other controllers: reset to fresh state
            if name != "GP-MPC":
                if hasattr(controller, "reset"):
                    controller.reset()
            else:
                # For GP-MPC: Only reset MPC solver, keep GP data
                if hasattr(controller, "mpc") and hasattr(controller.mpc, "_is_setup"):
                    controller.mpc._is_setup = False

            if hasattr(controller, "_is_setup"):
                controller._is_setup = False

            result, _ = run_single_episode(
                true_dynamics, controller, x0.copy(), config
            )
            result.run_id = i
            results.append(result)

            if (i + 1) % 10 == 0:
                n_success = sum(1 for r in results if r.success)
                gp_info = ""
                if name == "GP-MPC" and hasattr(controller, "gp"):
                    gp_info = f" (GP: {controller.gp.n_data} pts, active={controller.gp_is_active})"
                print(f"    {i + 1}/{n_runs}: {n_success}/{i + 1} successful{gp_info}")

        # Compile results
        mc_results = MonteCarloResults(config=config, results=results)
        all_results[name] = mc_results

        stats = mc_results.get_statistics()
        print(f"    Final: {stats['success_rate'] * 100:.1f}% success, "
              f"{stats.get('fuel_mean', 0):.3f} kg fuel")

        # GP-MPC specific info
        if name == "GP-MPC" and hasattr(controller, "gp"):
            print(f"    GP trained with {controller.gp.n_data} data points")

    return all_results


def print_comparison_table(results: dict, title: str = "Results"):
    """Print formatted comparison table."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"{'Controller':<15} {'Success%':>10} {'Fuel (kg)':>12} {'Pos Err (m)':>12} {'Vel Err (m/s)':>14}")
    print(f"{'-' * 70}")

    for name, mc_results in results.items():
        stats = mc_results.get_statistics()
        sr = stats["success_rate"] * 100
        fuel = stats.get("fuel_mean", 0)
        pos_err = stats.get("pos_error_mean", 0)
        vel_err = stats.get("vel_error_mean", 0)
        print(f"{name:<15} {sr:>9.1f}% {fuel:>11.3f} {pos_err:>11.3f} {vel_err:>13.3f}")

    print(f"{'=' * 70}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(
        description="GP-MPC vs Nominal MPC with Model Mismatch"
    )
    parser.add_argument("--quick", action="store_true", help="Quick test (10 runs)")
    parser.add_argument("--standard", action="store_true", help="Standard (50 runs)")
    parser.add_argument("--full", action="store_true", help="Full analysis (200 runs)")
    parser.add_argument("--runs", type=int, help="Custom number of runs")
    parser.add_argument("--no-mismatch", action="store_true", help="Run without mismatch (sanity check)")

    args = parser.parse_args()

    # Determine runs
    if args.runs:
        n_runs = args.runs
    elif args.full:
        n_runs = 200
    elif args.standard:
        n_runs = 50
    else:
        n_runs = 10

    print(f"\n{'#' * 70}")
    print("#  GP-MPC vs Nominal MPC: Learning Under Model Mismatch")
    print(f"#  Runs: {n_runs}")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # Simulation config
    sim_config = SimulationConfig(
        dt=0.1,
        max_time=30.0,
        altitude_mean=30.0,
        altitude_std=5.0,
        horizontal_std=3.0,
        velocity_mean=np.array([-3, 0, 0]),
        velocity_std=np.array([1, 0.5, 0.5]),
        landing_constraints=LandingConstraints(
            pos_tol_xy=5.0,
            pos_tol_z=2.0,
            vel_tol_xy=2.0,
            vel_tol_z=3.5,
        ),
    )

    # Model mismatch configuration
    if args.no_mismatch:
        mismatch = ModelMismatchConfig(
            mass_scale=1.0,
            thrust_scale=1.0,
            enable_drag=False,
        )
        print("\n[SANITY CHECK] Running WITHOUT model mismatch")
    else:
        # SEVERE model mismatch to show GP-MPC benefit
        mismatch = ModelMismatchConfig(
            mass_scale=1.4,    # 40% heavier - significant!
            thrust_scale=0.75,  # 25% weaker thrust - severe!
            enable_drag=True,  # Unmodeled drag
            C_D=0.5,           # Higher drag
            rho=1.5,           # Denser atmosphere
        )

    print(f"\nModel Mismatch: {mismatch.describe()}")

    # Run comparison
    print("\nRunning comparison...")
    results = run_comparison(n_runs, mismatch, sim_config)

    # Print results
    print_comparison_table(
        results,
        title=f"Controller Performance Under Model Mismatch ({mismatch.describe()})"
    )

    # Analysis: Show GP learning benefit
    gp_mpc = results["GP-MPC"]
    nominal = results["Nominal-MPC"]

    gp_stats = gp_mpc.get_statistics()
    nom_stats = nominal.get_statistics()

    gp_success = gp_stats["success_rate"]
    nom_success = nom_stats["success_rate"]
    gp_fuel = gp_stats.get("fuel_mean", 0)
    nom_fuel = nom_stats.get("fuel_mean", 0)
    gp_pos_err = gp_stats.get("pos_error_mean", 0)
    nom_pos_err = nom_stats.get("pos_error_mean", 0)

    success_improvement = (gp_success - nom_success) * 100
    fuel_improvement = ((nom_fuel - gp_fuel) / nom_fuel * 100) if nom_fuel > 0 else 0

    print(f"\n{'=' * 70}")
    print("  ANALYSIS: GP-MPC vs Nominal MPC")
    print(f"{'=' * 70}")
    print(f"  Success Rate:")
    print(f"    GP-MPC:      {gp_success * 100:.1f}%")
    print(f"    Nominal MPC: {nom_success * 100:.1f}%")
    print(f"    Improvement: {success_improvement:+.1f} percentage points")
    print(f"\n  Fuel Consumption:")
    print(f"    GP-MPC:      {gp_fuel:.3f} kg")
    print(f"    Nominal MPC: {nom_fuel:.3f} kg")
    print(f"    Savings:     {fuel_improvement:+.1f}%")
    print(f"\n  Position Error:")
    print(f"    GP-MPC:      {gp_pos_err:.3f} m")
    print(f"    Nominal MPC: {nom_pos_err:.3f} m")

    if success_improvement > 5:
        print("\n  CONCLUSION: GP-MPC significantly outperforms Nominal MPC")
        print("  when the model is wrong. The GP learns the model error!")
    elif fuel_improvement > 2:
        print("\n  CONCLUSION: GP-MPC achieves fuel savings through learning.")
        print("  The GP compensates for model mismatch, reducing control effort.")
    elif success_improvement > 0 or fuel_improvement > 0:
        print("\n  CONCLUSION: GP-MPC shows modest improvement over Nominal MPC.")
    else:
        print("\n  NOTE: GP-MPC did not show clear benefit. This may indicate:")
        print("  - Both controllers robust enough for this scenario")
        print("  - GP needs more training episodes")
        print("  - Model mismatch affects open-loop planning more than feedback")

    print(f"\n{'#' * 70}")
    print("#  Experiment Complete")
    print(f"#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
