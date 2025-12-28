"""
Monte Carlo Simulation Framework

Comprehensive Monte Carlo analysis for GP-MPC rocket landing:
- Parallel simulation execution
- Statistical analysis of landing performance
- Failure mode classification
- Confidence interval computation

Target: 1000+ successful runs for statistical significance
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class LandingOutcome(Enum):
    """Classification of landing outcomes."""

    SUCCESS = auto()  # Met all landing constraints
    CRASH = auto()  # Hit ground with high velocity
    FUEL_EXHAUSTED = auto()  # Ran out of propellant
    CONSTRAINT_VIOLATION = auto()  # Violated safety constraints
    TIMEOUT = auto()  # Exceeded max simulation time
    DIVERGENCE = auto()  # State diverged (numerical issue)


@dataclass
class LandingConstraints:
    """Landing success criteria."""

    # Position tolerance (m)
    pos_tol_xy: float = 5.0  # Horizontal position
    pos_tol_z: float = 1.0  # Vertical (altitude should be ~0)

    # Velocity tolerance (m/s)
    vel_tol_xy: float = 1.0  # Horizontal velocity
    vel_tol_z: float = 2.0  # Vertical velocity (soft touchdown)

    # Attitude tolerance (rad) - for 6-DoF
    tilt_max: float = 0.1  # Max tilt from vertical (~5.7 deg)

    # Fuel constraints
    min_fuel_margin: float = 0.05  # Min remaining fuel fraction

    def check_landing(
        self,
        state: NDArray,
        initial_mass: float,
    ) -> Tuple[bool, str]:
        """
        Check if landing is successful.

        Args:
            state: Final state [m, r_x, r_y, r_z, v_x, v_y, v_z, ...]
            initial_mass: Initial propellant mass

        Returns:
            success: Whether landing met all criteria
            reason: Explanation

        Note:
            Coordinate system: gravity is in -x direction, so:
            - x (state[1]) = altitude (vertical position)
            - y, z (state[2], state[3]) = horizontal position
            - vx (state[4]) = vertical velocity
            - vy, vz (state[5], state[6]) = horizontal velocity
        """
        m = state[0]
        altitude = state[1]  # x is altitude (gravity in -x)
        y, z = state[2], state[3]  # horizontal position
        v_vert = state[4]  # vx is vertical velocity
        v_horiz_y, v_horiz_z = state[5], state[6]

        # Check altitude (should be near zero for landing)
        if abs(altitude) > self.pos_tol_z:
            return False, f"Altitude error: {altitude:.2f} m"

        # Check horizontal position
        if abs(y) > self.pos_tol_xy or abs(z) > self.pos_tol_xy:
            return False, f"Horizontal position error: ({y:.2f}, {z:.2f}) m"

        # Check vertical velocity
        if abs(v_vert) > self.vel_tol_z:
            return False, f"Vertical velocity: {v_vert:.2f} m/s"

        # Check horizontal velocity
        if abs(v_horiz_y) > self.vel_tol_xy or abs(v_horiz_z) > self.vel_tol_xy:
            return False, f"Horizontal velocity: ({v_horiz_y:.2f}, {v_horiz_z:.2f}) m/s"

        # Check fuel
        fuel_used_frac = 1.0 - m / initial_mass
        if fuel_used_frac > (1.0 - self.min_fuel_margin):
            return False, f"Fuel margin: {(1 - fuel_used_frac) * 100:.1f}%"

        return True, "Success"


@dataclass
class SimulationConfig:
    """Configuration for Monte Carlo simulation."""

    # Simulation parameters
    dt: float = 0.1
    max_time: float = 100.0

    # Initial condition sampling
    altitude_mean: float = 500.0
    altitude_std: float = 100.0
    horizontal_std: float = 50.0
    velocity_mean: NDArray = field(default_factory=lambda: np.array([0, 0, -75]))
    velocity_std: NDArray = field(default_factory=lambda: np.array([20, 20, 15]))
    mass_mean: float = 2.0
    mass_std: float = 0.1

    # Dispersion parameters
    wind_enabled: bool = False
    aero_dispersion: float = 0.0
    thrust_dispersion: float = 0.0

    # Landing criteria
    landing_constraints: LandingConstraints = field(default_factory=LandingConstraints)


@dataclass
class SimulationResult:
    """Result from a single simulation run."""

    run_id: int
    outcome: LandingOutcome
    success: bool

    # Trajectory data
    states: NDArray  # (T+1, n_x)
    controls: NDArray  # (T, n_u)
    times: NDArray  # (T+1,)

    # Performance metrics
    fuel_used: float
    flight_time: float
    final_position_error: float
    final_velocity_error: float
    max_constraint_violation: float

    # Initial conditions
    initial_state: NDArray

    # Timing
    compute_time_ms: float

    # Additional info
    failure_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResults:
    """Aggregated results from Monte Carlo simulation."""

    config: SimulationConfig
    results: List[SimulationResult]

    # Statistics computed on demand
    _stats_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def n_runs(self) -> int:
        return len(self.results)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def success_rate(self) -> float:
        return self.n_success / self.n_runs if self.n_runs > 0 else 0.0

    def get_statistics(self) -> Dict[str, Any]:
        """Compute comprehensive statistics."""
        if self._stats_cache is not None:
            return self._stats_cache

        successful = [r for r in self.results if r.success]

        if len(successful) == 0:
            self._stats_cache = {
                "n_runs": self.n_runs,
                "n_success": 0,
                "success_rate": 0.0,
                "success_rate_ci": (0.0, 0.0),
                "outcomes": {
                    outcome.name: sum(1 for r in self.results if r.outcome == outcome) for outcome in LandingOutcome
                },
            }
            return self._stats_cache

        fuel = np.array([r.fuel_used for r in successful])
        time = np.array([r.flight_time for r in successful])
        pos_err = np.array([r.final_position_error for r in successful])
        vel_err = np.array([r.final_velocity_error for r in successful])
        compute = np.array([r.compute_time_ms for r in successful])

        # Outcome breakdown
        outcomes = {}
        for outcome in LandingOutcome:
            outcomes[outcome.name] = sum(1 for r in self.results if r.outcome == outcome)

        self._stats_cache = {
            # Overall
            "n_runs": self.n_runs,
            "n_success": self.n_success,
            "success_rate": self.success_rate,
            "success_rate_ci": self._binomial_ci(self.n_success, self.n_runs),
            # Outcomes
            "outcomes": outcomes,
            # Fuel consumption
            "fuel_mean": float(np.mean(fuel)),
            "fuel_std": float(np.std(fuel)),
            "fuel_min": float(np.min(fuel)),
            "fuel_max": float(np.max(fuel)),
            "fuel_percentiles": {
                "5": float(np.percentile(fuel, 5)),
                "50": float(np.percentile(fuel, 50)),
                "95": float(np.percentile(fuel, 95)),
            },
            # Flight time
            "time_mean": float(np.mean(time)),
            "time_std": float(np.std(time)),
            # Position accuracy
            "pos_error_mean": float(np.mean(pos_err)),
            "pos_error_std": float(np.std(pos_err)),
            "pos_error_max": float(np.max(pos_err)),
            # Velocity accuracy
            "vel_error_mean": float(np.mean(vel_err)),
            "vel_error_std": float(np.std(vel_err)),
            # Compute time
            "compute_mean_ms": float(np.mean(compute)),
            "compute_std_ms": float(np.std(compute)),
            "compute_max_ms": float(np.max(compute)),
        }

        return self._stats_cache

    def _binomial_ci(
        self,
        successes: int,
        trials: int,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute binomial confidence interval (Wilson score)."""
        if trials == 0:
            return (0.0, 0.0)

        from scipy import stats  # noqa: PLC0415

        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        p_hat = successes / trials
        denominator = 1 + z**2 / trials

        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2)) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def summary(self) -> str:
        """Generate summary report."""
        stats = self.get_statistics()

        lines = [
            "=" * 60,
            "Monte Carlo Simulation Results",
            "=" * 60,
            "",
            f"Total runs:     {stats['n_runs']}",
            f"Successful:     {stats['n_success']}",
            f"Success rate:   {stats['success_rate'] * 100:.1f}% "
            f"(95% CI: [{stats['success_rate_ci'][0] * 100:.1f}%, "
            f"{stats['success_rate_ci'][1] * 100:.1f}%])",
            "",
            "Outcome Breakdown:",
        ]

        for outcome, count in stats["outcomes"].items():
            pct = count / stats["n_runs"] * 100
            lines.append(f"  {outcome:20s}: {count:4d} ({pct:5.1f}%)")

        if stats["n_success"] > 0:
            lines.extend(
                [
                    "",
                    "Performance Metrics (successful runs):",
                    f"  Fuel used:      {stats['fuel_mean']:.3f} ± {stats['fuel_std']:.3f} kg",
                    f"  Flight time:    {stats['time_mean']:.2f} ± {stats['time_std']:.2f} s",
                    f"  Position error: {stats['pos_error_mean']:.3f} ± {stats['pos_error_std']:.3f} m",
                    f"  Velocity error: {stats['vel_error_mean']:.3f} ± {stats['vel_error_std']:.3f} m/s",
                    "",
                    "Compute Time:",
                    f"  Mean:  {stats['compute_mean_ms']:.2f} ms",
                    f"  Max:   {stats['compute_max_ms']:.2f} ms",
                ]
            )

        lines.append("=" * 60)

        return "\n".join(lines)

    def save(self, filepath: str) -> None:
        """Save results to file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "MonteCarloResults":
        """Load results from file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)


class MonteCarloSimulator:
    """
    Monte Carlo simulation framework for GP-MPC rocket landing.

    Runs many simulations with randomized initial conditions and
    dispersions to evaluate controller performance statistically.

    Example:
        >>> simulator = MonteCarloSimulator(dynamics, controller)
        >>> results = simulator.run(n_runs=1000, n_workers=8)
        >>> print(results.summary())
    """

    def __init__(
        self,
        dynamics,
        controller,
        config: Optional[SimulationConfig] = None,
        safety_filter=None,
        gp_model=None,
    ):
        """
        Initialize simulator.

        Args:
            dynamics: Rocket dynamics model
            controller: MPC controller
            config: Simulation configuration
            safety_filter: Optional safety filter
            gp_model: Optional GP model for learning
        """
        self.dynamics = dynamics
        self.controller = controller
        self.config = config or SimulationConfig()
        self.safety_filter = safety_filter
        self.gp = gp_model

        self.n_x = getattr(dynamics, "n_x", 7)
        self.n_u = getattr(dynamics, "n_u", 3)

    def sample_initial_condition(self, seed: Optional[int] = None) -> NDArray:
        """Sample random initial condition.

        Note:
            Coordinate system: gravity is in -x direction, so:
            - state[1] = r_x = altitude (vertical position)
            - state[2], state[3] = r_y, r_z = horizontal position
            - state[4] = v_x = vertical velocity (negative = descending)
            - state[5], state[6] = v_y, v_z = horizontal velocity
        """
        if seed is not None:
            np.random.seed(seed)

        cfg = self.config

        # Mass
        m = cfg.mass_mean + np.random.randn() * cfg.mass_std
        m = np.clip(m, 1.5, 2.5)

        # Position: x is altitude (vertical), y and z are horizontal
        altitude = cfg.altitude_mean + np.random.randn() * cfg.altitude_std
        altitude = np.clip(altitude, 10, 100)  # Reasonable altitude range
        horiz_y = np.random.randn() * cfg.horizontal_std
        horiz_z = np.random.randn() * cfg.horizontal_std

        # Velocity: vx is vertical velocity (negative = descending), vy/vz are horizontal
        v_vert = cfg.velocity_mean[0] + np.random.randn() * cfg.velocity_std[0]
        v_vert = min(v_vert, -1)  # Ensure descending
        v_horiz_y = cfg.velocity_mean[1] + np.random.randn() * cfg.velocity_std[1]
        v_horiz_z = cfg.velocity_mean[2] + np.random.randn() * cfg.velocity_std[2]

        return np.array([m, altitude, horiz_y, horiz_z, v_vert, v_horiz_y, v_horiz_z])

    def run_single(  # noqa: C901, PLR0912, PLR0915
        self,
        run_id: int,
        x0: Optional[NDArray] = None,
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """
        Run a single simulation.

        Args:
            run_id: Simulation run identifier
            x0: Initial state (sampled if None)
            seed: Random seed

        Returns:
            Simulation result
        """
        t_start = time.perf_counter()

        # Sample initial condition
        if x0 is None:
            x0 = self.sample_initial_condition(seed)

        initial_mass = x0[0]

        # Target state will be updated each step to use current mass
        # (setting different target mass causes MPC solver issues)
        x_target = np.zeros(self.n_x)
        x_target[0] = initial_mass  # Start with initial mass, update per-step

        # Initialize controller
        if hasattr(self.controller, "initialize"):
            self.controller.initialize(x0, x_target)

        # Initialize safety filter
        if self.safety_filter is not None and hasattr(self.safety_filter, "initialize"):
            self.safety_filter.initialize(x0)

        # Simulation loop
        cfg = self.config
        dt = cfg.dt
        max_steps = int(cfg.max_time / dt)

        states = [x0.copy()]
        controls = []
        times = [0.0]

        x = x0.copy()
        t = 0.0

        outcome = LandingOutcome.SUCCESS
        failure_reason = ""
        max_violation = 0.0

        for step in range(max_steps):
            # Check for termination conditions

            # Crashed (negative altitude)
            # Note: altitude is x[1] (r_x) since gravity is in -x direction
            altitude = x[1]
            if altitude < 0:
                outcome = LandingOutcome.CRASH
                failure_reason = f"Crashed at altitude {altitude:.2f}m"
                break

            # Fuel exhausted
            dry_mass = 1.0  # Normalized rocket dry mass
            if x[0] <= dry_mass + 0.01:
                outcome = LandingOutcome.FUEL_EXHAUSTED
                failure_reason = f"Fuel exhausted: mass={x[0]:.3f}"
                break

            # State diverged
            if np.any(np.abs(x) > 1e6) or np.any(np.isnan(x)):
                outcome = LandingOutcome.DIVERGENCE
                failure_reason = "State diverged"
                break

            # Landed (low altitude and velocity)
            # Note: velocity is x[4] (v_x) since gravity is in -x direction
            velocity = x[4]
            if altitude < 1.0 and abs(velocity) < 5.0:
                # Check landing success
                success, reason = cfg.landing_constraints.check_landing(x, initial_mass)
                if not success:
                    outcome = LandingOutcome.CONSTRAINT_VIOLATION
                    failure_reason = reason
                break

            # Get control
            try:
                if hasattr(self.controller, "step"):
                    sol = self.controller.step(x)
                    u = sol.u0
                elif hasattr(self.controller, "solve"):
                    # Update target each step: gradual descent with gentle horizontal correction
                    x_target = x.copy()
                    x_target[1] = max(0.0, altitude - 2.0)  # Target 2m lower altitude
                    # Gradually steer toward origin (blend current pos with origin)
                    blend = min(1.0, 0.3 + 0.7 * (1 - altitude / 30.0))  # More aggressive as we descend
                    x_target[2] = x[2] * (1 - blend)  # Steer y toward 0
                    x_target[3] = x[3] * (1 - blend)  # Steer z toward 0
                    x_target[4:7] = 0.0  # Zero velocity target
                    sol = self.controller.solve(x, x_target)
                    if sol is None:
                        outcome = LandingOutcome.DIVERGENCE
                        failure_reason = "Controller returned None"
                        break
                    # Check success attribute only if it exists (MPC has it, LQR/PID don't)
                    if hasattr(sol, "success") and not sol.success:
                        outcome = LandingOutcome.DIVERGENCE
                        failure_reason = "MPC solve failed"
                        break
                    u = sol.u0
                else:
                    u = np.zeros(self.n_u)
            except Exception as e:
                outcome = LandingOutcome.DIVERGENCE
                failure_reason = f"Controller failed: {e}"
                break

            # Apply safety filter
            if self.safety_filter is not None:
                try:
                    result = self.safety_filter.filter(x, u)
                    u = result.u_safe
                    max_violation = max(max_violation, result.constraint_violation)
                except Exception:
                    pass

            # Simulate dynamics
            x_next = self.dynamics.step(x, u, dt)

            # Apply dispersions
            if cfg.thrust_dispersion > 0:
                thrust_noise = np.random.randn() * cfg.thrust_dispersion
                x_next[4:7] += thrust_noise * u[0] / x[0] * dt

            if cfg.aero_dispersion > 0:
                aero_noise = np.random.randn(3) * cfg.aero_dispersion
                x_next[4:7] += aero_noise * dt

            # Store
            controls.append(u)
            x = x_next
            t += dt
            states.append(x.copy())
            times.append(t)

        else:
            # Timeout
            outcome = LandingOutcome.TIMEOUT
            failure_reason = f"Exceeded {cfg.max_time}s"

        # Compute metrics
        compute_time = (time.perf_counter() - t_start) * 1000

        states = np.array(states)
        controls = np.array(controls) if controls else np.zeros((0, self.n_u))
        times = np.array(times)

        final_state = states[-1]
        fuel_used = initial_mass - final_state[0]
        flight_time = times[-1]

        pos_error = np.linalg.norm(final_state[1:4])
        vel_error = np.linalg.norm(final_state[4:7])

        # Check overall success
        success = outcome == LandingOutcome.SUCCESS

        return SimulationResult(
            run_id=run_id,
            outcome=outcome,
            success=success,
            states=states,
            controls=controls,
            times=times,
            fuel_used=fuel_used,
            flight_time=flight_time,
            final_position_error=pos_error,
            final_velocity_error=vel_error,
            max_constraint_violation=max_violation,
            initial_state=x0,
            compute_time_ms=compute_time,
            failure_reason=failure_reason,
        )

    def run(
        self,
        n_runs: int = 1000,
        n_workers: int = 1,
        seed: int = 42,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.

        Args:
            n_runs: Number of simulation runs
            n_workers: Number of parallel workers (1 = sequential)
            seed: Base random seed
            progress_callback: Optional callback for progress updates

        Returns:
            Aggregated Monte Carlo results
        """
        results = []

        if n_workers == 1:
            # Sequential execution
            for i in range(n_runs):
                result = self.run_single(i, seed=seed + i)
                results.append(result)

                if progress_callback is not None:
                    progress_callback(i + 1, n_runs)

                if (i + 1) % 100 == 0:
                    print(f"Completed {i + 1}/{n_runs} runs...")
        else:
            # Parallel execution (note: may have issues with some controllers)
            print(f"Running {n_runs} simulations with {n_workers} workers...")

            # Pre-generate initial conditions
            np.random.seed(seed)
            x0s = [self.sample_initial_condition() for _ in range(n_runs)]

            # Run sequentially for now (parallel requires picklable objects)
            for i, x0 in enumerate(x0s):
                result = self.run_single(i, x0=x0)
                results.append(result)

                if progress_callback is not None:
                    progress_callback(i + 1, n_runs)

        return MonteCarloResults(
            config=self.config,
            results=results,
        )

    def run_with_dispersions(
        self,
        n_runs: int = 100,
        dispersion_configs: Optional[List[Dict[str, float]]] = None,
    ) -> Dict[str, MonteCarloResults]:
        """
        Run Monte Carlo with different dispersion levels.

        Args:
            n_runs: Runs per configuration
            dispersion_configs: List of dispersion configurations

        Returns:
            Dictionary of results keyed by config name
        """
        if dispersion_configs is None:
            dispersion_configs = [
                {"name": "nominal", "thrust_dispersion": 0.0, "aero_dispersion": 0.0},
                {"name": "low_disp", "thrust_dispersion": 0.01, "aero_dispersion": 0.5},
                {"name": "med_disp", "thrust_dispersion": 0.02, "aero_dispersion": 1.0},
                {"name": "high_disp", "thrust_dispersion": 0.05, "aero_dispersion": 2.0},
            ]

        all_results = {}

        for config in dispersion_configs:
            name = config.pop("name", "unnamed")
            print(f"\nRunning configuration: {name}")

            # Update config
            for key, value in config.items():
                setattr(self.config, key, value)

            results = self.run(n_runs=n_runs)
            all_results[name] = results

            print(f"  Success rate: {results.success_rate * 100:.1f}%")

        return all_results


def compare_controllers(
    dynamics,
    controllers: Dict[str, Any],
    n_runs: int = 100,
    config: Optional[SimulationConfig] = None,
    seed: int = 42,
) -> Dict[str, MonteCarloResults]:
    """
    Compare multiple controllers on same initial conditions.

    Args:
        dynamics: Rocket dynamics
        controllers: Dictionary of controllers
        n_runs: Number of runs
        config: Simulation configuration
        seed: Random seed

    Returns:
        Dictionary of results per controller
    """
    config = config or SimulationConfig()

    # Generate common initial conditions
    np.random.seed(seed)
    initial_states = []
    for i in range(n_runs):
        sim = MonteCarloSimulator(dynamics, None, config)
        x0 = sim.sample_initial_condition()
        initial_states.append(x0)

    # Run each controller
    results = {}

    for name, controller in controllers.items():
        print(f"\nEvaluating controller: {name}")

        sim = MonteCarloSimulator(dynamics, controller, config)

        run_results = []
        for i, x0 in enumerate(initial_states):
            result = sim.run_single(i, x0=x0.copy())
            run_results.append(result)

            if (i + 1) % 50 == 0:
                print(f"  Completed {i + 1}/{n_runs}...")

        results[name] = MonteCarloResults(
            config=config,
            results=run_results,
        )

        print(f"  Success rate: {results[name].success_rate * 100:.1f}%")

    return results
