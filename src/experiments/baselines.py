"""
Baseline Controllers for Comparison Studies

Implements baseline controllers for benchmarking GP-MPC:
1. LQR - Linear Quadratic Regulator (linearized dynamics)
2. PID - Proportional-Integral-Derivative controller
3. Tube-MPC - Robust MPC without learning
4. Nominal MPC - MPC without GP correction
5. Open-loop - Pre-computed trajectory tracking

These baselines enable fair comparison of GP-MPC improvements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import linalg


@dataclass
class LQRConfig:
    """Configuration for LQR controller."""

    # Cost weights
    Q_pos: float = 10.0  # Position weight
    Q_vel: float = 1.0  # Velocity weight
    Q_mass: float = 0.01  # Mass weight (fuel penalty)
    R_thrust: float = 0.01  # Thrust magnitude weight
    R_gimbal: float = 0.1  # Gimbal angle weight

    # Reference point for linearization
    linearize_at_hover: bool = True

    # Gain scheduling
    use_gain_scheduling: bool = False


@dataclass
class LQRSolution:
    """Solution from LQR controller."""

    u0: NDArray  # Control input
    K: NDArray  # Feedback gain
    cost_to_go: float = 0.0


class LQRController:
    """
    Linear Quadratic Regulator for rocket landing.

    Linearizes dynamics at hover condition and computes infinite-horizon
    LQR gains. Simple but limited to near-hover operation.

    Example:
        >>> lqr = LQRController(dynamics)
        >>> lqr.initialize(x0, x_target)
        >>> u = lqr.solve(x).u0
    """

    def __init__(
        self,
        dynamics,
        config: Optional[LQRConfig] = None,
    ):
        """
        Initialize LQR controller.

        Args:
            dynamics: Rocket dynamics model
            config: LQR configuration
        """
        self.dynamics = dynamics
        self.config = config or LQRConfig()

        self.n_x = getattr(dynamics, "n_x", 7)
        self.n_u = getattr(dynamics, "n_u", 3)

        self.x_target: Optional[NDArray] = None
        self.K: Optional[NDArray] = None
        self.u_hover: Optional[NDArray] = None
        self._initialized = False

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:
        """
        Initialize controller and compute LQR gains.

        Args:
            x0: Initial state
            x_target: Target state
        """
        self.x_target = x_target.copy()

        # Hover control
        g = getattr(self.dynamics.params, "g", 1.0) if hasattr(self.dynamics, "params") else 1.0

        mass = x0[0]
        self.u_hover = np.array([mass * g, 0.0, 0.0])

        # Linearize dynamics
        A, B = self._linearize(x_target, self.u_hover)

        # Build cost matrices
        Q = self._build_Q()
        R = self._build_R()

        # Solve discrete-time algebraic Riccati equation
        try:
            P = linalg.solve_discrete_are(A, B, Q, R)
            self.K = np.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        except Exception:
            # Fallback to simple hand-tuned gains
            # Coordinate system: x (state[1]) = altitude, vx (state[4]) = vertical velocity
            self.K = np.zeros((self.n_u, self.n_x))
            # Altitude control: thrust responds to altitude error and vertical velocity
            self.K[0, 1] = -1.0  # Negative altitude error -> more thrust
            self.K[0, 4] = -2.0  # Negative vertical velocity -> more thrust
            # Horizontal control - aggressive gains to steer toward origin
            self.K[1, 2] = 0.5  # y position error -> thrust_y
            self.K[1, 5] = 1.0  # vy -> thrust_y
            self.K[2, 3] = 0.5  # z position error -> thrust_z
            self.K[2, 6] = 1.0  # vz -> thrust_z

        self._initialized = True

    def _linearize(
        self,
        x: NDArray,
        u: NDArray,
        eps: float = 1e-6,
    ) -> Tuple[NDArray, NDArray]:
        """Linearize dynamics at given point."""
        dt = 0.1  # Discretization time step

        A = np.zeros((self.n_x, self.n_x))
        B = np.zeros((self.n_x, self.n_u))

        x_nom = self.dynamics.step(x, u, dt)

        # Compute A matrix
        for i in range(self.n_x):
            x_pert = x.copy()
            x_pert[i] += eps
            x_next = self.dynamics.step(x_pert, u, dt)
            A[:, i] = (x_next - x_nom) / eps

        # Compute B matrix
        for i in range(self.n_u):
            u_pert = u.copy()
            u_pert[i] += eps
            x_next = self.dynamics.step(x, u_pert, dt)
            B[:, i] = (x_next - x_nom) / eps

        return A, B

    def _build_Q(self) -> NDArray:
        """Build state cost matrix."""
        cfg = self.config
        Q = np.diag(
            [
                cfg.Q_mass,  # mass
                cfg.Q_pos,  # x
                cfg.Q_pos,  # y
                cfg.Q_pos,  # z
                cfg.Q_vel,  # vx
                cfg.Q_vel,  # vy
                cfg.Q_vel,  # vz
            ][: self.n_x]
        )
        return Q

    def _build_R(self) -> NDArray:
        """Build control cost matrix."""
        cfg = self.config
        R = np.diag(
            [
                cfg.R_thrust,  # T
                cfg.R_gimbal,  # gimbal_x
                cfg.R_gimbal,  # gimbal_y
            ][: self.n_u]
        )
        return R

    def solve(self, x: NDArray, x_target: Optional[NDArray] = None) -> LQRSolution:
        """
        Compute LQR control.

        Args:
            x: Current state
            x_target: Target state (uses initialized if None)

        Returns:
            LQR solution
        """
        if not self._initialized:
            raise RuntimeError("Controller not initialized")

        target = x_target if x_target is not None else self.x_target

        # State error
        dx = x - target

        # LQR control
        du = -self.K @ dx

        # Add hover feedforward
        u = self.u_hover.copy()
        u += du

        # Clip to bounds
        if hasattr(self.dynamics, "params"):
            params = self.dynamics.params
            u[0] = np.clip(u[0], getattr(params, "T_min", 0.1), getattr(params, "T_max", 10.0))
            gimbal_max = getattr(params, "gimbal_max", 0.5)
            u[1:] = np.clip(u[1:], -gimbal_max, gimbal_max)
        else:
            u[0] = np.clip(u[0], 0.1, 10.0)
            u[1:] = np.clip(u[1:], -0.5, 0.5)

        return LQRSolution(u0=u, K=self.K)


@dataclass
class PIDConfig:
    """Configuration for PID controller."""

    # Altitude controller gains
    Kp_z: float = 2.0
    Ki_z: float = 0.1
    Kd_z: float = 1.5

    # Horizontal position gains - aggressive to steer toward origin
    Kp_xy: float = 1.0
    Ki_xy: float = 0.05
    Kd_xy: float = 0.8

    # Integral limits
    max_integral: float = 10.0

    # Control limits
    thrust_min: float = 0.3
    thrust_max: float = 5.0
    gimbal_max: float = 0.35


@dataclass
class PIDSolution:
    """Solution from PID controller."""

    u0: NDArray
    error: NDArray = field(default_factory=lambda: np.zeros(3))


class PIDController:
    """
    PID controller for rocket landing.

    Simple cascaded PID: outer loop for position, inner loop for velocity.
    Limited performance but robust and simple to tune.

    Example:
        >>> pid = PIDController(dynamics)
        >>> pid.initialize(x0, x_target)
        >>> u = pid.solve(x).u0
    """

    def __init__(
        self,
        dynamics,
        config: Optional[PIDConfig] = None,
    ):
        """Initialize PID controller."""
        self.dynamics = dynamics
        self.config = config or PIDConfig()

        self.n_x = getattr(dynamics, "n_x", 7)
        self.n_u = getattr(dynamics, "n_u", 3)

        self.x_target: Optional[NDArray] = None
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.dt = 0.1

        # Get gravity
        if hasattr(dynamics, "params"):
            self.g = getattr(dynamics.params, "g", 1.0)
        else:
            self.g = 1.0

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:  # noqa: ARG002
        """Initialize controller."""
        self.x_target = x_target.copy()
        self.integral_error = np.zeros(3)
        self.prev_error = np.zeros(3)

    def solve(self, x: NDArray, x_target: Optional[NDArray] = None) -> PIDSolution:
        """Compute PID control.

        Note:
            Coordinate system: gravity is in -x direction, so:
            - pos[0] (x) = altitude (vertical position)
            - pos[1], pos[2] (y, z) = horizontal position
            - vel[0] (vx) = vertical velocity
            - vel[1], vel[2] (vy, vz) = horizontal velocity
        """
        target = x_target if x_target is not None else self.x_target
        cfg = self.config

        # Position and velocity
        pos = x[1:4]  # [altitude, y, z]
        vel = x[4:7]  # [v_vert, v_y, v_z]

        pos_target = target[1:4]
        vel_target = target[4:7]

        pos_error = pos_target - pos
        vel_error = vel_target - vel

        # Update integral
        self.integral_error += pos_error * self.dt
        self.integral_error = np.clip(self.integral_error, -cfg.max_integral, cfg.max_integral)

        # Derivative (using velocity error)
        error_derivative = vel_error

        # Altitude control (x-axis, since gravity is in -x)
        # pos_error[0] = altitude error, error_derivative[0] = vertical velocity error
        thrust_vert = cfg.Kp_z * pos_error[0] + cfg.Ki_z * self.integral_error[0] + cfg.Kd_z * error_derivative[0]

        # Add gravity compensation
        mass = x[0]
        thrust_x = mass * (self.g + thrust_vert)
        thrust_x = np.clip(thrust_x, cfg.thrust_min, cfg.thrust_max)

        # Horizontal control (y and z axes) - direct thrust components
        # pos_error[1] = y error, pos_error[2] = z error
        thrust_y = cfg.Kp_xy * pos_error[1] + cfg.Ki_xy * self.integral_error[1] + cfg.Kd_xy * error_derivative[1]
        thrust_z = cfg.Kp_xy * pos_error[2] + cfg.Ki_xy * self.integral_error[2] + cfg.Kd_xy * error_derivative[2]

        # Scale horizontal thrust by mass
        thrust_y = thrust_y * mass
        thrust_z = thrust_z * mass

        # Clip horizontal thrust components
        thrust_y = np.clip(thrust_y, -cfg.thrust_max * 0.3, cfg.thrust_max * 0.3)
        thrust_z = np.clip(thrust_z, -cfg.thrust_max * 0.3, cfg.thrust_max * 0.3)

        u = np.array([thrust_x, thrust_y, thrust_z])

        self.prev_error = pos_error.copy()

        return PIDSolution(u0=u, error=pos_error)


class NominalMPCWrapper:
    """
    Wrapper to use NominalMPC as a baseline (without GP).

    Simply wraps the existing NominalMPC for consistent interface.
    """

    def __init__(self, mpc_controller):
        """Wrap existing MPC controller."""
        self.mpc = mpc_controller
        self.x_target = None

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:
        """Initialize controller."""
        self.x_target = x_target.copy()
        if hasattr(self.mpc, "initialize"):
            self.mpc.initialize(x0, x_target)

    def solve(self, x: NDArray, x_target: Optional[NDArray] = None) -> Any:
        """Solve MPC."""
        target = x_target if x_target is not None else self.x_target

        if hasattr(self.mpc, "step"):
            return self.mpc.step(x)
        else:
            return self.mpc.solve(x, target)


class OpenLoopController:
    """
    Open-loop trajectory tracking controller.

    Tracks a pre-computed reference trajectory. No feedback correction.
    Used to show importance of closed-loop control.
    """

    def __init__(
        self,
        reference_states: NDArray,
        reference_controls: NDArray,
        dt: float = 0.1,
    ):
        """
        Initialize open-loop controller.

        Args:
            reference_states: Reference state trajectory (T+1, n_x)
            reference_controls: Reference control trajectory (T, n_u)
            dt: Time step
        """
        self.X_ref = reference_states
        self.U_ref = reference_controls
        self.dt = dt
        self.t = 0.0
        self.step_idx = 0

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:  # noqa: ARG002
        """Reset controller."""
        self.t = 0.0
        self.step_idx = 0

    def solve(self, x: NDArray, x_target: Optional[NDArray] = None) -> Any:  # noqa: ARG002
        """Get next control from reference."""
        idx = min(self.step_idx, len(self.U_ref) - 1)
        u = self.U_ref[idx].copy()

        self.step_idx += 1
        self.t += self.dt

        class Solution:
            def __init__(self, u):
                self.u0 = u

        return Solution(u)


class TubeMPCWrapper:
    """
    Wrapper for Tube-MPC as a baseline.

    Tube-MPC uses robust constraints but no learning.
    """

    def __init__(self, tube_mpc_controller):
        """Wrap TubeMPC controller."""
        self.tube_mpc = tube_mpc_controller
        self.x_target = None

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:
        """Initialize controller."""
        self.x_target = x_target.copy()
        if hasattr(self.tube_mpc, "initialize"):
            self.tube_mpc.initialize(x0, x_target)

    def solve(self, x: NDArray, x_target: Optional[NDArray] = None) -> Any:
        """Solve Tube-MPC."""
        target = x_target if x_target is not None else self.x_target
        return self.tube_mpc.solve(x, target)


def create_baseline_controllers(
    dynamics,
    mpc_config=None,
    reference_trajectory: Optional[Tuple[NDArray, NDArray]] = None,
) -> Dict[str, Any]:
    """
    Create all baseline controllers for comparison.

    Args:
        dynamics: Rocket dynamics
        mpc_config: MPC configuration (for nominal MPC)
        reference_trajectory: (states, controls) for open-loop

    Returns:
        Dictionary of baseline controllers
    """
    baselines = {}

    # LQR
    baselines["LQR"] = LQRController(dynamics)

    # PID
    baselines["PID"] = PIDController(dynamics)

    # Nominal MPC (if config provided)
    if mpc_config is not None:
        try:
            from mpc import NominalMPC3DoF  # noqa: PLC0415

            nominal_mpc = NominalMPC3DoF(dynamics, mpc_config)
            baselines["Nominal_MPC"] = NominalMPCWrapper(nominal_mpc)
        except ImportError:
            pass

    # Open-loop (if reference provided)
    if reference_trajectory is not None:
        states, controls = reference_trajectory
        baselines["Open_Loop"] = OpenLoopController(states, controls)

    return baselines


@dataclass
class BaselineComparison:
    """Results from baseline comparison study."""

    controller_names: List[str]
    success_rates: Dict[str, float]
    fuel_means: Dict[str, float]
    fuel_stds: Dict[str, float]
    time_means: Dict[str, float]
    compute_means: Dict[str, float]

    def to_table(self) -> str:
        """Generate comparison table."""
        lines = [
            "Controller Comparison Results",
            "=" * 80,
            f"{'Controller':<20} {'Success%':>10} {'Fuel (kg)':>15} {'Time (s)':>12} {'Compute (ms)':>15}",
            "-" * 80,
        ]

        for name in self.controller_names:
            success = self.success_rates.get(name, 0) * 100
            fuel = self.fuel_means.get(name, 0)
            fuel_std = self.fuel_stds.get(name, 0)
            time = self.time_means.get(name, 0)
            compute = self.compute_means.get(name, 0)

            lines.append(f"{name:<20} {success:>9.1f}% {fuel:>7.3f}±{fuel_std:<6.3f} {time:>11.2f} {compute:>14.2f}")

        lines.append("=" * 80)
        return "\n".join(lines)
