"""
Backup Controller for Predictive Safety Filter

The backup controller provides a stabilizing control policy that
guarantees safety from within the backup invariant set.

Key requirements:
1. Must stabilize the system to a safe equilibrium
2. Must satisfy all constraints when applied
3. Should be computationally simple (for real-time verification)

Common choices:
- LQR: Linear Quadratic Regulator around hover
- Tube controller: Robust feedback for uncertainty
- Emergency landing: Maximum braking trajectory

Reference:
    Wabersich, K. P., & Zeilinger, M. N. (2021). Predictive Safety Filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import solve_discrete_are

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class BackupControllerConfig:
    """Configuration for backup controller."""

    # LQR weights
    Q_diag: Optional[NDArray] = None  # State cost diagonal
    R_diag: Optional[NDArray] = None  # Control cost diagonal

    # Control limits
    u_min: Optional[NDArray] = None
    u_max: Optional[NDArray] = None

    # Discretization
    dt: float = 0.1

    # Type
    controller_type: str = "lqr"  # "lqr", "pd", "emergency"


class LQRBackupController:
    """
    LQR Backup Controller for rocket landing.

    Provides a stabilizing linear feedback controller:
        u = u_eq - K @ (x - x_eq)

    where K is the LQR gain computed around the hover equilibrium.

    Example:
        >>> backup = LQRBackupController(dynamics, config)
        >>> backup.compute_gains(x_eq, u_eq)
        >>>
        >>> # Get backup control
        >>> u_backup = backup.get_control(x_current)
    """

    def __init__(
        self,
        dynamics,
        config: Optional[BackupControllerConfig] = None,
    ):
        """
        Initialize LQR backup controller.

        Args:
            dynamics: Rocket dynamics model
            config: Controller configuration
        """
        self.dynamics = dynamics
        self.config = config or BackupControllerConfig()

        # Dimensions
        self.n_x = 14  # 6-DoF state
        self.n_u = 3  # Thrust vector

        # Equilibrium point (set by compute_gains)
        self.x_eq: Optional[NDArray] = None
        self.u_eq: Optional[NDArray] = None

        # LQR gain
        self.K: Optional[NDArray] = None
        self.P: Optional[NDArray] = None  # Value function matrix

        # Set up weight matrices
        self._setup_weights()

    def _setup_weights(self) -> None:
        """Set up Q and R matrices."""
        if self.config.Q_diag is not None:
            self.Q = np.diag(self.config.Q_diag)
        else:
            # Default: penalize position and velocity, less on attitude
            Q_diag = np.array(
                [
                    0.0,  # mass (don't penalize)
                    10.0,
                    10.0,
                    10.0,  # position
                    5.0,
                    5.0,
                    5.0,  # velocity
                    1.0,
                    2.0,
                    2.0,
                    1.0,  # quaternion (penalize qx, qy)
                    0.5,
                    0.5,
                    0.5,  # angular rate
                ]
            )
            self.Q = np.diag(Q_diag)

        if self.config.R_diag is not None:
            self.R = np.diag(self.config.R_diag)
        else:
            # Default: moderate control effort penalty
            self.R = np.diag([0.1, 0.1, 0.1])

    def compute_gains(
        self,
        x_eq: NDArray,
        u_eq: Optional[NDArray] = None,
    ) -> None:
        """
        Compute LQR gains around equilibrium.

        Args:
            x_eq: Equilibrium state (hover point)
            u_eq: Equilibrium control (if None, computed for hover)
        """
        self.x_eq = x_eq.copy()

        # Compute equilibrium control if not provided
        if u_eq is None:
            mass = x_eq[0]
            g = abs(self.dynamics.params.g_I[0] * self.dynamics.params.g0)
            self.u_eq = np.array([mass * g, 0.0, 0.0])  # Thrust to hover
        else:
            self.u_eq = u_eq.copy()

        # Get linearized dynamics
        A, B = self._linearize_at_equilibrium()

        # Solve discrete-time Riccati equation
        if HAS_SCIPY:
            try:
                # Discrete ARE
                self.P = solve_discrete_are(A, B, self.Q, self.R)

                # LQR gain: K = (R + B'PB)^{-1} B'PA
                BtP = B.T @ self.P
                self.K = np.linalg.solve(self.R + BtP @ B, BtP @ A)

            except Exception as e:
                print(f"LQR computation failed: {e}, using simple PD gains")
                self._compute_pd_gains()
        else:
            self._compute_pd_gains()

    def _linearize_at_equilibrium(self) -> Tuple[NDArray, NDArray]:
        """Get linearized dynamics at equilibrium."""
        # Use dynamics linearization if available
        if hasattr(self.dynamics, "linearize"):
            A, B = self.dynamics.linearize(self.x_eq, self.u_eq, self.config.dt)
        else:
            # Numerical linearization
            A, B = self._numerical_linearization()

        return A, B

    def _numerical_linearization(
        self,
        eps: float = 1e-6,
    ) -> Tuple[NDArray, NDArray]:
        """Compute linearization numerically."""
        A = np.zeros((self.n_x, self.n_x))
        B = np.zeros((self.n_x, self.n_u))

        # Nominal next state
        x_next_nom = self.dynamics.step(self.x_eq, self.u_eq, self.config.dt)

        # Perturb states
        for i in range(self.n_x):
            x_pert = self.x_eq.copy()
            x_pert[i] += eps
            x_next = self.dynamics.step(x_pert, self.u_eq, self.config.dt)
            A[:, i] = (x_next - x_next_nom) / eps

        # Perturb controls
        for i in range(self.n_u):
            u_pert = self.u_eq.copy()
            u_pert[i] += eps
            x_next = self.dynamics.step(self.x_eq, u_pert, self.config.dt)
            B[:, i] = (x_next - x_next_nom) / eps

        return A, B

    def _compute_pd_gains(self) -> None:
        """Compute simple PD gains as fallback."""
        # Simple diagonal gains
        self.K = np.zeros((self.n_u, self.n_x))

        # Position feedback to thrust direction
        self.K[0, 1] = 2.0  # x position -> Tx
        self.K[1, 2] = 2.0  # y position -> Ty
        self.K[2, 3] = 2.0  # z position -> Tz

        # Velocity feedback
        self.K[0, 4] = 1.0  # vx -> Tx
        self.K[1, 5] = 1.0  # vy -> Ty
        self.K[2, 6] = 1.0  # vz -> Tz

        # Simple P matrix
        self.P = self.Q.copy()

    def get_control(self, x: NDArray) -> NDArray:
        """
        Get backup control for state x.

        Args:
            x: Current state

        Returns:
            u: Backup control (saturated to limits)
        """
        if self.K is None:
            raise RuntimeError("Must call compute_gains() first")

        # LQR control
        dx = x - self.x_eq
        u = self.u_eq - self.K @ dx

        # Saturate to control limits
        if self.config.u_min is not None:
            u = np.maximum(u, self.config.u_min)
        if self.config.u_max is not None:
            u = np.minimum(u, self.config.u_max)

        return u

    def get_control_batch(self, X: NDArray) -> NDArray:
        """Get backup controls for multiple states."""
        return np.array([self.get_control(x) for x in X])

    def get_value_function(self, x: NDArray) -> float:
        """
        Get Lyapunov value V(x) = (x - x_eq)' P (x - x_eq).

        This can be used to define the backup invariant set.
        """
        if self.P is None:
            raise RuntimeError("Must call compute_gains() first")

        dx = x - self.x_eq
        return float(dx.T @ self.P @ dx)

    def simulate_backup(
        self,
        x0: NDArray,
        n_steps: int = 50,
    ) -> Tuple[NDArray, NDArray]:
        """
        Simulate backup controller from initial state.

        Args:
            x0: Initial state
            n_steps: Number of simulation steps

        Returns:
            X: State trajectory (n_steps+1, n_x)
            U: Control trajectory (n_steps, n_u)
        """
        X = np.zeros((n_steps + 1, self.n_x))
        U = np.zeros((n_steps, self.n_u))

        X[0] = x0

        for k in range(n_steps):
            U[k] = self.get_control(X[k])
            X[k + 1] = self.dynamics.step(X[k], U[k], self.config.dt)

        return X, U


class PDBackupController:
    """
    Simple PD backup controller.

    Provides a proportional-derivative feedback:
        u = Kp @ (x_target - x_pos) + Kd @ (0 - x_vel) + gravity_comp
    """

    def __init__(
        self,
        dynamics,
        Kp: float = 2.0,
        Kd: float = 1.0,
    ):
        """
        Initialize PD controller.

        Args:
            dynamics: Rocket dynamics
            Kp: Proportional gain
            Kd: Derivative gain
        """
        self.dynamics = dynamics
        self.Kp = Kp
        self.Kd = Kd

        self.n_x = 14
        self.n_u = 3

        # Target position (landing pad)
        self.x_target = np.zeros(3)

    def get_control(self, x: NDArray) -> NDArray:
        """Get PD control."""
        mass = x[0]
        pos = x[1:4]
        vel = x[4:7]

        # Gravity compensation
        g = abs(self.dynamics.params.g_I[0] * self.dynamics.params.g0)

        # PD feedback
        u = np.zeros(3)
        u[0] = mass * g + self.Kp * (self.x_target[0] - pos[0]) - self.Kd * vel[0]
        u[1] = self.Kp * (self.x_target[1] - pos[1]) - self.Kd * vel[1]
        u[2] = self.Kp * (self.x_target[2] - pos[2]) - self.Kd * vel[2]

        # Clamp thrust magnitude
        T_min = getattr(self.dynamics.params, "T_min", 0.5)
        T_max = getattr(self.dynamics.params, "T_max", 5.0)

        T_mag = np.linalg.norm(u)
        if T_mag < T_min:
            u = u / max(T_mag, 1e-6) * T_min
        elif T_mag > T_max:
            u = u / T_mag * T_max

        return u


class EmergencyBrakingController:
    """
    Emergency braking controller.

    Applies maximum deceleration in the direction opposite to velocity.
    Used as a last-resort backup for immediate stopping.
    """

    def __init__(self, dynamics):
        """Initialize emergency controller."""
        self.dynamics = dynamics
        self.T_max = getattr(dynamics.params, "T_max", 5.0)

    def get_control(self, x: NDArray) -> NDArray:
        """Get emergency braking control."""
        mass = x[0]
        vel = x[4:7]

        # Gravity compensation in body frame
        g = abs(self.dynamics.params.g_I[0] * self.dynamics.params.g0)

        # Velocity magnitude
        v_mag = np.linalg.norm(vel)

        if v_mag < 0.1:
            # Near hover - just compensate gravity
            return np.array([mass * g, 0.0, 0.0])

        # Maximum braking + gravity compensation
        # Thrust in direction opposite to velocity
        brake_dir = -vel / v_mag

        # Available thrust after gravity compensation
        T_brake = self.T_max - mass * g

        u = np.array([mass * g, 0.0, 0.0]) + T_brake * brake_dir if T_brake > 0 else np.array([self.T_max, 0.0, 0.0])

        return u


def create_backup_controller(
    dynamics,
    controller_type: str = "lqr",
    config: Optional[BackupControllerConfig] = None,
):
    """
    Factory function to create backup controller.

    Args:
        dynamics: Rocket dynamics
        controller_type: "lqr", "pd", or "emergency"
        config: Controller configuration

    Returns:
        Backup controller instance
    """
    config = config or BackupControllerConfig()
    config.controller_type = controller_type

    if controller_type == "lqr":
        return LQRBackupController(dynamics, config)
    elif controller_type == "pd":
        return PDBackupController(dynamics)
    elif controller_type == "emergency":
        return EmergencyBrakingController(dynamics)
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
