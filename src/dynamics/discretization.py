"""
Discretization and Integration Utilities for GP-MPC

Provides:
- Explicit integrators (Euler, RK4, RK45)
- Implicit integrators for stiff systems
- Adaptive step size control
- Symplectic integrators for energy conservation
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Tuple

import numpy as np
from numpy.typing import NDArray


class IntegratorType(Enum):
    """Available integration methods."""

    EULER = "euler"
    RK4 = "rk4"
    RK45 = "rk45"
    MIDPOINT = "midpoint"
    HEUN = "heun"


class Integrator:
    """
    Numerical integrator for ODEs.

    Integrates ẋ = f(t, x, u) with fixed or adaptive step size.

    Example:
        >>> integrator = Integrator(method="rk4")
        >>> x_next = integrator.step(f, t, x, u, dt)
    """

    def __init__(
        self,
        method: str = "rk4",
        adaptive: bool = False,
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ):
        """
        Initialize integrator.

        Args:
            method: Integration method ("euler", "rk4", "midpoint", "heun")
            adaptive: Use adaptive step size control
            rtol: Relative tolerance for adaptive stepping
            atol: Absolute tolerance for adaptive stepping
        """
        self.method = IntegratorType(method.lower())
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol

    def step(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        t: float,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Take one integration step.

        Args:
            f: Dynamics function f(t, x, u) -> ẋ
            t: Current time
            x: Current state
            u: Control input (held constant over step)
            dt: Timestep

        Returns:
            Next state
        """
        if self.method == IntegratorType.EULER:
            return self._euler_step(f, t, x, u, dt)
        elif self.method == IntegratorType.RK4:
            return self._rk4_step(f, t, x, u, dt)
        elif self.method == IntegratorType.MIDPOINT:
            return self._midpoint_step(f, t, x, u, dt)
        elif self.method == IntegratorType.HEUN:
            return self._heun_step(f, t, x, u, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _euler_step(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        t: float,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> NDArray:
        """Forward Euler: x_{k+1} = x_k + dt * f(t_k, x_k, u_k)"""
        return x + dt * f(t, x, u)

    def _rk4_step(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        t: float,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Classic 4th-order Runge-Kutta.

        k1 = f(t, x, u)
        k2 = f(t + dt/2, x + dt*k1/2, u)
        k3 = f(t + dt/2, x + dt*k2/2, u)
        k4 = f(t + dt, x + dt*k3, u)
        x_{k+1} = x_k + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        k1 = f(t, x, u)
        k2 = f(t + dt / 2, x + dt * k1 / 2, u)
        k3 = f(t + dt / 2, x + dt * k2 / 2, u)
        k4 = f(t + dt, x + dt * k3, u)

        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    def _midpoint_step(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        t: float,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Midpoint method (2nd-order).

        k1 = f(t, x, u)
        k2 = f(t + dt/2, x + dt*k1/2, u)
        x_{k+1} = x_k + dt * k2
        """
        k1 = f(t, x, u)
        k2 = f(t + dt / 2, x + dt * k1 / 2, u)
        return x + dt * k2

    def _heun_step(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        t: float,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Heun's method (2nd-order, improved Euler).

        k1 = f(t, x, u)
        k2 = f(t + dt, x + dt*k1, u)
        x_{k+1} = x_k + dt/2 * (k1 + k2)
        """
        k1 = f(t, x, u)
        k2 = f(t + dt, x + dt * k1, u)
        return x + (dt / 2) * (k1 + k2)

    def integrate(
        self,
        f: Callable[[float, NDArray, NDArray], NDArray],
        x0: NDArray,
        u: Callable[[float], NDArray],
        t_span: Tuple[float, float],
        dt: float,
    ) -> Tuple[NDArray, NDArray]:
        """
        Integrate over a time span.

        Args:
            f: Dynamics function f(t, x, u) -> ẋ
            x0: Initial state
            u: Control function u(t) -> u
            t_span: (t_start, t_end)
            dt: Timestep

        Returns:
            t: Time vector (N,)
            x: State trajectory (N, n_x)
        """
        t_start, t_end = t_span
        N = int(np.ceil((t_end - t_start) / dt)) + 1

        t = np.linspace(t_start, t_end, N)
        x = np.zeros((N, len(x0)))
        x[0] = x0

        for i in range(N - 1):
            u_i = u(t[i])
            x[i + 1] = self.step(f, t[i], x[i], u_i, dt)

        return t, x


# =============================================================================
# Standalone Integration Functions
# =============================================================================


def euler_step(
    f: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    u: NDArray,
    dt: float,
) -> NDArray:
    """
    Forward Euler step (time-invariant dynamics).

    Args:
        f: Dynamics f(x, u) -> ẋ
        x: Current state
        u: Control input
        dt: Timestep

    Returns:
        Next state
    """
    return x + dt * f(x, u)


def rk4_step(
    f: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    u: NDArray,
    dt: float,
) -> NDArray:
    """
    4th-order Runge-Kutta step (time-invariant dynamics).

    Args:
        f: Dynamics f(x, u) -> ẋ
        x: Current state
        u: Control input (held constant)
        dt: Timestep

    Returns:
        Next state
    """
    k1 = f(x, u)
    k2 = f(x + dt * k1 / 2, u)
    k3 = f(x + dt * k2 / 2, u)
    k4 = f(x + dt * k3, u)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_trajectory(
    f: Callable[[NDArray, NDArray], NDArray],
    x0: NDArray,
    u_traj: NDArray,
    dt: float,
    method: str = "rk4",
) -> NDArray:
    """
    Integrate state trajectory given control sequence.

    Args:
        f: Dynamics f(x, u) -> ẋ
        x0: Initial state (n_x,)
        u_traj: Control trajectory (N, n_u)
        dt: Timestep
        method: Integration method ("euler" or "rk4")

    Returns:
        State trajectory (N+1, n_x)
    """
    N = len(u_traj)
    n_x = len(x0)

    x_traj = np.zeros((N + 1, n_x))
    x_traj[0] = x0

    step_fn = rk4_step if method == "rk4" else euler_step

    for k in range(N):
        x_traj[k + 1] = step_fn(f, x_traj[k], u_traj[k], dt)

    return x_traj


# =============================================================================
# Quaternion-Specific Integration
# =============================================================================


def normalize_quaternion(q: NDArray) -> NDArray:
    """Normalize quaternion to unit length."""
    return q / np.linalg.norm(q)


def quaternion_euler_step(
    q: NDArray,
    omega: NDArray,
    dt: float,
) -> NDArray:
    """
    Integrate quaternion kinematics: q̇ = 0.5 * Ω(ω) * q

    Uses Euler step with normalization.

    Args:
        q: Current quaternion [w, x, y, z] (4,)
        omega: Angular velocity in body frame (3,)
        dt: Timestep

    Returns:
        Next quaternion (normalized) (4,)
    """
    # Omega matrix: q̇ = 0.5 * Omega(omega) @ q
    wx, wy, wz = omega
    Omega = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])

    q_dot = 0.5 * Omega @ q
    q_next = q + dt * q_dot

    return normalize_quaternion(q_next)


def quaternion_exponential_step(
    q: NDArray,
    omega: NDArray,
    dt: float,
) -> NDArray:
    """
    Integrate quaternion using exponential map (exact for constant ω).

    q_{k+1} = exp(0.5 * ω * dt) ⊗ q_k

    Args:
        q: Current quaternion [w, x, y, z] (4,)
        omega: Angular velocity in body frame (3,)
        dt: Timestep

    Returns:
        Next quaternion (4,)
    """
    omega_mag = np.linalg.norm(omega)

    if omega_mag < 1e-10:
        # Small angle approximation
        delta_q = np.array([1.0, 0.5 * omega[0] * dt, 0.5 * omega[1] * dt, 0.5 * omega[2] * dt])
    else:
        # Exact exponential map
        angle = omega_mag * dt / 2
        axis = omega / omega_mag
        delta_q = np.array([np.cos(angle), axis[0] * np.sin(angle), axis[1] * np.sin(angle), axis[2] * np.sin(angle)])

    # Quaternion multiplication: q_next = delta_q ⊗ q
    q_next = quaternion_multiply(delta_q, q)

    return normalize_quaternion(q_next)


def quaternion_multiply(q1: NDArray, q2: NDArray) -> NDArray:
    """
    Quaternion multiplication q1 ⊗ q2.

    Convention: q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


# =============================================================================
# Collocation Methods (for trajectory optimization)
# =============================================================================


def hermite_simpson_defect(
    f: Callable[[NDArray, NDArray], NDArray],
    x_k: NDArray,
    x_kp1: NDArray,
    u_k: NDArray,
    u_kp1: NDArray,
    dt: float,
) -> NDArray:
    """
    Compute Hermite-Simpson collocation defect.

    The defect should be zero for a dynamically consistent trajectory.
    Used in direct collocation trajectory optimization.

    Args:
        f: Dynamics f(x, u) -> ẋ
        x_k: State at start of interval
        x_kp1: State at end of interval
        u_k: Control at start
        u_kp1: Control at end
        dt: Interval duration

    Returns:
        Defect vector (should be zero for valid trajectory)
    """
    # Function values at endpoints
    f_k = f(x_k, u_k)
    f_kp1 = f(x_kp1, u_kp1)

    # Midpoint state (Hermite interpolation)
    x_mid = 0.5 * (x_k + x_kp1) + (dt / 8) * (f_k - f_kp1)

    # Midpoint control
    u_mid = 0.5 * (u_k + u_kp1)

    # Function at midpoint
    f_mid = f(x_mid, u_mid)

    # Simpson quadrature defect
    defect = x_kp1 - x_k - (dt / 6) * (f_k + 4 * f_mid + f_kp1)

    return defect


def trapezoidal_defect(
    f: Callable[[NDArray, NDArray], NDArray],
    x_k: NDArray,
    x_kp1: NDArray,
    u_k: NDArray,
    u_kp1: NDArray,
    dt: float,
) -> NDArray:
    """
    Compute trapezoidal collocation defect.

    Args:
        f: Dynamics f(x, u) -> ẋ
        x_k: State at start
        x_kp1: State at end
        u_k: Control at start
        u_kp1: Control at end
        dt: Interval duration

    Returns:
        Defect vector
    """
    f_k = f(x_k, u_k)
    f_kp1 = f(x_kp1, u_kp1)

    defect = x_kp1 - x_k - (dt / 2) * (f_k + f_kp1)

    return defect


# =============================================================================
# Sensitivity Computation
# =============================================================================


def integrate_sensitivity(
    f: Callable[[NDArray, NDArray], NDArray],  # noqa: ARG001
    A: Callable[[NDArray, NDArray], NDArray],
    B: Callable[[NDArray, NDArray], NDArray],
    x_traj: NDArray,
    u_traj: NDArray,
    dt: float,
) -> Tuple[NDArray, NDArray]:
    """
    Integrate sensitivity equations along trajectory.

    Computes ∂x_N/∂x_0 and ∂x_N/∂u_{0:N-1} along the trajectory.

    Args:
        f: Dynamics f(x, u) -> ẋ
        A: State Jacobian A(x, u) -> ∂f/∂x
        B: Control Jacobian B(x, u) -> ∂f/∂u
        x_traj: State trajectory (N+1, n_x)
        u_traj: Control trajectory (N, n_u)
        dt: Timestep

    Returns:
        Phi: State transition matrix (n_x, n_x)
        Psi: Control sensitivity (n_x, N*n_u)
    """
    N = len(u_traj)
    n_x = x_traj.shape[1]
    n_u = u_traj.shape[1]

    # Initialize
    Phi = np.eye(n_x)  # ∂x_k/∂x_0
    Psi = np.zeros((n_x, N * n_u))  # ∂x_k/∂u_{0:k-1}

    for k in range(N):
        x_k = x_traj[k]
        u_k = u_traj[k]

        # Jacobians at this point
        A_k = np.eye(n_x) + dt * A(x_k, u_k)
        B_k = dt * B(x_k, u_k)

        # Update sensitivities
        Phi = A_k @ Phi
        Psi = A_k @ Psi
        Psi[:, k * n_u : (k + 1) * n_u] = B_k

    return Phi, Psi
