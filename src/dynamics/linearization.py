"""
Linearization Utilities for GP-MPC

Provides utilities for:
- Numerical Jacobian computation (finite differences)
- Jacobian verification
- Affine model construction
- Discretization of Jacobians
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional, Protocol, Tuple

import numpy as np
from numpy.typing import NDArray


class DynamicsProtocol(Protocol):
    """Protocol for dynamics classes."""

    def dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        """Continuous dynamics f(x, u)."""
        ...

    @property
    def n_state(self) -> int: ...

    @property
    def n_control(self) -> int: ...


class FiniteDifferenceMethod(Enum):
    """Finite difference methods for Jacobian approximation."""

    FORWARD = "forward"
    CENTRAL = "central"
    COMPLEX = "complex"  # Complex step derivative (high accuracy)


def numerical_jacobian_x(
    f: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    u: NDArray,
    eps: float = 1e-6,
    method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL,
) -> NDArray:
    """
    Compute state Jacobian ∂f/∂x numerically.

    Args:
        f: Dynamics function f(x, u) -> x_dot
        x: State vector (n_x,)
        u: Control vector (n_u,)
        eps: Perturbation size
        method: Finite difference method

    Returns:
        State Jacobian (n_x, n_x)
    """
    n_x = len(x)
    f0 = f(x, u)
    n_f = len(f0)

    A = np.zeros((n_f, n_x))

    if method == FiniteDifferenceMethod.FORWARD:
        for i in range(n_x):
            x_plus = x.copy()
            x_plus[i] += eps
            f_plus = f(x_plus, u)
            A[:, i] = (f_plus - f0) / eps

    elif method == FiniteDifferenceMethod.CENTRAL:
        for i in range(n_x):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            f_plus = f(x_plus, u)
            f_minus = f(x_minus, u)
            A[:, i] = (f_plus - f_minus) / (2 * eps)

    elif method == FiniteDifferenceMethod.COMPLEX:
        # Complex step derivative: Im(f(x + i*eps)) / eps
        # Only works for functions that can handle complex inputs
        x_complex = x.astype(complex)
        for i in range(n_x):
            x_pert = x_complex.copy()
            x_pert[i] += 1j * eps
            f_pert = f(x_pert.real, u)  # Fall back to real for now  # noqa: F841
            # For true complex step, the function must support complex arithmetic
            A[:, i] = (f(x_pert.real + eps * np.eye(n_x)[i], u) - f0) / eps

    return A


def numerical_jacobian_u(
    f: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    u: NDArray,
    eps: float = 1e-6,
    method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL,
) -> NDArray:
    """
    Compute control Jacobian ∂f/∂u numerically.

    Args:
        f: Dynamics function f(x, u) -> x_dot
        x: State vector (n_x,)
        u: Control vector (n_u,)
        eps: Perturbation size
        method: Finite difference method

    Returns:
        Control Jacobian (n_x, n_u)
    """
    n_u = len(u)
    f0 = f(x, u)
    n_f = len(f0)

    B = np.zeros((n_f, n_u))

    if method == FiniteDifferenceMethod.FORWARD:
        for i in range(n_u):
            u_plus = u.copy()
            u_plus[i] += eps
            f_plus = f(x, u_plus)
            B[:, i] = (f_plus - f0) / eps

    elif method == FiniteDifferenceMethod.CENTRAL:
        for i in range(n_u):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[i] += eps
            u_minus[i] -= eps
            f_plus = f(x, u_plus)
            f_minus = f(x, u_minus)
            B[:, i] = (f_plus - f_minus) / (2 * eps)

    return B


def numerical_jacobians(
    f: Callable[[NDArray, NDArray], NDArray],
    x: NDArray,
    u: NDArray,
    eps: float = 1e-6,
    method: FiniteDifferenceMethod = FiniteDifferenceMethod.CENTRAL,
) -> Tuple[NDArray, NDArray]:
    """
    Compute both Jacobians numerically.

    Args:
        f: Dynamics function f(x, u) -> x_dot
        x: State vector (n_x,)
        u: Control vector (n_u,)
        eps: Perturbation size
        method: Finite difference method

    Returns:
        A: State Jacobian (n_x, n_x)
        B: Control Jacobian (n_x, n_u)
    """
    A = numerical_jacobian_x(f, x, u, eps, method)
    B = numerical_jacobian_u(f, x, u, eps, method)
    return A, B


def verify_jacobians(
    dynamics: DynamicsProtocol,
    x: NDArray,
    u: NDArray,
    A_analytical: NDArray,
    B_analytical: NDArray,
    eps: float = 1e-6,
    rtol: float = 1e-4,
    atol: float = 1e-6,
    verbose: bool = True,
) -> Tuple[bool, dict]:
    """
    Verify analytical Jacobians against numerical approximation.

    Args:
        dynamics: Dynamics object with dynamics() method
        x: State vector
        u: Control vector
        A_analytical: Analytical state Jacobian
        B_analytical: Analytical control Jacobian
        eps: Perturbation for numerical differentiation
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Print detailed comparison

    Returns:
        (is_valid, info_dict)
    """
    A_numerical, B_numerical = numerical_jacobians(dynamics.dynamics, x, u, eps, FiniteDifferenceMethod.CENTRAL)

    # Compute errors
    A_error = np.abs(A_analytical - A_numerical)
    B_error = np.abs(B_analytical - B_numerical)

    A_rel_error = A_error / (np.abs(A_numerical) + atol)
    B_rel_error = B_error / (np.abs(B_numerical) + atol)

    A_max_error = np.max(A_error)
    B_max_error = np.max(B_error)
    A_max_rel = np.max(A_rel_error)
    B_max_rel = np.max(B_rel_error)

    A_valid = np.allclose(A_analytical, A_numerical, rtol=rtol, atol=atol)
    B_valid = np.allclose(B_analytical, B_numerical, rtol=rtol, atol=atol)

    if verbose:
        print("Jacobian Verification:")
        print("  State Jacobian (A):")
        print(f"    Max absolute error: {A_max_error:.2e}")
        print(f"    Max relative error: {A_max_rel:.2e}")
        print(f"    Valid: {A_valid}")
        print("  Control Jacobian (B):")
        print(f"    Max absolute error: {B_max_error:.2e}")
        print(f"    Max relative error: {B_max_rel:.2e}")
        print(f"    Valid: {B_valid}")

    info = {
        "A_numerical": A_numerical,
        "B_numerical": B_numerical,
        "A_error": A_error,
        "B_error": B_error,
        "A_max_error": A_max_error,
        "B_max_error": B_max_error,
        "A_max_rel_error": A_max_rel,
        "B_max_rel_error": B_max_rel,
        "A_valid": A_valid,
        "B_valid": B_valid,
    }

    return A_valid and B_valid, info


class AffineModel:
    """
    Affine approximation of dynamics around a reference point.

    x_{k+1} ≈ A @ x_k + B @ u_k + c

    where:
        A = ∂F/∂x |_{x_ref, u_ref}
        B = ∂F/∂u |_{x_ref, u_ref}
        c = F(x_ref, u_ref) - A @ x_ref - B @ u_ref
    """

    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        c: NDArray,
        x_ref: Optional[NDArray] = None,
        u_ref: Optional[NDArray] = None,
    ):
        """
        Initialize affine model.

        Args:
            A: State Jacobian (n_x, n_x)
            B: Control Jacobian (n_x, n_u)
            c: Affine offset (n_x,)
            x_ref: Reference state (for error bounds)
            u_ref: Reference control (for error bounds)
        """
        self.A = A
        self.B = B
        self.c = c
        self.x_ref = x_ref
        self.u_ref = u_ref

        self.n_state = A.shape[0]
        self.n_control = B.shape[1]

    def predict(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Predict next state using affine model.

        Args:
            x: Current state (n_x,)
            u: Control input (n_u,)

        Returns:
            Predicted next state (n_x,)
        """
        return self.A @ x + self.B @ u + self.c

    def __call__(self, x: NDArray, u: NDArray) -> NDArray:
        """Alias for predict()."""
        return self.predict(x, u)

    @classmethod
    def from_dynamics(
        cls,
        dynamics: DynamicsProtocol,
        x_ref: NDArray,
        u_ref: NDArray,
        dt: float,
    ) -> "AffineModel":
        """
        Create affine model by linearizing dynamics.

        Args:
            dynamics: Dynamics object
            x_ref: Reference state
            u_ref: Reference control
            dt: Timestep

        Returns:
            AffineModel instance
        """
        # Get Jacobians (assuming dynamics has these methods)
        A_c = dynamics.jacobian_x(x_ref, u_ref)
        B_c = dynamics.jacobian_u(x_ref, u_ref)

        # Discretize
        A_d = np.eye(dynamics.n_state) + A_c * dt
        B_d = B_c * dt

        # Compute affine term
        x_next = dynamics.step(x_ref, u_ref, dt)
        c = x_next - A_d @ x_ref - B_d @ u_ref

        return cls(A_d, B_d, c, x_ref, u_ref)


def discretize_jacobians(
    A_c: NDArray,
    B_c: NDArray,
    dt: float,
    method: str = "euler",
) -> Tuple[NDArray, NDArray]:
    """
    Discretize continuous-time Jacobians.

    Args:
        A_c: Continuous state Jacobian (n_x, n_x)
        B_c: Continuous control Jacobian (n_x, n_u)
        dt: Timestep
        method: Discretization method
            - "euler": First-order Euler (A_d = I + A_c*dt)
            - "zoh": Zero-order hold (exact for LTI)
            - "taylor2": Second-order Taylor

    Returns:
        A_d: Discrete state Jacobian
        B_d: Discrete control Jacobian
    """
    n = A_c.shape[0]

    if method == "euler":
        A_d = np.eye(n) + A_c * dt
        B_d = B_c * dt

    elif method == "zoh":
        # Zero-order hold: A_d = exp(A_c * dt)
        # B_d = A_c^{-1} (A_d - I) B_c (if A_c invertible)
        from scipy.linalg import expm  # noqa: PLC0415

        A_d = expm(A_c * dt)

        # Use series approximation for B_d if A_c is nearly singular
        if np.abs(np.linalg.det(A_c)) > 1e-10:
            A_c_inv = np.linalg.inv(A_c)
            B_d = A_c_inv @ (A_d - np.eye(n)) @ B_c
        else:
            # Series: B_d ≈ dt * (I + A_c*dt/2 + ...) @ B_c
            B_d = dt * (np.eye(n) + A_c * dt / 2) @ B_c

    elif method == "taylor2":
        # Second-order Taylor: A_d = I + A_c*dt + 0.5*A_c²*dt²
        A_d = np.eye(n) + A_c * dt + 0.5 * A_c @ A_c * dt**2
        B_d = (np.eye(n) + A_c * dt / 2) @ B_c * dt

    else:
        raise ValueError(f"Unknown discretization method: {method}")

    return A_d, B_d


def compute_discrete_affine_model(
    f: Callable[[NDArray, NDArray], NDArray],  # noqa: ARG001
    F: Callable[[NDArray, NDArray, float], NDArray],
    x_ref: NDArray,
    u_ref: NDArray,
    dt: float,
    eps: float = 1e-6,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute discrete affine model numerically.

    x_{k+1} ≈ A_d @ x_k + B_d @ u_k + c

    Args:
        f: Continuous dynamics f(x, u)
        F: Discrete dynamics F(x, u, dt)
        x_ref: Reference state
        u_ref: Reference control
        dt: Timestep
        eps: Perturbation for numerical differentiation

    Returns:
        A_d: Discrete state Jacobian
        B_d: Discrete control Jacobian
        c: Affine offset
    """

    # Numerical Jacobians of discrete dynamics
    def F_xu(x, u):
        return F(x, u, dt)

    A_d = numerical_jacobian_x(F_xu, x_ref, u_ref, eps)
    B_d = numerical_jacobian_u(F_xu, x_ref, u_ref, eps)

    # Affine offset
    x_next_ref = F(x_ref, u_ref, dt)
    c = x_next_ref - A_d @ x_ref - B_d @ u_ref

    return A_d, B_d, c


def trajectory_jacobians(
    dynamics: DynamicsProtocol,
    x_traj: NDArray,
    u_traj: NDArray,
    dt: float,
) -> Tuple[list, list, list]:
    """
    Compute Jacobians along a trajectory.

    Args:
        dynamics: Dynamics object
        x_traj: State trajectory (N+1, n_x)
        u_traj: Control trajectory (N, n_u)
        dt: Timestep

    Returns:
        A_list: List of state Jacobians [A_0, A_1, ..., A_{N-1}]
        B_list: List of control Jacobians [B_0, B_1, ..., B_{N-1}]
        c_list: List of affine offsets [c_0, c_1, ..., c_{N-1}]
    """
    N = len(u_traj)

    A_list = []
    B_list = []
    c_list = []

    for k in range(N):
        x_k = x_traj[k]
        u_k = u_traj[k]
        x_next = x_traj[k + 1]

        # Continuous Jacobians
        A_c = dynamics.jacobian_x(x_k, u_k)
        B_c = dynamics.jacobian_u(x_k, u_k)

        # Discretize
        A_d, B_d = discretize_jacobians(A_c, B_c, dt, method="euler")

        # Affine offset
        c = x_next - A_d @ x_k - B_d @ u_k

        A_list.append(A_d)
        B_list.append(B_d)
        c_list.append(c)

    return A_list, B_list, c_list
