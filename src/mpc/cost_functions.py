"""
Cost Functions for MPC

Implements various cost formulations:
- Quadratic tracking cost
- Fuel-optimal cost (minimize ||T||)
- LQR terminal cost (stability guarantee)
- Minimum-time cost

The standard MPC cost is:
    J = Σ_{k=0}^{N-1} [l(x_k, u_k)] + V_f(x_N)

where:
    l(x, u) = stage cost (tracking + control effort)
    V_f(x) = terminal cost (often from LQR)

Reference:
    Rawlings, J. B., & Mayne, D. Q. (2009). Model Predictive Control:
    Theory and Design. Nob Hill Publishing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_continuous_are, solve_discrete_are

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class CostWeights:
    """
    Weight matrices for MPC cost function.

    Stage cost: l(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u
    Terminal cost: V_f(x) = (x - x_ref)^T P (x - x_ref)
    """

    # State weights (14x14 for 6-DoF)
    Q: Optional[NDArray] = None

    # Control weights (3x3)
    R: Optional[NDArray] = None

    # Terminal weight (14x14)
    P: Optional[NDArray] = None

    # Individual weight scalars (used if matrices not provided)
    w_position: float = 10.0
    w_velocity: float = 1.0
    w_attitude: float = 5.0
    w_omega: float = 0.1
    w_mass: float = 0.0  # Usually don't penalize mass
    w_thrust: float = 0.01
    w_fuel: float = 0.1  # Fuel consumption weight

    # Terminal cost multiplier
    terminal_weight: float = 10.0

    def __post_init__(self):
        """Initialize weight matrices if not provided."""
        if self.Q is None:
            self.Q = self._build_Q_matrix()
        if self.R is None:
            self.R = self._build_R_matrix()
        if self.P is None:
            self.P = self.terminal_weight * self.Q

    def _build_Q_matrix(self) -> NDArray:
        """Build state weight matrix."""
        Q = np.zeros((14, 14))

        # Mass (index 0)
        Q[0, 0] = self.w_mass

        # Position (indices 1-3)
        Q[1:4, 1:4] = self.w_position * np.eye(3)

        # Velocity (indices 4-6)
        Q[4:7, 4:7] = self.w_velocity * np.eye(3)

        # Attitude quaternion (indices 7-10)
        # Only penalize qx, qy (deviation from upright)
        Q[8, 8] = self.w_attitude  # qx
        Q[9, 9] = self.w_attitude  # qy

        # Angular velocity (indices 11-13)
        Q[11:14, 11:14] = self.w_omega * np.eye(3)

        return Q

    def _build_R_matrix(self) -> NDArray:
        """Build control weight matrix."""
        return self.w_thrust * np.eye(3)


# =============================================================================
# Stage Cost Functions
# =============================================================================


def quadratic_stage_cost(
    x: NDArray,
    u: NDArray,
    x_ref: NDArray,
    weights: CostWeights,
) -> float:
    """
    Quadratic stage cost.

    l(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u

    Args:
        x: Current state
        u: Current control
        x_ref: Reference state
        weights: Cost weights

    Returns:
        Stage cost value
    """
    x_err = x - x_ref
    return float(x_err.T @ weights.Q @ x_err + u.T @ weights.R @ u)


def fuel_optimal_stage_cost(
    x: NDArray,
    u: NDArray,
    x_ref: NDArray,
    weights: CostWeights,
) -> float:
    """
    Fuel-optimal stage cost.

    l(x, u) = w_fuel * ||T|| + (x - x_ref)^T Q (x - x_ref) + u^T R u

    Minimizing ||T|| is equivalent to minimizing fuel consumption
    since dm/dt = -alpha||T||.
    """
    x_err = x - x_ref
    T_mag = np.linalg.norm(u)

    return float(weights.w_fuel * T_mag + x_err.T @ weights.Q @ x_err + u.T @ weights.R @ u)


def tracking_cost(
    x: NDArray,
    u: NDArray,
    x_ref: NDArray,
    u_ref: NDArray,
    weights: CostWeights,
) -> float:
    """
    Reference tracking cost.

    l(x, u) = (x - x_ref)^T Q (x - x_ref) + (u - u_ref)^T R (u - u_ref)

    Useful for tracking a pre-computed trajectory from SCVX.
    """
    x_err = x - x_ref
    u_err = u - u_ref
    return float(x_err.T @ weights.Q @ x_err + u_err.T @ weights.R @ u_err)


# =============================================================================
# Terminal Cost (LQR)
# =============================================================================


def compute_lqr_terminal_cost(
    A: NDArray,
    B: NDArray,
    Q: NDArray,
    R: NDArray,
    discrete: bool = True,
) -> NDArray:
    """
    Compute LQR terminal cost matrix P.

    Solves the discrete (or continuous) algebraic Riccati equation:

    Discrete: P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
    Continuous: A^T P + P A - P B R^{-1} B^T P + Q = 0

    The terminal cost V_f(x) = x^T P x provides a stability guarantee
    when the terminal region is control-invariant under LQR.

    Args:
        A: System matrix (n_x, n_x)
        B: Input matrix (n_x, n_u)
        Q: State cost matrix (n_x, n_x)
        R: Control cost matrix (n_u, n_u)
        discrete: Whether system is discrete-time

    Returns:
        P: Terminal cost matrix (n_x, n_x)
    """
    P = solve_discrete_are(A, B, Q, R) if discrete else solve_continuous_are(A, B, Q, R)

    return P


def compute_lqr_gain(
    A: NDArray,
    B: NDArray,
    Q: NDArray,
    R: NDArray,
    discrete: bool = True,
) -> Tuple[NDArray, NDArray]:
    """
    Compute LQR gain and terminal cost.

    Args:
        A, B, Q, R: System and cost matrices
        discrete: Whether discrete-time

    Returns:
        K: LQR gain (n_u, n_x)
        P: Terminal cost matrix (n_x, n_x)
    """
    P = compute_lqr_terminal_cost(A, B, Q, R, discrete)

    if discrete:  # noqa: SIM108
        # K = (R + B^T P B)^{-1} B^T P A
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    else:
        # K = R^{-1} B^T P
        K = np.linalg.solve(R, B.T @ P)

    return K, P


class LQRTerminalCost:
    """
    LQR-based terminal cost with linearization.

    Computes terminal cost P around a linearization point,
    then evaluates V_f(x) = (x - x_eq)^T P (x - x_eq).
    """

    def __init__(
        self,
        dynamics_linearizer: Callable[[NDArray, NDArray, float], Tuple[NDArray, NDArray]],
        Q: NDArray,
        R: NDArray,
        dt: float = 0.1,
    ):
        """
        Initialize LQR terminal cost.

        Args:
            dynamics_linearizer: Function (x, u, dt) -> (A_d, B_d)
            Q: State cost matrix
            R: Control cost matrix
            dt: Discretization timestep
        """
        self.linearizer = dynamics_linearizer
        self.Q = Q
        self.R = R
        self.dt = dt

        # Cached values
        self._x_eq: Optional[NDArray] = None
        self._u_eq: Optional[NDArray] = None
        self._P: Optional[NDArray] = None
        self._K: Optional[NDArray] = None

    def update_linearization(
        self,
        x_eq: NDArray,
        u_eq: NDArray,
    ) -> None:
        """
        Update linearization point and recompute LQR.

        Args:
            x_eq: Equilibrium state
            u_eq: Equilibrium control
        """
        self._x_eq = x_eq.copy()
        self._u_eq = u_eq.copy()

        # Get linearized dynamics
        A_d, B_d = self.linearizer(x_eq, u_eq, self.dt)

        # Compute LQR
        try:
            self._K, self._P = compute_lqr_gain(A_d, B_d, self.Q, self.R, discrete=True)
        except Exception as e:
            # Fallback to simple terminal cost if Riccati fails
            print(f"LQR computation failed: {e}, using Q as terminal cost")
            self._P = 10.0 * self.Q
            self._K = np.zeros((self.R.shape[0], self.Q.shape[0]))

    def evaluate(self, x: NDArray) -> float:
        """
        Evaluate terminal cost.

        V_f(x) = (x - x_eq)^T P (x - x_eq)
        """
        if self._P is None:
            raise RuntimeError("Must call update_linearization first")

        x_err = x - self._x_eq
        return float(x_err.T @ self._P @ x_err)

    def get_lqr_control(self, x: NDArray) -> NDArray:
        """
        Get LQR feedback control.

        u = u_eq - K (x - x_eq)
        """
        if self._K is None:
            raise RuntimeError("Must call update_linearization first")

        x_err = x - self._x_eq
        return self._u_eq - self._K @ x_err

    @property
    def P(self) -> Optional[NDArray]:
        """Terminal cost matrix."""
        return self._P

    @property
    def K(self) -> Optional[NDArray]:
        """LQR gain."""
        return self._K


# =============================================================================
# CasADi Cost Functions
# =============================================================================

if HAS_CASADI:

    class CasADiCostFunction:
        """
        CasADi symbolic cost functions for MPC.
        """

        def __init__(self, weights: CostWeights):
            """
            Initialize with weights.

            Args:
                weights: Cost weight matrices
            """
            self.weights = weights

            # Convert to CasADi
            self.Q_ca = ca.DM(weights.Q)
            self.R_ca = ca.DM(weights.R)
            self.P_ca = ca.DM(weights.P)

        def stage_cost(
            self,
            x: ca.MX,
            u: ca.MX,
            x_ref: ca.MX,
        ) -> ca.MX:
            """
            Quadratic stage cost.

            l(x, u) = (x - x_ref)^T Q (x - x_ref) + u^T R u
            """
            x_err = x - x_ref
            return ca.bilin(self.Q_ca, x_err, x_err) + ca.bilin(self.R_ca, u, u)

        def tracking_cost(
            self,
            x: ca.MX,
            u: ca.MX,
            x_ref: ca.MX,
            u_ref: ca.MX,
        ) -> ca.MX:
            """
            Reference tracking cost.
            """
            x_err = x - x_ref
            u_err = u - u_ref
            return ca.bilin(self.Q_ca, x_err, x_err) + ca.bilin(self.R_ca, u_err, u_err)

        def fuel_cost(
            self,
            x: ca.MX,
            u: ca.MX,
            x_ref: ca.MX,
        ) -> ca.MX:
            """
            Fuel-optimal cost with tracking.
            """
            x_err = x - x_ref
            T_norm = ca.norm_2(u)
            return self.weights.w_fuel * T_norm + ca.bilin(self.Q_ca, x_err, x_err) + ca.bilin(self.R_ca, u, u)

        def terminal_cost(
            self,
            x: ca.MX,
            x_ref: ca.MX,
        ) -> ca.MX:
            """
            Terminal cost.

            V_f(x) = (x - x_ref)^T P (x - x_ref)
            """
            x_err = x - x_ref
            return ca.bilin(self.P_ca, x_err, x_err)

        def set_terminal_cost_matrix(self, P: NDArray) -> None:
            """Update terminal cost matrix (e.g., from LQR)."""
            self.weights.P = P
            self.P_ca = ca.DM(P)


# =============================================================================
# Cost Function for Trajectory Optimization
# =============================================================================


class TrajectoryObjective:
    """
    Complete objective function for trajectory optimization.

    J = Σ_{k=0}^{N-1} l(x_k, u_k, x_ref_k, u_ref_k) + V_f(x_N)
    """

    def __init__(
        self,
        weights: CostWeights,
        N: int,
        include_fuel: bool = False,
    ):
        """
        Initialize trajectory objective.

        Args:
            weights: Cost weights
            N: Prediction horizon
            include_fuel: Whether to include fuel cost
        """
        self.weights = weights
        self.N = N
        self.include_fuel = include_fuel

    def evaluate(
        self,
        X: NDArray,
        U: NDArray,
        X_ref: NDArray,
        U_ref: Optional[NDArray] = None,
    ) -> float:
        """
        Evaluate total trajectory cost.

        Args:
            X: State trajectory (N+1, n_x)
            U: Control trajectory (N, n_u)
            X_ref: Reference states (N+1, n_x)
            U_ref: Reference controls (N, n_u), optional

        Returns:
            Total cost
        """
        if U_ref is None:
            U_ref = np.zeros_like(U)

        total = 0.0

        # Stage costs
        for k in range(self.N):
            if self.include_fuel:
                total += fuel_optimal_stage_cost(X[k], U[k], X_ref[k], self.weights)
            else:
                total += tracking_cost(X[k], U[k], X_ref[k], U_ref[k], self.weights)

        # Terminal cost
        x_err = X[self.N] - X_ref[self.N]
        total += float(x_err.T @ self.weights.P @ x_err)

        return total

    def evaluate_gradient(
        self,
        X: NDArray,
        U: NDArray,
        X_ref: NDArray,
        U_ref: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute gradient of cost w.r.t. X and U.

        Returns:
            dJ_dX: (N+1, n_x)
            dJ_dU: (N, n_u)
        """
        if U_ref is None:
            U_ref = np.zeros_like(U)

        dJ_dX = np.zeros_like(X)
        dJ_dU = np.zeros_like(U)

        # Stage cost gradients
        for k in range(self.N):
            x_err = X[k] - X_ref[k]
            u_err = U[k] - U_ref[k]

            dJ_dX[k] = 2 * self.weights.Q @ x_err
            dJ_dU[k] = 2 * self.weights.R @ u_err

            if self.include_fuel:
                T_mag = np.linalg.norm(U[k])
                if T_mag > 1e-10:
                    dJ_dU[k] += self.weights.w_fuel * U[k] / T_mag

        # Terminal cost gradient
        x_err_N = X[self.N] - X_ref[self.N]
        dJ_dX[self.N] = 2 * self.weights.P @ x_err_N

        return dJ_dX, dJ_dU
