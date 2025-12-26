"""
Invariant Sets and Tube Controller for Safety Filter

This module provides:
1. Ellipsoidal invariant set computation
2. Tube controller design via LMI
3. Robust positive invariant (RPI) set computation

The backup invariant set S is defined such that:
- For all x ∈ S, the backup controller keeps x in S
- All constraints are satisfied within S

Reference:
    Borrelli, F., Bemporad, A., & Morari, M. (2017). Predictive Control
    for Linear and Hybrid Systems. Cambridge University Press.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class InvariantSetConfig:
    """Configuration for invariant set computation."""

    # Ellipsoid scaling
    alpha: float = 1.0  # Scaling factor

    # Constraint tightening
    constraint_margin: float = 0.1

    # LMI solver settings
    max_iter: int = 100
    tol: float = 1e-6

    # Disturbance bounds
    w_max: float = 0.1  # Max disturbance magnitude


class EllipsoidalInvariantSet:
    """
    Ellipsoidal invariant set for backup controller.

    The set is defined as:
        S = {x : (x - x_eq)' P (x - x_eq) ≤ alpha}

    where P is the Lyapunov matrix from LQR and alpha is chosen
    such that constraints are satisfied within S.

    Example:
        >>> inv_set = EllipsoidalInvariantSet(n_x=14)
        >>> inv_set.compute_from_lqr(P_lqr, x_eq)
        >>>
        >>> # Check if state is in set
        >>> is_safe = inv_set.contains(x)
    """

    def __init__(
        self,
        n_x: int,
        config: Optional[InvariantSetConfig] = None,
    ):
        """
        Initialize ellipsoidal invariant set.

        Args:
            n_x: State dimension
            config: Configuration
        """
        self.n_x = n_x
        self.config = config or InvariantSetConfig()

        # Ellipsoid parameters
        self.P: Optional[NDArray] = None  # Shape matrix
        self.x_eq: Optional[NDArray] = None  # Center
        self.alpha: float = 1.0  # Level set value

    def compute_from_lqr(
        self,
        P_lqr: NDArray,
        x_eq: NDArray,
        alpha: Optional[float] = None,
    ) -> None:
        """
        Compute invariant set from LQR Lyapunov matrix.

        Args:
            P_lqr: LQR value function matrix
            x_eq: Equilibrium point
            alpha: Level set value (auto-computed if None)
        """
        self.P = P_lqr.copy()
        self.x_eq = x_eq.copy()

        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = self.config.alpha

    def compute_maximal_alpha(
        self,
        constraint_fn: callable,
        n_samples: int = 1000,
    ) -> float:
        """
        Compute maximal alpha such that constraints are satisfied.

        Uses sampling on the ellipsoid boundary to check constraints.

        Args:
            constraint_fn: Function that returns True if constraints satisfied
            n_samples: Number of boundary samples

        Returns:
            Maximal alpha value
        """
        if self.P is None:
            raise RuntimeError("Must call compute_from_lqr first")

        # Sample points on unit sphere
        samples = np.random.randn(n_samples, self.n_x)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

        # Transform to ellipsoid boundary
        # x = x_eq + sqrt(alpha) * P^{-1/2} @ z, where ||z|| = 1
        try:
            P_inv_sqrt = np.linalg.cholesky(np.linalg.inv(self.P)).T
        except np.linalg.LinAlgError:
            # Use eigendecomposition for non-PD matrices
            eigvals, eigvecs = np.linalg.eigh(self.P)
            eigvals = np.maximum(eigvals, 1e-10)
            P_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        # Binary search for maximal alpha
        alpha_low, alpha_high = 0.01, 100.0

        for _ in range(50):
            alpha_mid = (alpha_low + alpha_high) / 2

            # Check all boundary points
            all_feasible = True
            for z in samples:
                x = self.x_eq + np.sqrt(alpha_mid) * P_inv_sqrt @ z
                if not constraint_fn(x):
                    all_feasible = False
                    break

            if all_feasible:
                alpha_low = alpha_mid
            else:
                alpha_high = alpha_mid

        # Apply safety margin
        self.alpha = alpha_low * (1 - self.config.constraint_margin)
        return self.alpha

    def contains(self, x: NDArray) -> bool:
        """Check if state is in invariant set."""
        if self.P is None:
            return False

        dx = x - self.x_eq
        value = dx.T @ self.P @ dx
        return value <= self.alpha

    def get_value(self, x: NDArray) -> float:
        """Get Lyapunov value V(x)."""
        if self.P is None:
            return np.inf

        dx = x - self.x_eq
        return float(dx.T @ self.P @ dx)

    def project(self, x: NDArray) -> NDArray:
        """Project state onto invariant set boundary."""
        if self.P is None:
            return x

        dx = x - self.x_eq
        value = dx.T @ self.P @ dx

        if value <= self.alpha:
            return x  # Already inside

        # Scale to boundary
        scale = np.sqrt(self.alpha / value)
        return self.x_eq + scale * dx

    def get_boundary_points(self, n_points: int = 100) -> NDArray:
        """Sample points on ellipsoid boundary."""
        if self.P is None:
            return np.zeros((0, self.n_x))

        # Sample on unit sphere
        samples = np.random.randn(n_points, self.n_x)
        samples = samples / np.linalg.norm(samples, axis=1, keepdims=True)

        # Transform to ellipsoid
        try:
            L = np.linalg.cholesky(np.linalg.inv(self.P))
            boundary = self.x_eq + np.sqrt(self.alpha) * (samples @ L.T)
        except np.linalg.LinAlgError:
            boundary = self.x_eq + np.sqrt(self.alpha) * samples

        return boundary


class TubeController:
    """
    Tube controller for robust constraint satisfaction.

    The tube controller provides:
    1. Feedback gain K for uncertainty rejection
    2. RPI set computation for tube width
    3. Constraint tightening based on tube

    The controlled system is:
        x_{k+1} = (A + BK)x_k + w_k

    where w_k is bounded disturbance (including GP uncertainty).
    """

    def __init__(
        self,
        A: NDArray,
        B: NDArray,
        config: Optional[InvariantSetConfig] = None,
    ):
        """
        Initialize tube controller.

        Args:
            A: State matrix
            B: Input matrix
            config: Configuration
        """
        self.A = A
        self.B = B
        self.n_x = A.shape[0]
        self.n_u = B.shape[1]
        self.config = config or InvariantSetConfig()

        # Tube controller gain
        self.K: Optional[NDArray] = None

        # Closed-loop matrix
        self.A_cl: Optional[NDArray] = None

        # RPI set
        self.rpi_set: Optional[NDArray] = None

    def compute_tube_gain(
        self,
        Q: Optional[NDArray] = None,
        R: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Compute tube controller gain via LQR or pole placement.

        Args:
            Q: State weight (default: identity)
            R: Control weight (default: identity)

        Returns:
            K: Feedback gain
        """
        if Q is None:
            Q = np.eye(self.n_x)
        if R is None:
            R = np.eye(self.n_u)

        if HAS_SCIPY:
            try:
                P = solve_discrete_are(self.A, self.B, Q, R)
                BtP = self.B.T @ P
                self.K = -np.linalg.solve(R + BtP @ self.B, BtP @ self.A)
            except Exception:
                # Fallback to simple gain
                self.K = -0.1 * np.linalg.pinv(self.B) @ self.A
        else:
            self.K = -0.1 * np.linalg.pinv(self.B) @ self.A

        self.A_cl = self.A + self.B @ self.K

        return self.K

    def compute_rpi_set(
        self,
        W: NDArray,
        n_iterations: int = 20,
    ) -> NDArray:
        """
        Compute robust positive invariant (RPI) set.

        The RPI set is the minimal set Ω such that:
            A_cl @ Ω ⊕ W ⊆ Ω

        Approximated using geometric series:
            Ω ≈ W ⊕ A_cl W ⊕ A_cl² W ⊕ ...

        Args:
            W: Disturbance set (as vertices or ellipsoid matrix)
            n_iterations: Number of Minkowski sum iterations

        Returns:
            RPI set approximation
        """
        if self.A_cl is None:
            raise RuntimeError("Must call compute_tube_gain first")

        # For simplicity, assume W is a box and compute bounding box of RPI
        # More sophisticated: use zonotopes or polytopes

        if W.ndim == 1:  # noqa: SIM108
            # W is diagonal of bounding box
            w_bounds = W
        else:
            # W is a matrix - use diagonal
            w_bounds = np.abs(np.diag(W))

        # Geometric series: Ω = Σ A_cl^k W
        rpi_bounds = w_bounds.copy()
        A_pow = self.A_cl.copy()

        for _ in range(n_iterations):
            rpi_bounds += np.abs(A_pow) @ w_bounds
            A_pow = A_pow @ self.A_cl

            # Check convergence
            if np.max(np.abs(A_pow)) < 1e-10:
                break

        self.rpi_set = rpi_bounds
        return rpi_bounds

    def get_constraint_tightening(self) -> NDArray:
        """
        Get constraint tightening amounts based on RPI set.

        Returns:
            Tightening amounts for each state constraint
        """
        if self.rpi_set is None:
            return np.zeros(self.n_x)

        return self.rpi_set * (1 + self.config.constraint_margin)

    def get_control(self, x: NDArray, x_nominal: NDArray) -> NDArray:
        """
        Get tube control correction.

        u = K @ (x - x_nominal)

        Args:
            x: Actual state
            x_nominal: Nominal (planned) state

        Returns:
            Control correction
        """
        if self.K is None:
            return np.zeros(self.n_u)

        return self.K @ (x - x_nominal)


class PolytopeInvariantSet:
    """
    Polytopic invariant set representation.

    Set defined as: S = {x : Hx ≤ h}

    More general than ellipsoids but more complex.
    """

    def __init__(self):
        """Initialize polytope."""
        self.H: Optional[NDArray] = None  # Inequality matrix
        self.h: Optional[NDArray] = None  # Inequality bounds

    def from_box(self, x_min: NDArray, x_max: NDArray) -> None:
        """Create polytope from box constraints."""
        n = len(x_min)
        self.H = np.vstack([np.eye(n), -np.eye(n)])
        self.h = np.concatenate([x_max, -x_min])

    def contains(self, x: NDArray) -> bool:
        """Check if state is in polytope."""
        if self.H is None:
            return False
        return np.all(self.H @ x <= self.h)

    def get_violations(self, x: NDArray) -> NDArray:
        """Get constraint violations (positive = violated)."""
        if self.H is None:
            return np.array([])
        return self.H @ x - self.h


def compute_lmi_invariant_set(
    A: NDArray,
    B: NDArray,
    K: NDArray,
    state_bounds: Optional[Tuple[NDArray, NDArray]] = None,
) -> Tuple[NDArray, float]:
    """
    Compute invariant set via LMI (simplified).

    Solves for P such that:
        (A + BK)'P(A + BK) - P < 0  (stability)
        x'Px ≤ 1 => constraints satisfied

    This is a simplified version - full LMI requires cvxpy or similar.

    Args:
        A: State matrix
        B: Input matrix
        K: Feedback gain
        state_bounds: (x_min, x_max) for constraint checking

    Returns:
        P: Lyapunov matrix defining ellipsoid
        alpha: Level set value
    """
    A_cl = A + B @ K
    n_x = A.shape[0]

    # Solve Lyapunov equation: A_cl' P A_cl - P + Q = 0
    Q = np.eye(n_x)

    if HAS_SCIPY:
        try:
            P = solve_discrete_lyapunov(A_cl.T, Q)
        except Exception:
            P = np.eye(n_x)
    else:
        P = np.eye(n_x)

    # Compute maximal alpha for constraints
    alpha = 1.0

    if state_bounds is not None:
        x_min, x_max = state_bounds
        # Check corners of constraint box
        corners = []
        for i in range(2**n_x):
            corner = np.array([x_min[j] if (i >> j) & 1 else x_max[j] for j in range(n_x)])
            corners.append(corner)

        # Find minimum alpha
        min_alpha = np.inf
        for corner in corners[: min(100, len(corners))]:  # Limit for high dimensions
            val = corner.T @ P @ corner
            if val > 0:
                min_alpha = min(min_alpha, val)

        if min_alpha < np.inf:
            alpha = min_alpha * 0.9  # Safety margin

    return P, alpha
