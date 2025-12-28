"""
Convex Hull Terminal Constraints for LMPC

The terminal constraint in LMPC is:
    x_N ∈ Conv(SS_local)

where SS_local is the local safe set. This is implemented as:
    x_N = Σ λ_i v_i,  Σ λ_i = 1,  λ_i ≥ 0

where {v_i} are the vertices of the convex hull.

Key challenges:
1. High-dimensional convex hulls are expensive to compute
2. We use vertex representation instead of halfspace

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning MPC for Iterative Tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from src.terminal.safe_set import SampledSafeSet

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

try:
    from scipy.spatial import ConvexHull, Delaunay  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class ConvexHullConfig:
    """Configuration for convex hull constraints."""

    # Vertex selection
    n_vertices: int = 10  # Number of vertices to use

    # Constraint formulation
    use_slack: bool = True  # Use slack variables for soft constraint
    slack_weight: float = 1e4  # Penalty for constraint violation

    # Numerical settings
    convex_tol: float = 1e-6  # Tolerance for convexity check
    lambda_min: float = 0.0  # Minimum lambda value
    lambda_max: float = 1.0  # Maximum lambda value


class ConvexHullConstraint:
    """
    Convex hull terminal constraint for LMPC.

    Formulates the constraint:
        x_N = Σ_{i=1}^K λ_i v_i
        Σ λ_i = 1
        λ_i ≥ 0

    This ensures x_N is a convex combination of safe set vertices.

    Example:
        >>> ch = ConvexHullConstraint(vertices, q_values)
        >>>
        >>> # Add to CasADi optimization
        >>> lambdas = opti.variable(K)
        >>> ch.add_constraints(opti, x_N, lambdas)
        >>>
        >>> # Terminal cost from Q-function
        >>> terminal_cost = ch.get_terminal_cost(lambdas)
    """

    def __init__(
        self,
        vertices: NDArray,
        q_values: NDArray,
        config: Optional[ConvexHullConfig] = None,
    ):
        """
        Initialize convex hull constraint.

        Args:
            vertices: Safe set vertices (K, n_x)
            q_values: Q-values at vertices (K,)
            config: Configuration parameters
        """
        self.config = config or ConvexHullConfig()

        # Store vertices and Q-values
        self.vertices = vertices
        self.q_values = q_values
        self.K = len(vertices)
        self.n_x = vertices.shape[1] if len(vertices) > 0 else 0

        # CasADi data matrices
        if HAS_CASADI and self.K > 0:
            self._V_ca = ca.DM(vertices.T)  # (n_x, K)
            self._Q_ca = ca.DM(q_values)  # (K,)

    def update_vertices(
        self,
        vertices: NDArray,
        q_values: NDArray,
    ) -> None:
        """Update vertices (for dynamic safe set)."""
        self.vertices = vertices
        self.q_values = q_values
        self.K = len(vertices)

        if HAS_CASADI and self.K > 0:
            self._V_ca = ca.DM(vertices.T)
            self._Q_ca = ca.DM(q_values)

    def check_feasibility(self, x: NDArray) -> bool:
        """
        Check if x is inside the convex hull.

        Uses linear programming or Delaunay triangulation.
        """
        if self.K == 0:
            return False

        if self.n_x >= self.K:
            # Not enough points for full-dimensional hull
            # Check if x is close to any vertex
            distances = np.linalg.norm(self.vertices - x, axis=1)
            return np.min(distances) < self.config.convex_tol

        if HAS_SCIPY:
            try:
                hull = Delaunay(self.vertices)
                return hull.find_simplex(x) >= 0
            except Exception:
                # Delaunay failed (e.g., degenerate points)
                pass

        # Fallback: solve LP to check feasibility
        return self._check_feasibility_lp(x)

    def _check_feasibility_lp(self, x: NDArray) -> bool:
        """Check feasibility using linear programming."""
        if not HAS_CASADI:
            return False

        # Solve: find λ such that x = V λ, sum(λ) = 1, λ ≥ 0
        opti = ca.Opti()
        lambdas = opti.variable(self.K)

        # x = V @ λ
        opti.subject_to(self._V_ca @ lambdas == x)

        # sum(λ) = 1
        opti.subject_to(ca.sum1(lambdas) == 1)

        # λ ≥ 0
        opti.subject_to(lambdas >= 0)

        # Minimize 0 (feasibility only)
        opti.minimize(0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})

        try:
            opti.solve()
            return True
        except RuntimeError:
            return False

    def project_onto_hull(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Project x onto convex hull.

        Finds the point in Conv(vertices) closest to x.

        Returns:
            x_proj: Projected point
            lambdas: Convex combination weights
        """
        if self.K == 0:
            return x, np.array([])

        if not HAS_CASADI:
            # Simple fallback: nearest vertex
            distances = np.linalg.norm(self.vertices - x, axis=1)
            nearest_idx = np.argmin(distances)
            lambdas = np.zeros(self.K)
            lambdas[nearest_idx] = 1.0
            return self.vertices[nearest_idx], lambdas

        # Solve: min ||x - V λ||^2 s.t. sum(λ)=1, λ≥0
        opti = ca.Opti()
        lambdas_var = opti.variable(self.K)

        # Projected point
        x_proj = self._V_ca @ lambdas_var

        # Objective: minimize distance
        diff = x_proj - x
        opti.minimize(ca.dot(diff, diff))

        # Convexity constraints
        opti.subject_to(ca.sum1(lambdas_var) == 1)
        opti.subject_to(lambdas_var >= 0)

        opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": False})

        try:
            sol = opti.solve()
            lambdas = sol.value(lambdas_var)
            x_proj = self.vertices.T @ lambdas
            return x_proj, lambdas
        except RuntimeError:
            # Fallback to nearest vertex
            distances = np.linalg.norm(self.vertices - x, axis=1)
            nearest_idx = np.argmin(distances)
            lambdas = np.zeros(self.K)
            lambdas[nearest_idx] = 1.0
            return self.vertices[nearest_idx], lambdas

    def get_interpolated_q(self, lambdas: NDArray) -> float:
        """
        Get interpolated Q-value from convex combination.

        Q(x) = Σ λ_i Q(v_i)
        """
        return float(np.dot(lambdas, self.q_values))


if HAS_CASADI:

    class CasADiConvexHullConstraint:
        """
        CasADi symbolic convex hull constraint for MPC.

        Adds the terminal constraint x_N ∈ Conv(SS_local) to
        CasADi optimization problem.
        """

        def __init__(
            self,
            config: Optional[ConvexHullConfig] = None,
        ):
            """
            Initialize CasADi convex hull constraint.

            Args:
                config: Configuration parameters
            """
            self.config = config or ConvexHullConfig()

            # Storage for vertices (updated each solve)
            self._vertices: Optional[NDArray] = None
            self._q_values: Optional[NDArray] = None
            self._V_ca: Optional[ca.DM] = None
            self._Q_ca: Optional[ca.DM] = None

        def set_vertices(
            self,
            vertices: NDArray,
            q_values: NDArray,
        ) -> None:
            """
            Set convex hull vertices.

            Call this before solving MPC with updated safe set.

            Args:
                vertices: Safe set vertices (K, n_x)
                q_values: Q-values at vertices (K,)
            """
            self._vertices = vertices
            self._q_values = q_values
            self._V_ca = ca.DM(vertices.T)  # (n_x, K)
            self._Q_ca = ca.DM(q_values)

        def add_constraints(
            self,
            opti: ca.Opti,
            x_N: ca.MX,
            lambdas: ca.MX,
        ) -> None:
            """
            Add convex hull constraints to optimization.

            Args:
                opti: CasADi Opti instance
                x_N: Terminal state variable (n_x,)
                lambdas: Convex combination weights (K,)
            """
            if self._V_ca is None:
                raise RuntimeError("Must call set_vertices() first")


            # x_N = V @ λ
            opti.subject_to(x_N == self._V_ca @ lambdas)

            # sum(λ) = 1
            opti.subject_to(ca.sum1(lambdas) == 1)

            # λ ≥ 0
            opti.subject_to(lambdas >= self.config.lambda_min)
            opti.subject_to(lambdas <= self.config.lambda_max)

        def add_soft_constraints(
            self,
            opti: ca.Opti,
            x_N: ca.MX,
            lambdas: ca.MX,
        ) -> ca.MX:
            """
            Add soft convex hull constraint with slack.

            Returns slack penalty to add to cost.
            """
            if self._V_ca is None:
                raise RuntimeError("Must call set_vertices() first")

            n_x = self._V_ca.shape[0]

            # Slack variable
            slack = opti.variable(n_x)
            opti.subject_to(slack >= 0)

            # x_N = V @ λ + slack
            opti.subject_to(x_N == self._V_ca @ lambdas + slack)

            # sum(λ) = 1
            opti.subject_to(ca.sum1(lambdas) == 1)

            # λ ≥ 0
            opti.subject_to(lambdas >= 0)

            # Return slack penalty
            return self.config.slack_weight * ca.dot(slack, slack)

        def get_terminal_cost(self, lambdas: ca.MX) -> ca.MX:
            """
            Get terminal cost from Q-function interpolation.

            V_f(x_N) = Σ λ_i Q(v_i)
            """
            if self._Q_ca is None:
                raise RuntimeError("Must call set_vertices() first")

            return ca.dot(self._Q_ca, lambdas)

        @property
        def n_vertices(self) -> int:
            """Number of vertices."""
            return len(self._q_values) if self._q_values is not None else 0


class TerminalSetManager:
    """
    Manages terminal set for LMPC.

    Combines local safe set selection with convex hull constraints.
    Handles fuel-aware shrinking of terminal set.

    Example:
        >>> from src.lmpc import TerminalSetManager
        >>>
        >>> manager = TerminalSetManager(safe_set)
        >>>
        >>> # Get terminal constraint for MPC at state x
        >>> vertices, q_values = manager.get_terminal_set(x, available_fuel)
        >>>
        >>> # Check if state is in terminal set
        >>> is_safe = manager.is_in_terminal_set(x)
    """

    def __init__(
        self,
        safe_set: SampledSafeSet,
        n_vertices: int = 10,
        use_fuel_aware: bool = True,
    ):
        """
        Initialize terminal set manager.

        Args:
            safe_set: Global safe set
            n_vertices: Number of vertices for convex hull
            use_fuel_aware: Enable fuel-aware filtering
        """
        self.safe_set = safe_set
        self.n_vertices = n_vertices
        self.use_fuel_aware = use_fuel_aware

        # Local safe set for neighbor queries
        from .local_safe_set import LocalSafeSet, LocalSafeSetConfig  # noqa: PLC0415

        config = LocalSafeSetConfig(K=n_vertices)
        self._local_ss = LocalSafeSet(safe_set, config)

        # Convex hull constraint
        self._ch_constraint = ConvexHullConstraint(
            np.zeros((0, safe_set.n_x)),
            np.zeros(0),
        )

    def get_terminal_set(
        self,
        x: NDArray,
        available_fuel: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Get terminal set vertices for state x.

        Args:
            x: Current/query state
            available_fuel: Available fuel (for filtering)

        Returns:
            vertices: Terminal set vertices (K, n_x)
            q_values: Q-values at vertices (K,)
        """
        # Get local safe set
        if self.use_fuel_aware and available_fuel is not None:
            vertices, q_values, _ = self._local_ss.query(x, K=self.n_vertices, available_fuel=available_fuel)
        else:
            vertices, q_values, _ = self._local_ss.query(x, K=self.n_vertices)

        # Update convex hull
        self._ch_constraint.update_vertices(vertices, q_values)

        return vertices, q_values

    def is_in_terminal_set(
        self,
        x: NDArray,
        available_fuel: Optional[float] = None,
    ) -> bool:
        """Check if x is in terminal set."""
        vertices, _ = self.get_terminal_set(x, available_fuel)

        if len(vertices) == 0:
            return False

        return self._ch_constraint.check_feasibility(x)

    def get_terminal_cost(self, x: NDArray) -> float:
        """Get terminal cost (Q-value) at x."""
        return self._local_ss.interpolate_q(x)

    def invalidate(self) -> None:
        """Invalidate caches (call when safe set changes)."""
        self._local_ss.invalidate()
