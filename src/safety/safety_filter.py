"""
Predictive Safety Filter for Rocket Landing

The safety filter modifies potentially unsafe control inputs to
guarantee constraint satisfaction while staying close to the
original input.

Algorithm:
1. Receive nominal control u_nom from MPC
2. Predict future trajectory with u_nom
3. Check if trajectory remains safe (in backup invariant set)
4. If unsafe, solve QP to find closest safe control

Safety filter QP:
    min_u  ||u - u_nom||²
    s.t.   x_{k+1} = f(x_k, u)
           V(x_N) ≤ alpha  (reach backup set)
           g(x_k, u) ≥ 0  (path constraints)

Reference:
    Wabersich, K. P., & Zeilinger, M. N. (2021). A predictive safety
    filter for learning-based control of constrained nonlinear
    dynamical systems. Automatica.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

from .backup_controller import BackupControllerConfig, LQRBackupController
from .invariant_sets import EllipsoidalInvariantSet, InvariantSetConfig
from .tube_mpc import TubeMPCConfig, TubePropagator


@dataclass
class SafetyFilterConfig:
    """Configuration for safety filter."""

    # Prediction horizon
    N: int = 10
    dt: float = 0.1

    # Backup set
    backup_type: str = "ellipsoid"  # "ellipsoid", "polytope"
    backup_margin: float = 0.9  # Safety margin on backup set

    # Constraint handling
    use_gp_uncertainty: bool = True
    gp_confidence: float = 0.95

    # QP solver
    qp_max_iter: int = 50
    qp_tol: float = 1e-6

    # Filtering behavior
    filter_mode: str = "soft"  # "hard", "soft"
    soft_weight: float = 1e3  # Weight for soft constraint violations


@dataclass
class SafetyFilterResult:
    """Result from safety filter."""

    u_safe: NDArray  # Safe control output
    u_nominal: NDArray  # Original nominal control
    is_modified: bool  # Whether control was modified
    is_safe: bool  # Whether original was safe
    safety_margin: float  # Distance to backup set boundary
    solve_time: float  # Computation time
    backup_value: float  # Lyapunov value at terminal state
    constraint_violations: Dict[str, float]  # Constraint margins


class PredictiveSafetyFilter:
    """
    Predictive Safety Filter for rocket landing.

    Ensures safety by checking if the nominal control leads to
    a state from which the backup controller can recover.

    Example:
        >>> safety = PredictiveSafetyFilter(dynamics, config)
        >>> safety.initialize(x_eq)
        >>>
        >>> # Filter control from MPC
        >>> result = safety.filter(x_current, u_mpc)
        >>> u_apply = result.u_safe
    """

    def __init__(
        self,
        dynamics,
        config: Optional[SafetyFilterConfig] = None,
        constraint_params=None,
    ):
        """
        Initialize safety filter.

        Args:
            dynamics: Rocket dynamics model
            config: Safety filter configuration
            constraint_params: Constraint parameters
        """
        self.dynamics = dynamics
        self.config = config or SafetyFilterConfig()
        self.constraint_params = constraint_params

        self.n_x = 14
        self.n_u = 3

        # Backup controller
        backup_config = BackupControllerConfig(dt=self.config.dt)
        self.backup = LQRBackupController(dynamics, backup_config)

        # Backup invariant set
        self.invariant_set = EllipsoidalInvariantSet(
            self.n_x,
            InvariantSetConfig(),
        )

        # Tube propagator for uncertainty
        tube_config = TubeMPCConfig(
            N=self.config.N,
            dt=self.config.dt,
        )
        self.tube_propagator = TubePropagator(dynamics, tube_config)

        # GP model (optional)
        self._gp_model = None

        # Initialization flag
        self._initialized = False

    def initialize(
        self,
        x_eq: NDArray,
        u_eq: Optional[NDArray] = None,
        backup_alpha: Optional[float] = None,
    ) -> None:
        """
        Initialize safety filter components.

        Args:
            x_eq: Equilibrium state (hover point)
            u_eq: Equilibrium control
            backup_alpha: Size of backup invariant set
        """
        # Compute backup controller gains
        self.backup.compute_gains(x_eq, u_eq)

        # Set up invariant set
        self.invariant_set.compute_from_lqr(
            self.backup.P,
            x_eq,
            backup_alpha,
        )

        # Optionally compute maximal safe alpha
        if backup_alpha is None and self.constraint_params is not None:

            def constraint_check(x):
                return self._check_constraints(x, self.backup.get_control(x))

            self.invariant_set.compute_maximal_alpha(constraint_check)

        self._initialized = True

    def set_gp_model(self, gp_model) -> None:
        """Set GP model for uncertainty-aware filtering."""
        self._gp_model = gp_model

    def filter(
        self,
        x: NDArray,
        u_nominal: NDArray,
    ) -> SafetyFilterResult:
        """
        Filter control input for safety.

        Args:
            x: Current state
            u_nominal: Nominal control from MPC

        Returns:
            SafetyFilterResult with safe control
        """
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")

        start_time = time.perf_counter()

        # Check if nominal control is safe
        is_safe, safety_margin, violations = self._check_safety(x, u_nominal)

        if is_safe:
            # Nominal control is safe - use it
            return SafetyFilterResult(
                u_safe=u_nominal.copy(),
                u_nominal=u_nominal.copy(),
                is_modified=False,
                is_safe=True,
                safety_margin=safety_margin,
                solve_time=time.perf_counter() - start_time,
                backup_value=self._get_backup_value(x, u_nominal),
                constraint_violations=violations,
            )

        # Nominal is unsafe - solve QP to find safe alternative
        u_safe = self._solve_safety_qp(x, u_nominal)

        # Verify the filtered control
        is_safe_filtered, margin_filtered, violations_filtered = self._check_safety(x, u_safe)

        return SafetyFilterResult(
            u_safe=u_safe,
            u_nominal=u_nominal.copy(),
            is_modified=True,
            is_safe=is_safe_filtered,
            safety_margin=margin_filtered,
            solve_time=time.perf_counter() - start_time,
            backup_value=self._get_backup_value(x, u_safe),
            constraint_violations=violations_filtered,
        )

    def _check_safety(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Check if control u is safe from state x.

        Safety means:
        1. Immediate constraints satisfied
        2. Predicted trajectory reaches backup set
        """
        violations = {}

        # Check immediate constraints
        constraint_ok = self._check_constraints(x, u)
        if not constraint_ok:
            violations["immediate"] = -1.0

        # Predict trajectory with backup controller after first step
        X_pred = self._predict_with_backup(x, u)

        # Check if terminal state is in backup set
        x_terminal = X_pred[-1]
        backup_value = self.invariant_set.get_value(x_terminal)
        safety_margin = self.invariant_set.alpha - backup_value

        violations["terminal"] = safety_margin

        # Check path constraints along prediction
        for k, x_k in enumerate(X_pred[:-1]):
            u_k = self.backup.get_control(x_k) if k > 0 else u
            if not self._check_constraints(x_k, u_k):
                violations[f"path_{k}"] = -1.0

        is_safe = (
            constraint_ok and safety_margin > 0 and all(v >= 0 for k, v in violations.items() if k.startswith("path"))
        )

        return is_safe, safety_margin, violations

    def _predict_with_backup(
        self,
        x: NDArray,
        u_first: NDArray,
    ) -> NDArray:
        """
        Predict trajectory: first step with u_first, then backup.

        Args:
            x: Current state
            u_first: Control for first step

        Returns:
            X_pred: Predicted trajectory (N+1, n_x)
        """
        N = self.config.N
        dt = self.config.dt

        X_pred = np.zeros((N + 1, self.n_x))
        X_pred[0] = x

        # First step with provided control
        X_pred[1] = self.dynamics.step(x, u_first, dt)

        # Remaining steps with backup controller
        for k in range(1, N):
            u_backup = self.backup.get_control(X_pred[k])
            X_pred[k + 1] = self.dynamics.step(X_pred[k], u_backup, dt)

        return X_pred

    def _check_constraints(self, x: NDArray, u: NDArray) -> bool:
        """Check if state-control pair satisfies constraints."""
        if self.constraint_params is None:
            return True

        # Thrust magnitude
        T_mag = np.linalg.norm(u)
        T_min = getattr(self.constraint_params, "T_min", 0.5)
        T_max = getattr(self.constraint_params, "T_max", 5.0)

        if T_mag < T_min or T_mag > T_max:
            return False

        # Tilt angle
        q = x[7:11]
        cos_theta = 1 - 2 * (q[1] ** 2 + q[2] ** 2)  # From quaternion
        theta_max = getattr(self.constraint_params, "theta_max", 60.0)

        if cos_theta < np.cos(np.deg2rad(theta_max)):
            return False

        # Glideslope
        pos = x[1:4]
        gamma = getattr(self.constraint_params, "gamma_gs", 30.0)
        tan_gamma = np.tan(np.deg2rad(gamma))

        if pos[0] > 0.1:  # Above ground  # noqa: SIM102
            if pos[0] ** 2 * tan_gamma**2 < pos[1] ** 2 + pos[2] ** 2:
                return False

        return True

    def _get_backup_value(self, x: NDArray, u: NDArray) -> float:
        """Get Lyapunov value at predicted terminal state."""
        X_pred = self._predict_with_backup(x, u)
        return self.invariant_set.get_value(X_pred[-1])

    def _solve_safety_qp(
        self,
        x: NDArray,
        u_nominal: NDArray,
    ) -> NDArray:
        """
        Solve safety filter QP.

        min_u  ||u - u_nom||²
        s.t.   V(x_N(u)) ≤ alpha
               g(x, u) ≥ 0
        """
        if HAS_CASADI:
            return self._solve_casadi_qp(x, u_nominal)
        else:
            return self._solve_gradient_qp(x, u_nominal)

    def _solve_casadi_qp(
        self,
        x: NDArray,
        u_nominal: NDArray,
    ) -> NDArray:
        """Solve safety QP using CasADi."""
        opti = ca.Opti()

        # Decision variable: first control
        u = opti.variable(self.n_u)

        # Objective: minimize deviation from nominal
        cost = ca.dot(u - u_nominal, u - u_nominal)
        opti.minimize(cost)

        # Predict trajectory (simplified: linearized)
        x_next = self._casadi_step(x, u)

        # Apply backup for remaining steps
        x_curr = x_next
        for k in range(1, self.config.N):
            u_backup = self.backup.u_eq - self.backup.K @ (x_curr - self.backup.x_eq)
            x_curr = self._casadi_step_from_mx(x_curr, u_backup)

        # Terminal constraint: V(x_N) ≤ alpha
        x_terminal = x_curr
        dx = x_terminal - self.backup.x_eq
        V_terminal = ca.bilin(ca.DM(self.backup.P), dx, dx)

        if self.config.filter_mode == "hard":
            opti.subject_to(V_terminal <= self.invariant_set.alpha * self.config.backup_margin)
        else:
            # Soft constraint
            slack = opti.variable()
            opti.subject_to(slack >= 0)
            opti.subject_to(V_terminal <= self.invariant_set.alpha * self.config.backup_margin + slack)
            cost += self.config.soft_weight * slack**2
            opti.minimize(cost)

        # Control constraints
        T_min = getattr(self.constraint_params, "T_min", 0.5) if self.constraint_params else 0.5
        T_max = getattr(self.constraint_params, "T_max", 5.0) if self.constraint_params else 5.0

        opti.subject_to(ca.dot(u, u) >= T_min**2)
        opti.subject_to(ca.dot(u, u) <= T_max**2)

        # Solver
        opti.solver(
            "ipopt",
            {
                "ipopt.print_level": 0,
                "print_time": False,
                "ipopt.max_iter": self.config.qp_max_iter,
            },
        )

        opti.set_initial(u, u_nominal)

        try:
            sol = opti.solve()
            return sol.value(u)
        except RuntimeError:
            # QP infeasible - fall back to backup controller
            return self.backup.get_control(x)

    def _casadi_step(self, x: NDArray, u: ca.MX) -> ca.MX:
        """Single dynamics step with CasADi (linearized)."""
        # Numerical Jacobians at current state
        A, B = self._linearize(x, np.zeros(3))

        x_eq = self.backup.x_eq
        u_eq = self.backup.u_eq

        # Linearized dynamics: x_next ≈ A(x - x_eq) + B(u - u_eq) + x_next_nom
        x_next_nom = self.dynamics.step(x_eq, u_eq, self.config.dt)

        dx = ca.DM(x - x_eq)
        du = u - u_eq

        x_next = ca.DM(x_next_nom) + ca.DM(A) @ dx + ca.DM(B) @ du

        return x_next

    def _casadi_step_from_mx(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """Dynamics step when x is already MX."""
        # Simplified: use linearization around equilibrium
        A, B = self._linearize(self.backup.x_eq, self.backup.u_eq)

        x_eq = self.backup.x_eq
        u_eq = self.backup.u_eq
        x_next_nom = self.dynamics.step(x_eq, u_eq, self.config.dt)

        dx = x - x_eq
        du = u - u_eq

        return ca.DM(x_next_nom) + ca.DM(A) @ dx + ca.DM(B) @ du

    def _linearize(self, x: NDArray, u: NDArray) -> Tuple[NDArray, NDArray]:
        """Numerical linearization."""
        eps = 1e-6
        dt = self.config.dt

        A = np.zeros((self.n_x, self.n_x))
        B = np.zeros((self.n_x, self.n_u))

        x_next_nom = self.dynamics.step(x, u, dt)

        for i in range(self.n_x):
            x_pert = x.copy()
            x_pert[i] += eps
            x_next = self.dynamics.step(x_pert, u, dt)
            A[:, i] = (x_next - x_next_nom) / eps

        for i in range(self.n_u):
            u_pert = u.copy()
            u_pert[i] += eps
            x_next = self.dynamics.step(x, u_pert, dt)
            B[:, i] = (x_next - x_next_nom) / eps

        return A, B

    def _solve_gradient_qp(
        self,
        x: NDArray,
        u_nominal: NDArray,
        n_iters: int = 50,
        lr: float = 0.1,
    ) -> NDArray:
        """
        Solve safety QP using gradient descent (fallback).
        """
        u = u_nominal.copy()

        for _ in range(n_iters):
            # Compute gradient of safety violation
            X_pred = self._predict_with_backup(x, u)
            V_terminal = self.invariant_set.get_value(X_pred[-1])

            # If safe, we're done
            if V_terminal <= self.invariant_set.alpha * self.config.backup_margin:
                break

            # Gradient of V w.r.t. u (numerical)
            eps = 1e-4
            grad = np.zeros(self.n_u)
            for i in range(self.n_u):
                u_pert = u.copy()
                u_pert[i] += eps
                X_pert = self._predict_with_backup(x, u_pert)
                V_pert = self.invariant_set.get_value(X_pert[-1])
                grad[i] = (V_pert - V_terminal) / eps

            # Combined gradient: safety + deviation from nominal
            grad_total = grad + 2 * (u - u_nominal)

            # Update
            u = u - lr * grad_total

            # Project onto thrust constraints
            T_min = getattr(self.constraint_params, "T_min", 0.5) if self.constraint_params else 0.5
            T_max = getattr(self.constraint_params, "T_max", 5.0) if self.constraint_params else 5.0

            T_mag = np.linalg.norm(u)
            if T_mag < T_min:
                u = u / max(T_mag, 1e-6) * T_min
            elif T_mag > T_max:
                u = u / T_mag * T_max

        return u

    def simulate_filtered(
        self,
        x0: NDArray,
        U_nominal: NDArray,
    ) -> Tuple[NDArray, NDArray, List[SafetyFilterResult]]:
        """
        Simulate with safety filter applied.

        Args:
            x0: Initial state
            U_nominal: Nominal control trajectory (N, n_u)

        Returns:
            X: Filtered state trajectory
            U: Filtered control trajectory
            results: Safety filter results at each step
        """
        N = len(U_nominal)
        dt = self.config.dt

        X = np.zeros((N + 1, self.n_x))
        U = np.zeros((N, self.n_u))
        results = []

        X[0] = x0

        for k in range(N):
            result = self.filter(X[k], U_nominal[k])
            U[k] = result.u_safe
            results.append(result)

            X[k + 1] = self.dynamics.step(X[k], U[k], dt)

        return X, U, results


class SimpleSafetyFilter:
    """
    Simplified safety filter using constraint checking only.

    No backup controller - just checks immediate constraints
    and modifies control to satisfy them.
    """

    def __init__(
        self,
        dynamics,
        constraint_params=None,
    ):
        """Initialize simple safety filter."""
        self.dynamics = dynamics
        self.params = constraint_params

    def filter(self, x: NDArray, u: NDArray) -> NDArray:  # noqa: ARG002
        """Filter control to satisfy constraints."""
        u_safe = u.copy()

        # Thrust magnitude bounds
        T_min = getattr(self.params, "T_min", 0.5) if self.params else 0.5
        T_max = getattr(self.params, "T_max", 5.0) if self.params else 5.0

        T_mag = np.linalg.norm(u_safe)

        if T_mag < T_min:
            u_safe = u_safe / max(T_mag, 1e-6) * T_min
        elif T_mag > T_max:
            u_safe = u_safe / T_mag * T_max

        return u_safe
