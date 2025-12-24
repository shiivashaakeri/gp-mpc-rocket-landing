"""
Learning Model Predictive Control (LMPC) for Rocket Landing

LMPC iteratively improves control performance by:
1. Storing successful trajectories in a safe set
2. Using the safe set as terminal constraint
3. Using interpolated Q-function as terminal cost
4. Guaranteeing recursive feasibility and cost improvement

Key properties:
- Recursive feasibility: If feasible at iteration j, feasible at j+1
- Cost improvement: J^{j+1} ≤ J^j
- Convergence to local optimum

The LMPC optimization problem:
    min_{u_0,...,u_{N-1}} Σ l(x_k, u_k) + Q(x_N)
    s.t. x_{k+1} = f(x_k, u_k)
         g(x_k, u_k) ≥ 0
         x_N ∈ Conv(SS_local)

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control
    for Iterative Tasks. IEEE TAC.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

from terminal.convex_hull import TerminalSetManager
from terminal.q_function import QFunctionConfig, QFunctionManager
from terminal.safe_set import FuelAwareSafeSet, SampledSafeSet


@dataclass
class LMPCConfig:
    """Configuration for LMPC."""

    # Horizon
    N: int = 15
    dt: float = 0.1

    # Terminal set
    n_terminal_vertices: int = 10
    use_fuel_aware: bool = True

    # Q-function
    q_method: str = "inverse_distance"
    q_neighbors: int = 10

    # Solver
    max_iter: int = 100
    tol: float = 1e-5
    warm_start: bool = True
    verbose: bool = False

    # Cost weights
    Q_diag: Optional[NDArray] = None  # State cost diagonal
    R_diag: Optional[NDArray] = None  # Control cost diagonal

    # Soft constraints
    soft_terminal: bool = True
    terminal_slack_weight: float = 1e4


@dataclass
class LMPCSolution:
    """Solution from LMPC."""

    success: bool
    X_opt: NDArray  # Optimal state trajectory
    U_opt: NDArray  # Optimal control trajectory
    cost: float  # Total cost
    terminal_cost: float  # Terminal cost Q(x_N)
    lambdas: NDArray  # Terminal convex combination
    solve_time: float
    iterations: int
    status: str


class LMPC:
    """
    Learning Model Predictive Control.

    Combines MPC with learned terminal set and cost from
    historical trajectory data.

    Example:
        >>> from src.lmpc import LMPC, LMPCConfig
        >>>
        >>> # Initialize LMPC
        >>> lmpc = LMPC(dynamics, config)
        >>>
        >>> # Add initial trajectory (e.g., from SCVX)
        >>> lmpc.add_trajectory(X_init, U_init, costs)
        >>>
        >>> # Iterative improvement
        >>> for iteration in range(n_iterations):
        >>>     X, U = lmpc.run_episode(x0)
        >>>     # LMPC automatically adds trajectory to safe set
    """

    def __init__(
        self,
        dynamics,
        config: Optional[LMPCConfig] = None,
        constraint_params=None,
    ):
        """
        Initialize LMPC.

        Args:
            dynamics: Rocket dynamics model
            config: LMPC configuration
            constraint_params: MPC constraint parameters
        """
        self.dynamics = dynamics
        self.config = config or LMPCConfig()
        self.constraint_params = constraint_params

        # State and control dimensions
        self.n_x = 14  # 6-DoF
        self.n_u = 3

        # Initialize safe set
        if self.config.use_fuel_aware:
            self.safe_set = FuelAwareSafeSet(
                n_x=self.n_x,
                n_u=self.n_u,
                fuel_index=0,  # Mass is first element
            )
        else:
            self.safe_set = SampledSafeSet(
                n_x=self.n_x,
                n_u=self.n_u,
            )

        # Terminal set manager
        self._terminal_manager = TerminalSetManager(
            self.safe_set,
            n_vertices=self.config.n_terminal_vertices,
            use_fuel_aware=self.config.use_fuel_aware,
        )

        # Q-function manager
        q_config = QFunctionConfig(
            method=self.config.q_method,
            n_neighbors=self.config.q_neighbors,
        )
        self._q_manager = QFunctionManager(self.safe_set, q_config)

        # Cost matrices
        self._setup_cost_matrices()

        # Warm start storage
        self._X_warm: Optional[NDArray] = None
        self._U_warm: Optional[NDArray] = None

        # Iteration counter
        self._current_iteration = 0

    def _setup_cost_matrices(self) -> None:
        """Set up cost weight matrices."""
        if self.config.Q_diag is not None:
            self.Q = np.diag(self.config.Q_diag)
        else:
            # Default weights for rocket
            Q_diag = np.array(
                [
                    0,  # mass
                    10,
                    10,
                    10,  # position
                    1,
                    1,
                    1,  # velocity
                    0,
                    5,
                    5,
                    0,  # quaternion (penalize qx, qy)
                    0.1,
                    0.1,
                    0.1,  # angular rate
                ]
            )
            self.Q = np.diag(Q_diag)

        if self.config.R_diag is not None:
            self.R = np.diag(self.config.R_diag)
        else:
            self.R = np.diag([0.01, 0.01, 0.01])

    def add_trajectory(
        self,
        states: NDArray,
        controls: NDArray,
        stage_costs: Optional[NDArray] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add a successful trajectory to safe set.

        Args:
            states: State trajectory (T+1, n_x)
            controls: Control trajectory (T, n_u)
            stage_costs: Stage costs (T,), computed if None
            metadata: Additional info
        """
        T = len(controls)

        if stage_costs is None:
            # Compute stage costs
            stage_costs = np.zeros(T)
            for k in range(T):
                x_err = states[k]  # Assuming target is origin
                stage_costs[k] = x_err.T @ self.Q @ x_err + controls[k].T @ self.R @ controls[k]

        self.safe_set.add_trajectory(
            states=states,
            controls=controls,
            stage_costs=stage_costs,
            iteration=self._current_iteration,
            metadata=metadata,
        )

        # Update Q-function
        self._q_manager.update()

        # Invalidate terminal set cache
        self._terminal_manager.invalidate()

    def solve(  # noqa: C901, PLR0912, PLR0915
        self,
        x0: NDArray,
        x_target: Optional[NDArray] = None,  # noqa: ARG002
    ) -> LMPCSolution:
        """
        Solve LMPC optimization problem.

        Args:
            x0: Current state
            x_target: Target state (for cost computation)

        Returns:
            LMPC solution
        """
        if self.safe_set.num_states == 0:
            raise RuntimeError("Safe set is empty. Add initial trajectory first.")

        N = self.config.N
        dt = self.config.dt

        start_time = time.perf_counter()

        # Get terminal set vertices
        available_fuel = x0[0] if self.config.use_fuel_aware else None
        vertices, q_values = self._terminal_manager.get_terminal_set(x0, available_fuel)

        if len(vertices) == 0:
            return LMPCSolution(
                success=False,
                X_opt=np.zeros((N + 1, self.n_x)),
                U_opt=np.zeros((N, self.n_u)),
                cost=np.inf,
                terminal_cost=np.inf,
                lambdas=np.array([]),
                solve_time=0,
                iterations=0,
                status="No feasible terminal set",
            )

        K = len(vertices)

        # Build and solve optimization problem
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(self.n_x, N + 1)
        U = opti.variable(self.n_u, N)
        lambdas = opti.variable(K)

        # Initial condition
        opti.subject_to(X[:, 0] == x0)

        # Dynamics constraints
        for k in range(N):
            x_next = self._casadi_dynamics(X[:, k], U[:, k], dt)
            opti.subject_to(X[:, k + 1] == x_next)

        # Path constraints (simplified)
        if self.constraint_params is not None:
            T_min = self.constraint_params.T_min
            T_max = self.constraint_params.T_max
        else:
            T_min, T_max = 0.5, 5.0

        for k in range(N):
            T_sq = ca.dot(U[:, k], U[:, k])
            opti.subject_to(T_sq >= T_min**2)
            opti.subject_to(T_sq <= T_max**2)

        # Terminal constraint: x_N = V @ λ
        V_ca = ca.DM(vertices.T)
        Q_ca = ca.DM(q_values)

        if self.config.soft_terminal:
            # Soft terminal constraint
            slack = opti.variable(self.n_x)
            opti.subject_to(slack >= 0)
            opti.subject_to(X[:, N] == V_ca @ lambdas + slack)
            slack_cost = self.config.terminal_slack_weight * ca.dot(slack, slack)
        else:
            opti.subject_to(X[:, N] == V_ca @ lambdas)
            slack_cost = 0

        # Convexity constraints
        opti.subject_to(ca.sum1(lambdas) == 1)
        opti.subject_to(lambdas >= 0)

        # Cost function
        Q_ca_mat = ca.DM(self.Q)
        R_ca_mat = ca.DM(self.R)

        stage_cost = 0
        for k in range(N):
            x_k = X[:, k]
            u_k = U[:, k]
            stage_cost += ca.bilin(Q_ca_mat, x_k, x_k) + ca.bilin(R_ca_mat, u_k, u_k)

        # Terminal cost: Q(x_N) ≈ λ^T Q_values
        terminal_cost = ca.dot(Q_ca, lambdas)

        total_cost = stage_cost + terminal_cost + slack_cost
        opti.minimize(total_cost)

        # Warm start
        if self._X_warm is not None and self.config.warm_start:
            opti.set_initial(X, self._X_warm.T)
            opti.set_initial(U, self._U_warm.T)
        else:
            # Initialize with linear interpolation
            X_init = np.zeros((N + 1, self.n_x))
            for k in range(N + 1):
                alpha = k / N
                X_init[k] = (1 - alpha) * x0 + alpha * vertices[0]
            opti.set_initial(X, X_init.T)

        # Initialize lambdas (nearest vertex)
        lambda_init = np.zeros(K)
        lambda_init[0] = 1.0
        opti.set_initial(lambdas, lambda_init)

        # Solver options
        opts = {
            "ipopt.max_iter": self.config.max_iter,
            "ipopt.tol": self.config.tol,
            "ipopt.print_level": 5 if self.config.verbose else 0,
            "print_time": self.config.verbose,
        }
        opti.solver("ipopt", opts)

        # Solve
        try:
            sol = opti.solve()
            success = True
            status = "Optimal"
        except RuntimeError as e:
            sol = opti.debug
            success = False
            status = str(e)

        solve_time = time.perf_counter() - start_time

        # Extract solution
        X_opt = sol.value(X).T
        U_opt = sol.value(U).T
        lambdas_opt = sol.value(lambdas)

        try:
            cost = float(sol.value(total_cost))
            term_cost = float(sol.value(terminal_cost))
            iters = sol.stats()["iter_count"]
        except Exception:
            cost = np.inf
            term_cost = np.inf
            iters = 0

        # Save for warm start
        if success:
            self._X_warm = X_opt
            self._U_warm = U_opt

        return LMPCSolution(
            success=success,
            X_opt=X_opt,
            U_opt=U_opt,
            cost=cost,
            terminal_cost=term_cost,
            lambdas=lambdas_opt,
            solve_time=solve_time,
            iterations=iters,
            status=status,
        )

    def _casadi_dynamics(self, x: ca.MX, u: ca.MX, dt: float) -> ca.MX:
        """CasADi symbolic dynamics (simplified 6-DoF)."""
        # State: [m, r(3), v(3), q(4), ω(3)]
        m = x[0]
        r = x[1:4]
        v = x[4:7]
        q = x[7:11]
        omega = x[11:14]

        # Parameters - handle both normalized and physical units
        if hasattr(self.dynamics.params, "g_I"):
            g_I = ca.DM(self.dynamics.params.g_I) * self.dynamics.params.g0
        else:
            g_I = ca.DM([-9.81, 0, 0])  # Default Earth gravity

        alpha = self.dynamics.params.alpha
        J = ca.DM(self.dynamics.params.J_B)  # Already 3x3 matrix
        r_T_B = ca.DM(self.dynamics.params.r_T_B)

        # Rotation matrix from quaternion
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        C_IB = ca.vertcat(
            ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
        )

        # Thrust
        T_I = C_IB @ u
        T_mag = ca.norm_2(u)

        # Derivatives
        m_dot = -alpha * T_mag
        r_dot = v
        v_dot = T_I / m + g_I

        # Quaternion derivative
        q_dot = 0.5 * ca.vertcat(-ca.dot(omega, q[1:4]), q[0] * omega + ca.cross(omega, q[1:4]))

        # Angular acceleration
        torque = ca.cross(r_T_B, u)
        J_omega = ca.mtimes(J, omega)
        omega_cross_J_omega = ca.cross(omega, J_omega)
        omega_dot = ca.solve(J, torque - omega_cross_J_omega)

        # Euler integration
        m_next = m + dt * m_dot
        r_next = r + dt * r_dot
        v_next = v + dt * v_dot
        q_next = q + dt * q_dot
        q_next = q_next / ca.norm_2(q_next)  # Normalize
        omega_next = omega + dt * omega_dot

        return ca.vertcat(m_next, r_next, v_next, q_next, omega_next)

    def run_episode(
        self,
        x0: NDArray,
        max_steps: int = 100,
        stage_cost_fn: Optional[Callable] = None,
    ) -> Tuple[NDArray, NDArray, NDArray, bool]:
        """
        Run a closed-loop episode with LMPC.

        Args:
            x0: Initial state
            max_steps: Maximum steps
            stage_cost_fn: Custom stage cost function

        Returns:
            X: State trajectory
            U: Control trajectory
            costs: Stage costs
            success: Whether episode succeeded
        """
        X = [x0]
        U = []
        costs = []
        x_current = x0.copy()

        for step in range(max_steps):
            # Solve LMPC
            solution = self.solve(x_current)

            if not solution.success:
                print(f"LMPC failed at step {step}: {solution.status}")
                break

            # Apply first control
            u = solution.u0
            U.append(u)

            # Compute stage cost
            if stage_cost_fn is not None:
                cost = stage_cost_fn(x_current, u)
            else:
                cost = x_current.T @ self.Q @ x_current + u.T @ self.R @ u
            costs.append(cost)

            # Simulate
            x_next = self.dynamics.step(x_current, u, self.config.dt)
            X.append(x_next)
            x_current = x_next

            # Check termination (landing)
            if x_current[1] < 0.1:  # Altitude near zero
                break

            # Shift warm start
            if solution.success:
                self._X_warm = np.vstack([solution.X_opt[1:], solution.X_opt[-1:]])
                self._U_warm = np.vstack([solution.U_opt[1:], solution.U_opt[-1:]])

        X = np.array(X)
        U = np.array(U)
        costs = np.array(costs)

        # Check if successful (landed)
        success = x_current[1] < 0.5

        # Add trajectory to safe set if successful
        if success:
            self.add_trajectory(X, U, costs)
            self._current_iteration += 1

        return X, U, costs, success

    @property
    def iteration(self) -> int:
        """Current iteration number."""
        return self._current_iteration

    def get_statistics(self) -> Dict[str, Any]:
        """Get LMPC statistics."""
        ss_stats = self.safe_set.get_statistics()
        q_stats = self._q_manager.get_statistics()

        return {
            "iteration": self._current_iteration,
            "safe_set": ss_stats,
            "q_function": q_stats,
        }


class SimpleLMPC:
    """
    Simplified LMPC for 3-DoF rocket.

    Faster implementation for testing and prototyping.
    """

    def __init__(
        self,
        dynamics,
        N: int = 10,
        dt: float = 0.1,
    ):
        self.dynamics = dynamics
        self.N = N
        self.dt = dt
        self.n_x = 7  # 3-DoF: [m, r(3), v(3)]
        self.n_u = 3

        # Safe set
        self.safe_set = SampledSafeSet(self.n_x, self.n_u)

        # Q-function
        self._q_manager = QFunctionManager(self.safe_set)

        # Cost weights
        self.Q = np.diag([0, 10, 10, 10, 1, 1, 1])
        self.R = np.diag([0.01, 0.01, 0.01])

    def add_trajectory(self, X: NDArray, U: NDArray) -> None:
        """Add trajectory to safe set."""
        T = len(U)
        costs = np.zeros(T)
        for k in range(T):
            costs[k] = X[k] @ self.Q @ X[k] + U[k] @ self.R @ U[k]

        self.safe_set.add_trajectory(X, U, costs)
        self._q_manager.update()

    def get_terminal_cost(self, x: NDArray) -> float:
        """Get Q-function value."""
        return self._q_manager.evaluate(x)
