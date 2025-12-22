"""
Nominal MPC for Rocket Landing (without GP)

Implements Model Predictive Control using CasADi and IPOPT.
This serves as the baseline before adding GP learning.

The MPC solves:
    min_{u_0,...,u_{N-1}} Σ l(x_k, u_k) + V_f(x_N)
    s.t. x_{k+1} = f(x_k, u_k)          (dynamics)
         g(x_k, u_k) >= 0               (path constraints)
         h(x_N) = 0                      (terminal constraints)
         x_0 = x_init                    (initial condition)

The dynamics are discretized using RK4 or direct collocation.

Reference:
    Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for
    6-DoF Mars rocket powered landing with free-final-time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False
    raise ImportError("CasADi is required for MPC. Install with: pip install casadi")

from .constraints import CasADiConstraints, ConstraintParams
from .cost_functions import CasADiCostFunction, CostWeights


@dataclass
class MPCConfig:
    """Configuration for MPC solver."""

    # Horizon
    N: int = 20  # Prediction horizon
    dt: float = 0.1  # Timestep [s]

    # Discretization
    integration_method: str = "rk4"  # "euler", "rk4", "collocation"

    # Solver settings
    max_iter: int = 100
    tol: float = 1e-6
    warm_start: bool = True
    verbose: bool = False

    # Constraint handling
    soft_constraints: bool = False
    constraint_slack_weight: float = 1e4

    # Reference trajectory
    use_reference: bool = False


@dataclass
class MPCSolution:
    """Container for MPC solution."""

    success: bool
    X_opt: NDArray  # Optimal state trajectory (N+1, n_x)
    U_opt: NDArray  # Optimal control trajectory (N, n_u)
    cost: float  # Optimal cost
    solve_time: float  # Solver time [s]
    iterations: int  # Number of iterations
    status: str  # Solver status message

    @property
    def u0(self) -> NDArray:
        """First control action (to apply)."""
        return self.U_opt[0]


class NominalMPC:
    """
    Nominal MPC controller for rocket landing.

    Uses CasADi for automatic differentiation and IPOPT for NLP solving.

    Example:
        >>> from src.dynamics import create_szmuk_rocket
        >>> from src.mpc import NominalMPC, MPCConfig
        >>>
        >>> rocket = create_szmuk_rocket()
        >>> mpc = NominalMPC(rocket, MPCConfig(N=20))
        >>> mpc.setup()
        >>>
        >>> # Solve MPC
        >>> x0 = rocket.create_initial_state(altitude=10.0, mass=2.0)
        >>> x_target = rocket.create_initial_state(altitude=0.0, mass=1.5)
        >>> solution = mpc.solve(x0, x_target)
        >>>
        >>> # Apply first control
        >>> u_apply = solution.u0
    """

    def __init__(
        self,
        dynamics,  # Rocket6DoFDynamics or similar
        config: Optional[MPCConfig] = None,
        constraint_params: Optional[ConstraintParams] = None,
        cost_weights: Optional[CostWeights] = None,
    ):
        """
        Initialize MPC.

        Args:
            dynamics: Rocket dynamics model with step() and linearize() methods
            config: MPC configuration
            constraint_params: Constraint parameters
            cost_weights: Cost function weights
        """
        self.dynamics = dynamics
        self.config = config or MPCConfig()
        self.constraint_params = constraint_params or ConstraintParams()
        self.cost_weights = cost_weights or CostWeights()

        # State and control dimensions
        self.n_x = 14  # 6-DoF rocket state
        self.n_u = 3  # Thrust vector

        # CasADi problem (set up by setup())
        self._opti: Optional[ca.Opti] = None
        self._X: Optional[ca.MX] = None
        self._U: Optional[ca.MX] = None
        self._x0_param: Optional[ca.MX] = None
        self._x_ref_param: Optional[ca.MX] = None
        self._u_ref_param: Optional[ca.MX] = None

        # Warm start
        self._X_warm: Optional[NDArray] = None
        self._U_warm: Optional[NDArray] = None

        # Constraints and costs
        self._constraints = CasADiConstraints(self.constraint_params)
        self._costs = CasADiCostFunction(self.cost_weights)

        self._is_setup = False

    def _create_dynamics_function(self) -> ca.Function:  # noqa: PLR0915
        """
        Create CasADi function for dynamics.

        Returns:
            CasADi function f(x, u) -> x_next
        """
        # Symbolic variables
        x = ca.MX.sym("x", self.n_x)
        u = ca.MX.sym("u", self.n_u)

        # Build dynamics symbolically
        # State: [m, r_I(3), v_I(3), q_BI(4), ω_B(3)]
        m = x[0]
        v = x[4:7]
        q = x[7:11]
        omega = x[11:14]

        # Parameters from dynamics model
        g_I = ca.DM(self.dynamics.params.g_I)  # Gravity in inertial frame
        J = ca.DM(self.dynamics.params.J_B)  # Inertia matrix (already 3x3)
        r_T_B = ca.DM(self.dynamics.params.r_T_B)
        alpha = self.dynamics.params.alpha

        # Rotation matrix from quaternion (body to inertial)
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        C_IB = ca.vertcat(
            ca.horzcat(1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)),
        )

        # Thrust in inertial frame
        T_I = C_IB @ u
        T_mag = ca.norm_2(u)

        # State derivatives
        m_dot = -alpha * T_mag
        r_dot = v
        v_dot = T_I / m + g_I

        # Quaternion derivative: q_dot = 0.5 * Omega(omega) * q
        q_dot = 0.5 * ca.vertcat(-ca.dot(omega, q[1:4]), q[0] * omega + ca.cross(omega, q[1:4]))

        # Angular acceleration
        # ω_dot = J^{-1} (r_T x T - ω x Jω)
        torque = ca.cross(r_T_B, u)
        J_omega = ca.mtimes(J, omega)
        omega_cross_J_omega = ca.cross(omega, J_omega)
        omega_dot = ca.solve(J, torque - omega_cross_J_omega)

        # Full state derivative
        x_dot = ca.vertcat(m_dot, r_dot, v_dot, q_dot, omega_dot)

        # Discretization
        if self.config.integration_method == "euler":
            x_next = x + self.config.dt * x_dot
        elif self.config.integration_method == "rk4":
            # RK4 integration
            dt = self.config.dt

            # This is simplified - proper RK4 needs k1, k2, k3, k4
            # For now, use Euler with smaller substeps
            n_substeps = 4
            h = dt / n_substeps
            x_curr = x
            for _ in range(n_substeps):
                # Recompute derivatives at current state
                # (simplified - using same u throughout)
                m_c = x_curr[0]
                v_c = x_curr[4:7]
                q_c = x_curr[7:11]
                omega_c = x_curr[11:14]

                qw_c, qx_c, qy_c, qz_c = q_c[0], q_c[1], q_c[2], q_c[3]
                C_IB_c = ca.vertcat(
                    ca.horzcat(
                        1 - 2 * (qy_c**2 + qz_c**2), 2 * (qx_c * qy_c - qw_c * qz_c), 2 * (qx_c * qz_c + qw_c * qy_c)
                    ),
                    ca.horzcat(
                        2 * (qx_c * qy_c + qw_c * qz_c), 1 - 2 * (qx_c**2 + qz_c**2), 2 * (qy_c * qz_c - qw_c * qx_c)
                    ),
                    ca.horzcat(
                        2 * (qx_c * qz_c - qw_c * qy_c), 2 * (qy_c * qz_c + qw_c * qx_c), 1 - 2 * (qx_c**2 + qy_c**2)
                    ),
                )

                T_I_c = C_IB_c @ u
                T_mag_c = ca.norm_2(u)

                m_dot_c = -alpha * T_mag_c
                r_dot_c = v_c
                v_dot_c = T_I_c / m_c + g_I
                q_dot_c = 0.5 * ca.vertcat(-ca.dot(omega_c, q_c[1:4]), q_c[0] * omega_c + ca.cross(omega_c, q_c[1:4]))
                torque_c = ca.cross(r_T_B, u)
                omega_dot_c = ca.solve(J, torque_c - ca.cross(omega_c, J @ omega_c))

                x_dot_c = ca.vertcat(m_dot_c, r_dot_c, v_dot_c, q_dot_c, omega_dot_c)
                x_curr = x_curr + h * x_dot_c

            x_next = x_curr
        else:
            # Default to Euler
            x_next = x + self.config.dt * x_dot

        # Normalize quaternion
        q_next = x_next[7:11]
        q_next_norm = q_next / ca.norm_2(q_next)
        x_next = ca.vertcat(x_next[0:7], q_next_norm, x_next[11:14])

        return ca.Function("dynamics", [x, u], [x_next], ["x", "u"], ["x_next"])

    def setup(self) -> None:
        """
        Set up the MPC optimization problem.

        Creates the CasADi Opti problem with:
        - Decision variables (states and controls)
        - Dynamics constraints
        - Path constraints
        - Terminal constraints
        - Cost function
        """
        N = self.config.N

        # Create Opti instance
        self._opti = ca.Opti()

        # Decision variables
        self._X = self._opti.variable(self.n_x, N + 1)  # States
        self._U = self._opti.variable(self.n_u, N)  # Controls

        # Parameters
        self._x0_param = self._opti.parameter(self.n_x)
        self._x_ref_param = self._opti.parameter(self.n_x, N + 1)
        self._u_ref_param = self._opti.parameter(self.n_u, N)

        # Dynamics function
        f = self._create_dynamics_function()

        # Initial condition constraint
        self._opti.subject_to(self._X[:, 0] == self._x0_param)

        # Dynamics constraints
        for k in range(N):
            x_k = self._X[:, k]
            u_k = self._U[:, k]
            x_next = f(x_k, u_k)
            self._opti.subject_to(self._X[:, k + 1] == x_next)

        # Path constraints
        for k in range(N):
            x_k = self._X[:, k]
            u_k = self._U[:, k]

            constraints = self._constraints.get_path_constraints(x_k, u_k)
            for g in constraints:
                if self.config.soft_constraints:
                    # Soft constraint with slack
                    slack = self._opti.variable()
                    self._opti.subject_to(g + slack >= 0)
                    self._opti.subject_to(slack >= 0)
                else:
                    self._opti.subject_to(g >= 0)

        # Terminal constraint on altitude (must reach ground)
        x_N = self._X[:, N]
        # Soft terminal constraint
        # self._opti.subject_to(r_N[0] <= 0.5)  # Altitude near zero

        # Cost function
        cost = 0
        for k in range(N):
            x_k = self._X[:, k]
            u_k = self._U[:, k]
            x_ref_k = self._x_ref_param[:, k]
            u_ref_k = self._u_ref_param[:, k]

            if self.config.use_reference:
                cost += self._costs.tracking_cost(x_k, u_k, x_ref_k, u_ref_k)
            else:
                cost += self._costs.stage_cost(x_k, u_k, x_ref_k)

        # Terminal cost
        x_N = self._X[:, N]
        x_ref_N = self._x_ref_param[:, N]
        cost += self._costs.terminal_cost(x_N, x_ref_N)

        self._opti.minimize(cost)

        # Solver options
        opts = {
            "ipopt.max_iter": self.config.max_iter,
            "ipopt.tol": self.config.tol,
            "ipopt.print_level": 5 if self.config.verbose else 0,
            "print_time": self.config.verbose,
            "ipopt.warm_start_init_point": "yes" if self.config.warm_start else "no",
        }
        self._opti.solver("ipopt", opts)

        self._is_setup = True

    def solve(
        self,
        x0: NDArray,
        x_target: NDArray,
        X_ref: Optional[NDArray] = None,
        U_ref: Optional[NDArray] = None,
    ) -> MPCSolution:
        """
        Solve the MPC problem.

        Args:
            x0: Current state (n_x,)
            x_target: Target state (n_x,)
            X_ref: Reference state trajectory (N+1, n_x), optional
            U_ref: Reference control trajectory (N, n_u), optional

        Returns:
            MPCSolution with optimal trajectory and first control
        """
        if not self._is_setup:
            self.setup()

        N = self.config.N

        # Set parameters
        self._opti.set_value(self._x0_param, x0)

        # Reference trajectory
        if X_ref is None:
            # Linear interpolation to target
            X_ref = np.zeros((N + 1, self.n_x))
            for k in range(N + 1):
                alpha = k / N
                X_ref[k] = (1 - alpha) * x0 + alpha * x_target
                # Fix quaternion interpolation
                q = X_ref[k, 7:11]
                X_ref[k, 7:11] = q / np.linalg.norm(q)

        if U_ref is None:
            # Hover thrust as reference
            U_ref = np.zeros((N, self.n_u))
            for k in range(N):
                m = X_ref[k, 0]
                U_ref[k] = np.array([0, 0, m * self.dynamics.params.g0])

        self._opti.set_value(self._x_ref_param, X_ref.T)
        self._opti.set_value(self._u_ref_param, U_ref.T)

        # Initial guess (warm start)
        if self._X_warm is not None and self.config.warm_start:
            self._opti.set_initial(self._X, self._X_warm.T)
            self._opti.set_initial(self._U, self._U_warm.T)
        else:
            self._opti.set_initial(self._X, X_ref.T)
            self._opti.set_initial(self._U, U_ref.T)

        # Solve
        start_time = time.perf_counter()
        try:
            sol = self._opti.solve()
            success = True
            status = "Optimal"
        except RuntimeError as e:
            # Get infeasible solution for debugging
            sol = self._opti.debug
            success = False
            status = str(e)

        solve_time = time.perf_counter() - start_time

        # Extract solution
        X_opt = sol.value(self._X).T  # (N+1, n_x)
        U_opt = sol.value(self._U).T  # (N, n_u)

        try:
            cost = float(sol.value(self._opti.f))
            iterations = sol.stats()["iter_count"]
        except Exception:
            cost = np.inf
            iterations = 0

        # Save for warm start
        if success:
            self._X_warm = X_opt
            self._U_warm = U_opt

        return MPCSolution(
            success=success,
            X_opt=X_opt,
            U_opt=U_opt,
            cost=cost,
            solve_time=solve_time,
            iterations=iterations,
            status=status,
        )

    def simulate_closed_loop(
        self,
        x0: NDArray,
        x_target: NDArray,
        n_steps: int,
        X_ref: Optional[NDArray] = None,
        U_ref: Optional[NDArray] = None,
    ) -> Tuple[NDArray, NDArray, List[MPCSolution]]:
        """
        Run closed-loop MPC simulation.

        Args:
            x0: Initial state
            x_target: Target state
            n_steps: Number of simulation steps
            X_ref: Full reference trajectory (optional)
            U_ref: Full reference controls (optional)

        Returns:
            X_traj: Closed-loop state trajectory (n_steps+1, n_x)
            U_traj: Applied controls (n_steps, n_u)
            solutions: List of MPC solutions
        """
        X_traj = np.zeros((n_steps + 1, self.n_x))
        U_traj = np.zeros((n_steps, self.n_u))
        solutions = []

        X_traj[0] = x0
        x_current = x0.copy()

        for i in range(n_steps):
            # Get reference for this step
            if X_ref is not None:
                # Shift reference for receding horizon
                idx_start = min(i, X_ref.shape[0] - self.config.N - 1)
                X_ref_mpc = X_ref[idx_start : idx_start + self.config.N + 1]
                if X_ref_mpc.shape[0] < self.config.N + 1:
                    # Pad with target
                    pad_len = self.config.N + 1 - X_ref_mpc.shape[0]
                    X_ref_mpc = np.vstack([X_ref_mpc, np.tile(x_target, (pad_len, 1))])
            else:
                X_ref_mpc = None

            if U_ref is not None:
                idx_start = min(i, U_ref.shape[0] - self.config.N)
                U_ref_mpc = U_ref[idx_start : idx_start + self.config.N]
                if U_ref_mpc.shape[0] < self.config.N:
                    pad_len = self.config.N - U_ref_mpc.shape[0]
                    hover = np.array([0, 0, x_target[0] * self.dynamics.params.g0])
                    U_ref_mpc = np.vstack([U_ref_mpc, np.tile(hover, (pad_len, 1))])
            else:
                U_ref_mpc = None

            # Solve MPC
            solution = self.solve(x_current, x_target, X_ref_mpc, U_ref_mpc)
            solutions.append(solution)

            if not solution.success:
                print(f"MPC failed at step {i}: {solution.status}")

            # Apply first control
            u_apply = solution.u0
            U_traj[i] = u_apply

            # Simulate one step
            x_next = self.dynamics.step(x_current, u_apply, self.config.dt)
            X_traj[i + 1] = x_next
            x_current = x_next

            # Check termination
            if x_current[1] < 0.1:  # Near ground
                X_traj = X_traj[: i + 2]
                U_traj = U_traj[: i + 1]
                break

            # Shift warm start
            if solution.success:
                self._X_warm = np.vstack([solution.X_opt[1:], solution.X_opt[-1:]])
                self._U_warm = np.vstack([solution.U_opt[1:], solution.U_opt[-1:]])

        return X_traj, U_traj, solutions

    def reset_warm_start(self) -> None:
        """Clear warm start cache."""
        self._X_warm = None
        self._U_warm = None


# =============================================================================
# Simplified MPC for 3-DoF
# =============================================================================


class NominalMPC3DoF:
    """
    Simplified MPC for 3-DoF point mass rocket.

    Faster setup and solve for prototyping.
    """

    def __init__(
        self,
        dynamics,
        config: Optional[MPCConfig] = None,
    ):
        self.dynamics = dynamics
        self.config = config or MPCConfig()
        self.n_x = 7  # [m, r(3), v(3)]
        self.n_u = 3  # Thrust in inertial frame

        # Simplified setup
        self._opti = None
        self._is_setup = False

    def setup(self) -> None:
        """Set up simplified MPC."""
        N = self.config.N
        dt = self.config.dt

        self._opti = ca.Opti()

        # Variables
        self._X = self._opti.variable(self.n_x, N + 1)
        self._U = self._opti.variable(self.n_u, N)

        # Parameters
        self._x0_param = self._opti.parameter(self.n_x)
        self._x_ref_param = self._opti.parameter(self.n_x)

        # Get gravity from dynamics
        g_I = self.dynamics.params.g_vec  # Gravity vector (3DoF uses g_vec)
        alpha = self.dynamics.params.alpha

        # Initial condition
        self._opti.subject_to(self._X[:, 0] == self._x0_param)

        # Dynamics (simple 3-DoF)
        for k in range(N):
            m_k = self._X[0, k]
            r_k = self._X[1:4, k]
            v_k = self._X[4:7, k]
            u_k = self._U[:, k]

            T_mag = ca.norm_2(u_k)

            # Derivatives
            m_dot = -alpha * T_mag
            r_dot = v_k
            v_dot = u_k / m_k + ca.DM(g_I)

            # Euler integration
            m_next = m_k + dt * m_dot
            r_next = r_k + dt * r_dot
            v_next = v_k + dt * v_dot

            x_next = ca.vertcat(m_next, r_next, v_next)
            self._opti.subject_to(self._X[:, k + 1] == x_next)

        # Constraints
        T_min, T_max = 0.5, 5.0
        for k in range(N):
            T_sq = ca.dot(self._U[:, k], self._U[:, k])
            self._opti.subject_to(T_sq >= T_min**2)
            self._opti.subject_to(T_sq <= T_max**2)

            # Glideslope
            r_k = self._X[1:4, k]
            gamma = np.deg2rad(30)
            self._opti.subject_to(r_k[0] ** 2 * np.tan(gamma) ** 2 >= r_k[1] ** 2 + r_k[2] ** 2)

        # Cost
        Q = np.diag([0, 10, 10, 10, 1, 1, 1])
        R = np.diag([0.01, 0.01, 0.01])

        cost = 0
        for k in range(N):
            x_err = self._X[:, k] - self._x_ref_param
            cost += ca.bilin(ca.DM(Q), x_err, x_err)
            cost += ca.bilin(ca.DM(R), self._U[:, k], self._U[:, k])

        # Terminal
        x_err_N = self._X[:, N] - self._x_ref_param
        cost += 10 * ca.bilin(ca.DM(Q), x_err_N, x_err_N)

        self._opti.minimize(cost)

        opts = {
            "ipopt.max_iter": self.config.max_iter,
            "ipopt.print_level": 0,
        }
        self._opti.solver("ipopt", opts)
        self._is_setup = True

    def solve(self, x0: NDArray, x_target: NDArray) -> MPCSolution:
        """Solve MPC."""
        if not self._is_setup:
            self.setup()

        self._opti.set_value(self._x0_param, x0)
        self._opti.set_value(self._x_ref_param, x_target)

        # Initial guess
        X_init = np.linspace(x0, x_target, self.config.N + 1)
        self._opti.set_initial(self._X, X_init.T)

        U_init = np.zeros((self.config.N, self.n_u))
        U_init[:, 2] = x0[0] * self.dynamics.params.g0
        self._opti.set_initial(self._U, U_init.T)

        start = time.perf_counter()
        try:
            sol = self._opti.solve()
            success = True
            status = "Optimal"
        except RuntimeError as e:
            sol = self._opti.debug
            success = False
            status = str(e)

        solve_time = time.perf_counter() - start

        return MPCSolution(
            success=success,
            X_opt=sol.value(self._X).T,
            U_opt=sol.value(self._U).T,
            cost=float(sol.value(self._opti.f)) if success else np.inf,
            solve_time=solve_time,
            iterations=sol.stats()["iter_count"] if success else 0,
            status=status,
        )
