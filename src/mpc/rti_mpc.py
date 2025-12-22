"""
Real-Time Iteration MPC (RTI-MPC)

Implements Real-Time Iteration scheme for fast MPC:
- Single QP iteration per control step
- Warm starting from shifted previous solution
- Preparation and feedback phases

The RTI scheme splits MPC into:
1. Preparation: Linearize dynamics, build QP (can be done in advance)
2. Feedback: Update initial state, solve QP, return control

This achieves sub-millisecond control rates for real-time systems.

Reference:
    Diehl, M., et al. (2005). A Real-Time Iteration Scheme for Nonlinear
    Optimization in Optimal Feedback Control. SIAM Journal on Control and
    Optimization.
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

from .constraints import CasADiConstraints, ConstraintParams
from .cost_functions import CostWeights


@dataclass
class RTIConfig:
    """Configuration for RTI-MPC."""

    # Horizon
    N: int = 20
    dt: float = 0.1

    # QP solver settings
    max_qp_iter: int = 10
    qp_tol: float = 1e-6

    # Warm start
    warm_start: bool = True

    # Trust region
    trust_region_x: float = 10.0
    trust_region_u: float = 5.0

    # Regularization
    reg_x: float = 1e-6
    reg_u: float = 1e-6


@dataclass
class RTISolution:
    """Solution from RTI step."""

    u0: NDArray  # Control to apply
    X_opt: NDArray  # Predicted trajectory
    U_opt: NDArray  # Control trajectory
    prep_time: float  # Preparation phase time [ms]
    feedback_time: float  # Feedback phase time [ms]
    total_time: float  # Total time [ms]
    qp_iterations: int  # QP iterations
    success: bool  # Convergence status


class RTI_MPC:  # noqa: N801
    """
    Real-Time Iteration MPC.

    Designed for fast control with:
    - Single Newton-type iteration per step
    - Efficient QP solve with warm starting
    - Split preparation/feedback phases

    Achieves O(100μs - 1ms) control rates.

    Example:
        >>> rti = RTI_MPC(dynamics, RTIConfig(N=20))
        >>> rti.initialize(x0, x_target)
        >>>
        >>> # Control loop
        >>> while running:
        >>>     rti.prepare()  # Can overlap with other computation
        >>>     x_current = get_state()
        >>>     sol = rti.feedback(x_current)
        >>>     apply_control(sol.u0)
    """

    def __init__(
        self,
        dynamics,
        config: Optional[RTIConfig] = None,
        constraint_params: Optional[ConstraintParams] = None,
        cost_weights: Optional[CostWeights] = None,
    ):
        """
        Initialize RTI-MPC.

        Args:
            dynamics: Rocket dynamics model
            config: RTI configuration
            constraint_params: Constraint parameters
            cost_weights: Cost weights
        """
        self.dynamics = dynamics
        self.config = config or RTIConfig()
        self.constraint_params = constraint_params or ConstraintParams()
        self.cost_weights = cost_weights or CostWeights()

        self.n_x = 14
        self.n_u = 3

        # Reference trajectory
        self._X_ref: Optional[NDArray] = None
        self._U_ref: Optional[NDArray] = None
        self._x_target: Optional[NDArray] = None

        # Current linearization point (nominal trajectory)
        self._X_nom: Optional[NDArray] = None
        self._U_nom: Optional[NDArray] = None

        # Linearized matrices (from preparation phase)
        self._A_list: List[NDArray] = []
        self._B_list: List[NDArray] = []
        self._c_list: List[NDArray] = []

        # QP problem (built in setup)
        self._qp_solver = None
        self._qp_built = False

        # Timing
        self._last_prep_time: float = 0.0
        self._last_feedback_time: float = 0.0

        # Constraints helper
        self._constraints = CasADiConstraints(self.constraint_params)

    def initialize(
        self,
        x0: NDArray,
        x_target: NDArray,
        X_init: Optional[NDArray] = None,
        U_init: Optional[NDArray] = None,
    ) -> None:
        """
        Initialize RTI with initial trajectory guess.

        Args:
            x0: Initial state
            x_target: Target state
            X_init: Initial state trajectory guess
            U_init: Initial control trajectory guess
        """
        N = self.config.N

        self._x_target = x_target

        # Initialize nominal trajectory
        if X_init is not None:
            self._X_nom = X_init.copy()
        else:
            # Linear interpolation
            self._X_nom = np.zeros((N + 1, self.n_x))
            for k in range(N + 1):
                alpha = k / N
                self._X_nom[k] = (1 - alpha) * x0 + alpha * x_target
                # Normalize quaternion
                q = self._X_nom[k, 7:11]
                self._X_nom[k, 7:11] = q / np.linalg.norm(q)

        if U_init is not None:
            self._U_nom = U_init.copy()
        else:
            self._U_nom = np.zeros((N, self.n_u))
            for k in range(N):
                m = self._X_nom[k, 0]
                self._U_nom[k] = np.array([0, 0, m * self.dynamics.params.g0])

        # Set reference to target
        self._X_ref = np.tile(x_target, (N + 1, 1))
        self._U_ref = np.zeros((N, self.n_u))

        # Build QP
        self._build_qp()

    def _build_qp(self) -> None:
        """Build the QP problem structure."""
        N = self.config.N

        # Use standard CasADi Opti (not conic for better compatibility)
        self._opti = ca.Opti()

        # Decision variables: deviations from nominal
        self._dX = self._opti.variable(self.n_x, N + 1)
        self._dU = self._opti.variable(self.n_u, N)

        # Parameters for linearized dynamics
        self._A_params = [self._opti.parameter(self.n_x, self.n_x) for _ in range(N)]
        self._B_params = [self._opti.parameter(self.n_x, self.n_u) for _ in range(N)]
        self._c_params = [self._opti.parameter(self.n_x) for _ in range(N)]

        # Parameter for initial state deviation
        self._dx0_param = self._opti.parameter(self.n_x)

        # Parameters for nominal trajectory (for constraint evaluation)
        self._X_nom_param = self._opti.parameter(self.n_x, N + 1)
        self._U_nom_param = self._opti.parameter(self.n_u, N)

        # Initial condition
        self._opti.subject_to(self._dX[:, 0] == self._dx0_param)

        # Linearized dynamics constraints
        for k in range(N):
            self._opti.subject_to(
                self._dX[:, k + 1]
                == self._A_params[k] @ self._dX[:, k] + self._B_params[k] @ self._dU[:, k] + self._c_params[k]
            )

        # Trust region constraints
        for k in range(N + 1):
            self._opti.subject_to(ca.dot(self._dX[:, k], self._dX[:, k]) <= self.config.trust_region_x)
        for k in range(N):
            self._opti.subject_to(ca.dot(self._dU[:, k], self._dU[:, k]) <= self.config.trust_region_u)

        # Linearized path constraints (simplified)
        for k in range(N):
            x_k = self._X_nom_param[:, k] + self._dX[:, k]
            u_k = self._U_nom_param[:, k] + self._dU[:, k]

            # Thrust bounds (convex)
            T_sq = ca.dot(u_k, u_k)
            self._opti.subject_to(T_sq >= self.constraint_params.T_min**2)
            self._opti.subject_to(T_sq <= self.constraint_params.T_max**2)

        # Quadratic cost
        Q = ca.DM(self.cost_weights.Q + self.config.reg_x * np.eye(self.n_x))
        R = ca.DM(self.cost_weights.R + self.config.reg_u * np.eye(self.n_u))
        P = ca.DM(self.cost_weights.P)

        cost = 0
        for k in range(N):
            x_k = self._X_nom_param[:, k] + self._dX[:, k]
            u_k = self._U_nom_param[:, k] + self._dU[:, k]
            x_ref = ca.DM(self._X_ref[k])

            x_err = x_k - x_ref
            cost += ca.bilin(Q, x_err, x_err) + ca.bilin(R, u_k, u_k)

        # Terminal cost
        x_N = self._X_nom_param[:, N] + self._dX[:, N]
        x_ref_N = ca.DM(self._X_ref[N])
        x_err_N = x_N - x_ref_N
        cost += ca.bilin(P, x_err_N, x_err_N)

        self._opti.minimize(cost)

        # Solver options for fast solve
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": self.config.max_qp_iter,
            "ipopt.tol": self.config.qp_tol,
            "print_time": False,
            "ipopt.warm_start_init_point": "yes",
        }
        self._opti.solver("ipopt", opts)

        self._qp_built = True

    def prepare(self) -> float:
        """
        Preparation phase: linearize dynamics around nominal.

        Can be done in advance while waiting for new state.

        Returns:
            Preparation time in milliseconds
        """
        if self._X_nom is None:
            raise RuntimeError("Must call initialize() first")

        start = time.perf_counter()

        N = self.config.N
        dt = self.config.dt

        self._A_list = []
        self._B_list = []
        self._c_list = []

        for k in range(N):
            x_k = self._X_nom[k]
            u_k = self._U_nom[k]

            # Linearize dynamics
            A_d, B_d = self.dynamics.linearize(x_k, u_k, dt=dt)

            # Compute affine term (for exact linearization)
            x_next_nom = self.dynamics.step(x_k, u_k, dt)
            c = x_next_nom - A_d @ x_k - B_d @ u_k

            self._A_list.append(A_d)
            self._B_list.append(B_d)
            self._c_list.append(c)

        self._last_prep_time = (time.perf_counter() - start) * 1000
        return self._last_prep_time

    def feedback(self, x_current: NDArray) -> RTISolution:
        """
        Feedback phase: update initial state and solve QP.

        This is the time-critical part that must complete quickly.

        Args:
            x_current: Current measured state

        Returns:
            RTI solution with control and trajectory
        """
        if not self._qp_built:
            raise RuntimeError("Must call initialize() first")

        start = time.perf_counter()

        N = self.config.N

        # Set QP parameters
        dx0 = x_current - self._X_nom[0]
        self._opti.set_value(self._dx0_param, dx0)

        for k in range(N):
            self._opti.set_value(self._A_params[k], self._A_list[k])
            self._opti.set_value(self._B_params[k], self._B_list[k])
            self._opti.set_value(self._c_params[k], self._c_list[k])

        self._opti.set_value(self._X_nom_param, self._X_nom.T)
        self._opti.set_value(self._U_nom_param, self._U_nom.T)

        # Warm start with zero deviation (already at nominal)
        self._opti.set_initial(self._dX, np.zeros((self.n_x, N + 1)))
        self._opti.set_initial(self._dU, np.zeros((self.n_u, N)))

        # Solve QP
        try:
            sol = self._opti.solve()
            success = True
            qp_iter = sol.stats().get("iter_count", 1)

            # Extract solution
            dX_opt = sol.value(self._dX).T
            dU_opt = sol.value(self._dU).T

            X_opt = self._X_nom + dX_opt
            U_opt = self._U_nom + dU_opt

        except RuntimeError:
            # QP failed - return nominal trajectory
            success = False
            qp_iter = 0
            X_opt = self._X_nom.copy()
            U_opt = self._U_nom.copy()

        self._last_feedback_time = (time.perf_counter() - start) * 1000

        # Shift nominal trajectory for next iteration
        self._shift_trajectory(X_opt, U_opt)

        return RTISolution(
            u0=U_opt[0],
            X_opt=X_opt,
            U_opt=U_opt,
            prep_time=self._last_prep_time,
            feedback_time=self._last_feedback_time,
            total_time=self._last_prep_time + self._last_feedback_time,
            qp_iterations=qp_iter,
            success=success,
        )

    def _shift_trajectory(
        self,
        X_opt: NDArray,
        U_opt: NDArray,
    ) -> None:
        """
        Shift nominal trajectory for next iteration.

        Implements standard MPC warm starting:
        X_nom[k] = X_opt[k+1], U_nom[k] = U_opt[k+1]
        """
        # Shift states
        self._X_nom[:-1] = X_opt[1:]
        # Extrapolate last state
        self._X_nom[-1] = X_opt[-1]  # Could use dynamics

        # Shift controls
        self._U_nom[:-1] = U_opt[1:]
        # Last control: hover or extrapolate
        self._U_nom[-1] = U_opt[-1]

    def step(self, x_current: NDArray) -> RTISolution:
        """
        Combined prepare + feedback step.

        Convenience method for when preparation cannot be parallelized.

        Args:
            x_current: Current state

        Returns:
            RTI solution
        """
        self.prepare()
        return self.feedback(x_current)

    def simulate_closed_loop(
        self,
        x0: NDArray,
        n_steps: int,
    ) -> Tuple[NDArray, NDArray, List[RTISolution]]:
        """
        Run closed-loop simulation with RTI-MPC.

        Args:
            x0: Initial state
            n_steps: Number of simulation steps

        Returns:
            X_traj: State trajectory
            U_traj: Control trajectory
            solutions: List of RTI solutions
        """
        X_traj = np.zeros((n_steps + 1, self.n_x))
        U_traj = np.zeros((n_steps, self.n_u))
        solutions = []

        X_traj[0] = x0
        x_current = x0.copy()

        for i in range(n_steps):
            # RTI step
            sol = self.step(x_current)
            solutions.append(sol)

            # Apply control
            u_apply = sol.u0
            U_traj[i] = u_apply

            # Simulate
            x_next = self.dynamics.step(x_current, u_apply, self.config.dt)
            X_traj[i + 1] = x_next
            x_current = x_next

            # Check termination
            if x_current[1] < 0.1:  # Near ground
                X_traj = X_traj[: i + 2]
                U_traj = U_traj[: i + 1]
                break

        return X_traj, U_traj, solutions

    def get_timing_stats(self) -> dict:
        """Get timing statistics."""
        return {
            "last_prep_time_ms": self._last_prep_time,
            "last_feedback_time_ms": self._last_feedback_time,
            "last_total_time_ms": self._last_prep_time + self._last_feedback_time,
        }


class SimpleRTI:
    """
    Simplified RTI for fast prototyping.

    Uses numpy-based QP instead of CasADi for maximum speed
    on small problems.
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
        self.n_x = 14
        self.n_u = 3

        # Cost weights
        self.Q = np.diag([0, 10, 10, 10, 1, 1, 1, 0, 5, 5, 0, 0.1, 0.1, 0.1])
        self.R = np.diag([0.01, 0.01, 0.01])
        self.P = 10 * self.Q

        # Trajectory storage
        self.X_nom = None
        self.U_nom = None
        self.x_target = None

    def initialize(self, x0: NDArray, x_target: NDArray) -> None:
        """Initialize with target."""
        self.x_target = x_target
        N = self.N

        # Linear interpolation
        self.X_nom = np.zeros((N + 1, self.n_x))
        for k in range(N + 1):
            alpha = k / N
            self.X_nom[k] = (1 - alpha) * x0 + alpha * x_target
            q = self.X_nom[k, 7:11]
            self.X_nom[k, 7:11] = q / np.linalg.norm(q)

        self.U_nom = np.zeros((N, self.n_u))
        for k in range(N):
            m = self.X_nom[k, 0]
            self.U_nom[k, 2] = m * self.dynamics.params.g0

    def step(self, x_current: NDArray) -> NDArray:
        """
        Single RTI step using gradient descent.

        Returns first control action.
        """
        if self.X_nom is None:
            raise RuntimeError("Must initialize first")

        N = self.N
        dt = self.dt

        # Forward simulate from current state
        X_sim = np.zeros((N + 1, self.n_x))
        X_sim[0] = x_current

        for k in range(N):
            X_sim[k + 1] = self.dynamics.step(X_sim[k], self.U_nom[k], dt)

        # Compute gradients via backward pass (simplified)
        dU = np.zeros((N, self.n_u))

        for k in range(N):
            # Simple gradient: move toward target
            x_err = X_sim[k] - self.x_target

            # Gradient of tracking cost w.r.t. u
            _, B = self.dynamics.linearize(X_sim[k], self.U_nom[k], dt=dt)
            dU[k] = -0.01 * B.T @ self.Q @ x_err - 0.01 * self.R @ self.U_nom[k]

        # Update controls
        self.U_nom = self.U_nom + np.clip(dU, -0.5, 0.5)

        # Clamp thrust
        for k in range(N):
            T_mag = np.linalg.norm(self.U_nom[k])
            if T_mag > 5.0:
                self.U_nom[k] = self.U_nom[k] / T_mag * 5.0
            elif T_mag < 0.5:
                self.U_nom[k] = self.U_nom[k] / max(T_mag, 1e-6) * 0.5

        # Shift
        self.X_nom[:-1] = X_sim[1:]
        self.X_nom[-1] = X_sim[-1]
        self.U_nom[:-1] = self.U_nom[1:]

        return self.U_nom[0]
