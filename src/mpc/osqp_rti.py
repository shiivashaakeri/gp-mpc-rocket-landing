"""
High-Performance RTI-MPC with OSQP Backend

Optimized Real-Time Iteration MPC for 50Hz+ control rates:
- OSQP: First-order QP solver with warm starting
- Precomputed matrices where possible
- Efficient sparse matrix operations
- Memory-efficient data structures

Target: 20ms total loop time (50Hz)
- Preparation: < 10ms
- Feedback: < 5ms
- GP prediction: < 5ms

Reference:
    Stellato, B., et al. (2020). OSQP: An Operator Splitting Solver
    for Quadratic Programs. Mathematical Programming Computation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

try:
    import osqp

    HAS_OSQP = True
except ImportError:
    HAS_OSQP = False

try:
    import casadi as ca  # noqa: F401

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class OSQPRTIConfig:
    """Configuration for OSQP-based RTI-MPC."""

    # Horizon
    N: int = 15
    dt: float = 0.1

    # OSQP settings
    osqp_verbose: bool = False
    osqp_max_iter: int = 50
    osqp_eps_abs: float = 1e-4
    osqp_eps_rel: float = 1e-4
    osqp_polish: bool = False  # Disable for speed
    osqp_warm_start: bool = True
    osqp_scaling: int = 3  # Reduced scaling iterations

    # RTI settings
    max_rti_iter: int = 1  # Single iteration for RTI
    trust_region: float = 10.0

    # Problem dimensions (set automatically)
    n_x: int = 7  # 3-DoF state dimension
    n_u: int = 3  # Control dimension

    # Precomputation
    precompute_matrices: bool = True


@dataclass
class OSQPRTISolution:
    """Solution from OSQP RTI."""

    u0: NDArray  # Control to apply
    X_opt: NDArray  # Optimal trajectory
    U_opt: NDArray  # Optimal controls
    cost: float  # Optimal cost
    prep_time_ms: float  # Preparation time
    feedback_time_ms: float  # Feedback time
    total_time_ms: float  # Total time
    osqp_iterations: int  # OSQP iterations
    success: bool  # Solve success


class OSQPRTIMPC:
    """
    High-Performance RTI-MPC using OSQP.

    Formulates MPC as a sparse QP and uses OSQP for fast solving:

        min  Σ x'Qx + u'Ru + x_N'Q_f x_N
        s.t. x_{k+1} = A_k x_k + B_k u_k + c_k  (linearized dynamics)
             x_min ≤ x_k ≤ x_max
             u_min ≤ u_k ≤ u_max
             x_0 = x_init

    Converted to standard QP:
        min  0.5 z'Pz + q'z
        s.t. l ≤ Az ≤ u

    where z = [x_0, u_0, x_1, u_1, ..., x_N]

    Example:
        >>> rti = OSQPRTIMPC(dynamics, config)
        >>> rti.initialize(x0, x_target)
        >>>
        >>> # 50Hz control loop
        >>> while running:
        >>>     sol = rti.step(x_current)  # < 20ms
        >>>     apply_control(sol.u0)
    """

    def __init__(
        self,
        dynamics,
        config: Optional[OSQPRTIConfig] = None,
    ):
        """
        Initialize OSQP RTI-MPC.

        Args:
            dynamics: Rocket dynamics (3-DoF recommended for speed)
            config: RTI configuration
        """
        if not HAS_OSQP:
            raise ImportError("OSQP not available. Install with: pip install osqp")

        self.dynamics = dynamics
        self.config = config or OSQPRTIConfig()

        # Dimensions
        self.n_x = self.config.n_x
        self.n_u = self.config.n_u
        self.N = self.config.N

        # Total decision variables: N+1 states + N controls
        self.n_vars = (self.N + 1) * self.n_x + self.N * self.n_u

        # Cost matrices
        self._setup_cost_matrices()

        # Constraint matrices (will be updated)
        self._setup_constraint_structure()

        # OSQP solver
        self._solver: Optional[osqp.OSQP] = None

        # Reference trajectory
        self._x_ref: Optional[NDArray] = None
        self._u_ref: Optional[NDArray] = None

        # Previous solution for warm start
        self._X_prev: Optional[NDArray] = None
        self._U_prev: Optional[NDArray] = None

        # Linearization point
        self._X_lin: Optional[NDArray] = None
        self._U_lin: Optional[NDArray] = None

        # Precomputed sparse patterns
        self._P_sparse: Optional[sparse.csc_matrix] = None
        self._A_sparse: Optional[sparse.csc_matrix] = None

    def _setup_cost_matrices(self) -> None:
        """Set up cost weight matrices."""
        # State cost (penalize deviation from target)
        Q_diag = np.array([0.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1.0])  # 3-DoF
        if len(Q_diag) != self.n_x:
            Q_diag = np.ones(self.n_x)
            Q_diag[1:4] = 10.0  # Position
            Q_diag[4:7] = 1.0  # Velocity

        self.Q = np.diag(Q_diag)
        self.Q_f = self.Q * 10  # Terminal cost

        # Control cost
        R_diag = np.ones(self.n_u) * 0.01
        self.R = np.diag(R_diag)

    def _setup_constraint_structure(self) -> None:
        """Set up sparse constraint matrix structure."""
        N, n_x, n_u = self.N, self.n_x, self.n_u
        n_vars = self.n_vars

        # Dynamics constraints: n_x * N equations
        # Bound constraints: n_vars bounds
        # Initial condition: n_x equations

        self.n_eq = n_x * (N + 1)  # Dynamics + initial condition
        self.n_ineq = 2 * n_vars  # Variable bounds
        self.n_constraints = self.n_eq + N * n_u  # Dynamics + thrust bounds

        # Bounds
        self.x_min = np.array([-np.inf, -100, -100, -100, -50, -50, -50])[:n_x]
        self.x_max = np.array([np.inf, 500, 100, 100, 50, 50, 50])[:n_x]
        self.u_min = np.array([0.3, -5, -5])[:n_u]
        self.u_max = np.array([5.0, 5, 5])[:n_u]

    def _build_cost_matrix(self) -> Tuple[sparse.csc_matrix, NDArray]:  # noqa: C901
        """
        Build sparse cost matrix P and vector q.

        Cost: 0.5 z'Pz + q'z
        where z = [x_0, u_0, x_1, u_1, ..., x_N]
        """
        N, n_x, n_u = self.N, self.n_x, self.n_u
        n_vars = self.n_vars

        # P is block diagonal with Q, R, Q, R, ..., Q_f
        P_data = []
        P_rows = []
        P_cols = []

        idx = 0
        for k in range(N):
            # State cost Q
            for i in range(n_x):
                for j in range(n_x):
                    if self.Q[i, j] != 0:
                        P_data.append(self.Q[i, j])
                        P_rows.append(idx + i)
                        P_cols.append(idx + j)
            idx += n_x

            # Control cost R
            for i in range(n_u):
                for j in range(n_u):
                    if self.R[i, j] != 0:
                        P_data.append(self.R[i, j])
                        P_rows.append(idx + i)
                        P_cols.append(idx + j)
            idx += n_u

        # Terminal cost Q_f
        for i in range(n_x):
            for j in range(n_x):
                if self.Q_f[i, j] != 0:
                    P_data.append(self.Q_f[i, j])
                    P_rows.append(idx + i)
                    P_cols.append(idx + j)

        P = sparse.csc_matrix((P_data, (P_rows, P_cols)), shape=(n_vars, n_vars))

        # Linear cost q (reference tracking)
        q = np.zeros(n_vars)

        if self._x_ref is not None:
            idx = 0
            for k in range(N):
                q[idx : idx + n_x] = -self.Q @ self._x_ref[k]
                idx += n_x + n_u
            q[idx : idx + n_x] = -self.Q_f @ self._x_ref[N]

        return P, q

    def _build_constraint_matrix(  # noqa: C901, PLR0915
        self,
        X_lin: NDArray,
        U_lin: NDArray,
        x_init: NDArray,
    ) -> Tuple[sparse.csc_matrix, NDArray, NDArray]:
        """
        Build sparse constraint matrix A and bounds l, u.

        Constraints:
        1. Dynamics: x_{k+1} = A_k x_k + B_k u_k + c_k
        2. Initial condition: x_0 = x_init
        3. Bounds: x_min ≤ x ≤ x_max, u_min ≤ u ≤ u_max
        """
        N, n_x, n_u = self.N, self.n_x, self.n_u
        n_vars = self.n_vars
        dt = self.config.dt

        A_data = []
        A_rows = []
        A_cols = []

        row = 0

        # Initial condition: x_0 = x_init
        for i in range(n_x):
            A_data.append(1.0)
            A_rows.append(row + i)
            A_cols.append(i)
        row += n_x

        # Dynamics constraints
        idx = 0  # Variable index
        for k in range(N):
            # Linearize at current point
            A_k, B_k = self._linearize(X_lin[k], U_lin[k])

            # -x_{k+1} + A_k x_k + B_k u_k = -c_k
            # x_k coefficients
            for i in range(n_x):
                for j in range(n_x):
                    if abs(A_k[i, j]) > 1e-10:
                        A_data.append(A_k[i, j])
                        A_rows.append(row + i)
                        A_cols.append(idx + j)

            # u_k coefficients
            for i in range(n_x):
                for j in range(n_u):
                    if abs(B_k[i, j]) > 1e-10:
                        A_data.append(B_k[i, j])
                        A_rows.append(row + i)
                        A_cols.append(idx + n_x + j)

            # -x_{k+1} coefficients
            for i in range(n_x):
                A_data.append(-1.0)
                A_rows.append(row + i)
                A_cols.append(idx + n_x + n_u + i)

            row += n_x
            idx += n_x + n_u

        # Build sparse matrix
        n_eq = row
        A_eq = sparse.csc_matrix((A_data, (A_rows, A_cols)), shape=(n_eq, n_vars))

        # Bounds for equality constraints
        l_eq = np.zeros(n_eq)
        u_eq = np.zeros(n_eq)

        # Initial condition
        l_eq[:n_x] = x_init
        u_eq[:n_x] = x_init

        # Dynamics affine terms
        row = n_x
        for k in range(N):
            # c_k = f(x_lin, u_lin) - A_k @ x_lin - B_k @ u_lin
            x_next_lin = self.dynamics.step(X_lin[k], U_lin[k], dt)
            A_k, B_k = self._linearize(X_lin[k], U_lin[k])
            c_k = x_next_lin - A_k @ X_lin[k] - B_k @ U_lin[k]

            l_eq[row : row + n_x] = c_k
            u_eq[row : row + n_x] = c_k
            row += n_x

        # Variable bounds
        l_bounds = np.zeros(n_vars)
        u_bounds = np.zeros(n_vars)

        idx = 0
        for k in range(N):
            l_bounds[idx : idx + n_x] = self.x_min
            u_bounds[idx : idx + n_x] = self.x_max
            idx += n_x

            l_bounds[idx : idx + n_u] = self.u_min
            u_bounds[idx : idx + n_u] = self.u_max
            idx += n_u

        l_bounds[idx : idx + n_x] = self.x_min
        u_bounds[idx : idx + n_x] = self.x_max

        # Identity for bounds
        A_bounds = sparse.eye(n_vars, format="csc")

        # Stack constraints
        A = sparse.vstack([A_eq, A_bounds], format="csc")
        l = np.concatenate([l_eq, l_bounds])
        u = np.concatenate([u_eq, u_bounds])

        return A, l, u

    def _linearize(
        self,
        x: NDArray,
        u: NDArray,
        eps: float = 1e-6,
    ) -> Tuple[NDArray, NDArray]:
        """Numerical linearization of dynamics."""
        dt = self.config.dt
        n_x, n_u = self.n_x, self.n_u

        A = np.zeros((n_x, n_x))
        B = np.zeros((n_x, n_u))

        x_next_nom = self.dynamics.step(x, u, dt)

        for i in range(n_x):
            x_pert = x.copy()
            x_pert[i] += eps
            x_next = self.dynamics.step(x_pert, u, dt)
            A[:, i] = (x_next - x_next_nom) / eps

        for i in range(n_u):
            u_pert = u.copy()
            u_pert[i] += eps
            x_next = self.dynamics.step(x, u_pert, dt)
            B[:, i] = (x_next - x_next_nom) / eps

        return A, B

    def initialize(
        self,
        x0: NDArray,
        x_target: NDArray,
        X_init: Optional[NDArray] = None,
        U_init: Optional[NDArray] = None,
    ) -> None:
        """
        Initialize RTI solver.

        Args:
            x0: Initial state
            x_target: Target state
            X_init: Initial trajectory guess
            U_init: Initial control guess
        """
        N = self.N

        # Set reference
        self._x_ref = np.tile(x_target, (N + 1, 1))
        self._u_ref = np.zeros((N, self.n_u))

        # Default hover thrust
        if hasattr(self.dynamics, "params"):
            mass = x0[0]
            g = getattr(self.dynamics.params, "g", 1.0)
            self._u_ref[:, 0] = mass * g
        else:
            self._u_ref[:, 0] = x0[0] * 1.0

        # Initial trajectory guess
        if X_init is not None:
            self._X_lin = X_init.copy()
        else:
            # Linear interpolation
            self._X_lin = np.zeros((N + 1, self.n_x))
            for k in range(N + 1):
                alpha = k / N
                self._X_lin[k] = (1 - alpha) * x0 + alpha * x_target

        if U_init is not None:
            self._U_lin = U_init.copy()
        else:
            self._U_lin = self._u_ref.copy()

        self._X_prev = self._X_lin.copy()
        self._U_prev = self._U_lin.copy()

        # Build and setup OSQP
        self._setup_osqp(x0)

    def _setup_osqp(self, x_init: NDArray) -> None:
        """Setup OSQP solver."""
        # Build matrices
        P, q = self._build_cost_matrix()
        A, l, u = self._build_constraint_matrix(self._X_lin, self._U_lin, x_init)

        self._P_sparse = P
        self._A_sparse = A

        # Create solver
        self._solver = osqp.OSQP()
        self._solver.setup(
            P=P,
            q=q,
            A=A,
            l=l,
            u=u,
            verbose=self.config.osqp_verbose,
            max_iter=self.config.osqp_max_iter,
            eps_abs=self.config.osqp_eps_abs,
            eps_rel=self.config.osqp_eps_rel,
            polish=self.config.osqp_polish,
            warm_start=self.config.osqp_warm_start,
            scaling=self.config.osqp_scaling,
        )

    def prepare(self) -> float:
        """
        Preparation phase: re-linearize and update QP.

        Can be called before getting new measurement.

        Returns:
            Preparation time in ms
        """
        t_start = time.perf_counter()

        if self._solver is None:
            return 0.0

        # Update cost vector (reference tracking)
        _, q = self._build_cost_matrix()
        self._solver.update(q=q)

        prep_time = (time.perf_counter() - t_start) * 1000
        return prep_time

    def feedback(self, x_current: NDArray) -> OSQPRTISolution:
        """
        Feedback phase: update initial state and solve.

        Args:
            x_current: Current state measurement

        Returns:
            RTI solution with optimal control
        """
        t_start = time.perf_counter()

        # Update constraint matrix with new linearization
        A, l, u = self._build_constraint_matrix(self._X_lin, self._U_lin, x_current)

        # Update OSQP
        self._solver.update(l=l, u=u, Ax=A.data)

        t_update = time.perf_counter()

        # Warm start from shifted previous solution
        if self._X_prev is not None and self.config.osqp_warm_start:
            z_warm = self._solution_to_vector(self._X_prev, self._U_prev)
            self._solver.warm_start(x=z_warm)

        # Solve
        result = self._solver.solve()

        t_solve = time.perf_counter()

        # Extract solution
        if result.info.status in ["solved", "solved_inaccurate"]:
            X_opt, U_opt = self._vector_to_solution(result.x)
            u0 = U_opt[0]
            cost = result.info.obj_val
            success = True

            # Update linearization for next iteration
            self._X_lin = X_opt.copy()
            self._U_lin = U_opt.copy()

            # Shift for warm start
            self._X_prev = np.vstack([X_opt[1:], X_opt[-1:]])
            self._U_prev = np.vstack([U_opt[1:], U_opt[-1:]])

        else:
            # Fallback to previous solution
            X_opt = self._X_prev if self._X_prev is not None else self._X_lin
            U_opt = self._U_prev if self._U_prev is not None else self._U_lin
            u0 = U_opt[0]
            cost = np.inf
            success = False

        total_time = (time.perf_counter() - t_start) * 1000
        feedback_time = (t_solve - t_update) * 1000

        return OSQPRTISolution(
            u0=u0,
            X_opt=X_opt,
            U_opt=U_opt,
            cost=cost,
            prep_time_ms=0.0,
            feedback_time_ms=feedback_time,
            total_time_ms=total_time,
            osqp_iterations=result.info.iter,
            success=success,
        )

    def step(self, x_current: NDArray) -> OSQPRTISolution:
        """
        Combined prepare + feedback step.

        Args:
            x_current: Current state

        Returns:
            RTI solution
        """
        t_start = time.perf_counter()

        prep_time = self.prepare()

        t_prep = time.perf_counter()  # noqa: F841

        sol = self.feedback(x_current)

        total_time = (time.perf_counter() - t_start) * 1000

        return OSQPRTISolution(
            u0=sol.u0,
            X_opt=sol.X_opt,
            U_opt=sol.U_opt,
            cost=sol.cost,
            prep_time_ms=prep_time,
            feedback_time_ms=sol.feedback_time_ms,
            total_time_ms=total_time,
            osqp_iterations=sol.osqp_iterations,
            success=sol.success,
        )

    def _solution_to_vector(self, X: NDArray, U: NDArray) -> NDArray:
        """Convert trajectory to decision vector."""
        N, n_x, n_u = self.N, self.n_x, self.n_u
        z = np.zeros(self.n_vars)

        idx = 0
        for k in range(N):
            z[idx : idx + n_x] = X[k]
            idx += n_x
            z[idx : idx + n_u] = U[k]
            idx += n_u
        z[idx : idx + n_x] = X[N]

        return z

    def _vector_to_solution(self, z: NDArray) -> Tuple[NDArray, NDArray]:
        """Convert decision vector to trajectory."""
        N, n_x, n_u = self.N, self.n_x, self.n_u

        X = np.zeros((N + 1, n_x))
        U = np.zeros((N, n_u))

        idx = 0
        for k in range(N):
            X[k] = z[idx : idx + n_x]
            idx += n_x
            U[k] = z[idx : idx + n_u]
            idx += n_u
        X[N] = z[idx : idx + n_x]

        return X, U

    def update_reference(self, x_target: NDArray) -> None:
        """Update target reference."""
        self._x_ref = np.tile(x_target, (self.N + 1, 1))

    def get_predicted_trajectory(self) -> Tuple[NDArray, NDArray]:
        """Get current predicted trajectory."""
        return self._X_lin.copy(), self._U_lin.copy()


class FastRTI3DoF(OSQPRTIMPC):
    """
    Optimized RTI for 3-DoF rocket dynamics.

    Uses analytical Jacobians for faster linearization.
    """

    def __init__(self, dynamics, config: Optional[OSQPRTIConfig] = None):
        """Initialize fast 3-DoF RTI."""
        config = config or OSQPRTIConfig()
        config.n_x = 7  # 3-DoF state
        config.n_u = 3
        super().__init__(dynamics, config)

    def _linearize(
        self,
        x: NDArray,
        u: NDArray,
        eps: float = 1e-6,  # noqa: ARG002
    ) -> Tuple[NDArray, NDArray]:
        """Analytical linearization for 3-DoF dynamics."""
        dt = self.config.dt

        m, rx, ry, rz, vx, vy, vz = x
        Tx, Ty, Tz = u

        # State: [m, rx, ry, rz, vx, vy, vz]
        # Dynamics:
        #   m_dot = -alpha|T|
        #   r_dot = v
        #   v_dot = T/m + g

        # Get parameters
        if hasattr(self.dynamics, "params"):
            g = getattr(self.dynamics.params, "g", 1.0)
            alpha = getattr(self.dynamics.params, "alpha", 0.01)
        else:
            g = 1.0  # noqa: F841
            alpha = 0.01

        T_mag = np.sqrt(Tx**2 + Ty**2 + Tz**2) + 1e-10

        # A matrix (∂f/∂x)
        A = np.eye(7)

        # Position derivatives w.r.t. velocity
        A[1, 4] = dt
        A[2, 5] = dt
        A[3, 6] = dt

        # Velocity derivatives w.r.t. mass
        A[4, 0] = -Tx / m**2 * dt
        A[5, 0] = -Ty / m**2 * dt
        A[6, 0] = -Tz / m**2 * dt

        # B matrix (∂f/∂u)
        B = np.zeros((7, 3))

        # Mass derivative w.r.t. thrust
        B[0, 0] = -alpha * Tx / T_mag * dt
        B[0, 1] = -alpha * Ty / T_mag * dt
        B[0, 2] = -alpha * Tz / T_mag * dt

        # Velocity derivatives w.r.t. thrust
        B[4, 0] = dt / m
        B[5, 1] = dt / m
        B[6, 2] = dt / m

        return A, B
