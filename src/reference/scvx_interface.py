"""
Successive Convexification (SCVX) Interface for Reference Trajectory Generation

Implements powered descent guidance using successive convexification:
- Free-final-time formulation
- Trust region constraints
- Virtual control for infeasibility handling
- Convergence monitoring

Reference:
    Szmuk, M., & Açikmeşe, B. (2018). Successive Convexification for 6-DoF
    Mars Rocket Powered Landing with Free-Final-Time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class SCVXConfig:
    """Configuration for SCVX trajectory optimization."""

    # Discretization
    N: int = 50  # Number of nodes

    # Convergence
    max_iterations: int = 50
    convergence_tol: float = 1e-4

    # Trust region
    trust_region_x: float = 100.0
    trust_region_u: float = 10.0
    trust_region_sigma: float = 10.0  # Time dilation

    # Virtual control
    lambda_vc: float = 1e5  # Virtual control penalty

    # Solver settings
    solver: str = "ipopt"
    verbose: bool = False

    # Time bounds
    t_min: float = 1.0
    t_max: float = 100.0


@dataclass
class SCVXSolution:
    """Solution from SCVX optimization."""

    X: NDArray  # State trajectory (N+1, n_x)
    U: NDArray  # Control trajectory (N, n_u)
    t_f: float  # Final time
    dt: float  # Time step
    cost: float  # Optimal cost
    iterations: int  # SCVX iterations
    converged: bool  # Convergence status
    history: List[Dict[str, Any]] = field(default_factory=list)


class SCVXSolver:
    """
    Successive Convexification solver for rocket landing.

    Solves the free-final-time powered descent problem:

        min  ∫ ||T|| dt  (fuel optimal)
        s.t. ẋ = f(x, u)
             x(0) = x_0
             x(t_f) = x_target
             state and control constraints

    Uses successive convexification to handle:
    - Nonconvex dynamics
    - Nonconvex thrust constraints
    - Free final time

    Example:
        >>> scvx = SCVXSolver(dynamics, config)
        >>> solution = scvx.solve(x0, x_target)
        >>>
        >>> # Use as reference trajectory
        >>> X_ref, U_ref = solution.X, solution.U
    """

    def __init__(
        self,
        dynamics,
        config: Optional[SCVXConfig] = None,
    ):
        """
        Initialize SCVX solver.

        Args:
            dynamics: Rocket dynamics model
            config: SCVX configuration
        """
        if not HAS_CASADI:
            raise ImportError("CasADi required for SCVX. Install with: pip install casadi")

        self.dynamics = dynamics
        self.config = config or SCVXConfig()

        # Problem dimensions
        self.n_x = getattr(dynamics, "n_x", 7)
        self.n_u = getattr(dynamics, "n_u", 3)
        self.N = self.config.N

        # Bounds
        self._setup_bounds()

        # Reference trajectory (for linearization)
        self._X_ref: Optional[NDArray] = None
        self._U_ref: Optional[NDArray] = None
        self._sigma_ref: float = 1.0  # Time dilation factor

    def _setup_bounds(self) -> None:
        """Setup state and control bounds."""
        # State bounds (position, velocity)
        self.x_min = np.array([-np.inf, -1000, -500, -500, -100, -100, -100])[: self.n_x]
        self.x_max = np.array([np.inf, 1000, 500, 500, 100, 100, 100])[: self.n_x]

        # Control bounds (thrust)
        self.u_min = np.array([0.3, -5, -5])[: self.n_u]
        self.u_max = np.array([5.0, 5, 5])[: self.n_u]

        # Get from dynamics if available
        if hasattr(self.dynamics, "params"):
            params = self.dynamics.params
            if hasattr(params, "T_min"):
                self.u_min[0] = params.T_min
            if hasattr(params, "T_max"):
                self.u_max[0] = params.T_max

    def _initialize_trajectory(
        self,
        x0: NDArray,
        x_target: NDArray,
        t_guess: float = 10.0,
    ) -> Tuple[NDArray, NDArray, float]:
        """Generate initial trajectory guess."""
        N = self.N

        # Linear interpolation for states
        X = np.zeros((N + 1, self.n_x))
        for k in range(N + 1):
            alpha = k / N
            X[k] = (1 - alpha) * x0 + alpha * x_target

        # Constant hover thrust
        U = np.zeros((N, self.n_u))
        if hasattr(self.dynamics, "params"):
            mass = x0[0]
            g = getattr(self.dynamics.params, "g", 1.0)
            U[:, 0] = mass * g
        else:
            U[:, 0] = x0[0] * 1.0

        return X, U, t_guess

    def _linearize_dynamics(
        self,
        x: NDArray,
        u: NDArray,
        sigma: float,
        eps: float = 1e-6,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Linearize dynamics at reference point.

        Returns A, B, c for: x_dot ≈ A(x - x_ref) + B(u - u_ref) + f(x_ref, u_ref)
        """
        dt = sigma / self.N  # Scaled time step

        # Numerical Jacobians
        A = np.zeros((self.n_x, self.n_x))
        B = np.zeros((self.n_x, self.n_u))

        f_nom = self.dynamics.continuous_dynamics(x, u)

        for i in range(self.n_x):
            x_pert = x.copy()
            x_pert[i] += eps
            f_pert = self.dynamics.continuous_dynamics(x_pert, u)
            A[:, i] = (f_pert - f_nom) / eps

        for i in range(self.n_u):
            u_pert = u.copy()
            u_pert[i] += eps
            f_pert = self.dynamics.continuous_dynamics(x, u_pert)
            B[:, i] = (f_pert - f_nom) / eps

        # Discretize (Euler)
        A_d = np.eye(self.n_x) + dt * A
        B_d = dt * B
        c_d = dt * (f_nom - A @ x - B @ u)

        return A_d, B_d, c_d

    def _build_subproblem(
        self,
        x0: NDArray,
        x_target: NDArray,
        X_ref: NDArray,
        U_ref: NDArray,
        sigma_ref: float,
    ) -> Tuple[ca.Opti, Dict]:
        """Build convex subproblem."""
        N, n_x, n_u = self.N, self.n_x, self.n_u

        opti = ca.Opti()

        # Decision variables
        X = opti.variable(n_x, N + 1)
        U = opti.variable(n_u, N)
        sigma = opti.variable()  # Time dilation

        # Virtual control for infeasibility
        nu = opti.variable(n_x, N)  # Virtual control

        variables = {"X": X, "U": U, "sigma": sigma, "nu": nu}

        # Cost: fuel + virtual control penalty
        cost = 0
        dt_ref = sigma_ref / N

        for k in range(N):
            # Fuel cost (thrust magnitude)
            cost += dt_ref * ca.norm_2(U[:, k])

            # Virtual control penalty
            cost += self.config.lambda_vc * ca.norm_2(nu[:, k])

        opti.minimize(cost)

        # Initial condition
        opti.subject_to(X[:, 0] == x0)

        # Terminal condition
        opti.subject_to(X[:, N] == x_target)

        # Dynamics constraints (linearized)
        for k in range(N):
            A_k, B_k, c_k = self._linearize_dynamics(X_ref[k], U_ref[k], sigma_ref)

            x_next = A_k @ X[:, k] + B_k @ U[:, k] + c_k + nu[:, k]
            opti.subject_to(X[:, k + 1] == x_next)

        # State bounds
        for k in range(N + 1):
            opti.subject_to(opti.bounded(self.x_min, X[:, k], self.x_max))

        # Control bounds
        for k in range(N):
            opti.subject_to(opti.bounded(self.u_min, U[:, k], self.u_max))

        # Time bounds
        opti.subject_to(opti.bounded(self.config.t_min, sigma, self.config.t_max))

        # Trust regions
        for k in range(N + 1):
            opti.subject_to(ca.norm_2(X[:, k] - X_ref[k]) <= self.config.trust_region_x)

        for k in range(N):
            opti.subject_to(ca.norm_2(U[:, k] - U_ref[k]) <= self.config.trust_region_u)

        opti.subject_to(ca.fabs(sigma - sigma_ref) <= self.config.trust_region_sigma)

        return opti, variables

    def solve(
        self,
        x0: NDArray,
        x_target: NDArray,
        t_guess: Optional[float] = None,
        X_init: Optional[NDArray] = None,
        U_init: Optional[NDArray] = None,
    ) -> SCVXSolution:
        """
        Solve trajectory optimization problem.

        Args:
            x0: Initial state
            x_target: Target state
            t_guess: Initial guess for final time
            X_init: Initial trajectory guess
            U_init: Initial control guess

        Returns:
            SCVX solution
        """
        # Initialize
        if X_init is not None and U_init is not None:
            X_ref = X_init.copy()
            U_ref = U_init.copy()
            sigma_ref = t_guess or 10.0
        else:
            X_ref, U_ref, sigma_ref = self._initialize_trajectory(x0, x_target, t_guess or 10.0)

        history = []
        converged = False

        for iteration in range(self.config.max_iterations):
            # Build and solve subproblem
            opti, variables = self._build_subproblem(x0, x_target, X_ref, U_ref, sigma_ref)

            # Initial guess
            opti.set_initial(variables["X"], X_ref.T)
            opti.set_initial(variables["U"], U_ref.T)
            opti.set_initial(variables["sigma"], sigma_ref)

            # Solver options
            opts = {
                "ipopt.print_level": 0 if not self.config.verbose else 5,
                "ipopt.max_iter": 500,
                "print_time": False,
            }
            opti.solver("ipopt", opts)

            try:
                sol = opti.solve()

                X_new = sol.value(variables["X"]).T
                U_new = sol.value(variables["U"]).T
                sigma_new = sol.value(variables["sigma"])
                nu_val = sol.value(variables["nu"])

                cost = sol.value(opti.f)
                virtual_cost = np.sum(np.linalg.norm(nu_val, axis=0))

            except Exception as e:
                print(f"SCVX iteration {iteration} failed: {e}")
                break

            # Check convergence
            dx = np.max(np.abs(X_new - X_ref))
            du = np.max(np.abs(U_new - U_ref))
            ds = np.abs(sigma_new - sigma_ref)

            history.append(
                {
                    "iteration": iteration,
                    "cost": cost,
                    "virtual_cost": virtual_cost,
                    "dx": dx,
                    "du": du,
                    "ds": ds,
                    "sigma": sigma_new,
                }
            )

            if self.config.verbose:
                print(f"SCVX iter {iteration}: cost={cost:.4f}, dx={dx:.6f}, du={du:.6f}, ds={ds:.6f}")

            if dx < self.config.convergence_tol and du < self.config.convergence_tol and virtual_cost < 1e-3:
                converged = True
                break

            # Update reference
            X_ref = X_new
            U_ref = U_new
            sigma_ref = sigma_new

        return SCVXSolution(
            X=X_ref,
            U=U_ref,
            t_f=sigma_ref,
            dt=sigma_ref / self.N,
            cost=history[-1]["cost"] if history else np.inf,
            iterations=len(history),
            converged=converged,
            history=history,
        )

    def generate_landing_trajectory(
        self,
        x0: NDArray,
        x_target: Optional[NDArray] = None,
    ) -> SCVXSolution:
        """
        Generate a complete landing trajectory.

        Args:
            x0: Initial state [mass, x, y, z, vx, vy, vz]
            x_target: Target state (default: landed at origin)

        Returns:
            SCVX solution
        """
        if x_target is None:
            x_target = np.zeros(self.n_x)
            x_target[0] = x0[0] * 0.8  # Estimate final mass

        return self.solve(x0, x_target)


class SimpleSCVX:
    """
    Simplified SCVX for quick trajectory generation without CasADi.

    Uses direct collocation with numpy/scipy optimization.
    """

    def __init__(self, dynamics, N: int = 50):
        """Initialize simple SCVX."""
        self.dynamics = dynamics
        self.N = N
        self.n_x = getattr(dynamics, "n_x", 7)
        self.n_u = getattr(dynamics, "n_u", 3)

    def generate_reference(
        self,
        x0: NDArray,
        x_target: NDArray,
        t_f: float = 10.0,
    ) -> Tuple[NDArray, NDArray, float]:
        """
        Generate simple reference trajectory using forward simulation.

        Args:
            x0: Initial state
            x_target: Target state
            t_f: Final time

        Returns:
            X: State trajectory
            U: Control trajectory
            dt: Time step
        """
        N = self.N
        dt = t_f / N

        X = np.zeros((N + 1, self.n_x))
        U = np.zeros((N, self.n_u))

        X[0] = x0

        # Simple proportional controller to target
        for k in range(N):
            # Position and velocity errors
            pos_err = x_target[1:4] - X[k, 1:4]
            vel_err = x_target[4:7] - X[k, 4:7]

            # Proportional gains
            Kp = 0.1
            Kd = 0.5

            # Desired acceleration
            a_des = Kp * pos_err + Kd * vel_err

            # Add gravity compensation
            g = getattr(self.dynamics.params, "g", 1.0) if hasattr(self.dynamics, "params") else 1.0

            a_des[0] += g  # Gravity is in -z direction

            # Convert to thrust
            mass = X[k, 0]
            U[k, 0] = np.clip(mass * np.linalg.norm(a_des), 0.3, 5.0)

            # Thrust direction (simplified)
            a_norm = np.linalg.norm(a_des)
            if a_norm > 1e-6:
                U[k, 1:3] = np.clip(a_des[1:3] / a_norm * 0.1, -5, 5)

            # Simulate forward
            X[k + 1] = self.dynamics.step(X[k], U[k], dt)

        return X, U, dt
