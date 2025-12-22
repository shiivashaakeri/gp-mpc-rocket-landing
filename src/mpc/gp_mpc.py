"""
GP-MPC: Model Predictive Control with Gaussian Process Dynamics

Extends nominal MPC by:
1. Adding GP mean prediction to dynamics
2. Propagating uncertainty through prediction horizon
3. Tightening constraints based on uncertainty

The learned dynamics model becomes:
    x_{k+1} = f_nominal(x_k, u_k) + d_GP(x_k, u_k)

where d_GP is the GP-predicted residual with uncertainty.

For constraint satisfaction under uncertainty:
    P(g(x) >= 0) >= 1 - ε

We use constraint tightening:
    g(μ_x) - κ * sigma_g >= 0

Reference:
    Hewing, L., et al. (2020). Learning-Based Model Predictive Control:
    Toward Safe Learning in Control. Annual Review of Control, Robotics,
    and Autonomous Systems.
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

from .constraints import ConstraintParams, TightenedConstraints
from .cost_functions import CostWeights
from .nominal_mpc import MPCConfig, MPCSolution
from .uncertainty_prop import PropagatedUncertainty, UncertaintyPropagator


@dataclass
class GPMPCConfig(MPCConfig):
    """Configuration for GP-MPC."""

    # GP settings
    use_gp_mean: bool = True  # Add GP mean to dynamics
    use_gp_uncertainty: bool = True  # Propagate GP uncertainty

    # Constraint tightening
    confidence_level: float = 0.95  # Constraint satisfaction probability

    # Uncertainty handling
    max_variance_for_constraint: float = 1.0  # Skip tightening if variance too high

    # Robustification
    robust_horizon: int = -1  # Horizon for robustification (-1 = full)


class GPMPC:
    """
    GP-MPC: MPC with Gaussian Process learned dynamics.

    Combines nominal dynamics with GP-learned residuals:
        x_{k+1} = f_nominal(x_k, u_k) + G(x_k) @ d_GP(x_k, u_k)

    where d_GP = [d_v; d_ω] are the learned acceleration residuals.

    Features:
    - GP mean added to prediction
    - Uncertainty propagation through horizon
    - Constraint tightening for probabilistic satisfaction
    - Warm starting from previous solution

    Example:
        >>> gp = StructuredRocketGP(config)
        >>> # ... train GP with data ...
        >>>
        >>> mpc = GPMPC(rocket, gp, GPMPCConfig())
        >>> mpc.setup()
        >>> solution = mpc.solve(x0, x_target)
    """

    def __init__(
        self,
        dynamics,
        gp_model,  # StructuredRocketGP
        config: Optional[GPMPCConfig] = None,
        constraint_params: Optional[ConstraintParams] = None,
        cost_weights: Optional[CostWeights] = None,
    ):
        """
        Initialize GP-MPC.

        Args:
            dynamics: Nominal rocket dynamics model
            gp_model: Trained GP model for residuals
            config: GP-MPC configuration
            constraint_params: Constraint parameters
            cost_weights: Cost weights
        """
        self.dynamics = dynamics
        self.gp = gp_model
        self.config = config or GPMPCConfig()
        self.constraint_params = constraint_params or ConstraintParams()
        self.cost_weights = cost_weights or CostWeights()

        self.n_x = 14
        self.n_u = 3

        # Constraint tightening
        self._tightened_constraints = TightenedConstraints(
            base_params=self.constraint_params,
            confidence_level=self.config.confidence_level,
        )

        # Uncertainty propagator
        self._uncertainty_prop = UncertaintyPropagator(
            dynamics=dynamics,
            gp_model=gp_model,
        )

        # Cache for linearized GP (for CasADi)
        self._gp_mean_cache: dict = {}
        self._gp_var_cache: dict = {}

        # Warm start
        self._X_warm: Optional[NDArray] = None
        self._U_warm: Optional[NDArray] = None

        self._is_setup = False

    def _predict_with_gp(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predict next state using nominal dynamics + GP.

        Args:
            x: Current state
            u: Control

        Returns:
            x_next: Predicted next state
            variance: Prediction variance (14,)
        """
        # Nominal prediction
        x_nominal = self.dynamics.step(x, u, self.config.dt)

        if not self.config.use_gp_mean:
            return x_nominal, np.zeros(14)

        # GP prediction
        d_v, d_omega, var_v, var_omega = self.gp.predict(x, u)

        # Add GP mean to nominal prediction
        # d affects acceleration, which after dt integration affects velocity
        x_next = x_nominal.copy()
        x_next[4:7] += d_v * self.config.dt  # Velocity correction
        x_next[11:14] += d_omega * self.config.dt  # Angular velocity correction

        # Build variance vector
        variance = np.zeros(14)
        variance[4:7] = var_v * self.config.dt**2
        variance[11:14] = var_omega * self.config.dt**2

        return x_next, variance

    def _get_tightened_params(
        self,
        propagated_uncertainty: PropagatedUncertainty,
        k: int,
    ) -> ConstraintParams:
        """
        Get constraint parameters tightened based on uncertainty at step k.

        Args:
            propagated_uncertainty: Propagated uncertainty structure
            k: Timestep

        Returns:
            Tightened constraint parameters
        """
        if not self.config.use_gp_uncertainty:
            return self.constraint_params

        # Get standard deviations at step k
        cov_k = propagated_uncertainty.covariances[k]
        std_k = np.sqrt(np.diag(cov_k))

        # Extract relevant uncertainties
        position_std = np.mean(std_k[1:4])
        velocity_std = np.mean(std_k[4:7])
        # Attitude uncertainty from quaternion is tricky - use small angle approx
        attitude_std = np.mean(std_k[8:10])  # qx, qy
        omega_std = np.mean(std_k[11:14])

        # Check if uncertainty is too high
        if velocity_std > self.config.max_variance_for_constraint:
            return self.constraint_params  # Don't tighten if too uncertain

        return self._tightened_constraints.get_tightened_params(
            position_std=position_std,
            velocity_std=velocity_std,
            attitude_std=attitude_std,
            omega_std=omega_std,
        )

    def setup(self) -> None:
        """
        Set up GP-MPC problem.

        Uses a two-stage approach:
        1. Predict trajectory with GP mean (nonlinear)
        2. Solve QP around this prediction with tightened constraints
        """
        # For now, we use a simpler approach:
        # Solve nominal MPC, then verify constraints with GP uncertainty
        self._is_setup = True

    def solve(  # noqa: C901, PLR0912
        self,
        x0: NDArray,
        x_target: NDArray,
        X_ref: Optional[NDArray] = None,
        U_ref: Optional[NDArray] = None,
    ) -> MPCSolution:
        """
        Solve GP-MPC problem.

        Strategy:
        1. Predict trajectory using nominal dynamics + GP mean
        2. Propagate uncertainty through horizon
        3. Solve MPC with tightened constraints

        Args:
            x0: Current state
            x_target: Target state
            X_ref: Reference trajectory (optional)
            U_ref: Reference controls (optional)

        Returns:
            MPC solution
        """
        N = self.config.N
        dt = self.config.dt

        start_time = time.perf_counter()

        # Step 1: Get initial trajectory prediction with GP
        X_pred = np.zeros((N + 1, self.n_x))
        U_pred = np.zeros((N, self.n_u))
        variances = np.zeros((N + 1, self.n_x))

        X_pred[0] = x0

        # Use warm start or reference for initial controls
        if self._U_warm is not None:
            U_init = self._U_warm
        elif U_ref is not None:
            U_init = U_ref[:N]
        else:
            # Simple hover control
            U_init = np.zeros((N, self.n_u))
            for k in range(N):
                m = X_pred[0, 0] if k == 0 else X_pred[k, 0]
                U_init[k] = np.array([0, 0, m * self.dynamics.params.g0])

        # Forward simulate with GP
        for k in range(N):
            u_k = U_init[k] if k < len(U_init) else U_init[-1]
            U_pred[k] = u_k
            X_pred[k + 1], variances[k + 1] = self._predict_with_gp(X_pred[k], u_k)

        # Step 2: Propagate uncertainty
        if self.config.use_gp_uncertainty:
            prop_uncertainty = self._uncertainty_prop.propagate(
                x0=x0,
                U=U_pred,
                Sigma_0=np.eye(self.n_x) * 1e-6,  # Small initial uncertainty
            )
        else:
            prop_uncertainty = None

        # Step 3: Solve constrained optimization
        # For simplicity, use iterative linearization approach

        max_iters = 10
        converged = False

        for iteration in range(max_iters):
            # Linearize dynamics around current trajectory
            A_list, B_list, c_list = [], [], []

            for k in range(N):
                A_d, B_d = self.dynamics.linearize(X_pred[k], U_pred[k], dt=dt)

                # Add GP Jacobians if significant
                # (simplified: just use nominal Jacobians)

                # Affine term from GP mean
                if self.config.use_gp_mean:
                    d_v, d_omega, _, _ = self.gp.predict(X_pred[k], U_pred[k])
                    c = np.zeros(self.n_x)
                    c[4:7] = d_v * dt
                    c[11:14] = d_omega * dt
                else:
                    c = np.zeros(self.n_x)

                A_list.append(A_d)
                B_list.append(B_d)
                c_list.append(c)

            # Solve QP subproblem
            X_new, U_new, cost = self._solve_qp(
                x0,
                x_target,
                X_pred,
                U_pred,
                A_list,
                B_list,
                c_list,
                prop_uncertainty,
                X_ref,
                U_ref,
            )

            # Check convergence
            X_change = np.max(np.abs(X_new - X_pred))
            U_change = np.max(np.abs(U_new - U_pred))

            X_pred = X_new
            U_pred = U_new

            if X_change < 1e-4 and U_change < 1e-4:
                converged = True
                break

            # Re-propagate uncertainty if needed
            if self.config.use_gp_uncertainty and iteration < max_iters - 1:
                prop_uncertainty = self._uncertainty_prop.propagate(
                    x0=x0,
                    U=U_pred,
                    Sigma_0=np.eye(self.n_x) * 1e-6,
                )

        solve_time = time.perf_counter() - start_time

        # Save for warm start
        self._X_warm = X_pred.copy()
        self._U_warm = U_pred.copy()

        return MPCSolution(
            success=converged,
            X_opt=X_pred,
            U_opt=U_pred,
            cost=cost,
            solve_time=solve_time,
            iterations=iteration + 1,
            status="Converged" if converged else "Max iterations",
        )

    def _solve_qp(  # noqa: PLR0915
        self,
        x0: NDArray,
        x_target: NDArray,
        X_nom: NDArray,
        U_nom: NDArray,
        A_list: List[NDArray],
        B_list: List[NDArray],
        c_list: List[NDArray],
        uncertainty: Optional[PropagatedUncertainty],
        X_ref: Optional[NDArray],
        U_ref: Optional[NDArray],
    ) -> Tuple[NDArray, NDArray, float]:
        """
        Solve QP subproblem around nominal trajectory.

        Returns:
            X_opt: Optimal states
            U_opt: Optimal controls
            cost: Optimal cost
        """
        N = self.config.N

        # Use CasADi for QP
        opti = ca.Opti()

        # Decision variables: deviations from nominal
        dX = opti.variable(self.n_x, N + 1)
        dU = opti.variable(self.n_u, N)

        # Initial condition
        opti.subject_to(dX[:, 0] == x0 - X_nom[0])

        # Linearized dynamics
        for k in range(N):
            A_k = ca.DM(A_list[k])
            B_k = ca.DM(B_list[k])
            c_k = ca.DM(c_list[k])

            # x_{k+1} - x_nom_{k+1} = A(x_k - x_nom_k) + B(u_k - u_nom_k) + c
            opti.subject_to(dX[:, k + 1] == A_k @ dX[:, k] + B_k @ dU[:, k] + c_k)

        # Get constraint parameters (possibly tightened)
        params_k = self._get_tightened_params(uncertainty, 0) if uncertainty is not None else self.constraint_params

        # Constraints on actual values
        for k in range(N):
            x_k = X_nom[k] + dX[:, k]
            u_k = U_nom[k] + dU[:, k]

            # Thrust bounds
            T_sq = ca.dot(u_k, u_k)
            opti.subject_to(T_sq >= params_k.T_min**2)
            opti.subject_to(T_sq <= params_k.T_max**2)

            # Glideslope
            r_k = x_k[1:4]
            gamma = params_k.gamma_gs_rad
            opti.subject_to(r_k[0] ** 2 * np.tan(gamma) ** 2 >= r_k[1] ** 2 + r_k[2] ** 2)

        # Trust region
        for k in range(N + 1):
            opti.subject_to(ca.dot(dX[:, k], dX[:, k]) <= 10.0)
        for k in range(N):
            opti.subject_to(ca.dot(dU[:, k], dU[:, k]) <= 5.0)

        # Cost function
        Q = ca.DM(self.cost_weights.Q)
        R = ca.DM(self.cost_weights.R)
        P = ca.DM(self.cost_weights.P)

        if X_ref is None:
            X_ref = np.tile(x_target, (N + 1, 1))
        if U_ref is None:
            U_ref = np.zeros((N, self.n_u))

        cost = 0
        for k in range(N):
            x_k = X_nom[k] + dX[:, k]
            u_k = U_nom[k] + dU[:, k]
            x_err = x_k - X_ref[k]
            u_err = u_k - U_ref[k]
            cost += ca.bilin(Q, x_err, x_err) + ca.bilin(R, u_err, u_err)

        # Terminal cost
        x_N = X_nom[N] + dX[:, N]
        x_err_N = x_N - X_ref[N] if len(X_ref) > N else x_N - x_target
        cost += ca.bilin(P, x_err_N, x_err_N)

        opti.minimize(cost)

        opts = {
            "ipopt.max_iter": 50,
            "ipopt.print_level": 0,
            "print_time": False,
        }
        opti.solver("ipopt", opts)

        try:
            sol = opti.solve()
            dX_opt = sol.value(dX).T
            dU_opt = sol.value(dU).T
            cost_val = float(sol.value(cost))

            X_opt = X_nom + dX_opt
            U_opt = U_nom + dU_opt

        except RuntimeError:
            # Return nominal if QP fails
            X_opt = X_nom
            U_opt = U_nom
            cost_val = np.inf

        return X_opt, U_opt, cost_val

    def get_uncertainty_at_horizon(
        self,
        k: int,  # noqa: ARG002
    ) -> Optional[NDArray]:
        """Get propagated covariance at step k from last solve."""
        # Would return cached uncertainty from last solve
        return None

    def reset_warm_start(self) -> None:
        """Clear warm start."""
        self._X_warm = None
        self._U_warm = None


# =============================================================================
# Simple GP-Enhanced Prediction (for testing)
# =============================================================================


class SimpleGPPredictor:
    """
    Simple wrapper that adds GP predictions to dynamics.

    Useful for testing GP integration before full GP-MPC.
    """

    def __init__(self, dynamics, gp_model):
        self.dynamics = dynamics
        self.gp = gp_model

    def predict(
        self,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Predict with GP-augmented dynamics.

        Returns:
            x_next: Next state
            d_mean: GP residual mean
            d_var: GP residual variance
        """
        # Nominal prediction
        x_nom = self.dynamics.step(x, u, dt)

        # GP prediction
        d_v, d_omega, var_v, var_omega = self.gp.predict(x, u)

        # Augment
        x_next = x_nom.copy()
        x_next[4:7] += d_v * dt
        x_next[11:14] += d_omega * dt

        d_mean = np.zeros(14)
        d_mean[4:7] = d_v
        d_mean[11:14] = d_omega

        d_var = np.zeros(14)
        d_var[4:7] = var_v
        d_var[11:14] = var_omega

        return x_next, d_mean, d_var

    def simulate(
        self,
        x0: NDArray,
        U: NDArray,
        dt: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Simulate trajectory with GP.

        Returns:
            X: State trajectory
            D_mean: GP residuals
            D_var: GP variances
        """
        N = len(U)
        X = np.zeros((N + 1, len(x0)))
        D_mean = np.zeros((N, 14))
        D_var = np.zeros((N, 14))

        X[0] = x0
        for k in range(N):
            X[k + 1], D_mean[k], D_var[k] = self.predict(X[k], U[k], dt)

        return X, D_mean, D_var
