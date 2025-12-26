"""
Tube MPC with GP Uncertainty Propagation

Tube MPC ensures robust constraint satisfaction by:
1. Computing uncertainty tubes around nominal trajectories
2. Tightening constraints based on tube width
3. Using GP uncertainty for tube computation

The tube is defined as:
    X_k = {x : ||x - x_k^nom|| ≤ e_k}

where e_k is the tube width at time k.

Reference:
    Mayne, D. Q. (2014). Model Predictive Control: Recent developments
    and future promise. Automatica.
"""

from __future__ import annotations  # noqa: I001

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
from numpy.typing import NDArray


@dataclass
class TubeMPCConfig:
    """Configuration for Tube MPC."""

    # Horizon
    N: int = 15
    dt: float = 0.1

    # Tube computation
    tube_method: str = "linear"  # "linear", "nonlinear", "gp"

    # Disturbance bounds
    w_max: float = 0.1  # Maximum disturbance

    # GP uncertainty scaling
    gp_confidence: float = 0.95  # Confidence level for GP bounds

    # Constraint tightening
    tightening_method: str = "additive"  # "additive", "multiplicative"


class TubePropagator:
    """
    Propagates uncertainty tubes along trajectories.

    Given a nominal trajectory and uncertainty model,
    computes the tube (reachable set) at each time step.

    Example:
        >>> propagator = TubePropagator(dynamics, config)
        >>>
        >>> # Propagate tube with GP uncertainty
        >>> tubes = propagator.propagate_with_gp(X_nom, U_nom, gp_model)
    """

    def __init__(
        self,
        dynamics,
        config: Optional[TubeMPCConfig] = None,
    ):
        """
        Initialize tube propagator.

        Args:
            dynamics: Rocket dynamics model
            config: Configuration
        """
        self.dynamics = dynamics
        self.config = config or TubeMPCConfig()

        self.n_x = 14  # 6-DoF
        self.n_u = 3

    def propagate_linear(
        self,
        X_nom: NDArray,
        U_nom: NDArray,
        w_bounds: Optional[NDArray] = None,
        K: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Propagate tube using linear uncertainty growth.

        e_{k+1} = |A_cl| @ e_k + w_max

        Args:
            X_nom: Nominal trajectory (N+1, n_x)
            U_nom: Nominal controls (N, n_u)
            w_bounds: Disturbance bounds (n_x,)
            K: Tube controller gain

        Returns:
            tubes: Tube widths at each time (N+1, n_x)
        """
        N = len(U_nom)

        if w_bounds is None:
            w_bounds = np.ones(self.n_x) * self.config.w_max

        tubes = np.zeros((N + 1, self.n_x))
        tubes[0] = 0  # No uncertainty at initial state

        for k in range(N):
            # Linearize around nominal
            A, B = self._linearize(X_nom[k], U_nom[k])

            # Closed-loop matrix
            A_cl = A + B @ K if K is not None else A

            # Propagate tube
            tubes[k + 1] = np.abs(A_cl) @ tubes[k] + w_bounds

        return tubes

    def propagate_with_gp(
        self,
        X_nom: NDArray,
        U_nom: NDArray,
        gp_model,
        n_sigma: float = 2.0,
    ) -> Tuple[NDArray, NDArray]:
        """
        Propagate tube using GP uncertainty.

        Uses GP predictive variance to bound uncertainty.

        Args:
            X_nom: Nominal trajectory (N+1, n_x)
            U_nom: Nominal controls (N, n_u)
            gp_model: Trained GP model with predict method
            n_sigma: Number of standard deviations for tube

        Returns:
            tubes: Tube widths (N+1, n_x)
            gp_vars: GP variances along trajectory (N, n_gp_outputs)
        """
        N = len(U_nom)

        tubes = np.zeros((N + 1, self.n_x))
        gp_vars = []

        tubes[0] = 0  # No uncertainty at initial state

        for k in range(N):
            # Get GP prediction and variance
            if hasattr(gp_model, "predict_with_uncertainty"):
                _, d_var = gp_model.predict_with_uncertainty(X_nom[k : k + 1], U_nom[k : k + 1])
            elif hasattr(gp_model, "predict"):
                # Assume predict returns (mean, var)
                _, d_var = gp_model.predict(X_nom[k : k + 1])
                if isinstance(d_var, tuple):
                    d_var = d_var[0]
            else:
                # No GP - use constant bounds
                d_var = np.ones(6) * self.config.w_max**2

            gp_vars.append(d_var.flatten())

            # Convert GP variance to bounds
            # GP outputs are typically [d_v(3), d_omega(3)]
            gp_std = np.sqrt(d_var.flatten())

            # Build full state disturbance bounds
            w_bounds = np.zeros(self.n_x)
            if len(gp_std) >= 6:
                w_bounds[4:7] = gp_std[:3] * n_sigma  # Velocity uncertainty
                w_bounds[11:14] = gp_std[3:6] * n_sigma  # Angular rate uncertainty
            else:
                w_bounds[4:7] = gp_std[: min(3, len(gp_std))] * n_sigma

            # Linearize
            A, B = self._linearize(X_nom[k], U_nom[k])

            # Propagate: e_{k+1} = |A| @ e_k + w_k
            tubes[k + 1] = np.abs(A) @ tubes[k] + w_bounds

        return tubes, np.array(gp_vars)

    def propagate_monte_carlo(
        self,
        X_nom: NDArray,
        U_nom: NDArray,
        gp_model,
        n_samples: int = 100,
        quantile: float = 0.95,
    ) -> NDArray:
        """
        Propagate tube using Monte Carlo sampling.

        More accurate but more expensive than linear propagation.

        Args:
            X_nom: Nominal trajectory
            U_nom: Nominal controls
            gp_model: GP model for sampling
            n_samples: Number of Monte Carlo samples
            quantile: Quantile for tube bounds

        Returns:
            tubes: Tube widths (N+1, n_x)
        """
        N = len(U_nom)

        # Initialize particles at nominal
        particles = np.tile(X_nom[0], (n_samples, 1))

        tubes = np.zeros((N + 1, self.n_x))
        tubes[0] = 0

        for k in range(N):
            # Propagate each particle
            next_particles = np.zeros_like(particles)

            for i, x in enumerate(particles):
                # Nominal dynamics
                x_next = self.dynamics.step(x, U_nom[k], self.config.dt)

                # Add GP-sampled disturbance
                if hasattr(gp_model, "sample"):
                    d = gp_model.sample(x.reshape(1, -1), U_nom[k : k + 1])
                    x_next[4:7] += d[:3].flatten() * self.config.dt
                    x_next[11:14] += d[3:6].flatten() * self.config.dt
                else:
                    # Add random noise
                    x_next += np.random.randn(self.n_x) * self.config.w_max

                next_particles[i] = x_next

            particles = next_particles

            # Compute tube as quantile of deviation from nominal
            deviations = np.abs(particles - X_nom[k + 1])
            tubes[k + 1] = np.quantile(deviations, quantile, axis=0)

        return tubes

    def _linearize(
        self,
        x: NDArray,
        u: NDArray,
        eps: float = 1e-6,
    ) -> Tuple[NDArray, NDArray]:
        """Numerical linearization."""
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


class TubeConstraintTightener:
    """
    Tightens constraints based on tube width.

    Given original constraints g(x) ≥ 0, computes tightened
    constraints ḡ(x_nom) ≥ 0 such that g(x) ≥ 0 for all x in tube.
    """

    def __init__(self, constraint_params=None):
        """
        Initialize constraint tightener.

        Args:
            constraint_params: Original constraint parameters
        """
        self.params = constraint_params

    def tighten_thrust_bounds(
        self,
        tube_u: NDArray,
    ) -> Tuple[float, float]:
        """
        Tighten thrust magnitude constraints.

        Args:
            tube_u: Control tube width

        Returns:
            T_min_tight, T_max_tight
        """
        T_min = getattr(self.params, "T_min", 0.5) if self.params else 0.5
        T_max = getattr(self.params, "T_max", 5.0) if self.params else 5.0

        # Reduce bounds by control uncertainty
        margin = np.linalg.norm(tube_u)

        return T_min + margin, T_max - margin

    def tighten_glideslope(
        self,
        tube_x: NDArray,
        gamma: float,
    ) -> float:
        """
        Tighten glideslope angle constraint.

        Args:
            tube_x: State tube width
            gamma: Original glideslope angle [deg]

        Returns:
            Tightened glideslope angle [deg]
        """
        # Position uncertainty
        pos_uncertainty = np.linalg.norm(tube_x[1:4])

        # Reduce angle to account for uncertainty
        gamma_rad = np.deg2rad(gamma)

        # Conservative tightening
        if pos_uncertainty > 0:
            altitude = max(tube_x[1], 1.0)  # Avoid division by zero
            angle_margin = np.arctan(pos_uncertainty / altitude)
            gamma_tight = gamma_rad - angle_margin
            return np.rad2deg(max(gamma_tight, np.deg2rad(10)))

        return gamma

    def tighten_tilt_angle(
        self,
        tube_x: NDArray,
        theta_max: float,
    ) -> float:
        """
        Tighten maximum tilt angle constraint.

        Args:
            tube_x: State tube width
            theta_max: Original max tilt [deg]

        Returns:
            Tightened max tilt [deg]
        """
        # Quaternion uncertainty affects tilt
        quat_uncertainty = np.linalg.norm(tube_x[7:11])

        # Angular rate uncertainty contributes to future tilt
        omega_uncertainty = np.linalg.norm(tube_x[11:14])

        # Conservative reduction
        theta_margin = np.rad2deg(quat_uncertainty + 0.1 * omega_uncertainty)

        return max(theta_max - theta_margin, 10.0)

    def get_tightened_params(
        self,
        tubes: NDArray,
        timestep: int,
    ) -> dict:
        """
        Get all tightened constraint parameters for a timestep.

        Args:
            tubes: Tube widths (N+1, n_x)
            timestep: Current timestep

        Returns:
            Dictionary of tightened parameters
        """
        tube_x = tubes[timestep]

        T_min = getattr(self.params, "T_min", 0.5) if self.params else 0.5  # noqa: F841
        T_max = getattr(self.params, "T_max", 5.0) if self.params else 5.0  # noqa: F841
        gamma = getattr(self.params, "gamma_gs", 30.0) if self.params else 30.0
        theta = getattr(self.params, "theta_max", 60.0) if self.params else 60.0

        # Assume control tube is proportional to state tube
        tube_u = tube_x[4:7] * 0.5  # Heuristic

        T_min_t, T_max_t = self.tighten_thrust_bounds(tube_u)
        gamma_t = self.tighten_glideslope(tube_x, gamma)
        theta_t = self.tighten_tilt_angle(tube_x, theta)

        return {
            "T_min": T_min_t,
            "T_max": T_max_t,
            "gamma_gs": gamma_t,
            "theta_max": theta_t,
            "tube_width": tube_x,
        }


class RobustTubeMPC:
    """
    Robust Tube MPC for rocket landing.

    Combines tube propagation, constraint tightening, and
    nominal MPC to ensure robust constraint satisfaction.

    Example:
        >>> tube_mpc = RobustTubeMPC(dynamics, config)
        >>> tube_mpc.set_gp_model(gp)
        >>>
        >>> solution = tube_mpc.solve(x0, x_target)
    """

    def __init__(
        self,
        dynamics,
        config: Optional[TubeMPCConfig] = None,
        constraint_params=None,
    ):
        """
        Initialize Robust Tube MPC.

        Args:
            dynamics: Rocket dynamics
            config: Tube MPC configuration
            constraint_params: Constraint parameters
        """
        self.dynamics = dynamics
        self.config = config or TubeMPCConfig()

        # Components
        self.propagator = TubePropagator(dynamics, self.config)
        self.tightener = TubeConstraintTightener(constraint_params)

        # GP model (optional)
        self._gp_model = None

        # Tube controller
        self._K_tube: Optional[NDArray] = None

    def set_gp_model(self, gp_model) -> None:
        """Set GP model for uncertainty propagation."""
        self._gp_model = gp_model

    def set_tube_controller(self, K: NDArray) -> None:
        """Set tube feedback controller gain."""
        self._K_tube = K

    def compute_tubes(
        self,
        X_nom: NDArray,
        U_nom: NDArray,
    ) -> NDArray:
        """
        Compute tubes for trajectory.

        Args:
            X_nom: Nominal state trajectory
            U_nom: Nominal control trajectory

        Returns:
            tubes: Tube widths
        """
        if self._gp_model is not None:
            tubes, _ = self.propagator.propagate_with_gp(X_nom, U_nom, self._gp_model)
        else:
            tubes = self.propagator.propagate_linear(X_nom, U_nom, K=self._K_tube)

        return tubes

    def get_tightened_constraints(
        self,
        tubes: NDArray,
    ) -> List[dict]:
        """
        Get tightened constraints for each timestep.

        Args:
            tubes: Tube widths (N+1, n_x)

        Returns:
            List of tightened constraint dicts
        """
        return [self.tightener.get_tightened_params(tubes, k) for k in range(len(tubes))]

    def check_tube_containment(
        self,
        x_actual: NDArray,
        x_nominal: NDArray,
        tube_width: NDArray,
    ) -> bool:
        """
        Check if actual state is within tube of nominal.

        Args:
            x_actual: Actual state
            x_nominal: Nominal state
            tube_width: Tube width

        Returns:
            True if x_actual is within tube
        """
        deviation = np.abs(x_actual - x_nominal)
        return np.all(deviation <= tube_width * 1.1)  # Small margin
