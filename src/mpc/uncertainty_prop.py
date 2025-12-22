"""
Uncertainty Propagation for GP-MPC

Propagates state uncertainty through the prediction horizon for:
1. Chance constraint satisfaction
2. Constraint tightening
3. Risk-aware planning

Methods:
- Linear covariance propagation (fast, approximate)
- Unscented Transform (more accurate, moderate cost)
- Monte Carlo (accurate, expensive)

The uncertainty comes from:
- Initial state uncertainty
- GP prediction uncertainty (process noise)
- Measurement noise

Reference:
    Hewing, L., et al. (2020). Cautious Model Predictive Control using
    Gaussian Process Regression. IEEE Transactions on Control Systems
    Technology.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PropagatedUncertainty:
    """Container for propagated uncertainty through horizon."""

    means: NDArray  # Mean trajectory (N+1, n_x)
    covariances: NDArray  # Covariance matrices (N+1, n_x, n_x)

    def get_std(self, k: int) -> NDArray:
        """Get standard deviation at step k."""
        return np.sqrt(np.diag(self.covariances[k]))

    def get_confidence_bounds(
        self,
        k: int,
        confidence: float = 0.95,
    ) -> Tuple[NDArray, NDArray]:
        """Get confidence bounds at step k."""
        from scipy.stats import norm  # noqa: PLC0415

        kappa = norm.ppf((1 + confidence) / 2)
        std = self.get_std(k)
        return self.means[k] - kappa * std, self.means[k] + kappa * std


class UncertaintyPropagator:
    """
    Propagates uncertainty through dynamics with GP.

    Uses linearized covariance propagation:
        Σ_{k+1} = A_k Σ_k A_k^T + Q_k

    where Q_k is the GP prediction covariance.
    """

    def __init__(
        self,
        dynamics,
        gp_model,
        method: str = "linear",
    ):
        """
        Initialize propagator.

        Args:
            dynamics: Nominal dynamics model
            gp_model: GP model for residuals
            method: Propagation method ("linear", "unscented", "monte_carlo")
        """
        self.dynamics = dynamics
        self.gp = gp_model
        self.method = method

        self.n_x = 14
        self.n_u = 3

    def propagate(
        self,
        x0: NDArray,
        U: NDArray,
        Sigma_0: Optional[NDArray] = None,
        dt: float = 0.1,
    ) -> PropagatedUncertainty:
        """
        Propagate uncertainty through horizon.

        Args:
            x0: Initial state
            U: Control trajectory (N, n_u)
            Sigma_0: Initial covariance (n_x, n_x)
            dt: Timestep

        Returns:
            PropagatedUncertainty with means and covariances
        """
        if self.method == "linear":
            return self._propagate_linear(x0, U, Sigma_0, dt)
        elif self.method == "unscented":
            return self._propagate_unscented(x0, U, Sigma_0, dt)
        elif self.method == "monte_carlo":
            return self._propagate_monte_carlo(x0, U, Sigma_0, dt)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _propagate_linear(
        self,
        x0: NDArray,
        U: NDArray,
        Sigma_0: Optional[NDArray],
        dt: float,
    ) -> PropagatedUncertainty:
        """
        Linear covariance propagation.

        Σ_{k+1} = A_k Σ_k A_k^T + Q_k^{GP}

        Fast O(N * n_x^3) but assumes linear uncertainty growth.
        """
        N = len(U)

        if Sigma_0 is None:
            Sigma_0 = np.eye(self.n_x) * 1e-6

        means = np.zeros((N + 1, self.n_x))
        covariances = np.zeros((N + 1, self.n_x, self.n_x))

        means[0] = x0
        covariances[0] = Sigma_0

        x_k = x0.copy()
        Sigma_k = Sigma_0.copy()

        for k in range(N):
            u_k = U[k]

            # Get linearized dynamics
            A_d, B_d = self.dynamics.linearize(x_k, u_k, dt=dt)

            # Get GP prediction and variance
            d_v, d_omega, var_v, var_omega = self.gp.predict(x_k, u_k)

            # Build GP covariance matrix (process noise)
            Q_gp = np.zeros((self.n_x, self.n_x))
            Q_gp[4:7, 4:7] = np.diag(var_v) * dt**2
            Q_gp[11:14, 11:14] = np.diag(var_omega) * dt**2

            # Nominal prediction with GP mean
            x_nom = self.dynamics.step(x_k, u_k, dt)
            x_next = x_nom.copy()
            x_next[4:7] += d_v * dt
            x_next[11:14] += d_omega * dt

            # Propagate covariance
            # Σ_{k+1} = A Σ_k A^T + Q_{GP}
            Sigma_next = A_d @ Sigma_k @ A_d.T + Q_gp

            # Store
            means[k + 1] = x_next
            covariances[k + 1] = Sigma_next

            # Update for next iteration
            x_k = x_next
            Sigma_k = Sigma_next

        return PropagatedUncertainty(means=means, covariances=covariances)

    def _propagate_unscented(
        self,
        x0: NDArray,
        U: NDArray,
        Sigma_0: Optional[NDArray],
        dt: float,
    ) -> PropagatedUncertainty:
        """
        Unscented Transform propagation.

        More accurate than linear for nonlinear systems.
        Uses sigma points to capture mean and covariance accurately.
        """
        N = len(U)
        n = self.n_x

        if Sigma_0 is None:
            Sigma_0 = np.eye(n) * 1e-6

        # UT parameters
        alpha = 1e-3
        beta = 2
        kappa = 0
        lambda_ = alpha**2 * (n + kappa) - n

        # Weights
        w_m = np.zeros(2 * n + 1)
        w_c = np.zeros(2 * n + 1)
        w_m[0] = lambda_ / (n + lambda_)
        w_c[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)
        for i in range(1, 2 * n + 1):
            w_m[i] = 1 / (2 * (n + lambda_))
            w_c[i] = 1 / (2 * (n + lambda_))

        means = np.zeros((N + 1, n))
        covariances = np.zeros((N + 1, n, n))

        means[0] = x0
        covariances[0] = Sigma_0

        x_k = x0.copy()
        Sigma_k = Sigma_0.copy()

        for k in range(N):
            u_k = U[k]

            # Generate sigma points
            sqrt_Sigma = np.linalg.cholesky((n + lambda_) * Sigma_k + 1e-10 * np.eye(n))

            sigma_points = np.zeros((2 * n + 1, n))
            sigma_points[0] = x_k
            for i in range(n):
                sigma_points[i + 1] = x_k + sqrt_Sigma[:, i]
                sigma_points[n + i + 1] = x_k - sqrt_Sigma[:, i]

            # Propagate sigma points
            sigma_points_next = np.zeros_like(sigma_points)
            for i in range(2 * n + 1):
                x_nom = self.dynamics.step(sigma_points[i], u_k, dt)
                d_v, d_omega, _, _ = self.gp.predict(sigma_points[i], u_k)
                sigma_points_next[i] = x_nom.copy()
                sigma_points_next[i, 4:7] += d_v * dt
                sigma_points_next[i, 11:14] += d_omega * dt

            # Compute mean and covariance
            x_next = np.sum(w_m[:, None] * sigma_points_next, axis=0)

            Sigma_next = np.zeros((n, n))
            for i in range(2 * n + 1):
                diff = sigma_points_next[i] - x_next
                Sigma_next += w_c[i] * np.outer(diff, diff)

            # Add GP variance as process noise
            _, _, var_v, var_omega = self.gp.predict(x_k, u_k)
            Q_gp = np.zeros((n, n))
            Q_gp[4:7, 4:7] = np.diag(var_v) * dt**2
            Q_gp[11:14, 11:14] = np.diag(var_omega) * dt**2
            Sigma_next += Q_gp

            means[k + 1] = x_next
            covariances[k + 1] = Sigma_next

            x_k = x_next
            Sigma_k = Sigma_next

        return PropagatedUncertainty(means=means, covariances=covariances)

    def _propagate_monte_carlo(
        self,
        x0: NDArray,
        U: NDArray,
        Sigma_0: Optional[NDArray],
        dt: float,
        n_samples: int = 100,
    ) -> PropagatedUncertainty:
        """
        Monte Carlo propagation.

        Most accurate but expensive. Samples from distributions
        and propagates particles.
        """
        N = len(U)
        n = self.n_x

        if Sigma_0 is None:
            Sigma_0 = np.eye(n) * 1e-6

        # Sample initial particles
        particles = np.random.multivariate_normal(x0, Sigma_0, n_samples)

        means = np.zeros((N + 1, n))
        covariances = np.zeros((N + 1, n, n))

        means[0] = x0
        covariances[0] = Sigma_0

        for k in range(N):
            u_k = U[k]

            # Propagate each particle
            particles_next = np.zeros_like(particles)
            for i in range(n_samples):
                x_nom = self.dynamics.step(particles[i], u_k, dt)
                d_v, d_omega, var_v, var_omega = self.gp.predict(particles[i], u_k)

                # Sample from GP posterior
                d_v_sample = d_v + np.sqrt(var_v) * np.random.randn(3)
                d_omega_sample = d_omega + np.sqrt(var_omega) * np.random.randn(3)

                particles_next[i] = x_nom.copy()
                particles_next[i, 4:7] += d_v_sample * dt
                particles_next[i, 11:14] += d_omega_sample * dt

            # Compute statistics
            means[k + 1] = np.mean(particles_next, axis=0)
            diff = particles_next - means[k + 1]
            covariances[k + 1] = (diff.T @ diff) / (n_samples - 1)

            particles = particles_next

        return PropagatedUncertainty(means=means, covariances=covariances)


class ConstraintTightening:
    """
    Computes constraint tightening based on propagated uncertainty.

    For chance constraint P(g(x) >= 0) >= 1 - ε:
        g(μ) - κ * sigma_g >= 0

    where κ = Φ^{-1}(1 - ε) for Gaussian.
    """

    def __init__(
        self,
        confidence: float = 0.95,
    ):
        """
        Initialize constraint tightening.

        Args:
            confidence: Desired constraint satisfaction probability
        """
        from scipy.stats import norm  # noqa: PLC0415

        self.confidence = confidence
        self.kappa = norm.ppf(confidence)  # ≈ 1.64 for 95%, 2.33 for 99%

    def tighten_linear_constraint(
        self,
        a: NDArray,
        b: float,
        mu: NDArray,  # noqa: ARG002
        Sigma: NDArray,
    ) -> float:
        """
        Tighten linear constraint a^T x >= b.

        Under x ~ N(μ, Σ):
            a^T x ~ N(a^T μ, a^T Σ a)

        Tightened: a^T μ - κ * √(a^T Σ a) >= b

        Returns the tightened bound.
        """
        sigma_ax = np.sqrt(a.T @ Sigma @ a)
        return b + self.kappa * sigma_ax

    def tighten_quadratic_constraint(
        self,
        x_mu: NDArray,
        x_Sigma: NDArray,
        constraint_func: Callable[[NDArray], float],
        n_samples: int = 100,
    ) -> float:
        """
        Tighten nonlinear constraint using sampling.

        Estimates the constraint distribution and finds
        the tightening amount.
        """
        # Sample from state distribution
        samples = np.random.multivariate_normal(x_mu, x_Sigma, n_samples)

        # Evaluate constraint on samples
        g_samples = np.array([constraint_func(s) for s in samples])

        # Find quantile
        return np.percentile(g_samples, (1 - self.confidence) * 100)

    def compute_back_offs(
        self,
        uncertainty: PropagatedUncertainty,
        constraint_gradients: List[NDArray],  # List of gradients at each step
    ) -> NDArray:
        """
        Compute constraint back-offs for each timestep.

        Args:
            uncertainty: Propagated uncertainty
            constraint_gradients: Constraint gradients w.r.t. state

        Returns:
            back_offs: (N, n_constraints) array of back-off amounts
        """
        N = len(uncertainty.means) - 1
        n_constraints = len(constraint_gradients[0]) if constraint_gradients else 0

        back_offs = np.zeros((N, n_constraints))

        for k in range(N):
            Sigma_k = uncertainty.covariances[k]

            for j, grad in enumerate(constraint_gradients[k] if k < len(constraint_gradients) else []):
                sigma_g = np.sqrt(grad.T @ Sigma_k @ grad)
                back_offs[k, j] = self.kappa * sigma_g

        return back_offs


class TubeBasedRobustness:
    """
    Tube-based robustness for bounded disturbances.

    For systems with bounded disturbances ||d|| <= d_max,
    computes the reachable tube around nominal trajectory.
    """

    def __init__(
        self,
        dynamics,
        d_max: float = 0.1,
    ):
        self.dynamics = dynamics
        self.d_max = d_max

    def compute_tube(
        self,
        X_nom: NDArray,
        U_nom: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Compute tube widths along trajectory.

        Returns:
            tube_widths: (N+1, n_x) array of tube half-widths
        """
        N = len(U_nom)
        n_x = X_nom.shape[1]

        tube_widths = np.zeros((N + 1, n_x))

        # Initial tube is zero (known initial state)
        width_k = np.zeros(n_x)

        for k in range(N):
            # Get dynamics Jacobian
            A_d, _ = self.dynamics.linearize(X_nom[k], U_nom[k], dt=dt)

            # Tube propagation: w_{k+1} = |A| w_k + d_max
            # Using L1 norm for robustness
            width_next = np.abs(A_d) @ width_k
            width_next[4:7] += self.d_max * dt  # Disturbance affects velocity
            width_next[11:14] += self.d_max * dt  # And angular velocity

            tube_widths[k + 1] = width_next
            width_k = width_next

        return tube_widths
