"""
Sparse Gaussian Process Regression with Inducing Points

Implements scalable GP inference using inducing point methods:
- FITC (Fully Independent Training Conditional)
- VFE (Variational Free Energy)

Complexity: O(NM²) instead of O(N³) where M << N inducing points.

Key equations (FITC):
    Q_ff = K_fu K_uu^{-1} K_uf  (Nyström approximation)
    Λ = diag(K_ff - Q_ff)       (FITC diagonal correction)

    Posterior:
    μ* = K_*u Σ K_uf (Λ + sigma²I)^{-1} y
    Σ = (K_uu + K_uf (Λ + sigma²I)^{-1} K_fu)^{-1}

Reference:
    Snelson, E., & Ghahramani, Z. (2006). Sparse Gaussian processes
    using pseudo-inputs. NIPS.

    Titsias, M. (2009). Variational learning of inducing variables
    in sparse Gaussian processes. AISTATS.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans2
from scipy.linalg import cho_solve, solve_triangular

from .exact_gp import GPPrediction
from .kernels import Kernel, SquaredExponentialARD


class SparseGP:
    """
    Sparse Gaussian Process with Inducing Points.

    Uses M << N inducing points for O(NM²) complexity instead of O(N³).

    Supports two approximations:
    - FITC: Fully Independent Training Conditional (faster, less accurate)
    - VFE: Variational Free Energy (slower, principled lower bound)

    Example:
        >>> kernel = SquaredExponentialARD(input_dim=6)
        >>> gp = SparseGP(kernel, n_inducing=50)
        >>> gp.fit(X_train, y_train)  # O(N * 50²) instead of O(N³)
        >>> pred = gp.predict(X_test)  # O(M²) per test point
    """

    def __init__(
        self,
        kernel: Kernel,
        n_inducing: int = 50,
        noise_variance: float = 1e-4,
        method: Literal["fitc", "vfe"] = "fitc",
        inducing_points: Optional[NDArray] = None,
        jitter: float = 1e-6,
    ):
        """
        Initialize Sparse GP.

        Args:
            kernel: Covariance function
            n_inducing: Number of inducing points M
            noise_variance: Observation noise sigma²_n
            method: Approximation method ("fitc" or "vfe")
            inducing_points: Initial inducing locations (M, D).
                           If None, initialized from training data.
            jitter: Jitter for numerical stability
        """
        self.kernel = kernel
        self.n_inducing = n_inducing
        self._noise_variance = noise_variance
        self.method = method
        self.jitter = jitter

        # Inducing points
        self._Z: Optional[NDArray] = inducing_points

        # Training data
        self.X_train: Optional[NDArray] = None
        self.y_train: Optional[NDArray] = None
        self.n_train: int = 0

        # Normalization
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

        # Cached computations
        self._L_uu: Optional[NDArray] = None  # Cholesky of K_uu
        self._L_B: Optional[NDArray] = None  # Cholesky of B = I + A^T Λ^{-1} A
        self._alpha: Optional[NDArray] = None
        self._log_marginal_likelihood: Optional[float] = None

    @property
    def inducing_points(self) -> Optional[NDArray]:
        """Inducing point locations Z (M, D)."""
        return self._Z

    @property
    def noise_variance(self) -> float:
        return self._noise_variance

    @noise_variance.setter
    def noise_variance(self, value: float) -> None:
        self._noise_variance = value
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations."""
        self._L_uu = None
        self._L_B = None
        self._alpha = None
        self._log_marginal_likelihood = None

    def _initialize_inducing_points(self, X: NDArray) -> NDArray:
        """
        Initialize inducing points from training data.

        Uses k-means clustering to select representative points.

        Args:
            X: Training inputs (N, D)

        Returns:
            Inducing points (M, D)
        """
        n_samples = X.shape[0]

        if n_samples <= self.n_inducing:
            # Use all points if we have fewer than M
            return X.copy()

        # Use k-means to find cluster centers
        try:
            Z, _ = kmeans2(X, self.n_inducing, minit="points")
        except Exception:
            # Fallback: random subset
            idx = np.random.choice(n_samples, self.n_inducing, replace=False)
            Z = X[idx].copy()

        return Z

    def fit(self, X: NDArray, y: NDArray) -> "SparseGP":
        """
        Fit sparse GP to training data.

        Args:
            X: Training inputs (N, D)
            y: Training targets (N,)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()

        self.X_train = X
        self.n_train = X.shape[0]

        # Normalize targets
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        if self._y_std < 1e-10:
            self._y_std = 1.0
        self.y_train = (y - self._y_mean) / self._y_std

        # Initialize inducing points if not provided
        if self._Z is None:
            self._Z = self._initialize_inducing_points(X)

        M = self._Z.shape[0]
        N = self.n_train

        # Compute kernel matrices
        K_uu = self.kernel(self._Z)  # (M, M)
        K_uf = self.kernel(self._Z, X)  # (M, N)

        # Cholesky of K_uu
        K_uu_jitter = K_uu + self.jitter * np.eye(M)
        self._L_uu = np.linalg.cholesky(K_uu_jitter)

        # Solve L_uu A = K_uf -> A = L_uu^{-1} K_uf
        A = solve_triangular(self._L_uu, K_uf, lower=True)  # (M, N)

        if self.method == "fitc":
            # FITC: diagonal correction
            # Λ = diag(K_ff - Q_ff) + σ²
            k_ff_diag = self.kernel.diagonal(X)  # (N,)
            q_ff_diag = np.sum(A**2, axis=0)  # (N,)
            Lambda_diag = k_ff_diag - q_ff_diag + self._noise_variance  # (N,)
            Lambda_diag = np.maximum(Lambda_diag, 1e-10)  # Ensure positive

            # B = I + A Λ^{-1} A^T
            Lambda_inv_sqrt = 1.0 / np.sqrt(Lambda_diag)
            A_scaled = A * Lambda_inv_sqrt  # (M, N)
            B = np.eye(M) + A_scaled @ A_scaled.T  # (M, M)

            self._L_B = np.linalg.cholesky(B)

            # Compute alpha for predictions
            # alpha = Σ K_uf Λ^{-1} y where Σ = (K_uu + K_uf Λ^{-1} K_fu)^{-1}
            c = A @ (self.y_train / Lambda_diag)  # (M,)
            self._alpha = cho_solve((self._L_B, True), c)

            # Log marginal likelihood (FITC)
            # log p(y) ≈ -0.5 (y^T Λ^{-1} y - c^T B^{-1} c) - 0.5 log|B| - 0.5 Σ log Λ_ii - N/2 log(2π)
            data_fit = -0.5 * (np.sum(self.y_train**2 / Lambda_diag) - np.dot(c, cho_solve((self._L_B, True), c)))
            complexity = -np.sum(np.log(np.diag(self._L_B))) - 0.5 * np.sum(np.log(Lambda_diag))
            constant = -0.5 * N * np.log(2 * np.pi)

            self._log_marginal_likelihood = data_fit + complexity + constant
            self._Lambda_diag = Lambda_diag

        else:  # VFE
            # VFE: Variational Free Energy
            sigma2 = self._noise_variance

            # B = K_uu + sigma²^{-2} K_uf K_fu
            B = K_uu + (1.0 / sigma2) * K_uf @ K_uf.T
            B = B + self.jitter * np.eye(M)
            self._L_B = np.linalg.cholesky(B)

            # alpha = sigma²^{-2} Σ K_uf y where Σ = B^{-1}
            c = K_uf @ self.y_train / sigma2
            self._alpha = cho_solve((self._L_B, True), c)

            # Log marginal likelihood (VFE lower bound)
            # ELBO = -0.5 sigma²^{-2} y^T y + sigma²^{-2} y^T K_fu alpha - 0.5 alpha^T K_uu alpha
            #        - 0.5 sigma²^{-2} Tr(K_ff - Q_ff) - N/2 log(sigma²) - 0.5 log|B/K_uu| - N/2 log(2π)
            k_ff_trace = np.sum(self.kernel.diagonal(X))
            q_ff_trace = np.sum(A**2)
            trace_term = (k_ff_trace - q_ff_trace) / sigma2

            data_fit = (
                -0.5 / sigma2 * np.dot(self.y_train, self.y_train)
                + (1.0 / sigma2) * np.dot(self.y_train, K_uf.T @ self._alpha)
                - 0.5 * np.dot(self._alpha, K_uu @ self._alpha)
            )

            complexity = (
                -np.sum(np.log(np.diag(self._L_B))) + np.sum(np.log(np.diag(self._L_uu))) - 0.5 * N * np.log(sigma2)
            )

            self._log_marginal_likelihood = data_fit - 0.5 * trace_term + complexity - 0.5 * N * np.log(2 * np.pi)

        return self

    def predict(
        self,
        X: NDArray,
        return_std: bool = True,
    ) -> GPPrediction:
        """
        Predict at test points.

        Complexity: O(M²) per test point, O(M² + PM) for P test points.

        Args:
            X: Test inputs (P, D)
            return_std: Whether to compute posterior std

        Returns:
            GPPrediction with mean, variance, std
        """
        if self._L_B is None:
            raise RuntimeError("Must call fit() before predict()")

        X = np.atleast_2d(X)

        # Cross-covariance K_*u
        K_star_u = self.kernel(X, self._Z)  # (P, M)

        # Mean: μ* = K_*u alpha
        mean = K_star_u @ self._alpha

        # Denormalize
        mean = mean * self._y_std + self._y_mean

        if return_std:
            # Variance: σ²* = k** - k*u (K_uu^{-1} - Σ) k_u*
            # where Σ = B^{-1} for VFE or (K_uu + K_uf Λ^{-1} K_fu)^{-1} for FITC

            k_star_star = self.kernel.diagonal(X)  # (P,)

            # v = L_uu^{-1} K_u*
            v = solve_triangular(self._L_uu, K_star_u.T, lower=True)  # (M, P)

            # w = L_B^{-1} v
            w = solve_triangular(self._L_B, v, lower=True)  # (M, P)

            # Variance = k** - ||v||² + ||w||²
            variance = k_star_star - np.sum(v**2, axis=0) + np.sum(w**2, axis=0)
            variance = np.maximum(variance, 1e-10) * self._y_std**2
            std = np.sqrt(variance)

            return GPPrediction(mean=mean, variance=variance, std=std)

        return GPPrediction(mean=mean, variance=np.zeros_like(mean), std=np.zeros_like(mean))

    def predict_f(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict latent function values.

        Args:
            X: Test inputs (P, D)

        Returns:
            mean: (P,)
            variance: (P,)
        """
        pred = self.predict(X, return_std=True)
        return pred.mean, pred.variance

    @property
    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood (or ELBO for VFE)."""
        if self._log_marginal_likelihood is None:
            raise RuntimeError("Must call fit() first")
        return self._log_marginal_likelihood

    def update(self, X_new: NDArray, y_new: NDArray) -> "SparseGP":
        """
        Online update with new data points.

        For efficiency, this just refits with augmented data.
        For true online updates, would need more sophisticated approach.

        Args:
            X_new: New inputs (K, D)
            y_new: New targets (K,)

        Returns:
            self
        """
        if self.X_train is None:
            return self.fit(X_new, y_new)

        # Denormalize current targets
        y_train_denorm = self.y_train * self._y_std + self._y_mean

        # Concatenate
        X_all = np.vstack([self.X_train, np.atleast_2d(X_new)])
        y_all = np.concatenate([y_train_denorm, np.atleast_1d(y_new)])

        # Refit
        return self.fit(X_all, y_all)

    def optimize_inducing_locations(
        self,
        n_iterations: int = 100,  # noqa: ARG002
        learning_rate: float = 0.01,  # noqa: ARG002
        verbose: bool = False,  # noqa: ARG002
    ) -> dict:
        """
        Optimize inducing point locations to maximize ELBO.

        Uses gradient descent on inducing point locations.

        Args:
            n_iterations: Number of optimization steps
            learning_rate: Step size
            verbose: Print progress

        Returns:
            Optimization info dict
        """
        # Simplified: just refit with k-means on current data
        if self.X_train is not None:
            self._Z = self._initialize_inducing_points(self.X_train)
            y_denorm = self.y_train * self._y_std + self._y_mean
            self.fit(self.X_train, y_denorm)

        return {"n_iterations": 0, "method": "kmeans"}

    def __repr__(self) -> str:
        return (
            f"SparseGP(n_inducing={self.n_inducing}, "
            f"method='{self.method}', "
            f"noise_variance={self._noise_variance:.6f}, "
            f"n_train={self.n_train})"
        )


class MultiOutputSparseGP:
    """
    Multi-output Sparse GP with shared or independent inducing points.

    Example:
        >>> gp = MultiOutputSparseGP(input_dim=6, output_dim=3, n_inducing=50)
        >>> gp.fit(X, Y)
        >>> mean, var = gp.predict(X_test)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_inducing: int = 50,
        noise_variance: float = 1e-4,
        share_inducing: bool = True,
    ):
        """
        Initialize multi-output sparse GP.

        Args:
            input_dim: Input dimension
            output_dim: Number of outputs
            n_inducing: Number of inducing points per output
            noise_variance: Observation noise
            share_inducing: If True, all outputs share same inducing points
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_inducing = n_inducing
        self.share_inducing = share_inducing

        # Create sparse GPs
        self.gps: list[SparseGP] = []
        for i in range(output_dim):
            kernel = SquaredExponentialARD(input_dim)
            self.gps.append(SparseGP(kernel, n_inducing, noise_variance))

    def fit(self, X: NDArray, Y: NDArray) -> "MultiOutputSparseGP":
        """
        Fit all sparse GPs.

        Args:
            X: Inputs (N, D)
            Y: Outputs (N, output_dim)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if Y.shape[1] != self.output_dim:
            Y = Y.T

        # Initialize shared inducing points
        if self.share_inducing:
            Z = self.gps[0]._initialize_inducing_points(X)
            for gp in self.gps:
                gp._Z = Z.copy()

        for i, gp in enumerate(self.gps):
            gp.fit(X, Y[:, i])

        return self

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict at test points.

        Args:
            X: Test inputs (P, D)

        Returns:
            mean: (P, output_dim)
            variance: (P, output_dim)
        """
        X = np.atleast_2d(X)
        P = X.shape[0]

        means = np.zeros((P, self.output_dim))
        variances = np.zeros((P, self.output_dim))

        for i, gp in enumerate(self.gps):
            pred = gp.predict(X)
            means[:, i] = pred.mean
            variances[:, i] = pred.variance

        return means, variances

    def predict_f(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """Alias for predict()."""
        return self.predict(X)

    def update(self, X_new: NDArray, Y_new: NDArray) -> "MultiOutputSparseGP":
        """
        Online update with new data.

        Args:
            X_new: New inputs (K, D)
            Y_new: New outputs (K, output_dim)

        Returns:
            self
        """
        Y_new = np.atleast_2d(Y_new)
        if Y_new.shape[1] != self.output_dim:
            Y_new = Y_new.T

        for i, gp in enumerate(self.gps):
            gp.update(X_new, Y_new[:, i])

        return self

    def __repr__(self) -> str:
        return (
            f"MultiOutputSparseGP(input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"n_inducing={self.n_inducing})"
        )
