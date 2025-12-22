"""
Exact Gaussian Process Regression

Implements standard GP regression with O(N³) complexity.
Used for validation and small datasets before switching to sparse GP.

The GP models:
    y = f(x) + ε,  ε ~ N(0, sigma²_n)
    f ~ GP(m(x), k(x, x'))

Posterior predictive:
    μ(x*) = k(x*, X) [K + sigma²_n I]^{-1} y
    σ²(x*) = k(x*, x*) - k(x*, X) [K + sigma²_n I]^{-1} k(X, x*)

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
    for Machine Learning. MIT Press. Chapter 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from .kernels import Kernel, SquaredExponentialARD


@dataclass
class GPPrediction:
    """Container for GP predictions."""

    mean: NDArray  # Posterior mean (N,) or (N, D)
    variance: NDArray  # Posterior variance (N,) or (N, D)
    std: NDArray  # Posterior std (N,) or (N, D)

    @property
    def confidence_bounds(self) -> Tuple[NDArray, NDArray]:
        """95% confidence bounds."""
        return self.mean - 1.96 * self.std, self.mean + 1.96 * self.std


class ExactGP:
    """
    Exact Gaussian Process Regression.

    Provides full GP inference with:
    - Posterior mean and variance prediction
    - Log marginal likelihood for hyperparameter optimization
    - Gradients for gradient-based optimization

    Complexity: O(N³) for training, O(N²) for prediction

    Example:
        >>> kernel = SquaredExponentialARD(input_dim=3)
        >>> gp = ExactGP(kernel, noise_variance=0.01)
        >>> gp.fit(X_train, y_train)
        >>> pred = gp.predict(X_test)
        >>> print(pred.mean, pred.std)
    """

    def __init__(
        self,
        kernel: Kernel,
        noise_variance: float = 1e-4,
        mean_function: Optional[Callable[[NDArray], NDArray]] = None,
        normalize_y: bool = True,
    ):
        """
        Initialize Exact GP.

        Args:
            kernel: Covariance function
            noise_variance: Observation noise sigma²_n
            mean_function: Prior mean function m(x). If None, uses zero mean.
            normalize_y: Whether to normalize targets to zero mean, unit variance
        """
        self.kernel = kernel
        self._noise_variance = noise_variance
        self.mean_function = mean_function
        self.normalize_y = normalize_y

        # Training data (set by fit())
        self.X_train: Optional[NDArray] = None
        self.y_train: Optional[NDArray] = None
        self.n_train: int = 0

        # Normalization parameters
        self._y_mean: float = 0.0
        self._y_std: float = 1.0

        # Cached computations (set by fit())
        self._L: Optional[NDArray] = None  # Cholesky factor
        self._alpha: Optional[NDArray] = None  # L^{-T} L^{-1} y
        self._log_marginal_likelihood: Optional[float] = None

    @property
    def noise_variance(self) -> float:
        """Observation noise variance sigma²_n."""
        return self._noise_variance

    @noise_variance.setter
    def noise_variance(self, value: float) -> None:
        assert value > 0, "Noise variance must be positive"
        self._noise_variance = value
        self._invalidate_cache()

    def _invalidate_cache(self) -> None:
        """Invalidate cached computations when hyperparameters change."""
        self._L = None
        self._alpha = None
        self._log_marginal_likelihood = None

    def fit(self, X: NDArray, y: NDArray) -> "ExactGP":
        """
        Fit GP to training data.

        Computes and caches the Cholesky decomposition for efficient
        subsequent predictions.

        Args:
            X: Training inputs (N, D)
            y: Training targets (N,) or (N, 1)

        Returns:
            self (for chaining)
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()

        assert X.shape[0] == len(y), "X and y must have same number of samples"

        self.X_train = X
        self.n_train = X.shape[0]

        # Normalize targets
        if self.normalize_y:
            self._y_mean = np.mean(y)
            self._y_std = np.std(y)
            if self._y_std < 1e-10:
                self._y_std = 1.0
            self.y_train = (y - self._y_mean) / self._y_std
        else:
            self._y_mean = 0.0
            self._y_std = 1.0
            self.y_train = y

        # Subtract prior mean if provided
        if self.mean_function is not None:
            self.y_train = self.y_train - self.mean_function(X) / self._y_std

        # Compute kernel matrix
        K = self.kernel(X)

        # Add noise variance to diagonal
        K_noise = K + self._noise_variance * np.eye(self.n_train)

        # Cholesky decomposition: K + sigma²I = L L^T
        try:
            self._L = np.linalg.cholesky(K_noise)
        except np.linalg.LinAlgError:
            # Add jitter if not positive definite
            jitter = 1e-6
            while jitter < 1.0:
                try:
                    self._L = np.linalg.cholesky(K_noise + jitter * np.eye(self.n_train))
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            else:
                raise ValueError("Kernel matrix is not positive definite even with jitter")

        # Solve L alpha = y, then L^T alpha = (L^{-1} y)
        # alpha = (K + sigma²I)^{-1} y
        self._alpha = cho_solve((self._L, True), self.y_train)

        # Compute log marginal likelihood
        self._compute_log_marginal_likelihood()

        return self

    def _compute_log_marginal_likelihood(self) -> None:
        """
        Compute log marginal likelihood.

        log p(y|X) = -0.5 y^T (K + sigma²I)^{-1} y - 0.5 log|K + sigma²I| - n/2 log(2π)
        """
        if self._L is None or self._alpha is None:
            return

        # Data fit term: -0.5 y^T alpha
        data_fit = -0.5 * np.dot(self.y_train, self._alpha)

        # Complexity term: -0.5 log|K + sigma²I| = -sum(log(diag(L)))
        complexity = -np.sum(np.log(np.diag(self._L)))

        # Constant term
        constant = -0.5 * self.n_train * np.log(2 * np.pi)

        self._log_marginal_likelihood = data_fit + complexity + constant

    @property
    def log_marginal_likelihood(self) -> float:
        """Log marginal likelihood of the training data."""
        if self._log_marginal_likelihood is None:
            raise RuntimeError("Must call fit() before accessing log_marginal_likelihood")
        return self._log_marginal_likelihood

    def predict(
        self,
        X: NDArray,
        return_std: bool = True,
        return_cov: bool = False,
    ) -> Union[GPPrediction, Tuple[NDArray, NDArray]]:
        """
        Predict at test points.

        Args:
            X: Test inputs (M, D)
            return_std: Whether to compute posterior std
            return_cov: Whether to return full covariance matrix

        Returns:
            GPPrediction with mean, variance, std
            or (mean, cov) if return_cov=True
        """
        if self._L is None:
            raise RuntimeError("Must call fit() before predict()")

        X = np.atleast_2d(X)

        # Cross-covariance k(X*, X)
        K_star = self.kernel(X, self.X_train)  # (M, N)

        # Posterior mean: μ* = K* alpha
        mean = K_star @ self._alpha

        # Add back prior mean and denormalize
        if self.mean_function is not None:
            mean = mean + self.mean_function(X) / self._y_std
        mean = mean * self._y_std + self._y_mean

        if return_cov:
            # Full posterior covariance
            # Σ* = K** - K* (K + sigma²I)^{-1} K*^T
            K_star_star = self.kernel(X)
            v = solve_triangular(self._L, K_star.T, lower=True)
            cov = K_star_star - v.T @ v
            cov = cov * self._y_std**2
            return mean, cov

        if return_std:
            # Posterior variance (diagonal only)
            # sigma²* = k** - k*^T (K + sigma²I)^{-1} k*
            k_star_star = self.kernel.diagonal(X)  # (M,)
            v = solve_triangular(self._L, K_star.T, lower=True)  # (N, M)
            variance = k_star_star - np.sum(v**2, axis=0)
            variance = np.maximum(variance, 1e-10)  # Numerical stability
            variance = variance * self._y_std**2
            std = np.sqrt(variance)

            return GPPrediction(mean=mean, variance=variance, std=std)

        return GPPrediction(mean=mean, variance=np.zeros_like(mean), std=np.zeros_like(mean))

    def predict_f(
        self,
        X: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Predict latent function (without observation noise).

        Same as predict() but returns (mean, variance) tuple.

        Args:
            X: Test inputs (M, D)

        Returns:
            mean: Posterior mean (M,)
            variance: Posterior variance (M,)
        """
        pred = self.predict(X, return_std=True)
        return pred.mean, pred.variance

    def sample_prior(
        self,
        X: NDArray,
        n_samples: int = 1,
        random_state: Optional[int] = None,
    ) -> NDArray:
        """
        Sample from prior distribution.

        Args:
            X: Input points (M, D)
            n_samples: Number of samples
            random_state: Random seed

        Returns:
            Samples (n_samples, M)
        """
        if random_state is not None:
            np.random.seed(random_state)

        X = np.atleast_2d(X)
        K = self.kernel(X)

        # Add jitter for numerical stability
        K = K + 1e-10 * np.eye(K.shape[0])

        L = np.linalg.cholesky(K)
        samples = L @ np.random.randn(X.shape[0], n_samples)

        # Add prior mean
        if self.mean_function is not None:
            samples = samples + self.mean_function(X)[:, None]

        return samples.T

    def sample_posterior(
        self,
        X: NDArray,
        n_samples: int = 1,
        random_state: Optional[int] = None,
    ) -> NDArray:
        """
        Sample from posterior distribution.

        Args:
            X: Input points (M, D)
            n_samples: Number of samples
            random_state: Random seed

        Returns:
            Samples (n_samples, M)
        """
        if self._L is None:
            raise RuntimeError("Must call fit() before sample_posterior()")

        if random_state is not None:
            np.random.seed(random_state)

        mean, cov = self.predict(X, return_cov=True)

        # Add jitter
        cov = cov + 1e-10 * np.eye(cov.shape[0])

        L = np.linalg.cholesky(cov)
        samples = mean[:, None] + L @ np.random.randn(X.shape[0], n_samples)

        return samples.T

    def optimize_hyperparameters(
        self,
        n_restarts: int = 5,
        verbose: bool = False,
    ) -> dict:
        """
        Optimize kernel hyperparameters by maximizing log marginal likelihood.

        Args:
            n_restarts: Number of random restarts
            verbose: Print optimization progress

        Returns:
            Dictionary with optimization results
        """
        if self.X_train is None:
            raise RuntimeError("Must call fit() before optimize_hyperparameters()")

        def objective(params):
            """Negative log marginal likelihood."""
            # Set kernel parameters
            n_kernel = self.kernel.n_params
            self.kernel.set_params(params[:n_kernel])
            self._noise_variance = np.exp(params[n_kernel])

            # Refit with new parameters
            try:
                self.fit(self.X_train, self.y_train * self._y_std + self._y_mean)
                return -self._log_marginal_likelihood
            except (np.linalg.LinAlgError, ValueError):
                return np.inf

        # Initial parameters
        initial_params = np.concatenate([self.kernel.get_params(), [np.log(self._noise_variance)]])

        best_result = None
        best_nll = np.inf

        for restart in range(n_restarts):
            if restart == 0:  # noqa: SIM108
                params0 = initial_params
            else:
                # Random initialization
                params0 = initial_params + 0.5 * np.random.randn(len(initial_params))

            try:
                result = minimize(objective, params0, method="L-BFGS-B", options={"maxiter": 100, "disp": verbose})

                if result.fun < best_nll:
                    best_nll = result.fun
                    best_result = result

            except Exception as e:
                if verbose:
                    print(f"Restart {restart} failed: {e}")

        if best_result is not None:
            # Set best parameters
            objective(best_result.x)

        return {
            "success": best_result is not None and best_result.success,
            "log_marginal_likelihood": -best_nll if best_result else None,
            "n_iterations": best_result.nit if best_result else 0,
        }

    def __repr__(self) -> str:
        return f"ExactGP(kernel={self.kernel}, noise_variance={self._noise_variance:.6f}, n_train={self.n_train})"


class MultiOutputExactGP:
    """
    Multi-output Exact GP using independent GPs per output.

    For outputs that share the same input, trains separate GPs
    for each output dimension.

    Example:
        >>> gp = MultiOutputExactGP(input_dim=6, output_dim=3)
        >>> gp.fit(X, Y)  # Y is (N, 3)
        >>> pred = gp.predict(X_test)  # Returns (M, 3) mean and variance
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel: Optional[Kernel] = None,
        noise_variance: float = 1e-4,
        share_hyperparameters: bool = False,
    ):
        """
        Initialize multi-output GP.

        Args:
            input_dim: Input dimension
            output_dim: Number of outputs
            kernel: Kernel to use (creates SE-ARD if None)
            noise_variance: Observation noise
            share_hyperparameters: If True, all outputs share same kernel
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.share_hyperparameters = share_hyperparameters

        # Create GPs for each output
        self.gps: list[ExactGP] = []
        for i in range(output_dim):
            if kernel is None:
                k = SquaredExponentialARD(input_dim)
            elif share_hyperparameters:
                k = kernel
            else:
                # Create copy of kernel
                k = SquaredExponentialARD(input_dim)
                k.set_params(kernel.get_params())

            self.gps.append(ExactGP(k, noise_variance=noise_variance))

    def fit(self, X: NDArray, Y: NDArray) -> "MultiOutputExactGP":
        """
        Fit all GPs.

        Args:
            X: Inputs (N, D)
            Y: Outputs (N, output_dim)

        Returns:
            self
        """
        X = np.atleast_2d(X)
        Y = np.atleast_2d(Y)

        if Y.shape[1] != self.output_dim:
            if Y.shape[0] == self.output_dim:
                Y = Y.T
            else:
                raise ValueError(f"Y must have {self.output_dim} columns")

        for i, gp in enumerate(self.gps):
            gp.fit(X, Y[:, i])

        return self

    def predict(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict at test points.

        Args:
            X: Test inputs (M, D)

        Returns:
            mean: (M, output_dim)
            variance: (M, output_dim)
        """
        X = np.atleast_2d(X)
        M = X.shape[0]

        means = np.zeros((M, self.output_dim))
        variances = np.zeros((M, self.output_dim))

        for i, gp in enumerate(self.gps):
            pred = gp.predict(X)
            means[:, i] = pred.mean
            variances[:, i] = pred.variance

        return means, variances

    def predict_f(self, X: NDArray) -> Tuple[NDArray, NDArray]:
        """Alias for predict()."""
        return self.predict(X)

    @property
    def log_marginal_likelihood(self) -> float:
        """Sum of log marginal likelihoods."""
        return sum(gp.log_marginal_likelihood for gp in self.gps)

    def __repr__(self) -> str:
        return f"MultiOutputExactGP(input_dim={self.input_dim}, output_dim={self.output_dim})"
