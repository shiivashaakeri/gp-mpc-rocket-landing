"""
Optimized GP Prediction for Real-Time Control

Performance optimizations for GP inference:
1. Precomputed Cholesky factorization
2. Vectorized kernel evaluations
3. Cached kernel matrices
4. Sparse approximations
5. Numba JIT compilation (optional)

Target: < 5ms for GP prediction in MPC loop

Reference:
    Quinonero-Candela, J., & Rasmussen, C. E. (2005). A Unifying View
    of Sparse Approximate Gaussian Process Regression. JMLR.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.linalg import solve_triangular

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from numba import jit, prange

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


@dataclass
class FastGPConfig:
    """Configuration for fast GP prediction."""

    # Numerical settings
    jitter: float = 1e-6

    # Caching
    cache_kernel_matrix: bool = True
    cache_cholesky: bool = True

    # Approximations
    use_sparse: bool = False
    n_inducing: int = 100

    # Vectorization
    batch_size: int = 100

    # JIT compilation
    use_numba: bool = False


class FastGPPredictor:
    """
    Optimized GP predictor for real-time control.

    Precomputes as much as possible during training:
    - Cholesky factorization of K + sigma²I
    - Alpha vector: alpha = (K + sigma²I)^{-1} y

    Prediction then only requires:
    - k* = k(x*, X)  [O(n)]
    - mean = k*' alpha   [O(n)]
    - var = k(x*, x*) - k*' L^{-1} k*  [O(n²) or O(n) with caching]

    Example:
        >>> fast_gp = FastGPPredictor(config)
        >>> fast_gp.fit(X_train, Y_train, lengthscales, variance, noise)
        >>>
        >>> # Fast prediction (< 1ms)
        >>> mean, var = fast_gp.predict(x_test)
    """

    def __init__(self, config: Optional[FastGPConfig] = None):
        """
        Initialize fast GP predictor.

        Args:
            config: Configuration parameters
        """
        self.config = config or FastGPConfig()

        # Training data
        self._X: Optional[NDArray] = None
        self._Y: Optional[NDArray] = None
        self._n_train = 0
        self._n_outputs = 0

        # Kernel parameters
        self._lengthscales: Optional[NDArray] = None
        self._variance: float = 1.0
        self._noise: float = 0.01

        # Precomputed quantities
        self._L: Optional[NDArray] = None  # Cholesky factor
        self._alpha: Optional[NDArray] = None  # (K + sigma²I)^{-1} y
        self._K_inv: Optional[NDArray] = None  # Cached inverse (optional)

        # Kernel function
        self._kernel_fn: Optional[Callable] = None

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        lengthscales: Optional[NDArray] = None,
        variance: float = 1.0,
        noise: float = 0.01,
    ) -> float:
        """
        Fit GP and precompute for fast prediction.

        Args:
            X: Training inputs (n, d)
            Y: Training outputs (n, m)
            lengthscales: Kernel lengthscales
            variance: Kernel variance
            noise: Observation noise

        Returns:
            Fitting time in ms
        """
        t_start = time.perf_counter()

        self._X = np.ascontiguousarray(X)
        self._Y = np.ascontiguousarray(Y.reshape(len(X), -1))
        self._n_train = len(X)
        self._n_outputs = self._Y.shape[1]

        # Kernel parameters
        self._variance = variance
        self._noise = noise

        if lengthscales is not None:
            self._lengthscales = np.ascontiguousarray(lengthscales)
        else:
            self._lengthscales = np.ones(X.shape[1])

        # Compute kernel matrix
        K = self._compute_kernel_matrix(self._X, self._X)
        K += (self._noise + self.config.jitter) * np.eye(self._n_train)

        # Cholesky factorization
        try:
            self._L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add more jitter if needed
            K += 1e-4 * np.eye(self._n_train)
            self._L = np.linalg.cholesky(K)

        # Precompute alpha = L'\(L\y) for each output
        self._alpha = np.zeros((self._n_train, self._n_outputs))
        for i in range(self._n_outputs):
            v = solve_triangular(self._L, self._Y[:, i], lower=True)
            self._alpha[:, i] = solve_triangular(self._L.T, v, lower=False)

        fit_time = (time.perf_counter() - t_start) * 1000
        return fit_time

    def predict(
        self,
        X_test: NDArray,
        return_var: bool = True,
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """
        Fast GP prediction.

        Args:
            X_test: Test inputs (n_test, d)
            return_var: Whether to return variance

        Returns:
            mean: Predictive mean (n_test, m)
            var: Predictive variance (n_test, m) or None
        """
        X_test = np.atleast_2d(X_test)
        n_test = len(X_test)

        # Compute cross-covariance k(X_test, X_train)
        K_star = self._compute_kernel_matrix(X_test, self._X)

        # Mean: K_star @ alpha
        mean = K_star @ self._alpha

        if not return_var:
            return mean, None

        # Variance computation
        var = np.zeros((n_test, self._n_outputs))

        # v = L^{-1} K_star^T
        v = solve_triangular(self._L, K_star.T, lower=True)

        # var = k(x*, x*) - v^T v
        K_ss = self._variance * np.ones(n_test)  # Diagonal of k(X_test, X_test)
        var_diag = K_ss - np.sum(v**2, axis=0)
        var_diag = np.maximum(var_diag, 0)  # Numerical stability

        for i in range(self._n_outputs):
            var[:, i] = var_diag

        return mean, var

    def predict_single(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """
        Predict for single point (optimized).

        Args:
            x: Single test point (d,)

        Returns:
            mean: (m,)
            var: (m,)
        """
        x = x.reshape(1, -1)
        mean, var = self.predict(x, return_var=True)
        return mean[0], var[0]

    def _compute_kernel_matrix(
        self,
        X1: NDArray,
        X2: NDArray,
    ) -> NDArray:
        """
        Compute RBF kernel matrix efficiently.

        k(x, x') = σ² exp(-0.5 * Σ (x_i - x'_i)² / l_i²)
        """
        # Scaled inputs
        X1_scaled = X1 / self._lengthscales
        X2_scaled = X2 / self._lengthscales

        # Squared distances using broadcasting
        # ||x - x'||² = ||x||² + ||x'||² - 2 x·x'
        sq1 = np.sum(X1_scaled**2, axis=1, keepdims=True)
        sq2 = np.sum(X2_scaled**2, axis=1, keepdims=True)

        sq_dist = sq1 + sq2.T - 2 * X1_scaled @ X2_scaled.T
        sq_dist = np.maximum(sq_dist, 0)  # Numerical stability

        return self._variance * np.exp(-0.5 * sq_dist)

    def update(self, X_new: NDArray, Y_new: NDArray) -> float:
        """
        Update GP with new data (recomputes factorization).

        For truly online updates, use rank-1 updates instead.

        Args:
            X_new: New inputs
            Y_new: New outputs

        Returns:
            Update time in ms
        """
        X_combined = np.vstack([self._X, X_new])
        Y_combined = np.vstack([self._Y, Y_new.reshape(-1, self._n_outputs)])

        return self.fit(X_combined, Y_combined, self._lengthscales, self._variance, self._noise)


class CachedGPPredictor(FastGPPredictor):
    """
    GP predictor with query caching for repeated predictions.

    Useful when MPC queries similar states repeatedly.
    """

    def __init__(
        self,
        config: Optional[FastGPConfig] = None,
        cache_size: int = 1000,
        cache_tolerance: float = 1e-4,
    ):
        """Initialize cached GP predictor."""
        super().__init__(config)

        self._cache_size = cache_size
        self._cache_tol = cache_tolerance

        # Cache: list of (x, mean, var) tuples
        self._cache: List[Tuple[NDArray, NDArray, NDArray]] = []

        # Statistics
        self._cache_hits = 0
        self._cache_misses = 0

    def predict_single(self, x: NDArray) -> Tuple[NDArray, NDArray]:
        """Predict with caching."""
        # Check cache
        for x_cached, mean_cached, var_cached in self._cache:
            if np.linalg.norm(x - x_cached) < self._cache_tol:
                self._cache_hits += 1
                return mean_cached, var_cached

        self._cache_misses += 1

        # Compute prediction
        mean, var = super().predict_single(x)

        # Add to cache
        if len(self._cache) >= self._cache_size:
            self._cache.pop(0)  # Remove oldest
        self._cache.append((x.copy(), mean, var))

        return mean, var

    def clear_cache(self) -> None:
        """Clear prediction cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / max(1, total)

        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
        }


class SparseGPPredictor:
    """
    Sparse GP predictor using inducing points.

    Reduces prediction complexity from O(n³) to O(m²n)
    where m << n is the number of inducing points.
    """

    def __init__(
        self,
        n_inducing: int = 100,
        config: Optional[FastGPConfig] = None,
    ):
        """
        Initialize sparse GP predictor.

        Args:
            n_inducing: Number of inducing points
            config: Configuration
        """
        self.n_inducing = n_inducing
        self.config = config or FastGPConfig()

        # Inducing points
        self._Z: Optional[NDArray] = None

        # Precomputed quantities
        self._Lm: Optional[NDArray] = None  # Cholesky of K_mm
        self._A: Optional[NDArray] = None  # Woodbury factor
        self._mean_weights: Optional[NDArray] = None

        # Kernel parameters
        self._lengthscales: Optional[NDArray] = None
        self._variance: float = 1.0
        self._noise: float = 0.01

    def fit(
        self,
        X: NDArray,
        Y: NDArray,
        lengthscales: Optional[NDArray] = None,
        variance: float = 1.0,
        noise: float = 0.01,
    ) -> float:
        """
        Fit sparse GP.

        Uses FITC approximation for efficiency.
        """
        t_start = time.perf_counter()

        n = len(X)
        m = min(self.n_inducing, n)

        # Select inducing points (k-means or random subset)
        indices = np.random.choice(n, m, replace=False)
        self._Z = X[indices].copy()

        self._lengthscales = lengthscales if lengthscales is not None else np.ones(X.shape[1])
        self._variance = variance
        self._noise = noise

        # Compute kernel matrices
        K_mm = self._compute_kernel(self._Z, self._Z)
        K_mm += self.config.jitter * np.eye(m)

        K_nm = self._compute_kernel(X, self._Z)

        # FITC diagonal approximation
        K_nn_diag = variance * np.ones(n)
        Q_nn_diag = np.sum(K_nm @ np.linalg.inv(K_mm) * K_nm, axis=1)
        Lambda = np.diag(K_nn_diag - Q_nn_diag + noise)

        # Woodbury identity for inversion
        self._Lm = np.linalg.cholesky(K_mm)

        # A = K_mm + K_mn @ Lambda^{-1} @ K_nm
        Lambda_inv = np.diag(1.0 / np.diag(Lambda))
        A = K_mm + K_nm.T @ Lambda_inv @ K_nm
        self._A = np.linalg.cholesky(A + self.config.jitter * np.eye(m))

        # Precompute mean weights
        Y = Y.reshape(n, -1)
        self._mean_weights = np.linalg.solve(self._A.T, np.linalg.solve(self._A, K_nm.T @ Lambda_inv @ Y))

        fit_time = (time.perf_counter() - t_start) * 1000
        return fit_time

    def predict(
        self,
        X_test: NDArray,
        return_var: bool = True,
    ) -> Tuple[NDArray, Optional[NDArray]]:
        """Fast sparse GP prediction."""
        X_test = np.atleast_2d(X_test)

        K_star_m = self._compute_kernel(X_test, self._Z)

        # Mean
        mean = K_star_m @ self._mean_weights

        if not return_var:
            return mean, None

        # Variance
        v = solve_triangular(self._Lm, K_star_m.T, lower=True)
        w = solve_triangular(self._A, K_star_m.T, lower=True)

        var_diag = self._variance - np.sum(v**2, axis=0) + np.sum(w**2, axis=0)
        var_diag = np.maximum(var_diag, 0)

        var = np.tile(var_diag.reshape(-1, 1), (1, mean.shape[1]))

        return mean, var

    def _compute_kernel(self, X1: NDArray, X2: NDArray) -> NDArray:
        """Compute RBF kernel."""
        X1_scaled = X1 / self._lengthscales
        X2_scaled = X2 / self._lengthscales

        sq1 = np.sum(X1_scaled**2, axis=1, keepdims=True)
        sq2 = np.sum(X2_scaled**2, axis=1, keepdims=True)
        sq_dist = sq1 + sq2.T - 2 * X1_scaled @ X2_scaled.T

        return self._variance * np.exp(-0.5 * np.maximum(sq_dist, 0))


# Numba-accelerated kernel computation (optional)
if HAS_NUMBA:

    @jit(nopython=True, parallel=True, cache=True)
    def _rbf_kernel_numba(
        X1: NDArray,
        X2: NDArray,
        lengthscales: NDArray,
        variance: float,
    ) -> NDArray:
        """Numba-accelerated RBF kernel."""
        n1, d = X1.shape
        n2 = X2.shape[0]

        K = np.empty((n1, n2))

        for i in prange(n1):
            for j in range(n2):
                sq_dist = 0.0
                for k in range(d):
                    diff = (X1[i, k] - X2[j, k]) / lengthscales[k]
                    sq_dist += diff * diff
                K[i, j] = variance * np.exp(-0.5 * sq_dist)

        return K


class NumbaGPPredictor(FastGPPredictor):
    """GP predictor with Numba-accelerated kernel computation."""

    def _compute_kernel_matrix(
        self,
        X1: NDArray,
        X2: NDArray,
    ) -> NDArray:
        """Use Numba kernel if available."""
        if HAS_NUMBA:
            return _rbf_kernel_numba(
                np.ascontiguousarray(X1),
                np.ascontiguousarray(X2),
                self._lengthscales,
                self._variance,
            )
        else:
            return super()._compute_kernel_matrix(X1, X2)


def create_fast_gp(
    gp_type: str = "exact",
    n_inducing: int = 100,
    use_cache: bool = True,  # noqa: ARG001
    use_numba: bool = True,
) -> FastGPPredictor:
    """
    Factory function for fast GP predictors.

    Args:
        gp_type: "exact", "sparse", "cached", "numba"
        n_inducing: Number of inducing points (for sparse)
        use_cache: Enable prediction caching
        use_numba: Use Numba acceleration

    Returns:
        Fast GP predictor instance
    """
    config = FastGPConfig(use_numba=use_numba and HAS_NUMBA)

    if gp_type == "sparse":
        return SparseGPPredictor(n_inducing, config)
    elif gp_type == "cached":
        return CachedGPPredictor(config)
    elif gp_type == "numba" and HAS_NUMBA:
        return NumbaGPPredictor(config)
    else:
        return FastGPPredictor(config)
