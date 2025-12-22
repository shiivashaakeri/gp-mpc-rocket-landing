"""
Kernel Functions for Gaussian Processes

Implements kernel (covariance) functions for GP regression:
- Squared Exponential (RBF) with Automatic Relevance Determination (ARD)
- Matérn kernels (1/2, 3/2, 5/2)
- Kernel composition (sum, product)
- Gradient computation for hyperparameter optimization

The SE-ARD kernel is the primary kernel for the GP-MPC framework:
    k(x, x') = σ² exp(-0.5 Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)

where σ² is the signal variance and lᵢ are per-dimension lengthscales.

Reference:
    Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian Processes
    for Machine Learning. MIT Press.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Base Kernel Class
# =============================================================================


class Kernel(ABC):
    """
    Abstract base class for kernel functions.

    All kernels implement:
    - __call__(X1, X2): Compute kernel matrix K(X1, X2)
    - diagonal(X): Compute diagonal k(xᵢ, xᵢ) efficiently
    - gradients(X1, X2): Compute gradients w.r.t. hyperparameters
    """

    @abstractmethod
    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Compute kernel matrix.

        Args:
            X1: First set of points (N1, D)
            X2: Second set of points (N2, D). If None, compute K(X1, X1).

        Returns:
            Kernel matrix (N1, N2)
        """
        pass

    @abstractmethod
    def diagonal(self, X: NDArray) -> NDArray:
        """
        Compute diagonal of kernel matrix k(xᵢ, xᵢ).

        More efficient than computing full matrix when only diagonal needed.

        Args:
            X: Input points (N, D)

        Returns:
            Diagonal values (N,)
        """
        pass

    @abstractmethod
    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        """
        Compute gradients of kernel matrix w.r.t. hyperparameters.

        Args:
            X1: First set of points (N1, D)
            X2: Second set of points (N2, D)

        Returns:
            Dictionary mapping parameter names to gradient matrices
        """
        pass

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of hyperparameters."""
        pass

    @property
    @abstractmethod
    def param_names(self) -> List[str]:
        """Names of hyperparameters."""
        pass

    @abstractmethod
    def get_params(self) -> NDArray:
        """Get hyperparameters as array (in log space for positive params)."""
        pass

    @abstractmethod
    def set_params(self, params: NDArray) -> None:
        """Set hyperparameters from array (in log space for positive params)."""
        pass

    def __add__(self, other: "Kernel") -> "SumKernel":
        """Add two kernels: k(x,x') = k1(x,x') + k2(x,x')"""
        return SumKernel(self, other)

    def __mul__(self, other: "Kernel") -> "ProductKernel":
        """Multiply two kernels: k(x,x') = k1(x,x') * k2(x,x')"""
        return ProductKernel(self, other)


# =============================================================================
# Squared Exponential Kernel with ARD
# =============================================================================


class SquaredExponentialARD(Kernel):
    """
    Squared Exponential (RBF) kernel with Automatic Relevance Determination.

    k(x, x') = σ² exp(-0.5 Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)

    Also known as the Radial Basis Function (RBF) or Gaussian kernel.

    ARD means each input dimension has its own lengthscale, allowing
    the GP to automatically determine which features are relevant.

    Hyperparameters:
        - signal_variance (σ²): Output scale, controls function amplitude
        - lengthscales (l): Per-dimension lengthscales, controls smoothness

    Example:
        >>> kernel = SquaredExponentialARD(input_dim=3)
        >>> X = np.random.randn(10, 3)
        >>> K = kernel(X)  # (10, 10) kernel matrix
        >>> k_diag = kernel.diagonal(X)  # (10,) diagonal
    """

    def __init__(
        self,
        input_dim: int,
        signal_variance: float = 1.0,
        lengthscales: Optional[NDArray] = None,
        learn_signal_variance: bool = True,
        learn_lengthscales: bool = True,
    ):
        """
        Initialize SE-ARD kernel.

        Args:
            input_dim: Number of input dimensions
            signal_variance: Initial signal variance σ²
            lengthscales: Initial lengthscales (input_dim,).
                         If None, initialized to 1.0 for all dimensions.
            learn_signal_variance: Whether to optimize signal variance
            learn_lengthscales: Whether to optimize lengthscales
        """
        self.input_dim = input_dim
        self._signal_variance = signal_variance

        if lengthscales is None:
            self._lengthscales = np.ones(input_dim)
        else:
            self._lengthscales = np.asarray(lengthscales).flatten()
            assert len(self._lengthscales) == input_dim

        self.learn_signal_variance = learn_signal_variance
        self.learn_lengthscales = learn_lengthscales

    @property
    def signal_variance(self) -> float:
        """Signal variance σ²."""
        return self._signal_variance

    @signal_variance.setter
    def signal_variance(self, value: float) -> None:
        assert value > 0, "Signal variance must be positive"
        self._signal_variance = value

    @property
    def lengthscales(self) -> NDArray:
        """Lengthscales l (one per input dimension)."""
        return self._lengthscales

    @lengthscales.setter
    def lengthscales(self, value: NDArray) -> None:
        value = np.asarray(value).flatten()
        assert len(value) == self.input_dim
        assert np.all(value > 0), "Lengthscales must be positive"
        self._lengthscales = value

    def _compute_scaled_distance_sq(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Compute squared scaled Euclidean distance.

        r²(x, x') = Σᵢ (xᵢ - x'ᵢ)² / lᵢ²

        Args:
            X1: (N1, D)
            X2: (N2, D) or None

        Returns:
            Squared distances (N1, N2)
        """
        # Scale inputs by lengthscales
        X1_scaled = X1 / self._lengthscales

        X2_scaled = X1_scaled if X2 is None else X2 / self._lengthscales

        # Compute ||x1 - x2||² = ||x1||² + ||x2||² - 2 * x1·x2
        X1_sq = np.sum(X1_scaled**2, axis=1, keepdims=True)  # (N1, 1)
        X2_sq = np.sum(X2_scaled**2, axis=1, keepdims=True)  # (N2, 1)

        dist_sq = X1_sq + X2_sq.T - 2 * X1_scaled @ X2_scaled.T

        # Numerical stability: ensure non-negative
        dist_sq = np.maximum(dist_sq, 0.0)

        return dist_sq

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Compute kernel matrix K(X1, X2).

        K[i,j] = σ² exp(-0.5 * r²(x1ᵢ, x2ⱼ))

        Args:
            X1: First inputs (N1, D)
            X2: Second inputs (N2, D). If None, uses X2 = X1.

        Returns:
            Kernel matrix (N1, N2)
        """
        X1 = np.atleast_2d(X1)
        if X2 is not None:
            X2 = np.atleast_2d(X2)

        dist_sq = self._compute_scaled_distance_sq(X1, X2)
        K = self._signal_variance * np.exp(-0.5 * dist_sq)

        return K

    def diagonal(self, X: NDArray) -> NDArray:
        """
        Compute diagonal k(xᵢ, xᵢ) = σ².

        For stationary kernels, the diagonal is constant.

        Args:
            X: Input points (N, D)

        Returns:
            Diagonal (N,) with value σ² everywhere
        """
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._signal_variance)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        """
        Compute gradients of kernel matrix w.r.t. hyperparameters.

        For hyperparameter optimization, we compute:
        - ∂K/∂(log σ²) = K (gradient w.r.t. log signal variance)
        - ∂K/∂(log lᵢ) = K ⊙ (xᵢ - x'ᵢ)² / lᵢ² (gradient w.r.t. log lengthscale)

        Args:
            X1: First inputs (N1, D)
            X2: Second inputs (N2, D)

        Returns:
            Dictionary with gradient matrices
        """
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2) if X2 is not None else X1

        # Compute kernel matrix
        K = self(X1, X2)

        grads = {}

        # Gradient w.r.t. log signal variance: ∂K/∂(log σ²) = K
        if self.learn_signal_variance:
            grads["log_signal_variance"] = K.copy()

        # Gradient w.r.t. log lengthscales
        if self.learn_lengthscales:
            for i in range(self.input_dim):
                # (x1ᵢ - x2ᵢ)² / lᵢ²
                diff_i = (X1[:, i : i + 1] - X2[:, i : i + 1].T) ** 2 / (self._lengthscales[i] ** 2)
                # ∂K/∂(log lᵢ) = K ⊙ (x1ᵢ - x2ᵢ)² / lᵢ²
                grads[f"log_lengthscale_{i}"] = K * diff_i

        return grads

    @property
    def n_params(self) -> int:
        """Number of hyperparameters."""
        n = 0
        if self.learn_signal_variance:
            n += 1
        if self.learn_lengthscales:
            n += self.input_dim
        return n

    @property
    def param_names(self) -> List[str]:
        """Names of hyperparameters."""
        names = []
        if self.learn_signal_variance:
            names.append("log_signal_variance")
        if self.learn_lengthscales:
            for i in range(self.input_dim):
                names.append(f"log_lengthscale_{i}")
        return names

    def get_params(self) -> NDArray:
        """
        Get hyperparameters as array (log space for positivity).

        Returns:
            Array of [log(σ²), log(l₀), log(l₁), ...]
        """
        params = []
        if self.learn_signal_variance:
            params.append(np.log(self._signal_variance))
        if self.learn_lengthscales:
            params.extend(np.log(self._lengthscales))
        return np.array(params)

    def set_params(self, params: NDArray) -> None:
        """
        Set hyperparameters from array (log space).

        Args:
            params: Array of [log(σ²), log(l₀), log(l₁), ...]
        """
        params = np.asarray(params).flatten()
        idx = 0

        if self.learn_signal_variance:
            self._signal_variance = np.exp(params[idx])
            idx += 1

        if self.learn_lengthscales:
            self._lengthscales = np.exp(params[idx : idx + self.input_dim])
            idx += self.input_dim

    def __repr__(self) -> str:
        return (
            f"SquaredExponentialARD("
            f"input_dim={self.input_dim}, "
            f"signal_variance={self._signal_variance:.4f}, "
            f"lengthscales={self._lengthscales})"
        )


# Alias for convenience
RBF = SquaredExponentialARD
SE_ARD = SquaredExponentialARD


# =============================================================================
# Isotropic SE Kernel (single lengthscale)
# =============================================================================


class SquaredExponential(Kernel):
    """
    Isotropic Squared Exponential kernel (single lengthscale for all dimensions).

    k(x, x') = σ² exp(-||x - x'||² / (2l²))

    Simpler than ARD when all features are equally relevant.
    """

    def __init__(
        self,
        signal_variance: float = 1.0,
        lengthscale: float = 1.0,
    ):
        self._signal_variance = signal_variance
        self._lengthscale = lengthscale

    @property
    def signal_variance(self) -> float:
        return self._signal_variance

    @property
    def lengthscale(self) -> float:
        return self._lengthscale

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        X1 = np.atleast_2d(X1)
        X2 = X1 if X2 is None else np.atleast_2d(X2)

        # Squared Euclidean distance
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0.0)

        K = self._signal_variance * np.exp(-dist_sq / (2 * self._lengthscale**2))
        return K

    def diagonal(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._signal_variance)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        X1 = np.atleast_2d(X1)
        X2 = X1 if X2 is None else np.atleast_2d(X2)

        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0.0)

        K = self._signal_variance * np.exp(-dist_sq / (2 * self._lengthscale**2))

        return {
            "log_signal_variance": K,
            "log_lengthscale": K * dist_sq / (self._lengthscale**2),
        }

    @property
    def n_params(self) -> int:
        return 2

    @property
    def param_names(self) -> List[str]:
        return ["log_signal_variance", "log_lengthscale"]

    def get_params(self) -> NDArray:
        return np.array([np.log(self._signal_variance), np.log(self._lengthscale)])

    def set_params(self, params: NDArray) -> None:
        self._signal_variance = np.exp(params[0])
        self._lengthscale = np.exp(params[1])

    def __repr__(self) -> str:
        return f"SquaredExponential(σ²={self._signal_variance:.4f}, l={self._lengthscale:.4f})"


# =============================================================================
# Matérn Kernels
# =============================================================================


class Matern32(Kernel):
    """
    Matérn 3/2 kernel with ARD.

    k(x, x') = σ² (1 + √3 r) exp(-√3 r)

    where r = √(Σᵢ (xᵢ - x'ᵢ)² / lᵢ²)

    Less smooth than SE (once differentiable), often more realistic for
    physical systems.
    """

    def __init__(
        self,
        input_dim: int,
        signal_variance: float = 1.0,
        lengthscales: Optional[NDArray] = None,
    ):
        self.input_dim = input_dim
        self._signal_variance = signal_variance

        if lengthscales is None:
            self._lengthscales = np.ones(input_dim)
        else:
            self._lengthscales = np.asarray(lengthscales).flatten()

    @property
    def signal_variance(self) -> float:
        return self._signal_variance

    @property
    def lengthscales(self) -> NDArray:
        return self._lengthscales

    def _compute_scaled_distance(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        """Compute scaled Euclidean distance (not squared)."""
        X1_scaled = X1 / self._lengthscales
        X2_scaled = X1_scaled if X2 is None else X2 / self._lengthscales

        X1_sq = np.sum(X1_scaled**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_scaled**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1_scaled @ X2_scaled.T
        dist_sq = np.maximum(dist_sq, 0.0)

        return np.sqrt(dist_sq)

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        X1 = np.atleast_2d(X1)
        if X2 is not None:
            X2 = np.atleast_2d(X2)

        r = self._compute_scaled_distance(X1, X2)
        sqrt3_r = np.sqrt(3) * r

        K = self._signal_variance * (1 + sqrt3_r) * np.exp(-sqrt3_r)
        return K

    def diagonal(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._signal_variance)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        # Simplified: return gradient w.r.t. log signal variance
        K = self(X1, X2)
        return {"log_signal_variance": K}

    @property
    def n_params(self) -> int:
        return 1 + self.input_dim

    @property
    def param_names(self) -> List[str]:
        return ["log_signal_variance"] + [f"log_lengthscale_{i}" for i in range(self.input_dim)]

    def get_params(self) -> NDArray:
        return np.concatenate([[np.log(self._signal_variance)], np.log(self._lengthscales)])

    def set_params(self, params: NDArray) -> None:
        self._signal_variance = np.exp(params[0])
        self._lengthscales = np.exp(params[1:])

    def __repr__(self) -> str:
        return f"Matern32(input_dim={self.input_dim}, σ²={self._signal_variance:.4f})"


class Matern52(Kernel):
    """
    Matérn 5/2 kernel with ARD.

    k(x, x') = σ² (1 + √5 r + 5r²/3) exp(-√5 r)

    Twice differentiable, good balance between SE smoothness and Matern32.
    """

    def __init__(
        self,
        input_dim: int,
        signal_variance: float = 1.0,
        lengthscales: Optional[NDArray] = None,
    ):
        self.input_dim = input_dim
        self._signal_variance = signal_variance

        if lengthscales is None:
            self._lengthscales = np.ones(input_dim)
        else:
            self._lengthscales = np.asarray(lengthscales).flatten()

    @property
    def signal_variance(self) -> float:
        return self._signal_variance

    @property
    def lengthscales(self) -> NDArray:
        return self._lengthscales

    def _compute_scaled_distance(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        X1_scaled = X1 / self._lengthscales
        X2_scaled = X1_scaled if X2 is None else X2 / self._lengthscales

        X1_sq = np.sum(X1_scaled**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2_scaled**2, axis=1, keepdims=True)
        dist_sq = X1_sq + X2_sq.T - 2 * X1_scaled @ X2_scaled.T
        dist_sq = np.maximum(dist_sq, 0.0)

        return np.sqrt(dist_sq)

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        X1 = np.atleast_2d(X1)
        if X2 is not None:
            X2 = np.atleast_2d(X2)

        r = self._compute_scaled_distance(X1, X2)
        sqrt5_r = np.sqrt(5) * r

        K = self._signal_variance * (1 + sqrt5_r + 5 * r**2 / 3) * np.exp(-sqrt5_r)
        return K

    def diagonal(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._signal_variance)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        K = self(X1, X2)
        return {"log_signal_variance": K}

    @property
    def n_params(self) -> int:
        return 1 + self.input_dim

    @property
    def param_names(self) -> List[str]:
        return ["log_signal_variance"] + [f"log_lengthscale_{i}" for i in range(self.input_dim)]

    def get_params(self) -> NDArray:
        return np.concatenate([[np.log(self._signal_variance)], np.log(self._lengthscales)])

    def set_params(self, params: NDArray) -> None:
        self._signal_variance = np.exp(params[0])
        self._lengthscales = np.exp(params[1:])

    def __repr__(self) -> str:
        return f"Matern52(input_dim={self.input_dim}, σ²={self._signal_variance:.4f})"


# =============================================================================
# Composite Kernels
# =============================================================================


class SumKernel(Kernel):
    """
    Sum of two kernels: k(x,x') = k1(x,x') + k2(x,x')

    Useful for combining different covariance structures.
    """

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def diagonal(self, X: NDArray) -> NDArray:
        return self.k1.diagonal(X) + self.k2.diagonal(X)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        grads = {}
        for name, grad in self.k1.gradients(X1, X2).items():
            grads[f"k1_{name}"] = grad
        for name, grad in self.k2.gradients(X1, X2).items():
            grads[f"k2_{name}"] = grad
        return grads

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    @property
    def param_names(self) -> List[str]:
        return [f"k1_{n}" for n in self.k1.param_names] + [f"k2_{n}" for n in self.k2.param_names]

    def get_params(self) -> NDArray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])

    def set_params(self, params: NDArray) -> None:
        n1 = self.k1.n_params
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])

    def __repr__(self) -> str:
        return f"SumKernel({self.k1}, {self.k2})"


class ProductKernel(Kernel):
    """
    Product of two kernels: k(x,x') = k1(x,x') * k2(x,x')

    Useful for modeling interactions between different input subspaces.
    """

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1 = k1
        self.k2 = k2

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        return self.k1(X1, X2) * self.k2(X1, X2)

    def diagonal(self, X: NDArray) -> NDArray:
        return self.k1.diagonal(X) * self.k2.diagonal(X)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        K1 = self.k1(X1, X2)
        K2 = self.k2(X1, X2)

        grads = {}
        for name, grad in self.k1.gradients(X1, X2).items():
            grads[f"k1_{name}"] = grad * K2
        for name, grad in self.k2.gradients(X1, X2).items():
            grads[f"k2_{name}"] = K1 * grad
        return grads

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    @property
    def param_names(self) -> List[str]:
        return [f"k1_{n}" for n in self.k1.param_names] + [f"k2_{n}" for n in self.k2.param_names]

    def get_params(self) -> NDArray:
        return np.concatenate([self.k1.get_params(), self.k2.get_params()])

    def set_params(self, params: NDArray) -> None:
        n1 = self.k1.n_params
        self.k1.set_params(params[:n1])
        self.k2.set_params(params[n1:])

    def __repr__(self) -> str:
        return f"ProductKernel({self.k1}, {self.k2})"


# =============================================================================
# White Noise Kernel
# =============================================================================


class WhiteNoise(Kernel):
    """
    White noise kernel (diagonal): k(x,x') = σ² δ(x,x')

    Adds independent noise to each observation.
    """

    def __init__(self, noise_variance: float = 1e-6):
        self._noise_variance = noise_variance

    @property
    def noise_variance(self) -> float:
        return self._noise_variance

    def __call__(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> NDArray:
        X1 = np.atleast_2d(X1)
        if X2 is None:
            return self._noise_variance * np.eye(X1.shape[0])
        else:
            X2 = np.atleast_2d(X2)
            # Non-zero only when X1 and X2 are the same points
            # For simplicity, return zeros for cross-covariance
            return np.zeros((X1.shape[0], X2.shape[0]))

    def diagonal(self, X: NDArray) -> NDArray:
        X = np.atleast_2d(X)
        return np.full(X.shape[0], self._noise_variance)

    def gradients(
        self,
        X1: NDArray,
        X2: Optional[NDArray] = None,
    ) -> dict[str, NDArray]:
        return {"log_noise_variance": self(X1, X2)}

    @property
    def n_params(self) -> int:
        return 1

    @property
    def param_names(self) -> List[str]:
        return ["log_noise_variance"]

    def get_params(self) -> NDArray:
        return np.array([np.log(self._noise_variance)])

    def set_params(self, params: NDArray) -> None:
        self._noise_variance = np.exp(params[0])

    def __repr__(self) -> str:
        return f"WhiteNoise(σ²={self._noise_variance:.6f})"


# =============================================================================
# Factory Functions
# =============================================================================


def create_se_ard_kernel(
    input_dim: int,
    signal_variance: float = 1.0,
    lengthscales: Optional[NDArray] = None,
) -> SquaredExponentialARD:
    """
    Create SE kernel with ARD.

    Args:
        input_dim: Number of input dimensions
        signal_variance: Initial signal variance
        lengthscales: Initial lengthscales (defaults to ones)

    Returns:
        SquaredExponentialARD kernel
    """
    return SquaredExponentialARD(
        input_dim=input_dim,
        signal_variance=signal_variance,
        lengthscales=lengthscales,
    )


def create_matern_kernel(
    input_dim: int,
    nu: float = 2.5,
    signal_variance: float = 1.0,
    lengthscales: Optional[NDArray] = None,
) -> Kernel:
    """
    Create Matérn kernel.

    Args:
        input_dim: Number of input dimensions
        nu: Smoothness parameter (0.5, 1.5, or 2.5)
        signal_variance: Initial signal variance
        lengthscales: Initial lengthscales

    Returns:
        Matern kernel (32 or 52)
    """
    if nu in (1.5, 3 / 2):
        return Matern32(input_dim, signal_variance, lengthscales)
    elif nu in (2.5, 5 / 2):
        return Matern52(input_dim, signal_variance, lengthscales)
    else:
        raise ValueError(f"Unsupported nu={nu}. Use 1.5 (Matern32) or 2.5 (Matern52)")
