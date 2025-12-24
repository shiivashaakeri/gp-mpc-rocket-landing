"""
Q-Function Approximation for LMPC

The Q-function (cost-to-go) is a key component of LMPC:
    Q(x) = min_u [l(x, u) + Q(f(x, u))]

In LMPC, we approximate Q using stored trajectory data:
1. Exact values at safe set points
2. Interpolation between points

Interpolation methods:
- K-nearest neighbor weighted average
- Gaussian Process regression
- Local polynomial fitting
- Neural network approximation

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning MPC for Iterative Tasks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .safe_set import SampledSafeSet


@dataclass
class QFunctionConfig:
    """Configuration for Q-function approximation."""

    # Interpolation settings
    method: str = "inverse_distance"  # "inverse_distance", "gp", "local_linear"
    n_neighbors: int = 10

    # Inverse distance weighting
    idw_power: float = 2.0  # Higher = more weight to nearest

    # Local linear regression
    local_linear_reg: float = 1e-4  # Regularization

    # GP settings (if method="gp")
    gp_lengthscale: float = 1.0
    gp_variance: float = 1.0
    gp_noise: float = 0.01

    # Bounds
    q_min: float = 0.0
    q_max: float = np.inf


class QFunctionApproximator(ABC):
    """Abstract base class for Q-function approximation."""

    @abstractmethod
    def fit(self, states: NDArray, q_values: NDArray) -> None:
        """Fit approximator to data."""
        pass

    @abstractmethod
    def predict(self, x: NDArray) -> float:
        """Predict Q-value at state x."""
        pass

    @abstractmethod
    def predict_batch(self, X: NDArray) -> NDArray:
        """Predict Q-values for multiple states."""
        pass


class InverseDistanceQFunction(QFunctionApproximator):
    """
    Inverse distance weighted Q-function approximation.

    Q(x) = Σ w_i Q(x_i) / Σ w_i

    where w_i = 1 / ||x - x_i||^p

    Simple, fast, and works well for smooth Q-functions.
    """

    def __init__(self, config: Optional[QFunctionConfig] = None):
        self.config = config or QFunctionConfig()
        self._states: Optional[NDArray] = None
        self._q_values: Optional[NDArray] = None

    def fit(self, states: NDArray, q_values: NDArray) -> None:
        """Store data (no fitting needed)."""
        self._states = states.copy()
        self._q_values = q_values.copy()

    def predict(self, x: NDArray) -> float:
        """Predict Q-value using inverse distance weighting."""
        if self._states is None or len(self._states) == 0:
            return self.config.q_max

        # Compute distances
        distances = np.linalg.norm(self._states - x, axis=1)

        # Check for exact match
        min_dist = np.min(distances)
        if min_dist < 1e-10:
            return float(self._q_values[np.argmin(distances)])

        # Use K nearest neighbors
        K = min(self.config.n_neighbors, len(self._states))
        nearest_idx = np.argpartition(distances, K - 1)[:K]

        nearest_dist = distances[nearest_idx]
        nearest_q = self._q_values[nearest_idx]

        # Inverse distance weights
        weights = 1.0 / (nearest_dist**self.config.idw_power)
        weights = weights / np.sum(weights)

        q_pred = float(np.dot(weights, nearest_q))
        return np.clip(q_pred, self.config.q_min, self.config.q_max)

    def predict_batch(self, X: NDArray) -> NDArray:
        """Predict Q-values for multiple states."""
        return np.array([self.predict(x) for x in X])


class LocalLinearQFunction(QFunctionApproximator):
    """
    Local linear regression Q-function.

    Fits a local linear model around each query point:
        Q(x) ≈ a + b^T (x - x_0)

    Uses weighted least squares with distance-based weights.
    """

    def __init__(self, config: Optional[QFunctionConfig] = None):
        self.config = config or QFunctionConfig()
        self._states: Optional[NDArray] = None
        self._q_values: Optional[NDArray] = None

    def fit(self, states: NDArray, q_values: NDArray) -> None:
        """Store data."""
        self._states = states.copy()
        self._q_values = q_values.copy()

    def predict(self, x: NDArray) -> float:
        """Predict using local linear regression."""
        if self._states is None or len(self._states) == 0:
            return self.config.q_max

        # Get K nearest neighbors
        distances = np.linalg.norm(self._states - x, axis=1)
        K = min(self.config.n_neighbors, len(self._states))
        nearest_idx = np.argpartition(distances, K - 1)[:K]

        X_local = self._states[nearest_idx]
        y_local = self._q_values[nearest_idx]
        d_local = distances[nearest_idx]

        # Weights
        weights = 1.0 / (d_local + 1e-10)
        W = np.diag(weights)

        # Centered coordinates
        X_centered = X_local - x

        # Add intercept term
        n_x = X_centered.shape[1]
        X_design = np.hstack([np.ones((K, 1)), X_centered])

        # Weighted least squares with regularization
        reg = self.config.local_linear_reg * np.eye(n_x + 1)
        reg[0, 0] = 0  # Don't regularize intercept

        try:
            A = X_design.T @ W @ X_design + reg
            b = X_design.T @ W @ y_local
            coeffs = np.linalg.solve(A, b)

            # Prediction at x (centered = 0)
            q_pred = coeffs[0]  # Just the intercept
        except np.linalg.LinAlgError:
            # Fallback to inverse distance
            weights = weights / np.sum(weights)
            q_pred = float(np.dot(weights, y_local))

        return np.clip(q_pred, self.config.q_min, self.config.q_max)

    def predict_batch(self, X: NDArray) -> NDArray:
        """Predict Q-values for multiple states."""
        return np.array([self.predict(x) for x in X])


class GPQFunction(QFunctionApproximator):
    """
    Gaussian Process Q-function approximation.

    Provides uncertainty estimates along with predictions.
    Uses a sparse GP for efficiency with large safe sets.
    """

    def __init__(self, config: Optional[QFunctionConfig] = None):
        self.config = config or QFunctionConfig()
        self._gp = None
        self._is_fitted = False

    def fit(self, states: NDArray, q_values: NDArray) -> None:
        """Fit GP to Q-function data."""
        try:
            from ..gp import SEKernelARD, SparseGP, SparseGPConfig  # noqa: PLC0415

            n_x = states.shape[1]
            n_data = len(states)

            # Create kernel
            kernel = SEKernelARD(
                lengthscales=np.full(n_x, self.config.gp_lengthscale),
                variance=self.config.gp_variance,
            )

            # Create sparse GP
            n_inducing = min(50, n_data // 2)
            gp_config = SparseGPConfig(n_inducing=n_inducing)

            self._gp = SparseGP(kernel, gp_config)
            self._gp.fit(states, q_values)
            self._is_fitted = True

        except ImportError:
            # Fallback to inverse distance
            print("GP module not available, using inverse distance")
            self._fallback = InverseDistanceQFunction(self.config)
            self._fallback.fit(states, q_values)
            self._is_fitted = False

    def predict(self, x: NDArray) -> float:
        """Predict Q-value with GP."""
        if self._is_fitted and self._gp is not None:
            mean, var = self._gp.predict(x.reshape(1, -1))
            return float(np.clip(mean[0], self.config.q_min, self.config.q_max))
        elif hasattr(self, "_fallback"):
            return self._fallback.predict(x)
        else:
            return self.config.q_max

    def predict_with_uncertainty(self, x: NDArray) -> Tuple[float, float]:
        """Predict Q-value and uncertainty."""
        if self._is_fitted and self._gp is not None:
            mean, var = self._gp.predict(x.reshape(1, -1))
            return float(mean[0]), float(var[0])
        else:
            return self.predict(x), 0.0

    def predict_batch(self, X: NDArray) -> NDArray:
        """Predict Q-values for multiple states."""
        if self._is_fitted and self._gp is not None:
            mean, _ = self._gp.predict(X)
            return np.clip(mean, self.config.q_min, self.config.q_max)
        elif hasattr(self, "_fallback"):
            return self._fallback.predict_batch(X)
        else:
            return np.full(len(X), self.config.q_max)


class QFunctionManager:
    """
    Manages Q-function approximation for LMPC.

    Handles:
    - Automatic updating when safe set changes
    - Multiple approximation methods
    - Iteration-aware Q-function (Q^j for iteration j)

    Example:
        >>> q_manager = QFunctionManager(safe_set, method="inverse_distance")
        >>>
        >>> # After adding trajectory to safe set
        >>> q_manager.update()
        >>>
        >>> # Query Q-value
        >>> q_value = q_manager.evaluate(x)
    """

    def __init__(
        self,
        safe_set: SampledSafeSet,
        config: Optional[QFunctionConfig] = None,
    ):
        """
        Initialize Q-function manager.

        Args:
            safe_set: Sampled safe set
            config: Q-function configuration
        """
        self.safe_set = safe_set
        self.config = config or QFunctionConfig()

        # Create approximator
        if self.config.method == "inverse_distance":
            self._approximator = InverseDistanceQFunction(self.config)
        elif self.config.method == "local_linear":
            self._approximator = LocalLinearQFunction(self.config)
        elif self.config.method == "gp":
            self._approximator = GPQFunction(self.config)
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

        self._is_fitted = False
        self._last_update_iteration = -1

    def update(self) -> None:
        """Update Q-function from safe set."""
        states = self.safe_set.get_all_states()
        q_values = self.safe_set.get_all_q_values()

        if len(states) > 0:
            self._approximator.fit(states, q_values)
            self._is_fitted = True

        self._last_update_iteration = self.safe_set.num_iterations

    def evaluate(self, x: NDArray) -> float:
        """Evaluate Q-function at state x."""
        if not self._is_fitted:
            self.update()

        return self._approximator.predict(x)

    def evaluate_batch(self, X: NDArray) -> NDArray:
        """Evaluate Q-function for multiple states."""
        if not self._is_fitted:
            self.update()

        return self._approximator.predict_batch(X)

    def needs_update(self) -> bool:
        """Check if Q-function needs updating."""
        return self.safe_set.num_iterations > self._last_update_iteration

    def get_min_q(self) -> float:
        """Get minimum Q-value in safe set."""
        if self.safe_set.num_states == 0:
            return np.inf
        return float(np.min(self.safe_set.get_all_q_values()))

    def get_statistics(self) -> dict:
        """Get Q-function statistics."""
        q_values = self.safe_set.get_all_q_values()

        if len(q_values) == 0:
            return {"n_points": 0}

        return {
            "n_points": len(q_values),
            "q_min": float(np.min(q_values)),
            "q_max": float(np.max(q_values)),
            "q_mean": float(np.mean(q_values)),
            "q_std": float(np.std(q_values)),
        }


class IterativeQFunction:
    """
    Iteration-aware Q-function for LMPC.

    Tracks Q-function improvement across iterations:
        Q^{j+1}(x) ≤ Q^j(x)

    This property ensures cost improvement in LMPC.
    """

    def __init__(self, safe_set: SampledSafeSet):
        """
        Initialize iterative Q-function.

        Args:
            safe_set: Sampled safe set
        """
        self.safe_set = safe_set
        self._q_by_iteration: dict = {}

    def get_q_at_iteration(self, iteration: int) -> QFunctionApproximator:
        """Get Q-function approximator for specific iteration."""
        if iteration not in self._q_by_iteration:
            # Build Q-function from trajectories up to this iteration
            states_list = []
            q_list = []

            for traj in self.safe_set._trajectories:
                if traj.iteration <= iteration:
                    states_list.append(traj.states)
                    q_list.append(traj.cost_to_go)

            if len(states_list) > 0:
                states = np.vstack(states_list)
                q_values = np.concatenate(q_list)

                approximator = InverseDistanceQFunction()
                approximator.fit(states, q_values)
                self._q_by_iteration[iteration] = approximator
            else:
                self._q_by_iteration[iteration] = None

        return self._q_by_iteration[iteration]

    def evaluate(self, x: NDArray, iteration: Optional[int] = None) -> float:
        """
        Evaluate Q-function.

        Args:
            x: State
            iteration: Specific iteration (None = latest)
        """
        if iteration is None:
            iteration = self.safe_set.num_iterations - 1

        approximator = self.get_q_at_iteration(iteration)

        if approximator is None:
            return np.inf

        return approximator.predict(x)

    def get_improvement(self, x: NDArray) -> float:
        """
        Get Q-function improvement from first to last iteration.

        Returns Q^0(x) - Q^J(x)
        """
        n_iter = self.safe_set.num_iterations

        if n_iter < 2:
            return 0.0

        q_first = self.evaluate(x, iteration=0)
        q_last = self.evaluate(x, iteration=n_iter - 1)

        return q_first - q_last
