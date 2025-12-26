"""
Novelty-Based Data Selection for Online Learning

Selects informative data points for GP training by prioritizing:
1. Novel states (far from existing data)
2. High-uncertainty regions
3. High-residual observations
4. Diverse coverage of state space

This prevents overfitting to redundant data and improves
sample efficiency of online learning.

Reference:
    Berkenkamp, F., et al. (2017). Safe Model-based Reinforcement Learning
    with Stability Guarantees. NeurIPS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class NoveltyConfig:
    """Configuration for novelty-based selection."""

    # Selection method
    method: str = "combined"  # "distance", "uncertainty", "residual", "combined"

    # Distance-based novelty
    distance_threshold: float = 0.1  # Minimum distance to existing data
    distance_weight: float = 1.0

    # Uncertainty-based selection
    uncertainty_threshold: float = 0.5  # Min GP variance for selection
    uncertainty_weight: float = 1.0

    # Residual-based selection
    residual_threshold: float = 0.1  # Min residual for selection
    residual_weight: float = 0.5

    # Combined scoring
    score_threshold: float = 0.3  # Minimum novelty score

    # Data limits
    max_points: int = 5000  # Maximum points to keep


class NoveltySelector:
    """
    Selects novel and informative data points.

    Uses multiple criteria to identify data points that
    will most improve GP model quality.

    Example:
        >>> selector = NoveltySelector(config)
        >>>
        >>> # Add existing data
        >>> selector.set_reference_data(X_existing)
        >>>
        >>> # Select from new candidates
        >>> indices = selector.select(X_new, n_select=100)
        >>> X_selected = X_new[indices]
    """

    def __init__(self, config: Optional[NoveltyConfig] = None):
        """
        Initialize novelty selector.

        Args:
            config: Configuration parameters
        """
        self.config = config or NoveltyConfig()

        # Reference data for distance computation
        self._reference_data: Optional[NDArray] = None
        self._kdtree: Optional[KDTree] = None

        # GP model for uncertainty (optional)
        self._gp_model = None

    def set_reference_data(self, X: NDArray) -> None:
        """
        Set reference data for novelty computation.

        Args:
            X: Existing data points (N, n_features)
        """
        self._reference_data = X.copy()

        if HAS_SCIPY and len(X) > 0:
            self._kdtree = KDTree(X)

    def set_gp_model(self, gp_model) -> None:
        """Set GP model for uncertainty-based selection."""
        self._gp_model = gp_model

    def compute_novelty_scores(
        self,
        X: NDArray,
        Y: Optional[NDArray] = None,
    ) -> NDArray:
        """
        Compute novelty scores for candidate points.

        Args:
            X: Candidate points (N, n_features)
            Y: Associated outputs (for residual scoring)

        Returns:
            scores: Novelty scores (N,)
        """
        n_points = len(X)

        if n_points == 0:
            return np.array([])

        # Initialize scores
        scores = np.zeros(n_points)

        # Distance-based novelty
        if self.config.method in ["distance", "combined"]:
            dist_scores = self._compute_distance_novelty(X)
            scores += self.config.distance_weight * dist_scores

        # Uncertainty-based novelty
        if self.config.method in ["uncertainty", "combined"] and self._gp_model is not None:
            unc_scores = self._compute_uncertainty_novelty(X)
            scores += self.config.uncertainty_weight * unc_scores

        # Residual-based novelty
        if self.config.method in ["residual", "combined"] and Y is not None:
            res_scores = self._compute_residual_novelty(Y)
            scores += self.config.residual_weight * res_scores

        # Normalize
        if np.max(scores) > 0:
            scores = scores / np.max(scores)

        return scores

    def _compute_distance_novelty(self, X: NDArray) -> NDArray:
        """Compute distance-based novelty scores."""
        if self._reference_data is None or len(self._reference_data) == 0:
            return np.ones(len(X))

        if self._kdtree is not None:
            # Fast KD-tree query
            distances, _ = self._kdtree.query(X, k=1)
        else:
            # Brute force
            distances = np.min(np.linalg.norm(X[:, np.newaxis] - self._reference_data, axis=2), axis=1)

        # Convert to novelty score (higher distance = higher novelty)
        # Use sigmoid-like function
        scores = 1.0 - np.exp(-distances / self.config.distance_threshold)

        return scores

    def _compute_uncertainty_novelty(self, X: NDArray) -> NDArray:
        """Compute uncertainty-based novelty scores."""
        if self._gp_model is None:
            return np.ones(len(X))

        try:
            _, var = self._gp_model.predict(X)

            # Sum variance across output dimensions
            if var.ndim > 1:
                var = np.sum(var, axis=1)

            # Normalize by threshold
            scores = var / self.config.uncertainty_threshold
            scores = np.minimum(scores, 1.0)

        except Exception:
            scores = np.ones(len(X))

        return scores

    def _compute_residual_novelty(self, Y: NDArray) -> NDArray:
        """Compute residual-based novelty scores."""
        # Larger residuals indicate model mismatch -> more informative
        residual_mag = np.linalg.norm(Y, axis=1) if Y.ndim > 1 else np.abs(Y)

        # Normalize
        scores = residual_mag / self.config.residual_threshold
        scores = np.minimum(scores, 1.0)

        return scores

    def select(
        self,
        X: NDArray,
        Y: Optional[NDArray] = None,
        n_select: Optional[int] = None,
        return_scores: bool = False,
    ) -> NDArray:
        """
        Select novel data points.

        Args:
            X: Candidate points (N, n_features)
            Y: Associated outputs
            n_select: Number to select (None = all above threshold)
            return_scores: Also return novelty scores

        Returns:
            indices: Selected indices
            scores: (if return_scores) Novelty scores
        """
        scores = self.compute_novelty_scores(X, Y)

        if n_select is not None:
            # Select top n_select by score
            indices = np.arange(len(scores)) if n_select >= len(scores) else np.argsort(scores)[-n_select:]
        else:
            # Select all above threshold
            indices = np.where(scores >= self.config.score_threshold)[0]

        if return_scores:
            return indices, scores
        return indices

    def select_diverse(
        self,
        X: NDArray,
        Y: Optional[NDArray] = None,
        n_select: int = 100,
    ) -> NDArray:
        """
        Select diverse set of novel points.

        Uses greedy farthest point sampling to ensure diversity.

        Args:
            X: Candidate points
            Y: Associated outputs
            n_select: Number to select

        Returns:
            indices: Selected indices
        """
        n_points = len(X)

        if n_select >= n_points:
            return np.arange(n_points)

        # Compute base novelty scores
        scores = self.compute_novelty_scores(X, Y)

        # Greedy selection with diversity
        selected = []
        remaining = list(range(n_points))

        # Start with highest novelty point
        first_idx = remaining[np.argmax(scores[remaining])]
        selected.append(first_idx)
        remaining.remove(first_idx)

        while len(selected) < n_select and len(remaining) > 0:
            # Compute distance to selected set
            X_selected = X[selected]

            best_idx = None
            best_score = -np.inf

            for idx in remaining:
                # Distance to nearest selected point
                dists = np.linalg.norm(X_selected - X[idx], axis=1)
                min_dist = np.min(dists)

                # Combined score: novelty + diversity
                combined = scores[idx] + 0.5 * min_dist

                if combined > best_score:
                    best_score = combined
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return np.array(selected)


class ActiveDataSelector:
    """
    Active learning-based data selector.

    Selects points that maximize expected information gain
    for the GP model.
    """

    def __init__(
        self,
        gp_model,
        acquisition: str = "uncertainty",
    ):
        """
        Initialize active selector.

        Args:
            gp_model: GP model for acquisition computation
            acquisition: Acquisition function type
        """
        self.gp_model = gp_model
        self.acquisition = acquisition

    def compute_acquisition(self, X: NDArray) -> NDArray:
        """
        Compute acquisition function values.

        Args:
            X: Candidate points

        Returns:
            values: Acquisition values (higher = more informative)
        """
        if self.acquisition == "uncertainty":
            # Pure uncertainty sampling
            _, var = self.gp_model.predict(X)
            if var.ndim > 1:
                return np.sum(var, axis=1)
            return var

        elif self.acquisition == "expected_improvement":
            # Expected improvement (for optimization)
            mean, var = self.gp_model.predict(X)

            # Best observed value
            y_best = np.min(self.gp_model._Y) if hasattr(self.gp_model, "_Y") else 0

            # EI computation
            std = np.sqrt(var)
            z = (y_best - mean) / (std + 1e-10)

            from scipy.stats import norm  # noqa: PLC0415

            ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)

            if ei.ndim > 1:
                return np.sum(ei, axis=1)
            return ei

        else:
            # Default to uncertainty
            _, var = self.gp_model.predict(X)
            if var.ndim > 1:
                return np.sum(var, axis=1)
            return var

    def select(self, X: NDArray, n_select: int) -> NDArray:
        """Select points with highest acquisition values."""
        values = self.compute_acquisition(X)

        if n_select >= len(values):
            return np.arange(len(values))

        return np.argsort(values)[-n_select:]


class DataBuffer:
    """
    Fixed-size data buffer with novelty-based replacement.

    When buffer is full, replaces least novel points with
    new data.
    """

    def __init__(
        self,
        max_size: int = 5000,
        novelty_selector: Optional[NoveltySelector] = None,
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum buffer size
            novelty_selector: Selector for novelty computation
        """
        self.max_size = max_size
        self.selector = novelty_selector or NoveltySelector()

        self._X: Optional[NDArray] = None
        self._Y: Optional[NDArray] = None

    def add(self, X_new: NDArray, Y_new: NDArray) -> int:
        """
        Add data to buffer.

        Args:
            X_new: New inputs
            Y_new: New outputs

        Returns:
            Number of points added
        """
        if self._X is None:
            self._X = X_new.copy()
            self._Y = Y_new.copy()
            return len(X_new)

        # Combine with existing
        X_combined = np.vstack([self._X, X_new])
        Y_combined = np.vstack([self._Y, Y_new])

        if len(X_combined) <= self.max_size:
            self._X = X_combined
            self._Y = Y_combined
            return len(X_new)

        # Need to prune - keep most novel
        self.selector.set_reference_data(X_combined)
        indices = self.selector.select_diverse(X_combined, Y_combined, n_select=self.max_size)

        self._X = X_combined[indices]
        self._Y = Y_combined[indices]

        # Count how many new points were kept
        n_new_kept = np.sum(indices >= len(self._X) - len(X_new))

        return n_new_kept

    def get_data(self) -> Tuple[NDArray, NDArray]:
        """Get all data in buffer."""
        if self._X is None:
            return np.array([]), np.array([])
        return self._X.copy(), self._Y.copy()

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._X) if self._X is not None else 0
