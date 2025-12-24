"""
K-Nearest Neighbor Local Safe Set for LMPC

The local safe set SS_local(x) is a subset of the full safe set
containing states "near" the query state x. This is used to:
1. Form convex hull terminal constraints
2. Interpolate Q-function values

Using K-nearest neighbors provides:
- Computational efficiency (don't use full safe set)
- Local approximation quality
- Adaptive density based on data

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control
    for Iterative Tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.spatial import KDTree

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .safe_set import FuelAwareSafeSet, SampledSafeSet


@dataclass
class LocalSafeSetConfig:
    """Configuration for local safe set computation."""

    # Number of neighbors
    K: int = 10  # Base number of neighbors
    K_min: int = 4  # Minimum neighbors for convex hull
    K_max: int = 50  # Maximum neighbors to consider

    # Distance weighting
    distance_metric: str = "weighted_euclidean"  # "euclidean", "weighted_euclidean", "mahalanobis"

    # State weights for distance computation
    # Higher weight = more importance in neighbor selection
    position_weight: float = 1.0
    velocity_weight: float = 0.5
    attitude_weight: float = 0.3
    angular_rate_weight: float = 0.2
    fuel_weight: float = 0.1

    # Adaptive K based on density
    adaptive_K: bool = True  # noqa: N815
    density_radius: float = 1.0  # Radius for density estimation

    # Filtering options
    filter_by_fuel: bool = True
    fuel_margin: float = 0.1


class LocalSafeSet:
    """
    K-Nearest Neighbor Local Safe Set.

    Efficiently finds nearby safe states for terminal constraint
    and Q-function interpolation in LMPC.

    Features:
    - KD-tree for O(log N) nearest neighbor queries
    - Weighted distance metric for state importance
    - Fuel-aware filtering
    - Adaptive K based on local density

    Example:
        >>> local_ss = LocalSafeSet(safe_set, config)
        >>>
        >>> # Find local safe set around query state
        >>> neighbors, q_values, distances = local_ss.query(x_query)
        >>>
        >>> # Get interpolated Q-value
        >>> q_interp = local_ss.interpolate_q(x_query)
    """

    def __init__(
        self,
        safe_set: SampledSafeSet,
        config: Optional[LocalSafeSetConfig] = None,
    ):
        """
        Initialize local safe set.

        Args:
            safe_set: Global sampled safe set
            config: Configuration parameters
        """
        self.safe_set = safe_set
        self.config = config or LocalSafeSetConfig()

        # KD-tree for efficient neighbor search
        self._kdtree: Optional[KDTree] = None
        self._weighted_states: Optional[NDArray] = None

        # State weighting
        self._weights = self._build_weight_vector()

        self._tree_valid = False

    def _build_weight_vector(self) -> NDArray:
        """Build weight vector for distance computation."""
        n_x = self.safe_set.n_x
        weights = np.ones(n_x)

        if n_x == 14:  # 6-DoF rocket
            # [m, r(3), v(3), q(4), ω(3)]
            weights[0] = self.config.fuel_weight  # Mass
            weights[1:4] = self.config.position_weight  # Position
            weights[4:7] = self.config.velocity_weight  # Velocity
            weights[7:11] = self.config.attitude_weight  # Quaternion
            weights[11:14] = self.config.angular_rate_weight  # Angular rate
        elif n_x == 7:  # 3-DoF rocket
            # [m, r(3), v(3)]
            weights[0] = self.config.fuel_weight
            weights[1:4] = self.config.position_weight
            weights[4:7] = self.config.velocity_weight

        return np.sqrt(weights)  # Square root for Euclidean distance

    def _rebuild_tree(self) -> None:
        """Rebuild KD-tree from safe set."""
        if self._tree_valid:
            return

        states = self.safe_set.get_all_states()

        if len(states) == 0:
            self._kdtree = None
            self._weighted_states = None
            self._tree_valid = True
            return

        # Apply weighting
        self._weighted_states = states * self._weights[np.newaxis, :]

        if HAS_SCIPY:
            self._kdtree = KDTree(self._weighted_states)

        self._tree_valid = True

    def invalidate(self) -> None:
        """Invalidate tree (call when safe set changes)."""
        self._tree_valid = False

    def query(
        self,
        x: NDArray,
        K: Optional[int] = None,
        available_fuel: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Find K nearest neighbors in safe set.

        Args:
            x: Query state (n_x,)
            K: Number of neighbors (uses config.K if None)
            available_fuel: Filter by fuel availability

        Returns:
            neighbors: Neighboring states (K, n_x)
            q_values: Q-values of neighbors (K,)
            distances: Distances to neighbors (K,)
        """
        self._rebuild_tree()

        if self._kdtree is None or self.safe_set.num_states == 0:
            return np.zeros((0, self.safe_set.n_x)), np.zeros(0), np.zeros(0)

        K = K or self.config.K

        # Get fuel-filtered states if needed
        if self.config.filter_by_fuel and available_fuel is not None:
            if isinstance(self.safe_set, FuelAwareSafeSet):
                states, q_values, indices = self.safe_set.get_feasible_states(available_fuel)
                if len(states) == 0:
                    return np.zeros((0, self.safe_set.n_x)), np.zeros(0), np.zeros(0)

                # Build temporary tree for filtered states
                weighted_filtered = states * self._weights[np.newaxis, :]
                tree = KDTree(weighted_filtered)
            else:
                tree = self._kdtree
                states = self.safe_set.get_all_states()
                q_values = self.safe_set.get_all_q_values()
        else:
            tree = self._kdtree
            states = self.safe_set.get_all_states()
            q_values = self.safe_set.get_all_q_values()

        # Adaptive K based on local density
        if self.config.adaptive_K:
            K = self._compute_adaptive_K(x, tree)

        # Clamp K to available points
        K = min(K, len(states))
        K = max(K, self.config.K_min)
        K = min(K, self.config.K_max)

        if K == 0:
            return np.zeros((0, self.safe_set.n_x)), np.zeros(0), np.zeros(0)

        # Query KD-tree
        x_weighted = x * self._weights
        distances, indices = tree.query(x_weighted, k=K)

        # Handle single neighbor case
        if K == 1:
            distances = np.array([distances])
            indices = np.array([indices])

        return states[indices], q_values[indices], distances

    def _compute_adaptive_K(
        self,
        x: NDArray,
        tree: KDTree,
    ) -> int:
        """
        Compute adaptive K based on local density.

        More neighbors in dense regions, fewer in sparse regions.
        """
        x_weighted = x * self._weights

        # Count points within density radius
        n_points = len(tree.data)
        if n_points == 0:
            return self.config.K_min

        # Query points within radius
        indices = tree.query_ball_point(x_weighted, self.config.density_radius)
        local_density = len(indices) / n_points

        # Scale K with density
        K = int(self.config.K * (1 + local_density * 2))
        return np.clip(K, self.config.K_min, self.config.K_max)

    def interpolate_q(
        self,
        x: NDArray,
        K: Optional[int] = None,
        method: str = "inverse_distance",
    ) -> float:
        """
        Interpolate Q-function at query point.

        Args:
            x: Query state
            K: Number of neighbors for interpolation
            method: Interpolation method
                - "inverse_distance": Inverse distance weighting
                - "nearest": Nearest neighbor value
                - "linear": Linear interpolation (barycentric)

        Returns:
            Interpolated Q-value
        """
        neighbors, q_values, distances = self.query(x, K)

        if len(neighbors) == 0:
            return np.inf  # No safe states available

        if method == "nearest":
            return q_values[0]

        elif method == "inverse_distance":
            # Handle zero distance (exact match)
            if distances[0] < 1e-10:
                return q_values[0]

            # Inverse distance weighting
            weights = 1.0 / (distances + 1e-10)
            weights = weights / np.sum(weights)
            return float(np.dot(weights, q_values))

        elif method == "linear":
            # Barycentric interpolation using convex combination
            # This is approximate - true barycentric requires convex hull
            if distances[0] < 1e-10:
                return q_values[0]

            weights = 1.0 / (distances + 1e-10)
            weights = weights / np.sum(weights)
            return float(np.dot(weights, q_values))

        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def get_convex_hull_data(
        self,
        x: NDArray,
        K: Optional[int] = None,
        available_fuel: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Get data for convex hull terminal constraint.

        Returns vertices that can be used to form:
            x_N ∈ Conv({v_1, v_2, ..., v_K})

        Args:
            x: Query state (for neighbor selection)
            K: Number of vertices
            available_fuel: Filter by fuel availability

        Returns:
            vertices: Convex hull vertices (K, n_x)
            q_values: Q-values at vertices (K,)
        """
        neighbors, q_values, _ = self.query(x, K, available_fuel)
        return neighbors, q_values


class MultiResolutionLocalSafeSet:
    """
    Multi-resolution local safe set for hierarchical queries.

    Uses multiple scales for adaptive precision:
    - Coarse level for global structure
    - Fine level for local accuracy

    Useful when the safe set spans large state space regions.
    """

    def __init__(
        self,
        safe_set: SampledSafeSet,
        n_levels: int = 3,
        K_per_level: Optional[List[int]] = None,
    ):
        """
        Initialize multi-resolution local safe set.

        Args:
            safe_set: Global safe set
            n_levels: Number of resolution levels
            K_per_level: K values for each level
        """
        self.safe_set = safe_set
        self.n_levels = n_levels
        self.K_per_level = K_per_level or [5 * (2**i) for i in range(n_levels)]

        # Create configs for each level
        self._local_sets: List[LocalSafeSet] = []
        for i, K in enumerate(self.K_per_level):
            config = LocalSafeSetConfig(K=K, adaptive_K=False)
            self._local_sets.append(LocalSafeSet(safe_set, config))

    def query_hierarchical(
        self,
        x: NDArray,
    ) -> List[Tuple[NDArray, NDArray, NDArray]]:
        """
        Query at all resolution levels.

        Returns list of (neighbors, q_values, distances) for each level.
        """
        results = []
        for local_ss in self._local_sets:
            results.append(local_ss.query(x))
        return results

    def interpolate_q_hierarchical(
        self,
        x: NDArray,
        weights: Optional[NDArray] = None,
    ) -> float:
        """
        Interpolate Q-value using multi-resolution approach.

        Combines estimates from all levels with given weights.
        """
        if weights is None:
            # Default: more weight to fine levels
            weights = np.array([2**i for i in range(self.n_levels)])
            weights = weights / np.sum(weights)

        q_estimates = []
        for local_ss in self._local_sets:
            q = local_ss.interpolate_q(x)
            if np.isfinite(q):
                q_estimates.append(q)
            else:
                q_estimates.append(np.inf)

        # Weighted combination (ignoring inf)
        q_estimates = np.array(q_estimates)
        valid_mask = np.isfinite(q_estimates)

        if not np.any(valid_mask):
            return np.inf

        valid_weights = weights[valid_mask]
        valid_weights = valid_weights / np.sum(valid_weights)

        return float(np.dot(valid_weights, q_estimates[valid_mask]))

    def invalidate(self) -> None:
        """Invalidate all levels."""
        for local_ss in self._local_sets:
            local_ss.invalidate()
