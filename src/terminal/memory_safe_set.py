"""
Memory-Optimized Safe Set for LMPC

Efficient storage and querying of safe set data:
1. Compressed storage using float32
2. Spatial indexing with k-d trees
3. Incremental updates
4. Trajectory pruning strategies

Memory targets:
- 10,000 states: < 10 MB
- 100,000 states: < 100 MB
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.spatial import KDTree

    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False


@dataclass
class MemoryOptimizedConfig:
    """Configuration for memory-optimized safe set."""

    # Storage
    use_float32: bool = True  # Use float32 instead of float64
    max_states: int = 50000  # Maximum states to store

    # Pruning
    pruning_strategy: str = "fifo"  # "fifo", "quality", "diversity"
    prune_threshold: float = 0.9  # Prune when at this fraction of max

    # Indexing
    rebuild_tree_threshold: int = 1000  # Rebuild KD-tree after this many additions

    # Compression
    compress_controls: bool = True  # Store controls at lower precision

    # Query optimization
    cache_queries: bool = True
    cache_size: int = 1000


class CompactTrajectory:
    """Memory-efficient trajectory storage."""

    __slots__ = ["controls", "iteration", "q_values", "states", "total_cost"]

    def __init__(
        self,
        states: NDArray,
        controls: NDArray,
        q_values: NDArray,
        iteration: int,
        total_cost: float,
        use_float32: bool = True,
    ):
        """
        Initialize compact trajectory.

        Args:
            states: State trajectory (T+1, n_x)
            controls: Control trajectory (T, n_u)
            q_values: Cost-to-go values (T+1,)
            iteration: LMPC iteration number
            total_cost: Total trajectory cost
            use_float32: Use float32 for storage
        """
        dtype = np.float32 if use_float32 else np.float64

        self.states = np.ascontiguousarray(states, dtype=dtype)
        self.controls = np.ascontiguousarray(controls, dtype=dtype)
        self.q_values = np.ascontiguousarray(q_values, dtype=dtype)
        self.iteration = iteration
        self.total_cost = float(total_cost)

    @property
    def length(self) -> int:
        """Number of timesteps."""
        return len(self.controls)

    @property
    def n_states(self) -> int:
        """Number of states in trajectory."""
        return len(self.states)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        return self.states.nbytes + self.controls.nbytes + self.q_values.nbytes + 16


class MemoryOptimizedSafeSet:
    """
    Memory-efficient safe set implementation.

    Optimizations:
    1. Float32 storage (50% memory reduction)
    2. Contiguous arrays for fast access
    3. K-d tree for O(log n) queries
    4. Automatic pruning when full
    5. Query caching

    Example:
        >>> ss = MemoryOptimizedSafeSet(n_x=14, n_u=3, config)
        >>> ss.add_trajectory(X, U, stage_costs)
        >>>
        >>> # Fast nearest-neighbor query
        >>> neighbors, q_values, dists = ss.query_knn(x, k=10)
    """

    def __init__(
        self,
        n_x: int,
        n_u: int,
        config: Optional[MemoryOptimizedConfig] = None,
    ):
        """
        Initialize memory-optimized safe set.

        Args:
            n_x: State dimension
            n_u: Control dimension
            config: Configuration
        """
        self.n_x = n_x
        self.n_u = n_u
        self.config = config or MemoryOptimizedConfig()

        self._dtype = np.float32 if self.config.use_float32 else np.float64

        # Trajectory storage
        self._trajectories: List[CompactTrajectory] = []

        # Flattened state array for fast queries
        self._all_states: Optional[NDArray] = None
        self._all_q_values: Optional[NDArray] = None
        self._state_to_traj: Optional[NDArray] = None  # Maps state index to trajectory

        # K-d tree for nearest neighbor queries
        self._kdtree: Optional[KDTree] = None
        self._tree_stale = True
        self._states_since_rebuild = 0

        # Query cache
        self._cache: Dict[int, Tuple] = {}

        # Statistics
        self._total_states = 0
        self._iteration = 0

    def add_trajectory(
        self,
        states: NDArray,
        controls: NDArray,
        stage_costs: Optional[NDArray] = None,
        iteration: Optional[int] = None,
    ) -> int:
        """
        Add trajectory to safe set.

        Args:
            states: State trajectory (T+1, n_x)
            controls: Control trajectory (T, n_u)
            stage_costs: Stage costs (T,)
            iteration: LMPC iteration

        Returns:
            Number of states added
        """
        T = len(controls)

        # Compute Q-values (cost-to-go)
        if stage_costs is None:
            stage_costs = np.ones(T)

        q_values = np.zeros(T + 1, dtype=self._dtype)
        q_values[T] = 0  # Terminal cost
        for k in range(T - 1, -1, -1):
            q_values[k] = stage_costs[k] + q_values[k + 1]

        total_cost = float(q_values[0])

        # Create compact trajectory
        traj = CompactTrajectory(
            states=states,
            controls=controls,
            q_values=q_values,
            iteration=iteration or self._iteration,
            total_cost=total_cost,
            use_float32=self.config.use_float32,
        )

        self._trajectories.append(traj)
        self._total_states += traj.n_states
        self._tree_stale = True
        self._states_since_rebuild += traj.n_states

        # Check if pruning needed
        if self._total_states > self.config.max_states * self.config.prune_threshold:
            self._prune()

        # Clear cache
        self._cache.clear()

        return traj.n_states

    def _prune(self) -> None:
        """Prune safe set based on configured strategy."""
        target_states = int(self.config.max_states * 0.8)  # Prune to 80%

        if self.config.pruning_strategy == "fifo":
            self._prune_fifo(target_states)
        elif self.config.pruning_strategy == "quality":
            self._prune_quality(target_states)
        elif self.config.pruning_strategy == "diversity":
            self._prune_diversity(target_states)

    def _prune_fifo(self, target_states: int) -> None:
        """Remove oldest trajectories."""
        while self._total_states > target_states and len(self._trajectories) > 1:
            removed = self._trajectories.pop(0)
            self._total_states -= removed.n_states

    def _prune_quality(self, target_states: int) -> None:
        """Remove trajectories with highest cost."""
        # Sort by total cost (ascending)
        self._trajectories.sort(key=lambda t: t.total_cost)

        # Remove worst trajectories
        while self._total_states > target_states and len(self._trajectories) > 1:
            removed = self._trajectories.pop()
            self._total_states -= removed.n_states

    def _prune_diversity(self, target_states: int) -> None:
        """Remove similar trajectories to maintain diversity."""
        # For now, use FIFO as fallback
        self._prune_fifo(target_states)

    def _rebuild_cache(self) -> None:
        """Rebuild flattened arrays and K-d tree."""
        if not self._tree_stale:
            return

        if len(self._trajectories) == 0:
            self._all_states = np.zeros((0, self.n_x), dtype=self._dtype)
            self._all_q_values = np.zeros(0, dtype=self._dtype)
            self._state_to_traj = np.zeros(0, dtype=np.int32)
            self._kdtree = None
            self._tree_stale = False
            return

        # Flatten all states
        all_states = []
        all_q = []
        state_to_traj = []

        for traj_idx, traj in enumerate(self._trajectories):
            all_states.append(traj.states)
            all_q.append(traj.q_values)
            state_to_traj.extend([traj_idx] * traj.n_states)

        self._all_states = np.vstack(all_states).astype(self._dtype)
        self._all_q_values = np.concatenate(all_q).astype(self._dtype)
        self._state_to_traj = np.array(state_to_traj, dtype=np.int32)

        # Build K-d tree
        if HAS_KDTREE and len(self._all_states) > 0:
            self._kdtree = KDTree(self._all_states)

        self._tree_stale = False
        self._states_since_rebuild = 0

    def query_knn(
        self,
        x: NDArray,
        k: int = 10,
        return_indices: bool = False,  # noqa: ARG002
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Query k nearest neighbors.

        Args:
            x: Query state
            k: Number of neighbors
            return_indices: Also return state indices

        Returns:
            neighbors: Nearest states (k, n_x)
            q_values: Q-values at neighbors (k,)
            distances: Distances to neighbors (k,)
        """
        self._rebuild_cache()

        if self._all_states is None or len(self._all_states) == 0:
            empty_states = np.zeros((0, self.n_x))
            empty_q = np.zeros(0)
            empty_d = np.zeros(0)
            return empty_states, empty_q, empty_d

        k = min(k, len(self._all_states))

        # Check cache
        cache_key = hash(x.tobytes()) if self.config.cache_queries else None
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Query K-d tree
        if self._kdtree is not None:
            distances, indices = self._kdtree.query(x.astype(np.float64), k=k)

            # Handle single result
            if k == 1:
                indices = np.array([indices])
                distances = np.array([distances])
        else:
            # Brute force fallback
            dists = np.linalg.norm(self._all_states - x, axis=1)
            indices = np.argsort(dists)[:k]
            distances = dists[indices]

        neighbors = self._all_states[indices]
        q_values = self._all_q_values[indices]

        result = (neighbors, q_values, distances)

        # Cache result
        if cache_key is not None and len(self._cache) < self.config.cache_size:
            self._cache[cache_key] = result

        return result

    def query_radius(
        self,
        x: NDArray,
        radius: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Query all states within radius.

        Args:
            x: Query state
            radius: Search radius

        Returns:
            neighbors: States within radius
            q_values: Q-values
            distances: Distances
        """
        self._rebuild_cache()

        if self._all_states is None or len(self._all_states) == 0:
            return np.zeros((0, self.n_x)), np.zeros(0), np.zeros(0)

        if self._kdtree is not None:
            indices = self._kdtree.query_ball_point(x.astype(np.float64), radius)
        else:
            dists = np.linalg.norm(self._all_states - x, axis=1)
            indices = np.where(dists <= radius)[0]

        if len(indices) == 0:
            return np.zeros((0, self.n_x)), np.zeros(0), np.zeros(0)

        neighbors = self._all_states[indices]
        q_values = self._all_q_values[indices]
        distances = np.linalg.norm(neighbors - x, axis=1)

        return neighbors, q_values, distances

    def get_convex_hull_vertices(
        self,
        x: NDArray,
        k: int = 20,
    ) -> Tuple[NDArray, NDArray]:
        """
        Get vertices for convex hull terminal constraint.

        Args:
            x: Current state (for local selection)
            k: Number of vertices

        Returns:
            vertices: Vertex states (k, n_x)
            q_values: Q-values at vertices (k,)
        """
        neighbors, q_values, _ = self.query_knn(x, k=k)
        return neighbors, q_values

    def interpolate_q(
        self,
        x: NDArray,
        k: int = 5,
        method: str = "inverse_distance",
    ) -> float:
        """
        Interpolate Q-value at state x.

        Args:
            x: Query state
            k: Number of neighbors for interpolation
            method: Interpolation method

        Returns:
            Interpolated Q-value
        """
        neighbors, q_values, distances = self.query_knn(x, k=k)

        if len(q_values) == 0:
            return np.inf

        if method == "inverse_distance":
            weights = 1.0 / (distances + 1e-10)
            return float(np.sum(weights * q_values) / np.sum(weights))
        elif method == "nearest":
            return float(q_values[0])
        else:
            return float(np.mean(q_values))

    @property
    def num_trajectories(self) -> int:
        """Number of stored trajectories."""
        return len(self._trajectories)

    @property
    def num_states(self) -> int:
        """Total number of states."""
        return self._total_states

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        traj_mem = sum(t.memory_bytes for t in self._trajectories)

        cache_mem = 0
        if self._all_states is not None:
            cache_mem += self._all_states.nbytes
            cache_mem += self._all_q_values.nbytes
            cache_mem += self._state_to_traj.nbytes

        return traj_mem + cache_mem

    @property
    def memory_mb(self) -> float:
        """Memory usage in megabytes."""
        return self.memory_bytes / 1e6

    def get_statistics(self) -> Dict:
        """Get safe set statistics."""
        return {
            "num_trajectories": self.num_trajectories,
            "num_states": self.num_states,
            "memory_mb": self.memory_mb,
            "tree_stale": self._tree_stale,
            "cache_size": len(self._cache),
        }

    def save(self, filepath: str) -> None:
        """Save safe set to file."""
        import pickle  # noqa: PLC0415

        data = {
            "n_x": self.n_x,
            "n_u": self.n_u,
            "trajectories": [
                {
                    "states": t.states,
                    "controls": t.controls,
                    "q_values": t.q_values,
                    "iteration": t.iteration,
                    "total_cost": t.total_cost,
                }
                for t in self._trajectories
            ],
            "iteration": self._iteration,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load safe set from file."""
        import pickle  # noqa: PLC0415

        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.n_x = data["n_x"]
        self.n_u = data["n_u"]
        self._iteration = data["iteration"]

        self._trajectories.clear()
        self._total_states = 0

        for traj_data in data["trajectories"]:
            traj = CompactTrajectory(
                states=traj_data["states"],
                controls=traj_data["controls"],
                q_values=traj_data["q_values"],
                iteration=traj_data["iteration"],
                total_cost=traj_data["total_cost"],
                use_float32=self.config.use_float32,
            )
            self._trajectories.append(traj)
            self._total_states += traj.n_states

        self._tree_stale = True
        self._cache.clear()


class StreamingSafeSet(MemoryOptimizedSafeSet):
    """
    Safe set with streaming updates for online learning.

    Optimized for frequent small updates during operation.
    """

    def __init__(
        self,
        n_x: int,
        n_u: int,
        config: Optional[MemoryOptimizedConfig] = None,
    ):
        """Initialize streaming safe set."""
        super().__init__(n_x, n_u, config)

        # Buffer for streaming updates
        self._buffer_states: List[NDArray] = []
        self._buffer_q: List[float] = []
        self._buffer_size = 100

    def add_state(
        self,
        state: NDArray,
        q_value: float,
    ) -> None:
        """Add single state to buffer."""
        self._buffer_states.append(state.astype(self._dtype))
        self._buffer_q.append(q_value)

        if len(self._buffer_states) >= self._buffer_size:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffer to main storage."""
        if len(self._buffer_states) == 0:
            return

        # Create pseudo-trajectory from buffer
        states = np.array(self._buffer_states)
        controls = np.zeros((len(states) - 1, self.n_u), dtype=self._dtype)
        q_values = np.array(self._buffer_q, dtype=self._dtype)

        traj = CompactTrajectory(
            states=states,
            controls=controls,
            q_values=q_values,
            iteration=self._iteration,
            total_cost=float(q_values[0]) if len(q_values) > 0 else 0.0,
            use_float32=self.config.use_float32,
        )

        self._trajectories.append(traj)
        self._total_states += len(states)
        self._tree_stale = True

        # Clear buffer
        self._buffer_states.clear()
        self._buffer_q.clear()
        self._cache.clear()
