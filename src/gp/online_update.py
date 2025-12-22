"""
Online GP Update Module

Handles real-time GP learning during flight:
- Data collection and buffering
- Novelty-based data selection (avoid redundant data)
- Efficient online GP updates
- Hyperparameter adaptation

The key challenge is maintaining GP accuracy while meeting
real-time constraints (<1ms for updates during 50Hz control).

Strategies:
1. Rank-1 updates for exact GP (limited scalability)
2. Inducing point updates for sparse GP
3. Sliding window with periodic refitting
4. Novelty filtering to limit data growth

Reference:
    Csató, L., & Opper, M. (2002). Sparse on-line Gaussian processes.
    Neural Computation, 14(3), 641-668.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .sparse_gp import SparseGP
from .structured_gp import StructuredRocketGP


@dataclass
class OnlineUpdateConfig:
    """Configuration for online GP updates."""

    # Data buffer settings
    buffer_size: int = 1000  # Maximum data points to store
    min_data_for_fit: int = 20  # Minimum data before fitting

    # Novelty filtering
    use_novelty_filter: bool = True
    novelty_threshold: float = 0.3  # Variance ratio threshold
    min_distance: float = 0.01  # Minimum distance to existing data

    # Update frequency
    update_interval: int = 10  # Fit every N new points
    refit_interval: int = 100  # Full refit every N points

    # Timing constraints
    max_update_time_ms: float = 5.0  # Max time for online update

    # Inducing point management
    add_inducing_on_novel: bool = True
    max_inducing_points: int = 100
    inducing_update_threshold: float = 0.5


@dataclass
class DataPoint:
    """Single data point for GP training."""

    x: NDArray  # State
    u: NDArray  # Control
    d: NDArray  # Residual (target)
    timestamp: float  # Collection time
    novelty: float = 0.0  # Novelty score when collected


class DataBuffer:
    """
    Circular buffer for training data with novelty filtering.

    Maintains a fixed-size buffer of training data, prioritizing
    novel (high-uncertainty) points over redundant ones.
    """

    def __init__(
        self,
        max_size: int = 1000,
        feature_dim: int = 10,
        target_dim: int = 3,
    ):
        """
        Initialize data buffer.

        Args:
            max_size: Maximum number of points to store
            feature_dim: Dimension of features
            target_dim: Dimension of targets
        """
        self.max_size = max_size
        self.feature_dim = feature_dim
        self.target_dim = target_dim

        # Storage
        self._features: Deque[NDArray] = deque(maxlen=max_size)
        self._targets: Deque[NDArray] = deque(maxlen=max_size)
        self._novelties: Deque[float] = deque(maxlen=max_size)
        self._timestamps: Deque[float] = deque(maxlen=max_size)

        # Statistics
        self._total_added: int = 0
        self._total_rejected: int = 0

    @property
    def size(self) -> int:
        """Current number of points in buffer."""
        return len(self._features)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    @property
    def is_full(self) -> bool:
        return self.size >= self.max_size

    def add(
        self,
        features: NDArray,
        targets: NDArray,
        novelty: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> bool:
        """
        Add data point to buffer.

        Args:
            features: Feature vector
            targets: Target vector
            novelty: Novelty score (higher = more novel)
            timestamp: Collection time

        Returns:
            True if point was added
        """
        if timestamp is None:
            timestamp = time.time()

        self._features.append(features.copy())
        self._targets.append(targets.copy())
        self._novelties.append(novelty)
        self._timestamps.append(timestamp)

        self._total_added += 1
        return True

    def add_if_novel(
        self,
        features: NDArray,
        targets: NDArray,
        novelty: float,
        threshold: float = 0.3,
        min_distance: Optional[float] = None,
    ) -> bool:
        """
        Add point only if it's novel enough.

        Args:
            features: Feature vector
            targets: Target vector
            novelty: Novelty score
            threshold: Minimum novelty to add
            min_distance: Minimum distance to existing points

        Returns:
            True if point was added
        """
        # Check novelty threshold
        if novelty < threshold:
            self._total_rejected += 1
            return False

        # Check minimum distance to existing data
        if min_distance is not None and self.size > 0:
            Z = self.get_features()
            distances = np.linalg.norm(Z - features, axis=1)
            if np.min(distances) < min_distance:
                self._total_rejected += 1
                return False

        return self.add(features, targets, novelty)

    def get_features(self) -> NDArray:
        """Get all features as array (N, feature_dim)."""
        if self.is_empty:
            return np.empty((0, self.feature_dim))
        return np.array(list(self._features))

    def get_targets(self) -> NDArray:
        """Get all targets as array (N, target_dim)."""
        if self.is_empty:
            return np.empty((0, self.target_dim))
        return np.array(list(self._targets))

    def get_data(self) -> Tuple[NDArray, NDArray]:
        """Get all data as (features, targets)."""
        return self.get_features(), self.get_targets()

    def get_recent(self, n: int) -> Tuple[NDArray, NDArray]:
        """Get n most recent points."""
        n = min(n, self.size)
        features = list(self._features)[-n:]
        targets = list(self._targets)[-n:]
        return np.array(features), np.array(targets)

    def clear(self) -> None:
        """Clear all data."""
        self._features.clear()
        self._targets.clear()
        self._novelties.clear()
        self._timestamps.clear()

    def get_statistics(self) -> dict:
        """Get buffer statistics."""
        return {
            "size": self.size,
            "max_size": self.max_size,
            "total_added": self._total_added,
            "total_rejected": self._total_rejected,
            "acceptance_rate": self._total_added / max(1, self._total_added + self._total_rejected),
            "mean_novelty": np.mean(list(self._novelties)) if self.size > 0 else 0.0,
        }


class OnlineGPUpdater:
    """
    Online updater for sparse GP.

    Handles efficient online updates while maintaining real-time performance.

    Example:
        >>> updater = OnlineGPUpdater(gp, config)
        >>>
        >>> # During control loop
        >>> for x, u, d in data_stream:
        >>>     updater.add_observation(x, u, d)
        >>>     if updater.should_update():
        >>>         updater.update()
    """

    def __init__(
        self,
        gp: SparseGP,
        config: Optional[OnlineUpdateConfig] = None,
        feature_extractor: Optional[Callable[[NDArray, NDArray], NDArray]] = None,
    ):
        """
        Initialize online updater.

        Args:
            gp: Sparse GP to update
            config: Update configuration
            feature_extractor: Function to extract features from (x, u)
        """
        self.gp = gp
        self.config = config or OnlineUpdateConfig()
        self.feature_extractor = feature_extractor or (lambda x, u: np.concatenate([x, u]))

        # Determine dimensions from GP or first data point
        self._feature_dim: Optional[int] = None
        self._target_dim: int = 1

        # Data buffer (initialized lazily)
        self._buffer: Optional[DataBuffer] = None

        # Update tracking
        self._points_since_update: int = 0
        self._points_since_refit: int = 0
        self._total_updates: int = 0
        self._last_update_time: float = 0.0

        # Performance tracking
        self._update_times: List[float] = []

    def _ensure_buffer(self, feature_dim: int, target_dim: int = 1) -> None:
        """Initialize buffer if needed."""
        if self._buffer is None:
            self._feature_dim = feature_dim
            self._target_dim = target_dim
            self._buffer = DataBuffer(
                max_size=self.config.buffer_size,
                feature_dim=feature_dim,
                target_dim=target_dim,
            )

    def add_observation(
        self,
        x: NDArray,
        u: NDArray,
        d: NDArray,
    ) -> bool:
        """
        Add new observation.

        Args:
            x: State
            u: Control
            d: Observed residual

        Returns:
            True if observation was added to buffer
        """
        # Extract features
        z = self.feature_extractor(x, u)
        d = np.atleast_1d(d)

        # Initialize buffer
        self._ensure_buffer(len(z), len(d))

        # Compute novelty if GP is fitted
        if self.config.use_novelty_filter and self.gp._L_B is not None:
            try:
                pred = self.gp.predict(z.reshape(1, -1))
                # Novelty = variance / prior_variance
                prior_var = self.gp.kernel.signal_variance
                novelty = float(np.mean(pred.variance) / prior_var)
            except Exception:
                novelty = 1.0  # Assume novel if prediction fails
        else:
            novelty = 1.0  # No filtering before first fit

        # Add to buffer
        if self.config.use_novelty_filter:
            added = self._buffer.add_if_novel(
                z,
                d,
                novelty,
                threshold=self.config.novelty_threshold,
                min_distance=self.config.min_distance,
            )
        else:
            added = self._buffer.add(z, d, novelty)

        if added:
            self._points_since_update += 1
            self._points_since_refit += 1

        return added

    def should_update(self) -> bool:
        """Check if GP should be updated."""
        if self._buffer is None or self._buffer.size < self.config.min_data_for_fit:
            return False

        return self._points_since_update >= self.config.update_interval

    def should_refit(self) -> bool:
        """Check if GP needs full refit."""
        if self._buffer is None:
            return False

        return self._points_since_refit >= self.config.refit_interval

    def update(self, force: bool = False) -> dict:
        """
        Perform online GP update.

        Args:
            force: Force update even if not scheduled

        Returns:
            Update statistics
        """
        if self._buffer is None or self._buffer.is_empty:
            return {"status": "no_data"}

        if not force and not self.should_update():
            return {"status": "skipped"}

        start_time = time.perf_counter()

        # Get data
        Z, D = self._buffer.get_data()

        # Refit GP
        try:
            if D.ndim == 1:
                self.gp.fit(Z, D)
            else:
                # For multi-output, fit each output
                self.gp.fit(Z, D[:, 0] if D.shape[1] == 1 else D.mean(axis=1))

            status = "success"
        except Exception as e:
            status = f"error: {e}"

        elapsed = (time.perf_counter() - start_time) * 1000
        self._update_times.append(elapsed)
        self._last_update_time = elapsed
        self._total_updates += 1
        self._points_since_update = 0

        if self.should_refit():
            self._points_since_refit = 0

        return {
            "status": status,
            "time_ms": elapsed,
            "n_data": self._buffer.size,
            "total_updates": self._total_updates,
        }

    def get_statistics(self) -> dict:
        """Get updater statistics."""
        stats = {
            "total_updates": self._total_updates,
            "points_since_update": self._points_since_update,
            "last_update_time_ms": self._last_update_time,
        }

        if self._buffer is not None:
            stats.update(self._buffer.get_statistics())

        if self._update_times:
            stats["mean_update_time_ms"] = np.mean(self._update_times)
            stats["max_update_time_ms"] = np.max(self._update_times)

        return stats


class OnlineStructuredGPUpdater:
    """
    Online updater for structured rocket GP.

    Handles both translational and rotational residuals.

    Example:
        >>> gp = StructuredRocketGP(config)
        >>> updater = OnlineStructuredGPUpdater(gp)
        >>>
        >>> # During flight
        >>> updater.add_observation(x, u, d_v, d_omega)
        >>> if updater.should_update():
        >>>     updater.update()
    """

    def __init__(
        self,
        gp: StructuredRocketGP,
        config: Optional[OnlineUpdateConfig] = None,
    ):
        """
        Initialize updater.

        Args:
            gp: Structured GP to update
            config: Update configuration
        """
        self.gp = gp
        self.config = config or OnlineUpdateConfig()

        # Tracking
        self._points_since_update: int = 0
        self._total_observations: int = 0
        self._total_updates: int = 0
        self._update_times: List[float] = []

    def add_observation(
        self,
        x: NDArray,
        u: NDArray,
        d_v: NDArray,
        d_omega: NDArray,
    ) -> bool:
        """
        Add new observation.

        Args:
            x: State (14,)
            u: Control (3,)
            d_v: Translational residual (3,)
            d_omega: Rotational residual (3,)

        Returns:
            True if observation was accepted
        """
        # Check novelty
        if self.config.use_novelty_filter and self.gp._is_fitted:
            is_novel = self.gp.is_novel(x, u)
            if not is_novel:
                return False

        # Add to GP's internal storage
        self.gp.add_data(
            x.reshape(1, -1),
            u.reshape(1, -1),
            d_v.reshape(1, -1),
            d_omega.reshape(1, -1),
        )

        self._points_since_update += 1
        self._total_observations += 1

        return True

    def should_update(self) -> bool:
        """Check if GP should be updated."""
        if self.gp.n_data < self.config.min_data_for_fit:
            return False
        return self._points_since_update >= self.config.update_interval

    def update(self, force: bool = False) -> dict:
        """
        Perform GP update.

        Args:
            force: Force update

        Returns:
            Update statistics
        """
        if not force and not self.should_update():
            return {"status": "skipped"}

        start_time = time.perf_counter()

        try:
            self.gp.fit()
            status = "success"
        except Exception as e:
            status = f"error: {e}"

        elapsed = (time.perf_counter() - start_time) * 1000
        self._update_times.append(elapsed)
        self._total_updates += 1
        self._points_since_update = 0

        return {
            "status": status,
            "time_ms": elapsed,
            "n_data": self.gp.n_data,
            "total_updates": self._total_updates,
        }

    def get_statistics(self) -> dict:
        """Get updater statistics."""
        stats = {
            "total_observations": self._total_observations,
            "total_updates": self._total_updates,
            "points_since_update": self._points_since_update,
            "n_data": self.gp.n_data,
            "gp_fitted": self.gp._is_fitted,
        }

        if self._update_times:
            stats["mean_update_time_ms"] = np.mean(self._update_times)
            stats["max_update_time_ms"] = np.max(self._update_times)

        return stats


class ResidualCollector:
    """
    Collects dynamics residuals from simulation for GP training.

    Computes d = x_actual - f_nominal(x, u) at each timestep.

    Example:
        >>> collector = ResidualCollector(dynamics)
        >>>
        >>> # During simulation
        >>> collector.record(x_k, u_k, x_kp1_actual, dt)
        >>>
        >>> # After simulation
        >>> X, U, D_v, D_omega = collector.get_training_data()
    """

    def __init__(
        self,
        nominal_dynamics: Callable[[NDArray, NDArray, float], NDArray],
        max_samples: int = 10000,
    ):
        """
        Initialize collector.

        Args:
            nominal_dynamics: Function f(x, u, dt) -> x_next_nominal
            max_samples: Maximum samples to store
        """
        self.nominal_dynamics = nominal_dynamics
        self.max_samples = max_samples

        # Storage
        self.states: List[NDArray] = []
        self.controls: List[NDArray] = []
        self.residuals_v: List[NDArray] = []
        self.residuals_omega: List[NDArray] = []

    def record(
        self,
        x: NDArray,
        u: NDArray,
        x_next_actual: NDArray,
        dt: float,
    ) -> None:
        """
        Record one transition and compute residual.

        Args:
            x: Current state (14,)
            u: Applied control (3,)
            x_next_actual: Actual next state (14,)
            dt: Timestep
        """
        if len(self.states) >= self.max_samples:
            # Remove oldest
            self.states.pop(0)
            self.controls.pop(0)
            self.residuals_v.pop(0)
            self.residuals_omega.pop(0)

        # Compute nominal prediction
        x_next_nominal = self.nominal_dynamics(x, u, dt)

        # Compute residual in state space
        residual = x_next_actual - x_next_nominal

        # Extract velocity and omega residuals (divided by dt to get acceleration)
        d_v = residual[4:7] / dt  # v residual -> acceleration
        d_omega = residual[11:14] / dt  # omega residual -> angular acceleration

        self.states.append(x.copy())
        self.controls.append(u.copy())
        self.residuals_v.append(d_v)
        self.residuals_omega.append(d_omega)

    def get_training_data(self) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Get collected data for GP training.

        Returns:
            X: States (N, 14)
            U: Controls (N, 3)
            D_v: Translational residuals (N, 3)
            D_omega: Rotational residuals (N, 3)
        """
        if not self.states:
            return (
                np.empty((0, 14)),
                np.empty((0, 3)),
                np.empty((0, 3)),
                np.empty((0, 3)),
            )

        return (
            np.array(self.states),
            np.array(self.controls),
            np.array(self.residuals_v),
            np.array(self.residuals_omega),
        )

    @property
    def n_samples(self) -> int:
        """Number of collected samples."""
        return len(self.states)

    def clear(self) -> None:
        """Clear all collected data."""
        self.states.clear()
        self.controls.clear()
        self.residuals_v.clear()
        self.residuals_omega.clear()

    def get_statistics(self) -> dict:
        """Get collector statistics."""
        if not self.states:
            return {"n_samples": 0}

        D_v = np.array(self.residuals_v)
        D_omega = np.array(self.residuals_omega)

        return {
            "n_samples": self.n_samples,
            "d_v_mean": np.mean(D_v, axis=0).tolist(),
            "d_v_std": np.std(D_v, axis=0).tolist(),
            "d_omega_mean": np.mean(D_omega, axis=0).tolist(),
            "d_omega_std": np.std(D_omega, axis=0).tolist(),
        }
