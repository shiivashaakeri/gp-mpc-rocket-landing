"""
Data Collection and Management for Online Learning

Handles collection, storage, and retrieval of trajectory data
for GP learning and safe set updates.

Key responsibilities:
1. Collect state-control-residual tuples during simulation
2. Store data efficiently for GP training
3. Manage data buffers with size limits
4. Compute model residuals for learning

Reference:
    Hewing, L., et al. (2020). Learning-Based Model Predictive Control:
    Toward Safe Learning in Control. Annual Review of Control.
"""

from __future__ import annotations

import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class DataPoint:
    """Single data point for GP training."""

    state: NDArray  # State x_k
    control: NDArray  # Control u_k
    next_state: NDArray  # Actual next state x_{k+1}
    predicted_next: NDArray  # Model-predicted next state
    residual: NDArray  # Prediction error (actual - predicted)
    timestamp: float  # Collection time
    episode: int  # Episode number
    timestep: int  # Timestep within episode
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def input(self) -> NDArray:
        """GP input: [state, control]."""
        return np.concatenate([self.state, self.control])

    @property
    def output(self) -> NDArray:
        """GP output: residual."""
        return self.residual


@dataclass
class EpisodeData:
    """Data from a complete episode (landing attempt)."""

    episode_id: int
    states: NDArray  # (T+1, n_x)
    controls: NDArray  # (T, n_u)
    residuals: NDArray  # (T, n_residual)
    success: bool  # Whether landing was successful
    total_cost: float
    landing_error: float  # Final position error
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        """Episode length."""
        return len(self.controls)


@dataclass
class DataManagerConfig:
    """Configuration for data manager."""

    # Buffer sizes
    max_buffer_size: int = 10000  # Maximum data points in buffer
    max_episodes: int = 100  # Maximum episodes to store

    # Data collection
    collect_every: int = 1  # Collect every N timesteps

    # Residual computation
    residual_type: str = "velocity"  # "velocity", "acceleration", "full"

    # Storage
    save_dir: Optional[str] = None
    auto_save_every: int = 10  # Save every N episodes


class DataManager:
    """
    Manages data collection and storage for online learning.

    Collects trajectory data during simulation and computes
    model residuals for GP training.

    Example:
        >>> data_mgr = DataManager(dynamics, config)
        >>>
        >>> # During simulation
        >>> for k in range(N):
        >>>     x_next = dynamics.step(x, u, dt)
        >>>     data_mgr.add_transition(x, u, x_next, episode=0, timestep=k)
        >>>
        >>> # Get data for GP training
        >>> X, Y = data_mgr.get_gp_training_data()
    """

    def __init__(
        self,
        dynamics,
        config: Optional[DataManagerConfig] = None,
    ):
        """
        Initialize data manager.

        Args:
            dynamics: Rocket dynamics model
            config: Configuration
        """
        self.dynamics = dynamics
        self.config = config or DataManagerConfig()

        # State/control dimensions - detect from dynamics if possible
        if hasattr(dynamics, "n_x"):
            self.n_x = dynamics.n_x
        else:
            self.n_x = 14  # Default 6-DoF

        if hasattr(dynamics, "n_u"):
            self.n_u = dynamics.n_u
        else:
            self.n_u = 3  # Default

        # Determine residual dimension
        if self.config.residual_type == "velocity":
            self.n_residual = 6  # [dv(3), dω(3)]
        elif self.config.residual_type == "acceleration":
            self.n_residual = 6
        else:
            self.n_residual = self.n_x

        # Data storage
        self._buffer: deque = deque(maxlen=self.config.max_buffer_size)
        self._episodes: List[EpisodeData] = []

        # Current episode tracking
        self._current_episode: List[DataPoint] = []
        self._episode_count = 0

        # Statistics
        self._total_points_collected = 0

    def add_transition(
        self,
        state: NDArray,
        control: NDArray,
        next_state: NDArray,
        episode: int,
        timestep: int,
        metadata: Optional[Dict] = None,
    ) -> DataPoint:
        """
        Add a state transition to the buffer.

        Args:
            state: Current state
            control: Applied control
            next_state: Actual next state
            episode: Episode number
            timestep: Timestep within episode
            metadata: Additional info

        Returns:
            Created data point
        """
        # Skip based on collection frequency
        if timestep % self.config.collect_every != 0:
            return None

        # Get model prediction
        dt = getattr(self.dynamics, "dt", 0.1)
        predicted_next = self.dynamics.step(state, control, dt)

        # Compute residual
        residual = self._compute_residual(state, control, next_state, predicted_next, dt)

        # Create data point
        point = DataPoint(
            state=state.copy(),
            control=control.copy(),
            next_state=next_state.copy(),
            predicted_next=predicted_next.copy(),
            residual=residual,
            timestamp=time.time(),
            episode=episode,
            timestep=timestep,
            metadata=metadata or {},
        )

        self._buffer.append(point)
        self._current_episode.append(point)
        self._total_points_collected += 1

        return point

    def _compute_residual(
        self,
        state: NDArray,
        control: NDArray,  # noqa: ARG002
        actual_next: NDArray,
        predicted_next: NDArray,
        dt: float,
    ) -> NDArray:
        """
        Compute model residual for GP learning.

        The residual represents unmodeled dynamics:
            x_{actual} = x_{predicted} + d(x, u) * dt

        Args:
            state: Current state
            control: Applied control
            actual_next: Actual next state
            predicted_next: Model-predicted next state
            dt: Timestep

        Returns:
            Residual vector
        """
        if self.config.residual_type == "velocity":
            # Residual in velocity and angular rate
            # d = (actual - predicted) / dt
            v_residual = (actual_next[4:7] - predicted_next[4:7]) / dt
            omega_residual = (actual_next[11:14] - predicted_next[11:14]) / dt
            return np.concatenate([v_residual, omega_residual])

        elif self.config.residual_type == "acceleration":
            # Residual as acceleration error
            v_actual = (actual_next[4:7] - state[4:7]) / dt
            v_predicted = (predicted_next[4:7] - state[4:7]) / dt

            omega_actual = (actual_next[11:14] - state[11:14]) / dt
            omega_predicted = (predicted_next[11:14] - state[11:14]) / dt

            return np.concatenate([v_actual - v_predicted, omega_actual - omega_predicted])

        else:  # full
            return (actual_next - predicted_next) / dt

    def end_episode(
        self,
        success: bool,
        total_cost: float,
        landing_error: float,
        metadata: Optional[Dict] = None,
    ) -> EpisodeData:
        """
        End current episode and store episode data.

        Args:
            success: Whether landing was successful
            total_cost: Total episode cost
            landing_error: Final position error
            metadata: Additional info

        Returns:
            Episode data
        """
        if len(self._current_episode) == 0:
            return None

        # Compile episode data
        states = np.array([p.state for p in self._current_episode])
        controls = np.array([p.control for p in self._current_episode])
        residuals = np.array([p.residual for p in self._current_episode])

        # Add final state
        if len(self._current_episode) > 0:
            final_state = self._current_episode[-1].next_state
            states = np.vstack([states, final_state])

        episode = EpisodeData(
            episode_id=self._episode_count,
            states=states,
            controls=controls,
            residuals=residuals,
            success=success,
            total_cost=total_cost,
            landing_error=landing_error,
            metadata=metadata or {},
        )

        self._episodes.append(episode)

        # Limit stored episodes
        if len(self._episodes) > self.config.max_episodes:
            self._episodes.pop(0)

        # Auto-save
        if self.config.save_dir and self._episode_count % self.config.auto_save_every == 0:
            self.save(self.config.save_dir)

        # Reset current episode
        self._current_episode = []
        self._episode_count += 1

        return episode

    def get_gp_training_data(
        self,
        max_points: Optional[int] = None,
        recent_only: bool = False,
        successful_only: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        """
        Get data for GP training.

        Args:
            max_points: Maximum points to return
            recent_only: Only use recent data
            successful_only: Only use data from successful episodes

        Returns:
            X: Input features (N, n_x + n_u)
            Y: Output targets (N, n_residual)
        """
        points = list(self._buffer)

        # Filter by episode success
        if successful_only:
            successful_episodes = {e.episode_id for e in self._episodes if e.success}
            points = [p for p in points if p.episode in successful_episodes]

        # Limit to recent data
        if recent_only and max_points:
            points = points[-max_points:]
        elif max_points:  # noqa: SIM102
            # Uniform sampling
            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = [points[i] for i in indices]

        if len(points) == 0:
            return np.zeros((0, self.n_x + self.n_u)), np.zeros((0, self.n_residual))

        X = np.array([p.input for p in points])
        Y = np.array([p.output for p in points])

        return X, Y

    def get_recent_data(self, n_points: int) -> Tuple[NDArray, NDArray]:
        """Get most recent n data points."""
        points = list(self._buffer)[-n_points:]

        if len(points) == 0:
            return np.zeros((0, self.n_x + self.n_u)), np.zeros((0, self.n_residual))

        X = np.array([p.input for p in points])
        Y = np.array([p.output for p in points])

        return X, Y

    def get_episode(self, episode_id: int) -> Optional[EpisodeData]:
        """Get specific episode data."""
        for ep in self._episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def get_successful_episodes(self) -> List[EpisodeData]:
        """Get all successful episodes."""
        return [ep for ep in self._episodes if ep.success]

    def get_statistics(self) -> Dict[str, Any]:
        """Get data collection statistics."""
        n_successful = sum(1 for ep in self._episodes if ep.success)

        return {
            "total_points": self._total_points_collected,
            "buffer_size": len(self._buffer),
            "n_episodes": len(self._episodes),
            "n_successful": n_successful,
            "success_rate": n_successful / max(1, len(self._episodes)),
        }

    def save(self, filepath: str) -> None:
        """Save data to file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "buffer": list(self._buffer),
            "episodes": self._episodes,
            "episode_count": self._episode_count,
            "total_points": self._total_points_collected,
            "config": self.config,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load data from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self._buffer = deque(data["buffer"], maxlen=self.config.max_buffer_size)
        self._episodes = data["episodes"]
        self._episode_count = data["episode_count"]
        self._total_points_collected = data["total_points"]

    def clear(self) -> None:
        """Clear all data."""
        self._buffer.clear()
        self._episodes.clear()
        self._current_episode.clear()
        self._episode_count = 0
        self._total_points_collected = 0


class StreamingDataCollector:
    """
    Streaming data collector for real-time learning.

    Collects data in a streaming fashion and triggers
    GP updates when enough new data is available.
    """

    def __init__(
        self,
        data_manager: DataManager,
        update_threshold: int = 50,
    ):
        """
        Initialize streaming collector.

        Args:
            data_manager: Data manager instance
            update_threshold: Points before triggering update
        """
        self.data_manager = data_manager
        self.update_threshold = update_threshold

        self._points_since_update = 0
        self._update_callbacks = []

    def add_callback(self, callback: callable) -> None:
        """Add callback for update trigger."""
        self._update_callbacks.append(callback)

    def collect(
        self,
        state: NDArray,
        control: NDArray,
        next_state: NDArray,
        episode: int,
        timestep: int,
    ) -> bool:
        """
        Collect data point and check for update trigger.

        Returns:
            True if update should be triggered
        """
        point = self.data_manager.add_transition(state, control, next_state, episode, timestep)

        if point is not None:
            self._points_since_update += 1

        if self._points_since_update >= self.update_threshold:
            self._trigger_update()
            return True

        return False

    def _trigger_update(self) -> None:
        """Trigger update callbacks."""
        for callback in self._update_callbacks:
            callback()

        self._points_since_update = 0
