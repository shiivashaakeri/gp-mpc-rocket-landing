"""
Online Learning Loop for GP-MPC Rocket Landing

Coordinates the complete online learning pipeline:
1. Data collection during simulation
2. Novelty-based data selection
3. Online GP updates
4. Safe set expansion after successful landings
5. Periodic hyperparameter retraining

The learning loop enables iterative improvement:
    Episode 0: Use nominal model
    Episode 1+: Use GP-enhanced model with learned corrections

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control.
    Hewing, L., et al. (2020). Learning-Based Model Predictive Control.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .data_manager import DataManager, DataManagerConfig, EpisodeData
from .hyperparameter_tuner import HyperparameterConfig, HyperparameterTuner
from .novelty_selector import DataBuffer, NoveltyConfig, NoveltySelector


@dataclass
class OnlineLearningConfig:
    """Configuration for online learning loop."""

    # Data collection
    data_config: DataManagerConfig = field(default_factory=DataManagerConfig)

    # Novelty selection
    novelty_config: NoveltyConfig = field(default_factory=NoveltyConfig)

    # Hyperparameter tuning
    hyperparam_config: HyperparameterConfig = field(default_factory=HyperparameterConfig)

    # GP update settings
    online_update_method: str = "incremental"  # "incremental", "batch", "sliding_window"
    update_frequency: int = 10  # Updates per episode
    window_size: int = 1000  # Sliding window size

    # Safe set update
    update_safe_set: bool = True
    safe_set_success_only: bool = True

    # Learning triggers
    min_episodes_for_gp: int = 1  # Episodes before using GP
    retrain_every_n_episodes: int = 5  # Full retrain interval

    # Logging
    verbose: bool = True
    log_dir: Optional[str] = None


@dataclass
class LearningStatistics:
    """Statistics from online learning."""

    total_episodes: int = 0
    successful_episodes: int = 0
    total_data_points: int = 0
    gp_updates: int = 0
    hyperparam_retrains: int = 0
    mean_episode_cost: float = 0.0
    mean_landing_error: float = 0.0
    cost_history: List[float] = field(default_factory=list)
    error_history: List[float] = field(default_factory=list)


class OnlineLearner:
    """
    Online Learning Loop Coordinator.

    Manages the complete learning pipeline for iterative
    improvement of GP-MPC rocket landing.

    Example:
        >>> learner = OnlineLearner(dynamics, gp_model, safe_set, config)
        >>>
        >>> # Run learning loop
        >>> for episode in range(n_episodes):
        >>>     # Run MPC with current GP
        >>>     X, U, cost = run_mpc_episode(learner.get_gp_model())
        >>>
        >>>     # Update learner
        >>>     learner.process_episode(X, U, success=True)
        >>>
        >>>     # Get updated models
        >>>     gp = learner.get_gp_model()
    """

    def __init__(
        self,
        dynamics,
        gp_model,
        safe_set=None,
        config: Optional[OnlineLearningConfig] = None,
    ):
        """
        Initialize online learner.

        Args:
            dynamics: Rocket dynamics model
            gp_model: GP model for learning corrections
            safe_set: Safe set for LMPC (optional)
            config: Learning configuration
        """
        self.dynamics = dynamics
        self.gp_model = gp_model
        self.safe_set = safe_set
        self.config = config or OnlineLearningConfig()

        # Components
        self.data_manager = DataManager(dynamics, self.config.data_config)
        self.novelty_selector = NoveltySelector(self.config.novelty_config)
        self.hyperparam_tuner = HyperparameterTuner(self.config.hyperparam_config)
        self.data_buffer = DataBuffer(
            max_size=self.config.data_config.max_buffer_size,
            novelty_selector=self.novelty_selector,
        )

        # State
        self._current_episode = 0
        self._gp_active = False
        self._statistics = LearningStatistics()

        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "episode_start": [],
            "episode_end": [],
            "gp_update": [],
            "safe_set_update": [],
        }

    def add_callback(self, event: str, callback: Callable) -> None:
        """Add callback for learning events."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, **kwargs) -> None:
        """Trigger callbacks for an event."""
        for callback in self._callbacks.get(event, []):
            callback(**kwargs)

    def start_episode(self) -> int:
        """
        Start a new episode.

        Returns:
            Episode number
        """
        self._trigger_callbacks("episode_start", episode=self._current_episode)

        if self.config.verbose:
            print(f"\n=== Episode {self._current_episode} ===")

        return self._current_episode

    def add_transition(
        self,
        state: NDArray,
        control: NDArray,
        next_state: NDArray,
        timestep: int,
    ) -> None:
        """
        Add a transition from the current episode.

        Args:
            state: Current state
            control: Applied control
            next_state: Resulting state
            timestep: Timestep within episode
        """
        self.data_manager.add_transition(
            state=state,
            control=control,
            next_state=next_state,
            episode=self._current_episode,
            timestep=timestep,
        )

        # Trigger online GP update at intervals
        if timestep > 0 and timestep % (100 // self.config.update_frequency) == 0 and self._gp_active:
            self._online_gp_update()

    def end_episode(
        self,
        states: NDArray,
        controls: NDArray,
        success: bool,
        total_cost: float,
        landing_error: float,
        metadata: Optional[Dict] = None,
    ) -> EpisodeData:
        """
        End current episode and process results.

        Args:
            states: Full state trajectory
            controls: Full control trajectory
            success: Whether landing was successful
            total_cost: Total episode cost
            landing_error: Final position error
            metadata: Additional info

        Returns:
            Episode data
        """
        # End episode in data manager
        episode_data = self.data_manager.end_episode(
            success=success,
            total_cost=total_cost,
            landing_error=landing_error,
            metadata=metadata,
        )

        # Update statistics
        self._update_statistics(success, total_cost, landing_error)

        # Update safe set if successful
        if success and self.config.update_safe_set and self.safe_set is not None:
            self._update_safe_set(states, controls, total_cost)

        # Batch GP update
        self._batch_gp_update()

        # Periodic hyperparameter retraining
        if self._current_episode > 0 and self._current_episode % self.config.retrain_every_n_episodes == 0:
            self._retrain_hyperparameters()

        # Activate GP after enough episodes
        # _current_episode is incremented AFTER this check
        # So if _current_episode=1 (just finished episode 1, i.e., 2nd episode),
        # and min_episodes=2, we should activate
        if (self._current_episode + 1) >= self.config.min_episodes_for_gp:
            self._gp_active = True

        self._trigger_callbacks(
            "episode_end",
            episode=self._current_episode,
            success=success,
            cost=total_cost,
        )

        if self.config.verbose:
            status = "SUCCESS" if success else "FAILED"
            print(f"Episode {self._current_episode}: {status}, Cost={total_cost:.2f}, Error={landing_error:.3f}m")

        self._current_episode += 1

        return episode_data

    def process_episode(
        self,
        states: NDArray,
        controls: NDArray,
        success: bool,
        total_cost: Optional[float] = None,
        landing_error: Optional[float] = None,
    ) -> EpisodeData:
        """
        Process a complete episode (simplified interface).

        Handles data collection and learning updates.

        Args:
            states: State trajectory (T+1, n_x)
            controls: Control trajectory (T, n_u)
            success: Whether landing was successful
            total_cost: Total cost (computed if None)
            landing_error: Final error (computed if None)

        Returns:
            Episode data
        """
        self.start_episode()

        # Add all transitions
        for k in range(len(controls)):
            self.add_transition(states[k], controls[k], states[k + 1], k)

        # Compute metrics if not provided
        if total_cost is None:
            total_cost = self._compute_cost(states, controls)

        if landing_error is None:
            landing_error = np.linalg.norm(states[-1, 1:4])  # Position error

        return self.end_episode(
            states=states,
            controls=controls,
            success=success,
            total_cost=total_cost,
            landing_error=landing_error,
        )

    def _compute_cost(self, states: NDArray, controls: NDArray) -> float:
        """Compute trajectory cost."""
        n_x = states.shape[1]
        n_u = controls.shape[1]

        # Create appropriate cost matrices
        if n_x == 14:  # 6-DoF
            Q = np.diag([0, 10, 10, 10, 1, 1, 1, 0, 1, 1, 0, 0.1, 0.1, 0.1])
        elif n_x == 7:  # 3-DoF
            Q = np.diag([0, 10, 10, 10, 1, 1, 1])
        else:
            Q = np.eye(n_x)

        R = np.eye(n_u) * 0.01

        cost = 0.0
        for k in range(len(controls)):
            cost += states[k] @ Q @ states[k] + controls[k] @ R @ controls[k]

        return cost

    def _online_gp_update(self) -> None:
        """Perform online (incremental) GP update."""
        if not self._gp_active:
            return

        # Get recent data
        X_new, Y_new = self.data_manager.get_recent_data(50)

        if len(X_new) == 0:
            return

        # Select novel points
        indices = self.novelty_selector.select(X_new, Y_new, n_select=20)

        if len(indices) == 0:
            return

        X_selected = X_new[indices]
        Y_selected = Y_new[indices]

        # Update GP
        if hasattr(self.gp_model, "update"):
            self.gp_model.update(X_selected, Y_selected)
            self._statistics.gp_updates += 1

            self._trigger_callbacks("gp_update", n_points=len(indices))

    def _batch_gp_update(self) -> None:
        """Perform batch GP update at end of episode."""
        if not self._gp_active:
            return

        # Get all training data
        X, Y = self.data_manager.get_gp_training_data(
            max_points=self.config.window_size,
            successful_only=False,
        )

        if len(X) < 10:
            return

        # Select diverse subset
        indices = self.novelty_selector.select_diverse(X, Y, n_select=min(len(X), self.config.window_size))

        X_train = X[indices]
        Y_train = Y[indices]

        # Update data buffer
        self.data_buffer.add(X_train, Y_train)

        # Get buffered data and update GP
        X_buffer, Y_buffer = self.data_buffer.get_data()

        if len(X_buffer) > 0 and hasattr(self.gp_model, "fit"):
            self.gp_model.fit(X_buffer, Y_buffer)

        # Update novelty selector reference
        self.novelty_selector.set_reference_data(X_buffer)
        self.novelty_selector.set_gp_model(self.gp_model)

        self._statistics.total_data_points = len(X_buffer)

    def _update_safe_set(
        self,
        states: NDArray,
        controls: NDArray,
        total_cost: float,  # noqa: ARG002
    ) -> None:
        """Update safe set with successful trajectory."""
        if self.safe_set is None:
            return

        n_x = states.shape[1]
        n_u = controls.shape[1]

        # Compute stage costs with appropriate dimensions
        if n_x == 14:  # 6-DoF
            Q = np.diag([0, 10, 10, 10, 1, 1, 1, 0, 1, 1, 0, 0.1, 0.1, 0.1])
        elif n_x == 7:  # 3-DoF
            Q = np.diag([0, 10, 10, 10, 1, 1, 1])
        else:
            Q = np.eye(n_x)

        R = np.eye(n_u) * 0.01

        T = len(controls)
        stage_costs = np.zeros(T)
        for k in range(T):
            stage_costs[k] = states[k] @ Q @ states[k] + controls[k] @ R @ controls[k]

        # Add to safe set
        self.safe_set.add_trajectory(
            states=states,
            controls=controls,
            stage_costs=stage_costs,
            iteration=self._current_episode,
        )

        self._trigger_callbacks(
            "safe_set_update",
            episode=self._current_episode,
            n_states=len(states),
        )

        if self.config.verbose:
            print(f"  Safe set updated: {self.safe_set.num_states} states")

    def _retrain_hyperparameters(self) -> None:
        """Retrain GP hyperparameters."""
        X, Y = self.data_buffer.get_data()

        if len(X) < self.config.hyperparam_config.min_data_for_retrain:
            return

        new_params = self.hyperparam_tuner.tune(self.gp_model, X, Y)

        # Apply new hyperparameters
        if hasattr(self.gp_model, "set_hyperparameters"):
            self.gp_model.set_hyperparameters(new_params)
        # Manual update
        elif hasattr(self.gp_model, "kernel"):
            if "lengthscales" in new_params:
                self.gp_model.kernel.lengthscales = new_params["lengthscales"]
            if "variance" in new_params:
                self.gp_model.kernel.variance = new_params["variance"]

        self._statistics.hyperparam_retrains += 1

        if self.config.verbose:
            print(f"  Hyperparameters retrained: {new_params}")

    def _update_statistics(
        self,
        success: bool,
        cost: float,
        error: float,
    ) -> None:
        """Update learning statistics."""
        self._statistics.total_episodes += 1
        if success:
            self._statistics.successful_episodes += 1

        self._statistics.cost_history.append(cost)
        self._statistics.error_history.append(error)

        # Running averages
        self._statistics.mean_episode_cost = np.mean(self._statistics.cost_history[-20:])
        self._statistics.mean_landing_error = np.mean(self._statistics.error_history[-20:])

    def get_gp_model(self):
        """Get current GP model."""
        return self.gp_model

    def get_safe_set(self):
        """Get current safe set."""
        return self.safe_set

    def get_statistics(self) -> LearningStatistics:
        """Get learning statistics."""
        return self._statistics

    def is_gp_active(self) -> bool:
        """Check if GP is being used."""
        return self._gp_active

    def save(self, path: str) -> None:
        """Save learner state."""
        Path(path).mkdir(parents=True, exist_ok=True)

        # Save data
        self.data_manager.save(f"{path}/data.pkl")

        # Save safe set
        if self.safe_set is not None:
            self.safe_set.save(f"{path}/safe_set.pkl")

        # Save statistics
        import pickle  # noqa: PLC0415

        with open(f"{path}/statistics.pkl", "wb") as f:
            pickle.dump(self._statistics, f)

    def load(self, path: str) -> None:
        """Load learner state."""
        self.data_manager.load(f"{path}/data.pkl")

        if self.safe_set is not None:
            with contextlib.suppress(FileNotFoundError):
                self.safe_set.load(f"{path}/safe_set.pkl")

        import pickle  # noqa: PLC0415

        try:
            with open(f"{path}/statistics.pkl", "rb") as f:
                self._statistics = pickle.load(f)
        except FileNotFoundError:
            pass


class IterativeLearningRunner:
    """
    Runs iterative learning episodes.

    Coordinates the full learning loop including:
    - MPC controller
    - Safety filter
    - Online learning
    - Performance tracking
    """

    def __init__(
        self,
        dynamics,
        mpc_controller,
        learner: OnlineLearner,
        safety_filter=None,
        dt: float = 0.1,
    ):
        """
        Initialize learning runner.

        Args:
            dynamics: Rocket dynamics
            mpc_controller: MPC controller
            learner: Online learner
            safety_filter: Safety filter (optional)
            dt: Simulation timestep
        """
        self.dynamics = dynamics
        self.mpc = mpc_controller
        self.learner = learner
        self.safety_filter = safety_filter
        self.dt = dt

    def run_episode(
        self,
        x0: NDArray,
        x_target: NDArray,
        max_steps: int = 100,
    ) -> Tuple[NDArray, NDArray, bool, float]:
        """
        Run a single learning episode.

        Args:
            x0: Initial state
            x_target: Target state
            max_steps: Maximum steps

        Returns:
            X: State trajectory
            U: Control trajectory
            success: Whether landing succeeded
            cost: Total cost
        """
        self.learner.start_episode()

        X = [x0]
        U = []
        x_current = x0.copy()
        total_cost = 0.0

        for step in range(max_steps):
            # Get MPC control
            try:
                solution = self.mpc.solve(x_current, x_target)
                u = solution.u0
            except Exception:
                break

            # Apply safety filter
            if self.safety_filter is not None:
                result = self.safety_filter.filter(x_current, u)
                u = result.u_safe

            U.append(u)

            # Simulate
            x_next = self.dynamics.step(x_current, u, self.dt)

            # Record transition
            self.learner.add_transition(x_current, u, x_next, step)

            X.append(x_next)
            x_current = x_next

            # Compute cost
            total_cost += float(x_current @ np.eye(len(x_current)) @ x_current * 0.01)

            # Check termination
            if x_current[1] < 0.1:  # Landed
                break

        X = np.array(X)
        U = np.array(U)

        # Determine success
        final_pos_error = np.linalg.norm(X[-1, 1:4])
        final_vel = np.linalg.norm(X[-1, 4:7])
        success = final_pos_error < 1.0 and final_vel < 1.0

        # Process episode
        self.learner.end_episode(
            states=X,
            controls=U,
            success=success,
            total_cost=total_cost,
            landing_error=final_pos_error,
        )

        return X, U, success, total_cost

    def run_learning_loop(
        self,
        x0_generator: Callable,
        x_target: NDArray,
        n_episodes: int,
        max_steps: int = 100,
    ) -> List[Dict]:
        """
        Run multiple learning episodes.

        Args:
            x0_generator: Function that generates initial states
            x_target: Target state
            n_episodes: Number of episodes
            max_steps: Max steps per episode

        Returns:
            List of episode results
        """
        results = []

        for episode in range(n_episodes):
            x0 = x0_generator()
            X, U, success, cost = self.run_episode(x0, x_target, max_steps)

            results.append(
                {
                    "episode": episode,
                    "X": X,
                    "U": U,
                    "success": success,
                    "cost": cost,
                    "landing_error": np.linalg.norm(X[-1, 1:4]),
                }
            )

        return results
