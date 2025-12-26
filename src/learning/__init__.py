"""
Online Learning Module for GP-MPC Rocket Landing

Coordinates the complete online learning pipeline:
1. Data collection during simulation
2. Novelty-based data selection
3. Online GP updates
4. Safe set expansion after successful landings
5. Periodic hyperparameter retraining

The learning loop enables iterative improvement of the GP model
and safe set, leading to better landing performance over time.

Components:
    data_manager: Data collection and storage
    novelty_selector: Novelty-based data selection
    hyperparameter_tuner: GP hyperparameter optimization
    online_learner: Main learning loop coordinator

Usage:
    >>> from src.learning import OnlineLearner, OnlineLearningConfig
    >>>
    >>> # Initialize learner
    >>> learner = OnlineLearner(dynamics, gp_model, safe_set, config)
    >>>
    >>> # Run learning episodes
    >>> for episode in range(n_episodes):
    >>>     X, U, cost = run_mpc_episode(...)
    >>>     learner.process_episode(X, U, success=True)
    >>>
    >>> # Get improved model
    >>> gp = learner.get_gp_model()
    >>> safe_set = learner.get_safe_set()

Reference:
    Hewing, L., et al. (2020). Learning-Based Model Predictive Control:
    Toward Safe Learning in Control. Annual Review of Control.
"""

from .data_manager import (
    DataManager,
    DataManagerConfig,
    DataPoint,
    EpisodeData,
    StreamingDataCollector,
)
from .hyperparameter_tuner import (
    AdaptiveHyperparameterScheduler,
    HyperparameterConfig,
    HyperparameterTuner,
)
from .novelty_selector import (
    ActiveDataSelector,
    DataBuffer,
    NoveltyConfig,
    NoveltySelector,
)
from .online_learner import (
    IterativeLearningRunner,
    LearningStatistics,
    OnlineLearner,
    OnlineLearningConfig,
)

__all__ = [
    "ActiveDataSelector",
    "AdaptiveHyperparameterScheduler",
    "DataBuffer",
    "DataManager",
    "DataManagerConfig",
    # Data Management
    "DataPoint",
    "EpisodeData",
    # Hyperparameter Tuning
    "HyperparameterConfig",
    "HyperparameterTuner",
    "IterativeLearningRunner",
    "LearningStatistics",
    # Novelty Selection
    "NoveltyConfig",
    "NoveltySelector",
    "OnlineLearner",
    # Online Learning
    "OnlineLearningConfig",
    "StreamingDataCollector",
]
