"""
LMPC Module for GP-MPC Rocket Landing

Learning Model Predictive Control (LMPC) combines MPC with data-driven
terminal constraints and costs for iterative improvement.

This module contains the main LMPC controller. Terminal set components
(safe set, Q-function, convex hull) are in the `terminal` module.

Usage:
    >>> from src.lmpc import LMPC, LMPCConfig
    >>> from src.terminal import SampledSafeSet  # Terminal set components
    >>>
    >>> # Initialize LMPC
    >>> lmpc = LMPC(dynamics, LMPCConfig(N=15))
    >>>
    >>> # Add initial trajectory (e.g., from SCVX)
    >>> lmpc.add_trajectory(X_init, U_init)
    >>>
    >>> # Run iterative improvement
    >>> for i in range(n_iterations):
    >>>     X, U, costs, success = lmpc.run_episode(x0)
    >>>     print(f"Iteration {i}: cost = {np.sum(costs):.2f}")

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control
    for Iterative Tasks. IEEE Transactions on Automatic Control.
"""

# Re-export terminal components for convenience
from src.terminal import (
    ConvexHullConfig,
    ConvexHullConstraint,
    FuelAwareSafeSet,
    GPQFunction,
    InverseDistanceQFunction,
    IterativeQFunction,
    LocalLinearQFunction,
    LocalSafeSet,
    LocalSafeSetConfig,
    MultiResolutionLocalSafeSet,
    QFunctionApproximator,
    QFunctionConfig,
    QFunctionManager,
    SampledSafeSet,
    TerminalSetManager,
    TrajectoryData,
)

from .lmpc import (
    LMPC,
    LMPCConfig,
    LMPCSolution,
    SimpleLMPC,
)

__all__ = [
    "LMPC",
    "ConvexHullConfig",
    "ConvexHullConstraint",
    "FuelAwareSafeSet",
    "GPQFunction",
    "InverseDistanceQFunction",
    "IterativeQFunction",
    # LMPC Controller
    "LMPCConfig",
    "LMPCSolution",
    "LocalLinearQFunction",
    "LocalSafeSet",
    "LocalSafeSetConfig",
    "MultiResolutionLocalSafeSet",
    "QFunctionApproximator",
    "QFunctionConfig",
    "QFunctionManager",
    "SampledSafeSet",
    "SimpleLMPC",
    "TerminalSetManager",
    # Re-exported from terminal (for backwards compatibility)
    "TrajectoryData",
]
