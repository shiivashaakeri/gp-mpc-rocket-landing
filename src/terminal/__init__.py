"""
Terminal Set Module for LMPC

This module provides terminal set management for Learning MPC:
- safe_set: Sampled safe set data structure
- local_safe_set: K-nearest neighbor local safe set
- convex_hull: Convex hull terminal constraints
- q_function: Q-function (cost-to-go) approximation

The terminal set SS contains states from which successful
trajectories to the target have been demonstrated.

Key concepts:
    - SS^j = Safe set at iteration j (grows with each success)
    - Q(x) = Cost-to-go from state x
    - Terminal constraint: x_N ∈ Conv(SS_local)
    - Terminal cost: V_f(x) = Q(x)

Usage:
    >>> from src.terminal import SampledSafeSet, LocalSafeSet, QFunctionManager
    >>>
    >>> # Create safe set
    >>> ss = SampledSafeSet(n_x=14, n_u=3)
    >>> ss.add_trajectory(X, U, costs)
    >>>
    >>> # Query local safe set
    >>> local_ss = LocalSafeSet(ss)
    >>> neighbors, q_values, distances = local_ss.query(x)
    >>>
    >>> # Get Q-function value
    >>> qm = QFunctionManager(ss)
    >>> q_val = qm.evaluate(x)

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control
    for Iterative Tasks. IEEE Transactions on Automatic Control.
"""

from .convex_hull import (
    ConvexHullConfig,
    ConvexHullConstraint,
    TerminalSetManager,
)
from .local_safe_set import (
    LocalSafeSet,
    LocalSafeSetConfig,
    MultiResolutionLocalSafeSet,
)
from .memory_safe_set import (
    CompactTrajectory,
    MemoryOptimizedConfig,
    MemoryOptimizedSafeSet,
    StreamingSafeSet,
)
from .q_function import (
    GPQFunction,
    InverseDistanceQFunction,
    IterativeQFunction,
    LocalLinearQFunction,
    QFunctionApproximator,
    QFunctionConfig,
    QFunctionManager,
)
from .safe_set import (
    FuelAwareSafeSet,
    SampledSafeSet,
    TrajectoryData,
)

# Check for CasADi
try:
    from .convex_hull import CasADiConvexHullConstraint  # noqa: F401

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

__all__ = [
    "CompactTrajectory",
    # Convex Hull
    "ConvexHullConfig",
    "ConvexHullConstraint",
    "FuelAwareSafeSet",
    "GPQFunction",
    "InverseDistanceQFunction",
    "IterativeQFunction",
    "LocalLinearQFunction",
    "LocalSafeSet",
    # Local Safe Set
    "LocalSafeSetConfig",
    # Memory Safe Set
    "MemoryOptimizedConfig",
    "MemoryOptimizedSafeSet",
    "MultiResolutionLocalSafeSet",
    "QFunctionApproximator",
    # Q-Function
    "QFunctionConfig",
    "QFunctionManager",
    "SampledSafeSet",
    "StreamingSafeSet",
    "TerminalSetManager",
    # Safe Set
    "TrajectoryData",
]

if HAS_CASADI:
    __all__.append("CasADiConvexHullConstraint")
