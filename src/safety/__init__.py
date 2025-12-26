"""
Safety Module for GP-MPC Rocket Landing

Predictive safety filter for constraint satisfaction under uncertainty.

Components:
    backup_controller: LQR and PD backup controllers
    invariant_sets: Ellipsoidal and polytopic invariant sets
    tube_mpc: Tube propagation with GP uncertainty
    safety_filter: Predictive safety filter QP

The safety filter ensures that:
1. The system can always recover to a safe state
2. All constraints are satisfied along the trajectory
3. Learning does not compromise safety

Algorithm:
    1. Receive control u from MPC
    2. Predict trajectory with u, then backup controller
    3. Check if terminal state reaches backup invariant set
    4. If unsafe, modify u to closest safe alternative

Usage:
    >>> from src.safety import PredictiveSafetyFilter, SafetyFilterConfig
    >>>
    >>> # Initialize safety filter
    >>> safety = PredictiveSafetyFilter(dynamics, SafetyFilterConfig())
    >>> safety.initialize(x_hover)
    >>>
    >>> # Filter MPC control
    >>> result = safety.filter(x_current, u_mpc)
    >>> u_safe = result.u_safe

Reference:
    Wabersich, K. P., & Zeilinger, M. N. (2021). A predictive safety
    filter for learning-based control. Automatica.
"""

from .backup_controller import (
    BackupControllerConfig,
    EmergencyBrakingController,
    LQRBackupController,
    PDBackupController,
    create_backup_controller,
)
from .invariant_sets import (
    EllipsoidalInvariantSet,
    InvariantSetConfig,
    PolytopeInvariantSet,
    TubeController,
    compute_lmi_invariant_set,
)
from .safety_filter import (
    PredictiveSafetyFilter,
    SafetyFilterConfig,
    SafetyFilterResult,
    SimpleSafetyFilter,
)
from .tube_mpc import (
    RobustTubeMPC,
    TubeConstraintTightener,
    TubeMPCConfig,
    TubePropagator,
)

__all__ = [
    # Backup Controller
    "BackupControllerConfig",
    "EllipsoidalInvariantSet",
    "EmergencyBrakingController",
    # Invariant Sets
    "InvariantSetConfig",
    "LQRBackupController",
    "PDBackupController",
    "PolytopeInvariantSet",
    "PredictiveSafetyFilter",
    "RobustTubeMPC",
    # Safety Filter
    "SafetyFilterConfig",
    "SafetyFilterResult",
    "SimpleSafetyFilter",
    "TubeConstraintTightener",
    "TubeController",
    # Tube MPC
    "TubeMPCConfig",
    "TubePropagator",
    "compute_lmi_invariant_set",
    "create_backup_controller",
]
