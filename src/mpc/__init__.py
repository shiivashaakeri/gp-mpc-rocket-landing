"""
MPC Module for GP-MPC Rocket Landing

This module provides Model Predictive Control implementations:
- constraints: Path and terminal constraints for rocket landing
- cost_functions: Quadratic costs, LQR terminal cost
- nominal_mpc: Basic MPC without GP learning
- gp_mpc: MPC with GP-augmented dynamics
- uncertainty_prop: Uncertainty propagation for constraint tightening
- rti_mpc: Real-Time Iteration MPC for fast control

Usage:
    >>> from src.mpc import NominalMPC, MPCConfig
    >>> from src.dynamics import create_szmuk_rocket
    >>>
    >>> rocket = create_szmuk_rocket()
    >>> mpc = NominalMPC(rocket, MPCConfig(N=20))
    >>> mpc.setup()
    >>>
    >>> solution = mpc.solve(x0, x_target)
    >>> u_apply = solution.u0
"""

from .constraints import (
    CasADiConstraints,
    ConstraintParams,
    TightenedConstraints,
    check_all_constraints,
    compute_constraint_jacobians,
    eval_angular_rate,
    eval_gimbal_angle,
    eval_glideslope,
    eval_thrust_magnitude,
    eval_tilt_angle,
)
from .cost_functions import (
    CasADiCostFunction,
    CostWeights,
    LQRTerminalCost,
    TrajectoryObjective,
    compute_lqr_gain,
    compute_lqr_terminal_cost,
    fuel_optimal_stage_cost,
    quadratic_stage_cost,
    tracking_cost,
)
from .gp_mpc import (
    GPMPC,
    GPMPCConfig,
    SimpleGPPredictor,
)
from .nominal_mpc import (
    MPCConfig,
    MPCSolution,
    NominalMPC,
    NominalMPC3DoF,
)
from .osqp_rti import (
    OSQPRTIMPC,
    FastRTI3DoF,
    OSQPRTIConfig,
    OSQPRTISolution,
)
from .rti_mpc import (
    RTI_MPC,
    RTIConfig,
    RTISolution,
    SimpleRTI,
)
from .uncertainty_prop import (
    ConstraintTightening,
    PropagatedUncertainty,
    TubeBasedRobustness,
    UncertaintyPropagator,
)

__all__ = [
    "GPMPC",
    "OSQPRTIMPC",
    "RTI_MPC",
    "CasADiConstraints",
    "CasADiCostFunction",
    # Constraints
    "ConstraintParams",
    "ConstraintTightening",
    # Cost Functions
    "CostWeights",
    # OSQP RTI-MPC
    "FastRTI3DoF",
    # GP-MPC
    "GPMPCConfig",
    "LQRTerminalCost",
    # Nominal MPC
    "MPCConfig",
    "MPCSolution",
    "NominalMPC",
    "NominalMPC3DoF",
    "OSQPRTIConfig",
    "OSQPRTISolution",
    # Uncertainty
    "PropagatedUncertainty",
    # RTI-MPC
    "RTIConfig",
    "RTISolution",
    "SimpleGPPredictor",
    "SimpleRTI",
    "TightenedConstraints",
    "TrajectoryObjective",
    "TubeBasedRobustness",
    "UncertaintyPropagator",
    "check_all_constraints",
    "compute_constraint_jacobians",
    "compute_lqr_gain",
    "compute_lqr_terminal_cost",
    "eval_angular_rate",
    "eval_gimbal_angle",
    "eval_glideslope",
    "eval_thrust_magnitude",
    "eval_tilt_angle",
    "fuel_optimal_stage_cost",
    "quadratic_stage_cost",
    "tracking_cost",
]
