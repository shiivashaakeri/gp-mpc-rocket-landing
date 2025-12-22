"""
Dynamics Module for GP-MPC Rocket Landing

This module provides dynamics models for powered descent guidance:

- Rocket6DoFDynamics: Full 6-DoF rigid body model (14 states)
- Rocket3DoFDynamics: Point-mass model (7 states) for prototyping

Both classes wrap simdyn's rocket models with a consistent interface
suitable for GP-MPC integration.

Usage:
    >>> from src.dynamics import Rocket6DoFDynamics, create_szmuk_rocket
    >>>
    >>> # Create 6-DoF rocket with Szmuk parameters
    >>> rocket = create_szmuk_rocket()
    >>>
    >>> # Create initial state
    >>> x0 = rocket.create_initial_state(altitude=10.0, mass=2.0)
    >>>
    >>> # Evaluate dynamics
    >>> u = rocket.hover_thrust(x0)
    >>> x_next = rocket.step(x0, u, dt=0.1)
    >>>
    >>> # Get Jacobians for MPC
    >>> A, B = rocket.linearize(x0, u)

Key Components:
    - rocket_6dof: Full 6-DoF dynamics (Szmuk et al. 2018)
    - rocket_3dof: Simplified 3-DoF point-mass model
    - linearization: Jacobian computation utilities
    - discretization: Integration methods (Euler, RK4, etc.)

References:
    Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for 6-DoF
    powered descent guidance with compound state-triggered constraints.
"""

from .discretization import (
    Integrator,
    IntegratorType,
    euler_step,
    hermite_simpson_defect,
    integrate_sensitivity,
    integrate_trajectory,
    quaternion_euler_step,
    quaternion_exponential_step,
    quaternion_multiply,
    rk4_step,
    trapezoidal_defect,
)
from .linearization import (
    AffineModel,
    FiniteDifferenceMethod,
    compute_discrete_affine_model,
    discretize_jacobians,
    numerical_jacobian_u,
    numerical_jacobian_x,
    numerical_jacobians,
    trajectory_jacobians,
    verify_jacobians,
)
from .rocket_3dof import (
    Rocket3DoFConfig,
    Rocket3DoFDynamics,
    create_normalized_rocket,
    create_rocket_3dof,
)
from .rocket_6dof import (
    Rocket6DoFConfig,
    Rocket6DoFDynamics,
    create_rocket_6dof,
    create_szmuk_rocket,
)

__all__ = [
    "AffineModel",
    "FiniteDifferenceMethod",
    # Discretization
    "Integrator",
    "IntegratorType",
    "Rocket3DoFConfig",
    # 3-DoF Rocket
    "Rocket3DoFDynamics",
    "Rocket6DoFConfig",
    # 6-DoF Rocket
    "Rocket6DoFDynamics",
    "compute_discrete_affine_model",
    "create_normalized_rocket",
    "create_rocket_3dof",
    "create_rocket_6dof",
    "create_szmuk_rocket",
    "discretize_jacobians",
    "euler_step",
    "hermite_simpson_defect",
    "integrate_sensitivity",
    "integrate_trajectory",
    "numerical_jacobian_u",
    # Linearization
    "numerical_jacobian_x",
    "numerical_jacobians",
    "quaternion_euler_step",
    "quaternion_exponential_step",
    "quaternion_multiply",
    "rk4_step",
    "trajectory_jacobians",
    "trapezoidal_defect",
    "verify_jacobians",
]
