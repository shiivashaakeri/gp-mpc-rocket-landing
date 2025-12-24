"""
GP-MPC Rocket Landing

Learning-Based Model Predictive Control for 6-DoF Rocket Powered Descent
with Safety Guarantees.

Modules:
    dynamics: 6-DoF and 3-DoF rocket dynamics models
    gp: Gaussian Process learning for model residuals
    mpc: Model Predictive Control with GP-corrected dynamics
    safety: Predictive safety filter for constraint satisfaction
    terminal: Terminal set management (LMPC-style)
    reference: Reference trajectory generation
    utils: Utility functions (quaternions, config loading, etc.)
"""

__version__ = "0.1.0"
__author__ = "GP-MPC Rocket Landing Team"

# Convenience imports
from . import dynamics, gp, lmpc, mpc, reference, safety, terminal, utils

__all__ = [
    "dynamics",
    "gp",
    "lmpc",
    "mpc",
    "reference",
    "safety",
    "terminal",
    "utils",
]
