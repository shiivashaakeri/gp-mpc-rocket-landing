"""
Gaussian Process Module for GP-MPC

This module provides GP regression components for learning rocket dynamics residuals:

- kernels: Covariance functions (SE-ARD, Matérn, etc.)
- exact_gp: Full GP regression O(N³) for validation
- sparse_gp: Sparse GP with inducing points O(NM²) for scalability
- features: Physics-informed feature extraction for rockets
- structured_gp: 6-output GP learning only acceleration residuals

Usage:
    >>> from src.gp import StructuredRocketGP, StructuredGPConfig
    >>>
    >>> # Create structured GP for rocket dynamics
    >>> config = StructuredGPConfig(n_inducing=50)
    >>> gp = StructuredRocketGP(config)
    >>>
    >>> # Add training data (states, controls, residuals)
    >>> gp.add_data(X, U, D_v, D_omega)
    >>> gp.fit()
    >>>
    >>> # Predict residuals with uncertainty
    >>> d_v, d_omega, var_v, var_omega = gp.predict(x, u)
"""

from .exact_gp import (
    ExactGP,
    GPPrediction,
    MultiOutputExactGP,
)
from .fast_gp import (
    CachedGPPredictor,
    FastGPConfig,
    FastGPPredictor,
    NumbaGPPredictor,
    SparseGPPredictor,
    create_fast_gp,
)
from .features import (
    AtmosphereModel,
    CombinedFeatureExtractor,
    RotationalFeatureExtractor,
    Simple3DoFFeatureExtractor,
    TranslationalFeatureExtractor,
)
from .kernels import (
    RBF,
    SE_ARD,
    # Base class
    Kernel,
    # Matérn kernels
    Matern32,
    Matern52,
    ProductKernel,
    SquaredExponential,
    # SE kernels
    SquaredExponentialARD,
    # Composite kernels
    SumKernel,
    # Noise kernel
    WhiteNoise,
    create_matern_kernel,
    # Factory functions
    create_se_ard_kernel,
)
from .online_update import (
    DataBuffer,
    OnlineGPUpdater,
    OnlineStructuredGPUpdater,
    OnlineUpdateConfig,
    ResidualCollector,
)
from .sparse_gp import (
    MultiOutputSparseGP,
    SparseGP,
)
from .structured_gp import (
    Simple3DoFGP,
    StructuredGPConfig,
    StructuredRocketGP,
)

__all__ = [
    "RBF",
    "SE_ARD",
    "AtmosphereModel",
    "CachedGPPredictor",
    "CombinedFeatureExtractor",
    "DataBuffer",
    # Exact GP
    "ExactGP",
    # Fast GP
    "FastGPConfig",
    "FastGPPredictor",
    "GPPrediction",
    # Kernels
    "Kernel",
    "Matern32",
    "Matern52",
    "MultiOutputExactGP",
    "MultiOutputSparseGP",
    "NumbaGPPredictor",
    "OnlineGPUpdater",
    "OnlineStructuredGPUpdater",
    # Online Update
    "OnlineUpdateConfig",
    "ProductKernel",
    "ResidualCollector",
    "RotationalFeatureExtractor",
    "Simple3DoFFeatureExtractor",
    "Simple3DoFGP",
    # Sparse GP
    "SparseGP",
    "SparseGPPredictor",
    "SquaredExponential",
    "SquaredExponentialARD",
    "StructuredGPConfig",
    # Structured GP
    "StructuredRocketGP",
    "SumKernel",
    # Feature Extraction
    "TranslationalFeatureExtractor",
    "WhiteNoise",
    "create_fast_gp",
    "create_matern_kernel",
    "create_se_ard_kernel",
]
