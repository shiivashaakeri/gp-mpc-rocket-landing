"""
Structured Gaussian Process for Rocket Dynamics

Physics-informed GP that only learns residuals for acceleration terms:
- d_v ∈ R³: Translational acceleration residual
- d_ω ∈ R³: Rotational acceleration residual

This exploits the structure of rocket dynamics:
- Mass dynamics: ṁ = - alpha ||T|| (exact, no residual needed)
- Position kinematics: ṙ = v (exact)
- Attitude kinematics: q̇ = 0.5 Ω(ω) q (exact)

Only the acceleration equations need learning:
- v̇ = (1/m) C_{I/B} T + g + d_v(x, u)
- ω̇ = J^{-1} (r_T x T - ω x Jω) + d_ω(x, u)

This reduces from 14 outputs to 6 outputs, significantly
reducing GP complexity and improving sample efficiency.

Reference:
    Torrente, G., et al. (2021). Data-driven MPC for quadrotors.
    Uses similar structured learning approach.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .exact_gp import MultiOutputExactGP
from .features import (
    AtmosphereModel,
    CombinedFeatureExtractor,
)
from .kernels import SquaredExponentialARD
from .sparse_gp import MultiOutputSparseGP


@dataclass
class StructuredGPConfig:
    """Configuration for structured rocket GP."""

    # GP settings
    n_inducing: int = 50
    noise_variance: float = 1e-4
    use_sparse: bool = True

    # Feature extraction
    reference_velocity: float = 10.0
    include_altitude: bool = True
    include_density: bool = True

    # Kernel settings
    signal_variance: float = 0.1  # Prior on residual magnitude
    lengthscales_translational: Optional[NDArray] = None
    lengthscales_rotational: Optional[NDArray] = None

    # Data management
    max_data_points: int = 1000
    novelty_threshold: float = 0.1


class StructuredRocketGP:
    """
    Physics-structured GP for 6-DoF rocket dynamics.

    Only learns 6 residual outputs instead of 14 state derivatives.
    Uses separate GPs for translational and rotational residuals
    with physics-informed feature extraction.

    The full dynamics model becomes:
        ẋ = f_nominal(x, u) + G(x) @ d(x, u)

    where:
        - f_nominal: Nominal dynamics from physics
        - G(x): Selection matrix (maps 6 residuals to 14 state derivatives)
        - d(x, u) = [d_v; d_ω]: Learned residuals

    Example:
        >>> gp = StructuredRocketGP(config)
        >>> gp.add_data(X, U, D_v, D_omega)
        >>> d_v_pred, d_omega_pred, sigma_v, sigma_omega = gp.predict(x, u)
    """

    def __init__(
        self,
        config: Optional[StructuredGPConfig] = None,
    ):
        """
        Initialize structured GP.

        Args:
            config: Configuration options
        """
        self.config = config or StructuredGPConfig()

        # Feature extractors
        self.feature_extractor = CombinedFeatureExtractor(
            atmosphere=AtmosphereModel(),
            reference_velocity=self.config.reference_velocity,
        )

        n_feat_v = self.feature_extractor.n_features_translational
        n_feat_omega = self.feature_extractor.n_features_rotational

        # Create kernels
        if self.config.lengthscales_translational is None:
            ls_v = np.ones(n_feat_v)
        else:
            ls_v = self.config.lengthscales_translational

        if self.config.lengthscales_rotational is None:
            ls_omega = np.ones(n_feat_omega)
        else:
            ls_omega = self.config.lengthscales_rotational

        kernel_v = SquaredExponentialARD(  # noqa: F841
            input_dim=n_feat_v,
            signal_variance=self.config.signal_variance,
            lengthscales=ls_v,
        )

        kernel_omega = SquaredExponentialARD(  # noqa: F841
            input_dim=n_feat_omega,
            signal_variance=self.config.signal_variance,
            lengthscales=ls_omega,
        )

        # Create GPs for each residual component
        if self.config.use_sparse:
            self.gp_v = MultiOutputSparseGP(
                input_dim=n_feat_v,
                output_dim=3,  # d_v ∈ R³
                n_inducing=self.config.n_inducing,
                noise_variance=self.config.noise_variance,
            )
            self.gp_omega = MultiOutputSparseGP(
                input_dim=n_feat_omega,
                output_dim=3,  # d_ω ∈ R³
                n_inducing=self.config.n_inducing,
                noise_variance=self.config.noise_variance,
            )
        else:
            self.gp_v = MultiOutputExactGP(
                input_dim=n_feat_v,
                output_dim=3,
                noise_variance=self.config.noise_variance,
            )
            self.gp_omega = MultiOutputExactGP(
                input_dim=n_feat_omega,
                output_dim=3,
                noise_variance=self.config.noise_variance,
            )

        # Data storage
        self.X_data: list = []
        self.U_data: list = []
        self.D_v_data: list = []
        self.D_omega_data: list = []
        self._is_fitted: bool = False

    @property
    def n_data(self) -> int:
        """Number of stored data points."""
        return len(self.X_data)

    def add_data(
        self,
        X: NDArray,
        U: NDArray,
        D_v: NDArray,
        D_omega: NDArray,
    ) -> None:
        """
        Add training data.

        Args:
            X: States (N, 14)
            U: Controls (N, 3)
            D_v: Translational residuals (N, 3)
            D_omega: Rotational residuals (N, 3)
        """
        X = np.atleast_2d(X)
        U = np.atleast_2d(U)
        D_v = np.atleast_2d(D_v)
        D_omega = np.atleast_2d(D_omega)

        for i in range(X.shape[0]):
            self.X_data.append(X[i])
            self.U_data.append(U[i])
            self.D_v_data.append(D_v[i])
            self.D_omega_data.append(D_omega[i])

        # Limit data size
        if len(self.X_data) > self.config.max_data_points:
            # Remove oldest data
            excess = len(self.X_data) - self.config.max_data_points
            self.X_data = self.X_data[excess:]
            self.U_data = self.U_data[excess:]
            self.D_v_data = self.D_v_data[excess:]
            self.D_omega_data = self.D_omega_data[excess:]

        self._is_fitted = False

    def fit(self) -> None:
        """Fit GPs to stored data."""
        if self.n_data == 0:
            raise RuntimeError("No data to fit")

        X = np.array(self.X_data)
        U = np.array(self.U_data)
        D_v = np.array(self.D_v_data)
        D_omega = np.array(self.D_omega_data)

        # Extract features
        Z_v = self.feature_extractor.extract_batch_translational(X, U)
        Z_omega = self.feature_extractor.extract_batch_rotational(X, U)

        # Fit GPs
        self.gp_v.fit(Z_v, D_v)
        self.gp_omega.fit(Z_omega, D_omega)

        self._is_fitted = True

    def predict(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Predict residuals at a single point.

        Args:
            x: State (14,)
            u: Control (3,)

        Returns:
            d_v_mean: Translational residual mean (3,)
            d_omega_mean: Rotational residual mean (3,)
            d_v_var: Translational residual variance (3,)
            d_omega_var: Rotational residual variance (3,)
        """
        if not self._is_fitted:
            if self.n_data > 0:
                self.fit()
            else:
                # Return zero residuals with prior variance
                zero = np.zeros(3)
                prior_var = np.full(3, self.config.signal_variance)
                return zero, zero, prior_var, prior_var

        # Extract features
        z_v = self.feature_extractor.extract_translational(x, u)
        z_omega = self.feature_extractor.extract_rotational(x, u)

        # Predict
        d_v_mean, d_v_var = self.gp_v.predict(z_v.reshape(1, -1))
        d_omega_mean, d_omega_var = self.gp_omega.predict(z_omega.reshape(1, -1))

        return (
            d_v_mean.flatten(),
            d_omega_mean.flatten(),
            d_v_var.flatten(),
            d_omega_var.flatten(),
        )

    def predict_batch(
        self,
        X: NDArray,
        U: NDArray,
    ) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Predict residuals for batch of points.

        Args:
            X: States (N, 14)
            U: Controls (N, 3)

        Returns:
            d_v_mean: (N, 3)
            d_omega_mean: (N, 3)
            d_v_var: (N, 3)
            d_omega_var: (N, 3)
        """
        if not self._is_fitted:
            if self.n_data > 0:
                self.fit()
            else:
                N = X.shape[0]
                zero = np.zeros((N, 3))
                prior_var = np.full((N, 3), self.config.signal_variance)
                return zero, zero, prior_var, prior_var

        # Extract features
        Z_v = self.feature_extractor.extract_batch_translational(X, U)
        Z_omega = self.feature_extractor.extract_batch_rotational(X, U)

        # Predict
        d_v_mean, d_v_var = self.gp_v.predict(Z_v)
        d_omega_mean, d_omega_var = self.gp_omega.predict(Z_omega)

        return d_v_mean, d_omega_mean, d_v_var, d_omega_var

    def get_full_residual(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Tuple[NDArray, NDArray]:
        """
        Get full 14-dimensional residual with uncertainty.

        Maps the 6D residual back to 14D state derivative space.

        Args:
            x: State (14,)
            u: Control (3,)

        Returns:
            d_mean: Residual mean (14,)
            d_var: Residual variance (14,)
        """
        d_v_mean, d_omega_mean, d_v_var, d_omega_var = self.predict(x, u)

        # Map to 14D: d affects [v̇, ω̇] only
        # State: [m, r, v, q, ω] indices: [0, 1:4, 4:7, 7:11, 11:14]
        d_mean = np.zeros(14)
        d_var = np.zeros(14)

        d_mean[4:7] = d_v_mean  # v̇ residual
        d_mean[11:14] = d_omega_mean  # ω̇ residual

        d_var[4:7] = d_v_var
        d_var[11:14] = d_omega_var

        return d_mean, d_var

    def is_novel(self, x: NDArray, u: NDArray) -> bool:
        """
        Check if point is novel (high uncertainty).

        Used for online data selection.

        Args:
            x: State (14,)
            u: Control (3,)

        Returns:
            True if point has high uncertainty
        """
        _, _, d_v_var, d_omega_var = self.predict(x, u)

        # Novelty based on variance relative to prior
        prior_var = self.config.signal_variance
        max_var = max(np.max(d_v_var), np.max(d_omega_var))

        return bool(max_var > self.config.novelty_threshold * prior_var)

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """
        Optimize kernel hyperparameters.

        Returns:
            Optimization results dict
        """
        # For sparse GPs, this would optimize via marginal likelihood
        # Simplified: just return current state
        return {
            "n_data": self.n_data,
            "is_fitted": self._is_fitted,
        }

    def save(self, path: str) -> None:
        """
        Save GP model to file.

        Args:
            path: File path
        """
        data = {
            "config": self.config,
            "X_data": self.X_data,
            "U_data": self.U_data,
            "D_v_data": self.D_v_data,
            "D_omega_data": self.D_omega_data,
            "is_fitted": self._is_fitted,
        }
        np.save(path, data, allow_pickle=True)

    def load(self, path: str) -> None:
        """
        Load GP model from file.

        Args:
            path: File path
        """
        data = np.load(path, allow_pickle=True).item()
        self.X_data = data["X_data"]
        self.U_data = data["U_data"]
        self.D_v_data = data["D_v_data"]
        self.D_omega_data = data["D_omega_data"]

        if data["is_fitted"]:
            self.fit()

    def __repr__(self) -> str:
        return (
            f"StructuredRocketGP(n_data={self.n_data}, n_inducing={self.config.n_inducing}, fitted={self._is_fitted})"
        )


class Simple3DoFGP:
    """
    Simplified GP for 3-DoF rocket (point mass).

    Only learns translational acceleration residual d_v ∈ R³.

    Useful for algorithm development before moving to 6-DoF.
    """

    def __init__(
        self,
        n_inducing: int = 50,
        noise_variance: float = 1e-4,
        use_sparse: bool = True,
    ):
        from .features import Simple3DoFFeatureExtractor  # noqa: PLC0415

        self.feature_extractor = Simple3DoFFeatureExtractor()
        n_feat = self.feature_extractor.n_features

        if use_sparse:
            self.gp = MultiOutputSparseGP(
                input_dim=n_feat,
                output_dim=3,
                n_inducing=n_inducing,
                noise_variance=noise_variance,
            )
        else:
            self.gp = MultiOutputExactGP(
                input_dim=n_feat,
                output_dim=3,
                noise_variance=noise_variance,
            )

        self.X_data: list = []
        self.U_data: list = []
        self.D_data: list = []
        self._is_fitted = False

    @property
    def n_data(self) -> int:
        return len(self.X_data)

    def add_data(self, X: NDArray, U: NDArray, D: NDArray) -> None:
        """Add training data."""
        X = np.atleast_2d(X)
        U = np.atleast_2d(U)
        D = np.atleast_2d(D)

        for i in range(X.shape[0]):
            self.X_data.append(X[i])
            self.U_data.append(U[i])
            self.D_data.append(D[i])

        self._is_fitted = False

    def fit(self) -> None:
        """Fit GP to data."""
        if self.n_data == 0:
            raise RuntimeError("No data")

        X = np.array(self.X_data)
        U = np.array(self.U_data)
        D = np.array(self.D_data)

        Z = self.feature_extractor.extract_batch(X, U)
        self.gp.fit(Z, D)
        self._is_fitted = True

    def predict(self, x: NDArray, u: NDArray) -> Tuple[NDArray, NDArray]:
        """Predict residual."""
        if not self._is_fitted:
            if self.n_data > 0:
                self.fit()
            else:
                return np.zeros(3), np.ones(3) * 0.1

        z = self.feature_extractor.extract(x, u)
        mean, var = self.gp.predict(z.reshape(1, -1))
        return mean.flatten(), var.flatten()

    def __repr__(self) -> str:
        return f"Simple3DoFGP(n_data={self.n_data}, fitted={self._is_fitted})"
