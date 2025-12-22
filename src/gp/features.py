"""
Feature Extraction for Rocket Dynamics GP

Extracts relevant features from rocket state for GP learning of residuals.
The key insight is that we don't need all 14 state dimensions - we can
construct informative features that capture aerodynamic effects.

Translational residual features (for d_v):
    z_v = [v_I, q_dyn, alpha, beta, T_B, h, rho]

    - v_I: Velocity (affects Reynolds number, dynamic pressure)
    - q_dyn: Dynamic pressure = 0.5 * rho * ||v||²
    - alpha, beta: Angle of attack and sideslip (aero angles)
    - T_B: Thrust vector (affects induced drag, plume effects)
    - h: Altitude (affects air density)
    - rho: Air density at altitude

Rotational residual features (for d_ω):
    z_ω = [ω_B, T_B, q_dyn, alpha, beta, v_B]

    - ω_B: Angular velocity (damping effects)
    - T_B: Thrust (gimbal torque errors)
    - q_dyn, alpha, beta: Aero torques
    - v_B: Velocity in body frame (aero torques)

Reference:
    Physics-informed feature selection reduces GP input dimension
    while preserving predictive accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Atmospheric Model
# =============================================================================


@dataclass
class AtmosphereModel:
    """
    Simple exponential atmosphere model.

    rho(h) = rho_0 * exp(-h / H)

    where H is the scale height.
    """

    rho_0: float = 1.225  # Sea level density [kg/m³]
    scale_height: float = 8500.0  # Scale height [m]

    def density(self, altitude: float) -> float:
        """Get air density at altitude."""
        return self.rho_0 * np.exp(-altitude / self.scale_height)

    def density_array(self, altitude: NDArray) -> NDArray:
        """Vectorized density computation."""
        return self.rho_0 * np.exp(-altitude / self.scale_height)


# =============================================================================
# Feature Extractors
# =============================================================================


class RocketFeatureExtractor:
    """
    Base class for rocket feature extraction.

    Extracts features from (x, u) pairs for GP learning.
    """

    def __init__(
        self,
        atmosphere: Optional[AtmosphereModel] = None,
        include_altitude: bool = True,
        include_density: bool = True,
        reference_velocity: float = 10.0,  # For normalization
    ):
        """
        Initialize feature extractor.

        Args:
            atmosphere: Atmospheric model for density
            include_altitude: Include altitude as feature
            include_density: Include air density as feature
            reference_velocity: Reference velocity for normalization
        """
        self.atmosphere = atmosphere or AtmosphereModel()
        self.include_altitude = include_altitude
        self.include_density = include_density
        self.v_ref = reference_velocity

        # Compute feature dimension
        self._compute_feature_dim()

    def _compute_feature_dim(self) -> None:
        """Compute output feature dimension."""
        raise NotImplementedError

    @property
    def n_features(self) -> int:
        """Number of output features."""
        return self._n_features

    @property
    def feature_names(self) -> List[str]:
        """Names of features for interpretability."""
        return self._feature_names

    def extract(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Extract features from state-control pair.

        Args:
            x: State vector (14,) for 6-DoF or (7,) for 3-DoF
            u: Control vector (3,)

        Returns:
            Feature vector (n_features,)
        """
        raise NotImplementedError

    def extract_batch(self, X: NDArray, U: NDArray) -> NDArray:
        """
        Extract features from batch of state-control pairs.

        Args:
            X: States (N, n_x)
            U: Controls (N, n_u)

        Returns:
            Features (N, n_features)
        """
        N = X.shape[0]
        Z = np.zeros((N, self.n_features))

        for i in range(N):
            Z[i] = self.extract(X[i], U[i])

        return Z


class TranslationalFeatureExtractor(RocketFeatureExtractor):
    """
    Feature extractor for translational dynamics residual d_v.

    Features:
        - Velocity magnitude and direction
        - Dynamic pressure
        - Angle of attack (alpha) and sideslip (beta)
        - Thrust vector
        - Altitude and density

    These capture the physics of aerodynamic drag and thrust effects.
    """

    def _compute_feature_dim(self) -> None:
        """Compute feature dimension."""
        # Base features: v_I (3), |v|, q_dyn, alpha, beta, T_B (3), |T|
        n = 3 + 1 + 1 + 1 + 1 + 3 + 1  # = 11

        if self.include_altitude:
            n += 1
        if self.include_density:
            n += 1

        self._n_features = n

        # Feature names
        self._feature_names = [
            "v_x",
            "v_y",
            "v_z",
            "speed",
            "q_dyn",
            "alpha",
            "beta",
            "T_x",
            "T_y",
            "T_z",
            "T_mag",
        ]
        if self.include_altitude:
            self._feature_names.append("altitude")
        if self.include_density:
            self._feature_names.append("density")

    def extract(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Extract translational features.

        Args:
            x: State vector (14,) [m, r_I(3), v_I(3), q(4), ω(3)]
            u: Thrust in body frame (3,)

        Returns:
            Feature vector (n_features,)
        """
        # Parse state
        r_I = x[1:4]  # Position (inertial)
        v_I = x[4:7]  # Velocity (inertial)
        q = x[7:11]  # Quaternion

        altitude = r_I[0]  # x is up in UEN
        speed = np.linalg.norm(v_I)

        # Dynamic pressure
        rho = self.atmosphere.density(altitude)
        q_dyn = 0.5 * rho * speed**2

        # Rotation matrix (body from inertial)
        C_BI = self._quat_to_dcm(q)

        # Velocity in body frame
        v_B = C_BI @ v_I

        # Angle of attack and sideslip
        if speed > 1e-3:
            # alpha = atan2(v_z, v_x) in body frame
            # β = asin(v_y / |v|)
            alpha = np.arctan2(-v_B[2], v_B[0])  # Note: depends on convention
            beta = np.arcsin(np.clip(v_B[1] / speed, -1, 1))
        else:
            alpha = 0.0
            beta = 0.0

        # Thrust
        T_B = u
        T_mag = np.linalg.norm(T_B)

        # Assemble features (normalized)
        features = [
            v_I[0] / self.v_ref,
            v_I[1] / self.v_ref,
            v_I[2] / self.v_ref,
            speed / self.v_ref,
            q_dyn / (0.5 * self.atmosphere.rho_0 * self.v_ref**2),  # Normalized
            alpha,  # Already in radians [-π, π]
            beta,  # Already in radians [-π/2, π/2]
            T_B[0] / 10.0,  # Assume max ~10 N for normalization
            T_B[1] / 10.0,
            T_B[2] / 10.0,
            T_mag / 10.0,
        ]

        if self.include_altitude:
            features.append(altitude / 100.0)  # Normalize by 100m

        if self.include_density:
            features.append(rho / self.atmosphere.rho_0)

        return np.array(features)

    def _quat_to_dcm(self, q: NDArray) -> NDArray:
        """Convert quaternion [w,x,y,z] to DCM."""
        w, x, y, z = q

        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y + w * z), 2 * (x * z - w * y)],
                [2 * (x * y - w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z + w * x)],
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )


class RotationalFeatureExtractor(RocketFeatureExtractor):
    """
    Feature extractor for rotational dynamics residual d_ω.

    Features:
        - Angular velocity
        - Thrust vector (gimbal effects)
        - Velocity in body frame (aero torques)
        - Dynamic pressure
    """

    def _compute_feature_dim(self) -> None:
        """Compute feature dimension."""
        # ω_B (3), |ω|, T_B (3), v_B (3), |v|, q_dyn
        n = 3 + 1 + 3 + 3 + 1 + 1  # = 12

        self._n_features = n
        self._feature_names = [
            "omega_x",
            "omega_y",
            "omega_z",
            "omega_mag",
            "T_x",
            "T_y",
            "T_z",
            "v_Bx",
            "v_By",
            "v_Bz",
            "speed",
            "q_dyn",
        ]

    def extract(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Extract rotational features.

        Args:
            x: State vector (14,)
            u: Thrust in body frame (3,)

        Returns:
            Feature vector (n_features,)
        """
        r_I = x[1:4]
        v_I = x[4:7]
        q = x[7:11]
        omega_B = x[11:14]

        altitude = r_I[0]
        speed = np.linalg.norm(v_I)
        omega_mag = np.linalg.norm(omega_B)

        # Dynamic pressure
        rho = self.atmosphere.density(altitude)
        q_dyn = 0.5 * rho * speed**2

        # Velocity in body frame
        C_BI = self._quat_to_dcm(q)
        v_B = C_BI @ v_I

        # Thrust
        T_B = u

        # Assemble features (normalized)
        omega_ref = 1.0  # rad/s reference

        features = [
            omega_B[0] / omega_ref,
            omega_B[1] / omega_ref,
            omega_B[2] / omega_ref,
            omega_mag / omega_ref,
            T_B[0] / 10.0,
            T_B[1] / 10.0,
            T_B[2] / 10.0,
            v_B[0] / self.v_ref,
            v_B[1] / self.v_ref,
            v_B[2] / self.v_ref,
            speed / self.v_ref,
            q_dyn / (0.5 * self.atmosphere.rho_0 * self.v_ref**2),
        ]

        return np.array(features)

    def _quat_to_dcm(self, q: NDArray) -> NDArray:
        """Convert quaternion [w,x,y,z] to DCM."""
        w, x, y, z = q
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y + w * z), 2 * (x * z - w * y)],
                [2 * (x * y - w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z + w * x)],
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )


class Simple3DoFFeatureExtractor(RocketFeatureExtractor):
    """
    Simple feature extractor for 3-DoF rocket.

    For point-mass model, features are simpler:
        z = [v_I, |v|, q_dyn, T_I, |T|, altitude]
    """

    def _compute_feature_dim(self) -> None:
        """Compute feature dimension."""
        # v_I (3), |v|, q_dyn, T_I (3), |T| = 9
        n = 9
        if self.include_altitude:
            n += 1
        if self.include_density:
            n += 1

        self._n_features = n

        self._feature_names = [
            "v_x",
            "v_y",
            "v_z",
            "speed",
            "q_dyn",
            "T_x",
            "T_y",
            "T_z",
            "T_mag",
        ]
        if self.include_altitude:
            self._feature_names.append("altitude")
        if self.include_density:
            self._feature_names.append("density")

    def extract(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Extract features for 3-DoF model.

        Args:
            x: State vector (7,) [m, r_I(3), v_I(3)]
            u: Thrust in inertial frame (3,)

        Returns:
            Feature vector (n_features,)
        """
        r_I = x[1:4]
        v_I = x[4:7]

        altitude = r_I[0]
        speed = np.linalg.norm(v_I)

        # Dynamic pressure
        rho = self.atmosphere.density(altitude)
        q_dyn = 0.5 * rho * speed**2

        # Thrust
        T_mag = np.linalg.norm(u)

        features = [
            v_I[0] / self.v_ref,
            v_I[1] / self.v_ref,
            v_I[2] / self.v_ref,
            speed / self.v_ref,
            q_dyn / (0.5 * self.atmosphere.rho_0 * self.v_ref**2),
            u[0] / 10.0,
            u[1] / 10.0,
            u[2] / 10.0,
            T_mag / 10.0,
        ]

        if self.include_altitude:
            features.append(altitude / 100.0)
        if self.include_density:
            features.append(rho / self.atmosphere.rho_0)

        return np.array(features)


class CombinedFeatureExtractor:
    """
    Combined feature extractor for both translational and rotational.

    Provides single interface for extracting features for the
    structured GP that learns both d_v and d_ω.
    """

    def __init__(
        self,
        atmosphere: Optional[AtmosphereModel] = None,
        reference_velocity: float = 10.0,
    ):
        self.translational = TranslationalFeatureExtractor(
            atmosphere=atmosphere,
            reference_velocity=reference_velocity,
        )
        self.rotational = RotationalFeatureExtractor(
            atmosphere=atmosphere,
            reference_velocity=reference_velocity,
        )

    @property
    def n_features_translational(self) -> int:
        return self.translational.n_features

    @property
    def n_features_rotational(self) -> int:
        return self.rotational.n_features

    def extract_translational(self, x: NDArray, u: NDArray) -> NDArray:
        """Extract features for translational residual."""
        return self.translational.extract(x, u)

    def extract_rotational(self, x: NDArray, u: NDArray) -> NDArray:
        """Extract features for rotational residual."""
        return self.rotational.extract(x, u)

    def extract_batch_translational(self, X: NDArray, U: NDArray) -> NDArray:
        """Batch extract translational features."""
        return self.translational.extract_batch(X, U)

    def extract_batch_rotational(self, X: NDArray, U: NDArray) -> NDArray:
        """Batch extract rotational features."""
        return self.rotational.extract_batch(X, U)
