"""
Dispersion Analysis Module

Analyzes GP-MPC robustness under various disturbances:
1. Wind disturbances (constant, gusts, turbulence)
2. Aerodynamic uncertainty (Cd variation)
3. Thrust dispersion (magnitude, direction)
4. Initial condition dispersions
5. Combined dispersions

Generates dispersion ellipses and Monte Carlo scatter plots.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class WindModel(Enum):
    """Wind disturbance models."""

    NONE = auto()
    CONSTANT = auto()  # Constant wind velocity
    GUST = auto()  # Discrete gusts
    DRYDEN = auto()  # Dryden turbulence
    CUSTOM = auto()  # User-defined


@dataclass
class WindConfig:
    """Wind disturbance configuration."""

    model: WindModel = WindModel.NONE

    # Constant wind
    wind_velocity: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # m/s

    # Gust parameters
    gust_amplitude: float = 10.0  # m/s
    gust_duration: float = 2.0  # s
    gust_probability: float = 0.1  # per second

    # Dryden turbulence
    turbulence_intensity: float = 1.0  # Low: 0.5, Med: 1.0, High: 2.0
    altitude_ref: float = 500.0  # Reference altitude

    def get_wind(
        self,
        t: float,
        pos: NDArray,
        seed: Optional[int] = None,
    ) -> NDArray:
        """
        Get wind velocity at given time and position.

        Args:
            t: Current time
            pos: Position [x, y, z]
            seed: Random seed for stochastic models

        Returns:
            Wind velocity [wx, wy, wz]
        """
        if self.model == WindModel.NONE:
            return np.zeros(3)

        elif self.model == WindModel.CONSTANT:
            return self.wind_velocity.copy()

        elif self.model == WindModel.GUST:
            # Discrete gusts at random times
            if seed is not None:
                np.random.seed(seed + int(t * 10))

            if np.random.rand() < self.gust_probability * 0.1:  # Per call
                direction = np.random.randn(3)
                direction /= np.linalg.norm(direction) + 1e-6
                return direction * self.gust_amplitude
            return np.zeros(3)

        elif self.model == WindModel.DRYDEN:
            # Simplified Dryden turbulence model
            altitude = max(pos[2], 10)
            Lu = 200 * (altitude / self.altitude_ref) ** 0.33  # noqa: F841
            sigma_u = self.turbulence_intensity * 3.0

            # Low-pass filtered white noise (simplified)
            if seed is not None:
                np.random.seed(seed + int(t * 100))

            wind = np.random.randn(3) * sigma_u
            # Add altitude dependence
            wind *= min(1.0, altitude / 100)

            return wind

        return np.zeros(3)


@dataclass
class AeroDispersionConfig:
    """Aerodynamic dispersion configuration."""

    # Drag coefficient uncertainty
    Cd_nominal: float = 0.5
    Cd_std: float = 0.1  # Standard deviation

    # Reference area uncertainty
    A_nominal: float = 1.0  # m^2
    A_std: float = 0.05

    # Center of pressure offset
    cp_offset_std: float = 0.05  # m

    enabled: bool = False

    def sample(self, seed: Optional[int] = None) -> Dict[str, float]:
        """Sample aerodynamic parameters."""
        if seed is not None:
            np.random.seed(seed)

        if not self.enabled:
            return {
                "Cd": self.Cd_nominal,
                "A": self.A_nominal,
                "cp_offset": np.zeros(3),
            }

        return {
            "Cd": self.Cd_nominal + np.random.randn() * self.Cd_std,
            "A": self.A_nominal + np.random.randn() * self.A_std,
            "cp_offset": np.random.randn(3) * self.cp_offset_std,
        }


@dataclass
class ThrustDispersionConfig:
    """Thrust dispersion configuration."""

    # Magnitude dispersion
    thrust_scale_std: float = 0.02  # 2% thrust uncertainty

    # Direction dispersion (misalignment)
    misalignment_std: float = 0.01  # rad (~0.5 deg)

    # Thrust fluctuation (high frequency)
    fluctuation_std: float = 0.01  # Per-step variation

    enabled: bool = False

    def apply_dispersion(
        self,
        thrust_cmd: NDArray,
        seed: Optional[int] = None,
    ) -> NDArray:
        """
        Apply thrust dispersion to commanded thrust.

        Args:
            thrust_cmd: Commanded thrust [T, gimbal_x, gimbal_y]
            seed: Random seed

        Returns:
            Actual thrust with dispersion
        """
        if not self.enabled:
            return thrust_cmd.copy()

        if seed is not None:
            np.random.seed(seed)

        thrust = thrust_cmd.copy()

        # Magnitude dispersion
        scale = 1.0 + np.random.randn() * self.thrust_scale_std
        thrust[0] *= scale

        # Misalignment
        thrust[1] += np.random.randn() * self.misalignment_std
        thrust[2] += np.random.randn() * self.misalignment_std

        # Fluctuation
        thrust[0] *= 1.0 + np.random.randn() * self.fluctuation_std

        return thrust


@dataclass
class InitialConditionDispersion:
    """Initial condition dispersion configuration."""

    # Position dispersions (m)
    x_std: float = 10.0
    y_std: float = 10.0
    z_std: float = 20.0

    # Velocity dispersions (m/s)
    vx_std: float = 5.0
    vy_std: float = 5.0
    vz_std: float = 10.0

    # Mass dispersion (kg)
    mass_std: float = 0.05

    def sample(
        self,
        nominal: NDArray,
        seed: Optional[int] = None,
    ) -> NDArray:
        """Sample dispersed initial condition."""
        if seed is not None:
            np.random.seed(seed)

        dispersed = nominal.copy()

        dispersed[0] += np.random.randn() * self.mass_std
        dispersed[1] += np.random.randn() * self.x_std
        dispersed[2] += np.random.randn() * self.y_std
        dispersed[3] += np.random.randn() * self.z_std
        dispersed[4] += np.random.randn() * self.vx_std
        dispersed[5] += np.random.randn() * self.vy_std
        dispersed[6] += np.random.randn() * self.vz_std

        return dispersed


@dataclass
class DispersionConfig:
    """Combined dispersion configuration."""

    wind: WindConfig = field(default_factory=WindConfig)
    aero: AeroDispersionConfig = field(default_factory=AeroDispersionConfig)
    thrust: ThrustDispersionConfig = field(default_factory=ThrustDispersionConfig)
    initial: InitialConditionDispersion = field(default_factory=InitialConditionDispersion)

    # Dispersion level presets
    @classmethod
    def nominal(cls) -> "DispersionConfig":
        """No dispersions (ideal conditions)."""
        return cls()

    @classmethod
    def low(cls) -> "DispersionConfig":
        """Low dispersion level."""
        return cls(
            wind=WindConfig(
                model=WindModel.CONSTANT,
                wind_velocity=np.array([2.0, 1.0, 0.0]),
            ),
            aero=AeroDispersionConfig(Cd_std=0.05, enabled=True),
            thrust=ThrustDispersionConfig(thrust_scale_std=0.01, enabled=True),
        )

    @classmethod
    def medium(cls) -> "DispersionConfig":
        """Medium dispersion level."""
        return cls(
            wind=WindConfig(
                model=WindModel.DRYDEN,
                turbulence_intensity=1.0,
            ),
            aero=AeroDispersionConfig(Cd_std=0.1, enabled=True),
            thrust=ThrustDispersionConfig(thrust_scale_std=0.02, misalignment_std=0.01, enabled=True),
        )

    @classmethod
    def high(cls) -> "DispersionConfig":
        """High dispersion level."""
        return cls(
            wind=WindConfig(
                model=WindModel.DRYDEN,
                turbulence_intensity=2.0,
            ),
            aero=AeroDispersionConfig(Cd_std=0.2, enabled=True),
            thrust=ThrustDispersionConfig(
                thrust_scale_std=0.05, misalignment_std=0.02, fluctuation_std=0.02, enabled=True
            ),
        )


class DispersedDynamics:
    """
    Dynamics wrapper that applies dispersions.

    Wraps base dynamics with wind, aero, and thrust disturbances.
    """

    def __init__(
        self,
        base_dynamics,
        dispersion_config: DispersionConfig,
        seed: int = 42,
    ):
        """
        Initialize dispersed dynamics.

        Args:
            base_dynamics: Base rocket dynamics
            dispersion_config: Dispersion configuration
            seed: Random seed
        """
        self.base = base_dynamics
        self.config = dispersion_config
        self.seed = seed
        self.step_count = 0

        # Sample fixed dispersions
        self.aero_params = self.config.aero.sample(seed)

        # Copy attributes from base
        self.n_x = getattr(base_dynamics, "n_x", 7)
        self.n_u = getattr(base_dynamics, "n_u", 3)
        if hasattr(base_dynamics, "params"):
            self.params = base_dynamics.params

    def step(self, x: NDArray, u: NDArray, dt: float) -> NDArray:
        """
        Step dynamics with dispersions.

        Args:
            x: Current state
            u: Control input
            dt: Time step

        Returns:
            Next state with dispersions applied
        """
        self.step_count += 1

        # Apply thrust dispersion
        u_actual = self.config.thrust.apply_dispersion(u, seed=self.seed + self.step_count)

        # Base dynamics step
        x_next = self.base.step(x, u_actual, dt)

        # Apply wind disturbance
        t = self.step_count * dt
        pos = x[1:4]
        wind = self.config.wind.get_wind(t, pos, self.seed + self.step_count + 1000)

        # Wind affects velocity
        x_next[4:7] += wind * dt

        # Apply aerodynamic dispersion
        if self.config.aero.enabled:
            # Simplified: extra drag
            vel = x[4:7]
            speed = np.linalg.norm(vel)
            if speed > 1.0:
                Cd = self.aero_params["Cd"]
                A = self.aero_params["A"]
                rho = 0.02  # Mars atmosphere
                drag = 0.5 * rho * Cd * A * speed**2
                drag_accel = drag / x[0] * (vel / speed)
                x_next[4:7] -= drag_accel * dt

        return x_next

    def continuous_dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        """Get continuous dynamics (for linearization)."""
        return self.base.continuous_dynamics(x, u)

    def reset(self) -> None:
        """Reset step counter."""
        self.step_count = 0


@dataclass
class DispersionAnalysisResults:
    """Results from dispersion analysis."""

    config_name: str
    dispersion_config: DispersionConfig

    # Landing accuracy
    final_positions: NDArray  # (N, 3)
    final_velocities: NDArray  # (N, 3)

    # Statistics
    position_mean: NDArray
    position_cov: NDArray
    velocity_mean: NDArray
    velocity_cov: NDArray

    # Success metrics
    n_runs: int
    n_success: int
    success_rate: float

    def get_dispersion_ellipse(
        self,
        confidence: float = 0.95,
    ) -> Tuple[NDArray, float, float, float]:
        """
        Compute dispersion ellipse for landing positions.

        Args:
            confidence: Confidence level

        Returns:
            center, semi_major, semi_minor, angle
        """
        from scipy import stats  # noqa: PLC0415

        # Use horizontal positions (x, y)
        positions_2d = self.final_positions[:, :2]

        center = np.mean(positions_2d, axis=0)
        cov_2d = np.cov(positions_2d.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)

        # Chi-squared value for confidence level
        chi2 = stats.chi2.ppf(confidence, 2)

        # Semi-axes
        semi_major = np.sqrt(eigenvalues[1] * chi2)
        semi_minor = np.sqrt(eigenvalues[0] * chi2)

        # Angle of major axis
        angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

        return center, semi_major, semi_minor, angle

    def get_3sigma_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get 3-sigma bounds for all state components."""
        bounds = {}

        for i, name in enumerate(["x", "y", "z"]):
            mean = self.position_mean[i]
            std = np.sqrt(self.position_cov[i, i])
            bounds[f"pos_{name}"] = (mean - 3 * std, mean + 3 * std)

        for i, name in enumerate(["vx", "vy", "vz"]):
            mean = self.velocity_mean[i]
            std = np.sqrt(self.velocity_cov[i, i])
            bounds[f"vel_{name}"] = (mean - 3 * std, mean + 3 * std)

        return bounds


class DispersionAnalysis:
    """
    Comprehensive dispersion analysis framework.

    Runs Monte Carlo with various dispersion levels and analyzes results.

    Example:
        >>> analyzer = DispersionAnalysis(dynamics, controller)
        >>> results = analyzer.run_sweep()
        >>> analyzer.plot_dispersion_ellipses(results)
    """

    def __init__(
        self,
        dynamics,
        controller,
        mc_config=None,
    ):
        """
        Initialize dispersion analysis.

        Args:
            dynamics: Base rocket dynamics
            controller: MPC controller
            mc_config: Monte Carlo configuration
        """
        self.base_dynamics = dynamics
        self.controller = controller
        self.mc_config = mc_config

    def run_single(
        self,
        dispersion_config: DispersionConfig,
        n_runs: int = 100,
        seed: int = 42,
    ) -> DispersionAnalysisResults:
        """
        Run analysis with single dispersion configuration.

        Args:
            dispersion_config: Dispersion settings
            n_runs: Number of Monte Carlo runs
            seed: Random seed

        Returns:
            Analysis results
        """
        from .monte_carlo import MonteCarloSimulator, SimulationConfig  # noqa: PLC0415

        final_positions = []
        final_velocities = []
        n_success = 0

        for i in range(n_runs):
            # Create dispersed dynamics for this run
            dispersed = DispersedDynamics(self.base_dynamics, dispersion_config, seed=seed + i)

            # Run simulation
            mc_config = self.mc_config or SimulationConfig()
            sim = MonteCarloSimulator(dispersed, self.controller, mc_config)

            # Sample initial condition with dispersion
            x0 = sim.sample_initial_condition(seed=seed + i)
            x0 = dispersion_config.initial.sample(x0, seed=seed + i + 10000)

            result = sim.run_single(i, x0=x0)

            if result.success:
                n_success += 1

            final_positions.append(result.states[-1, 1:4])
            final_velocities.append(result.states[-1, 4:7])

        final_positions = np.array(final_positions)
        final_velocities = np.array(final_velocities)

        return DispersionAnalysisResults(
            config_name="custom",
            dispersion_config=dispersion_config,
            final_positions=final_positions,
            final_velocities=final_velocities,
            position_mean=np.mean(final_positions, axis=0),
            position_cov=np.cov(final_positions.T),
            velocity_mean=np.mean(final_velocities, axis=0),
            velocity_cov=np.cov(final_velocities.T),
            n_runs=n_runs,
            n_success=n_success,
            success_rate=n_success / n_runs,
        )

    def run_sweep(
        self,
        n_runs_per_level: int = 100,
        seed: int = 42,
    ) -> Dict[str, DispersionAnalysisResults]:
        """
        Run analysis across dispersion levels.

        Args:
            n_runs_per_level: Runs per dispersion level
            seed: Random seed

        Returns:
            Results per dispersion level
        """
        levels = {
            "nominal": DispersionConfig.nominal(),
            "low": DispersionConfig.low(),
            "medium": DispersionConfig.medium(),
            "high": DispersionConfig.high(),
        }

        results = {}

        print("Dispersion Analysis Sweep")
        print("=" * 50)

        for name, config in levels.items():
            print(f"\nRunning {name} dispersion level...")
            result = self.run_single(config, n_runs_per_level, seed)
            result.config_name = name
            results[name] = result
            print(f"  Success rate: {result.success_rate * 100:.1f}%")
            print(f"  Position std: {np.sqrt(np.diag(result.position_cov))}")

        return results

    def summary_table(
        self,
        results: Dict[str, DispersionAnalysisResults],
    ) -> str:
        """Generate summary table."""
        lines = [
            "=" * 70,
            "Dispersion Analysis Summary",
            "=" * 70,
            f"{'Level':<12} {'Success%':>10} {'Pos σ_xy (m)':>15} {'Pos σ_z (m)':>12} {'Vel σ (m/s)':>12}",  # noqa: RUF001
            "-" * 70,
        ]

        for name, result in results.items():
            pos_std_xy = np.sqrt(result.position_cov[0, 0] + result.position_cov[1, 1])
            pos_std_z = np.sqrt(result.position_cov[2, 2])
            vel_std = np.sqrt(np.trace(result.velocity_cov))

            lines.append(
                f"{name:<12} {result.success_rate * 100:>9.1f}% {pos_std_xy:>14.2f} {pos_std_z:>11.2f} {vel_std:>11.2f}"
            )

        lines.append("=" * 70)

        return "\n".join(lines)
