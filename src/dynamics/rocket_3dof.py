"""
3-DoF Rocket Dynamics Wrapper for GP-MPC

Wraps simdyn's Rocket3DoF with a consistent interface for the GP-MPC framework.
Simpler than 6-DoF, useful for algorithm prototyping and validation.

State vector (n=7):
    x = [m, r_x, r_y, r_z, v_x, v_y, v_z]

Control vector (m=3):
    u = [T_x, T_y, T_z]  (thrust in inertial frame)

Reference:
    Blackmore, L., Açikmeşe, B., & Scharf, D. P. (2010). Minimum-landing-error
    powered-descent guidance for Mars landing using convex optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from simdyn import (
    Rocket3DoF as SimdynRocket3DoF,
)
from simdyn import (
    Rocket3DoFParams,
)


@dataclass
class Rocket3DoFConfig:
    """Configuration for 3-DoF rocket dynamics."""

    # Mass properties
    m_dry: float = 1.0
    m_wet: float = 2.0

    # Propulsion
    I_sp: float = 30.0
    g0: float = 1.0
    T_min: float = 0.0
    T_max: float = 6.5

    # Environment
    g_I: Optional[NDArray] = None  # Gravity vector  # noqa: N815

    # Drag (optional)
    enable_drag: bool = False
    rho: float = 1.0
    C_D: float = 0.5
    A_ref: float = 0.5

    # Constraints
    gamma_gs: float = np.deg2rad(30.0)  # Glide slope angle
    v_max: float = np.inf  # Maximum velocity

    # Integration
    default_dt: float = 0.1

    def __post_init__(self):
        if self.g_I is None:
            self.g_I = np.array([-1.0, 0.0, 0.0])

    @classmethod
    def normalized_defaults(cls) -> "Rocket3DoFConfig":
        """Create config with normalized (Szmuk-like) parameters."""
        return cls()

    @classmethod
    def fuel_optimal_defaults(cls) -> "Rocket3DoFConfig":
        """Parameters suitable for fuel-optimal trajectory optimization."""
        return cls(
            m_wet=2.0,
            m_dry=1.0,
            T_min=0.3,  # Minimum thrust > 0 for convexification
            T_max=5.0,
            I_sp=300.0,
            g_I=np.array([-9.81, 0.0, 0.0]),
        )


class Rocket3DoFDynamics:
    """
    3-DoF rocket dynamics wrapper providing consistent interface for GP-MPC.

    This is a point-mass model suitable for:
    - Trajectory optimization prototyping
    - Algorithm validation before 6-DoF
    - Understanding fundamental powered descent dynamics

    Example:
        >>> config = Rocket3DoFConfig.normalized_defaults()
        >>> rocket = Rocket3DoFDynamics(config)
        >>> x0 = rocket.create_initial_state(altitude=10.0, mass=2.0)
        >>> u = rocket.hover_thrust(x0)
        >>> x_next = rocket.step(x0, u, dt=0.1)
    """

    # State indices
    IDX_MASS = 0
    IDX_POS = slice(1, 4)
    IDX_VEL = slice(4, 7)

    # State dimensions
    N_STATE = 7
    N_CONTROL = 3

    def __init__(self, config: Optional[Rocket3DoFConfig] = None):
        """
        Initialize 3-DoF rocket dynamics.

        Args:
            config: Configuration parameters. Uses normalized defaults if None.
        """
        self.config = config or Rocket3DoFConfig.normalized_defaults()

        # Create simdyn Rocket3DoF
        self._rocket = self._create_simdyn_rocket()

    def _create_simdyn_rocket(self) -> SimdynRocket3DoF:
        """Create the underlying simdyn Rocket3DoF instance."""
        cfg = self.config

        params = Rocket3DoFParams(
            m_dry=cfg.m_dry,
            m_wet=cfg.m_wet,
            I_sp=cfg.I_sp,
            g0=cfg.g0,
            g_vec=np.array(cfg.g_I),  # simdyn uses g_vec, we use g_I for consistency
            T_min=cfg.T_min,
            T_max=cfg.T_max,
            gamma_gs=cfg.gamma_gs,
            enable_drag=cfg.enable_drag,
            rho=cfg.rho if cfg.enable_drag else 0.0,
            C_D=cfg.C_D if cfg.enable_drag else 0.0,
            A_ref=cfg.A_ref if cfg.enable_drag else 1.0,
        )

        return SimdynRocket3DoF(params)

    @property
    def params(self) -> Rocket3DoFParams:
        """Access underlying simdyn parameters."""
        return self._rocket.params

    @property
    def n_state(self) -> int:
        """State dimension."""
        return self.N_STATE

    @property
    def n_control(self) -> int:
        """Control dimension."""
        return self.N_CONTROL

    # =========================================================================
    # State Creation and Manipulation
    # =========================================================================

    def pack_state(
        self,
        mass: float,
        position: NDArray,
        velocity: NDArray,
    ) -> NDArray:
        """
        Pack state components into state vector.

        Args:
            mass: Vehicle mass [kg]
            position: Position in inertial frame [m] (3,)
            velocity: Velocity in inertial frame [m/s] (3,)

        Returns:
            State vector (7,)
        """
        return self._rocket.pack_state(
            mass=mass,
            position=np.asarray(position),
            velocity=np.asarray(velocity),
        )

    def create_initial_state(
        self,
        altitude: float = 10.0,
        downrange: float = 0.0,
        crossrange: float = 0.0,
        velocity: Optional[NDArray] = None,
        mass: Optional[float] = None,
    ) -> NDArray:
        """
        Create initial state with convenient parameters.

        Args:
            altitude: Initial altitude (r_x in UEN) [m]
            downrange: Initial downrange (r_z in UEN) [m]
            crossrange: Initial crossrange (r_y in UEN) [m]
            velocity: Initial velocity [m/s], defaults to zero
            mass: Initial mass [kg], defaults to m_wet

        Returns:
            Initial state vector (7,)
        """
        position = np.array([altitude, crossrange, downrange])

        velocity = np.zeros(3) if velocity is None else np.asarray(velocity)

        if mass is None:
            mass = self.config.m_wet

        return self.pack_state(mass, position, velocity)

    # =========================================================================
    # State Accessors
    # =========================================================================

    def get_mass(self, x: NDArray) -> float:
        """Extract mass from state."""
        return self._rocket.get_mass(x)

    def get_position(self, x: NDArray) -> NDArray:
        """Extract position from state."""
        return self._rocket.get_position(x)

    def get_velocity(self, x: NDArray) -> NDArray:
        """Extract velocity from state."""
        return self._rocket.get_velocity(x)

    def get_altitude(self, x: NDArray) -> float:
        """Get altitude (r_x component)."""
        return self._rocket.get_altitude(x)

    def get_speed(self, x: NDArray) -> float:
        """Get speed magnitude."""
        return self._rocket.get_speed(x)

    def get_thrust_magnitude(self, u: NDArray) -> float:
        """Get thrust magnitude."""
        return self._rocket.get_thrust_magnitude(u)

    def get_thrust_direction(self, u: NDArray) -> NDArray:
        """Get thrust direction (unit vector)."""
        return self._rocket.get_thrust_direction(u)

    def fuel_remaining(self, x: NDArray) -> float:
        """Get remaining fuel mass [kg]."""
        return self._rocket.fuel_remaining(x)

    def fuel_fraction(self, x: NDArray) -> float:
        """Get fuel fraction remaining (0 to 1)."""
        return self._rocket.fuel_fraction(x)

    # =========================================================================
    # Dynamics
    # =========================================================================

    def dynamics(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Evaluate continuous-time dynamics: ẋ = f(x, u)

        Args:
            x: State vector (7,)
            u: Control vector (thrust in inertial frame) (3,)

        Returns:
            State derivative (7,)
        """
        return self._rocket.f(x, u)

    def f(self, x: NDArray, u: NDArray) -> NDArray:
        """Alias for dynamics()."""
        return self.dynamics(x, u)

    def step(
        self,
        x: NDArray,
        u: NDArray,
        dt: Optional[float] = None,
    ) -> NDArray:
        """
        Integrate dynamics one timestep: x_{k+1} = F(x_k, u_k)

        Args:
            x: Current state (7,)
            u: Control input (3,)
            dt: Timestep [s], uses config default if None

        Returns:
            Next state (7,)
        """
        if dt is None:
            dt = self.config.default_dt

        return self._rocket.f_discrete(x, u, dt)

    def f_discrete(self, x: NDArray, u: NDArray, dt: float) -> NDArray:
        """Alias for step()."""
        return self.step(x, u, dt)

    # =========================================================================
    # Jacobians
    # =========================================================================

    def jacobian_x(self, x: NDArray, u: NDArray) -> NDArray:
        """
        State Jacobian: A = ∂f/∂x

        Args:
            x: State vector (7,)
            u: Control vector (3,)

        Returns:
            State Jacobian (7, 7)
        """
        return self._rocket.A(x, u)

    def jacobian_u(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Control Jacobian: B = ∂f/∂u

        Args:
            x: State vector (7,)
            u: Control vector (3,)

        Returns:
            Control Jacobian (7, 3)
        """
        return self._rocket.B(x, u)

    def A(self, x: NDArray, u: NDArray) -> NDArray:
        """Alias for jacobian_x()."""
        return self.jacobian_x(x, u)

    def B(self, x: NDArray, u: NDArray) -> NDArray:
        """Alias for jacobian_u()."""
        return self.jacobian_u(x, u)

    def linearize(
        self,
        x: NDArray,
        u: NDArray,
        dt: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Linearize dynamics around (x, u).

        Args:
            x: Linearization state (7,)
            u: Linearization control (3,)
            dt: If provided, returns discrete-time Jacobians

        Returns:
            A: State Jacobian (7, 7)
            B: Control Jacobian (7, 3)
        """
        A_c = self.jacobian_x(x, u)
        B_c = self.jacobian_u(x, u)

        if dt is not None:
            A_d = np.eye(self.N_STATE) + A_c * dt
            B_d = B_c * dt
            return A_d, B_d

        return A_c, B_c

    def linearize_discrete(
        self,
        x: NDArray,
        u: NDArray,
        dt: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Linearize discrete dynamics: x_{k+1} ≈ A_d x_k + B_d u_k + c

        Args:
            x: Linearization state (7,)
            u: Linearization control (3,)
            dt: Timestep

        Returns:
            A_d: Discrete state Jacobian (7, 7)
            B_d: Discrete control Jacobian (7, 3)
            c: Affine term (7,)
        """
        A_d, B_d = self.linearize(x, u, dt)
        x_next = self.step(x, u, dt)
        c = x_next - A_d @ x - B_d @ u
        return A_d, B_d, c

    # =========================================================================
    # Constraints
    # =========================================================================

    def thrust_constraint(self, u: NDArray) -> Tuple[float, float]:
        """
        Evaluate thrust magnitude constraint.

        Returns:
            (lower_violation, upper_violation) where negative = satisfied
        """
        T_mag = self.get_thrust_magnitude(u)
        lower = self.config.T_min - T_mag
        upper = T_mag - self.config.T_max
        return lower, upper

    def glide_slope_constraint(self, x: NDArray) -> float:
        """
        Evaluate glide slope constraint.

        Returns:
            Constraint value (negative = satisfied)
        """
        return self._rocket.glide_slope_constraint(x)

    def is_thrust_valid(self, u: NDArray) -> bool:
        """Check if thrust is within bounds."""
        return self._rocket.is_thrust_valid(u)

    def is_glide_slope_satisfied(self, x: NDArray) -> bool:
        """Check if glide slope constraint is satisfied."""
        return self._rocket.is_glide_slope_satisfied(x)

    def is_state_valid(self, x: NDArray) -> bool:
        """Check if state satisfies all state constraints."""
        return self._rocket.is_state_valid(x)

    def is_control_valid(self, u: NDArray) -> bool:
        """Check if control satisfies all control constraints."""
        return self._rocket.is_control_valid(u)

    def evaluate_constraints(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Dict[str, float]:
        """
        Evaluate all constraints at (x, u).

        Returns:
            Dictionary of constraint values (negative = satisfied)
        """
        T_lower, T_upper = self.thrust_constraint(u)
        return {
            "thrust_lower": T_lower,
            "thrust_upper": T_upper,
            "glide_slope": self.glide_slope_constraint(x),
        }

    # =========================================================================
    # Control Utilities
    # =========================================================================

    def hover_thrust(self, x: NDArray) -> NDArray:
        """
        Compute thrust required to hover.

        Args:
            x: Current state

        Returns:
            Thrust vector in inertial frame (3,)
        """
        return self._rocket.hover_thrust(x)

    def clamp_thrust(self, u: NDArray) -> NDArray:
        """
        Clamp thrust to valid range while preserving direction.

        Args:
            u: Desired thrust vector (3,)

        Returns:
            Clamped thrust vector (3,)
        """
        T_mag = np.linalg.norm(u)
        if T_mag < 1e-10:
            return np.array([self.config.T_min, 0.0, 0.0])

        T_mag_clamped = np.clip(T_mag, self.config.T_min, self.config.T_max)
        return u * (T_mag_clamped / T_mag)

    def get_control_bounds(self) -> Tuple[NDArray, NDArray]:
        """
        Get control bounds for optimization.

        Returns:
            (lower_bounds, upper_bounds) each (3,)
        """
        return self._rocket.get_control_bounds()

    def get_state_bounds(self) -> Tuple[NDArray, NDArray]:
        """
        Get state bounds for optimization.

        Returns:
            (lower_bounds, upper_bounds) each (7,)
        """
        return self._rocket.get_state_bounds()

    # =========================================================================
    # Simulation
    # =========================================================================

    def simulate(
        self,
        x0: NDArray,
        controller: Callable[[float, NDArray], NDArray],
        t_span: Tuple[float, float],
        dt: Optional[float] = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Simulate closed-loop trajectory.

        Args:
            x0: Initial state (7,)
            controller: Control law (t, x) -> u
            t_span: (t_start, t_end)
            dt: Timestep, uses config default if None

        Returns:
            t: Time vector (N,)
            x: State trajectory (N, 7)
            u: Control trajectory (N-1, 3)
        """
        if dt is None:
            dt = self.config.default_dt

        return self._rocket.simulate(x0, controller, t_span, dt=dt)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def energy(self, x: NDArray) -> Dict[str, float]:
        """
        Compute energy components.

        Returns:
            Dictionary with kinetic, potential, and total energy
        """
        return self._rocket.energy(x)

    def time_of_flight_estimate(self, x: NDArray) -> float:
        """
        Estimate time of flight to landing.

        Args:
            x: Current state

        Returns:
            Estimated time to landing [s]
        """
        return self._rocket.time_of_flight_estimate(x)

    def __repr__(self) -> str:
        return (
            f"Rocket3DoFDynamics("
            f"m_wet={self.config.m_wet}, "
            f"m_dry={self.config.m_dry}, "
            f"T_range=[{self.config.T_min}, {self.config.T_max}])"
        )


def create_normalized_rocket() -> Rocket3DoFDynamics:
    """
    Create 3-DoF rocket with normalized parameters.

    Returns:
        Configured Rocket3DoFDynamics instance
    """
    return Rocket3DoFDynamics(Rocket3DoFConfig.normalized_defaults())


def create_rocket_3dof(
    m_wet: float = 2.0,
    m_dry: float = 1.0,
    T_min: float = 0.0,
    T_max: float = 6.5,
    I_sp: float = 30.0,
    **kwargs,
) -> Rocket3DoFDynamics:
    """
    Create 3-DoF rocket with custom parameters.

    Args:
        m_wet: Wet mass [kg]
        m_dry: Dry mass [kg]
        T_min: Minimum thrust [N]
        T_max: Maximum thrust [N]
        I_sp: Specific impulse [s]
        **kwargs: Additional Rocket3DoFConfig parameters

    Returns:
        Configured Rocket3DoFDynamics instance
    """
    config = Rocket3DoFConfig(
        m_wet=m_wet,
        m_dry=m_dry,
        T_min=T_min,
        T_max=T_max,
        I_sp=I_sp,
        **kwargs,
    )
    return Rocket3DoFDynamics(config)
