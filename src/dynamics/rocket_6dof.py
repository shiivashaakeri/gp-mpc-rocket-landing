"""
6-DoF Rocket Dynamics Wrapper for GP-MPC

Wraps simdyn's Rocket6DoF with a consistent interface for the GP-MPC framework.
Provides dynamics evaluation, Jacobians, constraints, and discretization.

State vector (n=14):
    x = [m, r_I (3), v_I (3), q_BI (4), omega_B (3)]

Control vector (m=3):
    u = [T_Bx, T_By, T_Bz]  (thrust in body frame)

Reference:
    Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for 6-DoF
    powered descent guidance with compound state-triggered constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from simdyn import (
    Rocket6DoF as SimdynRocket6DoF,
)
from simdyn import (
    Rocket6DoFParams,
    quat_identity,
    quat_normalize,
)


@dataclass
class Rocket6DoFConfig:
    """Configuration for 6-DoF rocket dynamics."""

    # Mass properties
    m_dry: float = 1.0
    m_wet: float = 2.0

    # Inertia tensor (body frame)
    J_B: Optional[NDArray] = None

    # Propulsion
    I_sp: float = 30.0
    g0: float = 1.0
    T_min: float = 1.5
    T_max: float = 6.5

    # Geometry
    r_T_B: Optional[NDArray] = None  # Thrust application point  # noqa: N815
    r_cp_B: Optional[NDArray] = None  # Center of pressure  # noqa: N815

    # Environment
    g_I: Optional[NDArray] = None  # Gravity vector (inertial)  # noqa: N815

    # Constraints
    delta_max: float = np.deg2rad(20.0)  # Max gimbal angle
    theta_max: float = np.deg2rad(90.0)  # Max tilt angle
    gamma_gs: float = np.deg2rad(30.0)  # Glide slope angle
    omega_max: float = np.deg2rad(60.0)  # Max angular rate

    # Aerodynamics
    enable_aero: bool = False
    C_A: Optional[NDArray] = None  # Aero coefficient matrix
    rho: float = 0.0
    S_ref: float = 1.0

    # Integration
    default_dt: float = 0.1
    use_rk4: bool = True

    def __post_init__(self):
        """Set defaults for array parameters."""
        if self.J_B is None:
            self.J_B = np.diag([0.02, 1.0, 1.0]) * 0.168
        if self.r_T_B is None:
            self.r_T_B = np.array([-0.25, 0.0, 0.0])
        if self.r_cp_B is None:
            self.r_cp_B = np.array([0.05, 0.0, 0.0])
        if self.g_I is None:
            self.g_I = np.array([-1.0, 0.0, 0.0])

    @classmethod
    def szmuk_defaults(cls) -> "Rocket6DoFConfig":
        """Create config with Szmuk et al. (2018) parameters."""
        return cls()


class Rocket6DoFDynamics:
    """
    6-DoF rocket dynamics wrapper providing consistent interface for GP-MPC.

    This class wraps simdyn's Rocket6DoF and provides:
    - Continuous and discrete dynamics evaluation
    - Analytical Jacobians (state and control)
    - Constraint evaluation and checking
    - State packing/unpacking utilities
    - Integration with configurable methods

    Example:
        >>> config = Rocket6DoFConfig.szmuk_defaults()
        >>> rocket = Rocket6DoFDynamics(config)
        >>> x0 = rocket.create_initial_state(altitude=10.0, mass=2.0)
        >>> u = rocket.hover_thrust(x0)
        >>> x_next = rocket.step(x0, u, dt=0.1)
    """

    # State indices
    IDX_MASS = 0
    IDX_POS = slice(1, 4)
    IDX_VEL = slice(4, 7)
    IDX_QUAT = slice(7, 11)
    IDX_OMEGA = slice(11, 14)

    # State dimensions
    N_STATE = 14
    N_CONTROL = 3

    def __init__(self, config: Optional[Rocket6DoFConfig] = None):
        """
        Initialize 6-DoF rocket dynamics.

        Args:
            config: Configuration parameters. Uses Szmuk defaults if None.
        """
        self.config = config or Rocket6DoFConfig.szmuk_defaults()

        # Create simdyn Rocket6DoF
        self._rocket = self._create_simdyn_rocket()

        # Cache for Jacobians
        self._last_x: Optional[NDArray] = None
        self._last_u: Optional[NDArray] = None
        self._cached_A: Optional[NDArray] = None
        self._cached_B: Optional[NDArray] = None

    def _create_simdyn_rocket(self) -> SimdynRocket6DoF:
        """Create the underlying simdyn Rocket6DoF instance."""
        cfg = self.config

        params = Rocket6DoFParams(
            m_dry=cfg.m_dry,
            m_wet=cfg.m_wet,
            J_B=np.array(cfg.J_B),
            I_sp=cfg.I_sp,
            g0=cfg.g0,
            g_I=np.array(cfg.g_I),
            r_T_B=np.array(cfg.r_T_B),
            r_cp_B=np.array(cfg.r_cp_B),
            T_min=cfg.T_min,
            T_max=cfg.T_max,
            delta_max=cfg.delta_max,
            theta_max=cfg.theta_max,
            gamma_gs=cfg.gamma_gs,
            omega_max=cfg.omega_max,
            enable_aero=cfg.enable_aero,
        )

        return SimdynRocket6DoF(params)

    @property
    def params(self) -> Rocket6DoFParams:
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
        quaternion: NDArray,
        omega: NDArray,
    ) -> NDArray:
        """
        Pack state components into state vector.

        Args:
            mass: Vehicle mass [kg]
            position: Position in inertial frame [m] (3,)
            velocity: Velocity in inertial frame [m/s] (3,)
            quaternion: Attitude quaternion q_BI [w,x,y,z] (4,)
            omega: Angular velocity in body frame [rad/s] (3,)

        Returns:
            State vector (14,)
        """
        return self._rocket.pack_state(
            mass=mass,
            position=np.asarray(position),
            velocity=np.asarray(velocity),
            quaternion=np.asarray(quaternion),
            omega=np.asarray(omega),
        )

    def create_initial_state(
        self,
        altitude: float = 10.0,
        downrange: float = 0.0,
        crossrange: float = 0.0,
        velocity: Optional[NDArray] = None,
        tilt_angle: float = 0.0,
        tilt_axis: Optional[NDArray] = None,
        omega: Optional[NDArray] = None,
        mass: Optional[float] = None,
    ) -> NDArray:
        """
        Create initial state with convenient parameters.

        Args:
            altitude: Initial altitude (r_x in UEN) [m]
            downrange: Initial downrange (r_z in UEN) [m]
            crossrange: Initial crossrange (r_y in UEN) [m]
            velocity: Initial velocity [m/s], defaults to zero
            tilt_angle: Initial tilt from vertical [rad]
            tilt_axis: Axis for tilt rotation, defaults to [0,1,0]
            omega: Initial angular velocity [rad/s], defaults to zero
            mass: Initial mass [kg], defaults to m_wet

        Returns:
            Initial state vector (14,)
        """
        # Position
        position = np.array([altitude, crossrange, downrange])

        # Velocity
        velocity = np.zeros(3) if velocity is None else np.asarray(velocity)

        # Quaternion from tilt
        if tilt_angle != 0.0:
            if tilt_axis is None:
                tilt_axis = np.array([0.0, 1.0, 0.0])
            tilt_axis = np.asarray(tilt_axis)
            tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)

            # Quaternion from axis-angle
            half_angle = tilt_angle / 2.0
            quaternion = np.array(
                [
                    np.cos(half_angle),
                    tilt_axis[0] * np.sin(half_angle),
                    tilt_axis[1] * np.sin(half_angle),
                    tilt_axis[2] * np.sin(half_angle),
                ]
            )
        else:
            quaternion = quat_identity()

        # Angular velocity
        omega = np.zeros(3) if omega is None else np.asarray(omega)

        # Mass
        if mass is None:
            mass = self.config.m_wet

        return self.pack_state(mass, position, velocity, quaternion, omega)

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

    def get_quaternion(self, x: NDArray) -> NDArray:
        """Extract quaternion from state."""
        return self._rocket.get_quaternion(x)

    def get_omega(self, x: NDArray) -> NDArray:
        """Extract angular velocity from state."""
        return self._rocket.get_omega(x)

    def get_altitude(self, x: NDArray) -> float:
        """Get altitude (r_x component)."""
        return self._rocket.get_altitude(x)

    def get_speed(self, x: NDArray) -> float:
        """Get speed magnitude."""
        return self._rocket.get_speed(x)

    def get_dcm(self, x: NDArray) -> NDArray:
        """Get direction cosine matrix C_BI (body from inertial)."""
        return self._rocket.get_dcm(x)

    def get_tilt_angle(self, x: NDArray) -> float:
        """Get tilt angle from vertical [rad]."""
        return self._rocket.get_tilt_angle(x)

    def get_gimbal_angle(self, u: NDArray) -> float:
        """Get gimbal angle (angle between thrust and body x-axis) [rad]."""
        return self._rocket.get_gimbal_angle(u)

    def get_thrust_magnitude(self, u: NDArray) -> float:
        """Get thrust magnitude."""
        return self._rocket.get_thrust_magnitude(u)

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
            x: State vector (14,)
            u: Control vector (thrust in body frame) (3,)

        Returns:
            State derivative (14,)
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
            x: Current state (14,)
            u: Control input (3,)
            dt: Timestep [s], uses config default if None

        Returns:
            Next state (14,)
        """
        if dt is None:
            dt = self.config.default_dt

        x_next = self._rocket.f_discrete(x, u, dt)

        # Normalize quaternion to prevent drift
        x_next = self.normalize_state(x_next)

        return x_next

    def f_discrete(self, x: NDArray, u: NDArray, dt: float) -> NDArray:
        """Alias for step()."""
        return self.step(x, u, dt)

    def normalize_state(self, x: NDArray) -> NDArray:
        """Normalize quaternion in state to prevent numerical drift."""
        x_normalized = x.copy()
        q = x_normalized[self.IDX_QUAT]
        x_normalized[self.IDX_QUAT] = quat_normalize(q)
        return x_normalized

    # =========================================================================
    # Jacobians
    # =========================================================================

    def jacobian_x(self, x: NDArray, u: NDArray) -> NDArray:
        """
        State Jacobian: A = ∂f/∂x

        Args:
            x: State vector (14,)
            u: Control vector (3,)

        Returns:
            State Jacobian (14, 14)
        """
        return self._rocket.A(x, u)

    def jacobian_u(self, x: NDArray, u: NDArray) -> NDArray:
        """
        Control Jacobian: B = ∂f/∂u

        Args:
            x: State vector (14,)
            u: Control vector (3,)

        Returns:
            Control Jacobian (14, 3)
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

        Returns continuous-time Jacobians:
            ẋ ≈ f(x₀, u₀) + A(x - x₀) + B(u - u₀)

        Args:
            x: Linearization state (14,)
            u: Linearization control (3,)
            dt: If provided, returns discrete-time Jacobians

        Returns:
            A: State Jacobian (14, 14)
            B: Control Jacobian (14, 3)
        """
        A_c = self.jacobian_x(x, u)
        B_c = self.jacobian_u(x, u)

        if dt is not None:
            # Discretize using first-order approximation
            # A_d ≈ I + A_c * dt
            # B_d ≈ B_c * dt
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
            x: Linearization state (14,)
            u: Linearization control (3,)
            dt: Timestep

        Returns:
            A_d: Discrete state Jacobian (14, 14)
            B_d: Discrete control Jacobian (14, 3)
            c: Affine term (14,) such that F(x,u) = A_d @ x + B_d @ u + c
        """
        A_d, B_d = self.linearize(x, u, dt)
        x_next = self.step(x, u, dt)

        # c = F(x,u) - A_d @ x - B_d @ u
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
        lower = self.config.T_min - T_mag  # negative if T_mag >= T_min
        upper = T_mag - self.config.T_max  # negative if T_mag <= T_max
        return lower, upper

    def gimbal_constraint(self, u: NDArray) -> float:
        """
        Evaluate gimbal angle constraint.

        Returns:
            Constraint value (negative = satisfied)
        """
        return self._rocket.gimbal_constraint(u)

    def tilt_constraint(self, x: NDArray) -> float:
        """
        Evaluate tilt angle constraint.

        Returns:
            Constraint value (negative = satisfied)
        """
        return self._rocket.tilt_constraint(x)

    def glide_slope_constraint(self, x: NDArray) -> float:
        """
        Evaluate glide slope constraint.

        Returns:
            Constraint value (negative = satisfied)
        """
        return self._rocket.glide_slope_constraint(x)

    def angular_rate_constraint(self, x: NDArray) -> float:
        """
        Evaluate angular rate constraint.

        Returns:
            Constraint value (negative = satisfied)
        """
        return self._rocket.angular_rate_constraint(x)

    def is_thrust_valid(self, u: NDArray) -> bool:
        """Check if thrust is within bounds."""
        return self._rocket.is_thrust_valid(u)

    def is_gimbal_valid(self, u: NDArray) -> bool:
        """Check if gimbal angle is within bounds."""
        return self._rocket.is_gimbal_valid(u)

    def is_tilt_valid(self, x: NDArray) -> bool:
        """Check if tilt angle is within bounds."""
        return self._rocket.is_tilt_valid(x)

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
            "gimbal": self.gimbal_constraint(u),
            "tilt": self.tilt_constraint(x),
            "glide_slope": self.glide_slope_constraint(x),
            "angular_rate": self.angular_rate_constraint(x),
        }

    def max_constraint_violation(
        self,
        x: NDArray,
        u: NDArray,
    ) -> Tuple[str, float]:
        """
        Find maximum constraint violation.

        Returns:
            (constraint_name, violation_value)
        """
        constraints = self.evaluate_constraints(x, u)
        max_name = max(constraints, key=constraints.get)
        return max_name, constraints[max_name]

    # =========================================================================
    # Control Utilities
    # =========================================================================

    def hover_thrust(self, x: NDArray) -> NDArray:
        """
        Compute thrust required to hover.

        Args:
            x: Current state

        Returns:
            Thrust vector in body frame (3,)
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
            # Zero thrust - return minimum along body x
            return np.array([self.config.T_min, 0.0, 0.0])

        T_mag_clamped = np.clip(T_mag, self.config.T_min, self.config.T_max)
        return u * (T_mag_clamped / T_mag)

    def clamp_gimbal(self, u: NDArray) -> NDArray:
        """
        Clamp thrust to satisfy gimbal constraint while preserving magnitude.

        Args:
            u: Desired thrust vector (3,)

        Returns:
            Clamped thrust vector (3,)
        """
        T_mag = np.linalg.norm(u)
        if T_mag < 1e-10:
            return np.array([self.config.T_min, 0.0, 0.0])

        # Gimbal angle
        cos_delta = u[0] / T_mag
        cos_delta_min = np.cos(self.config.delta_max)

        if cos_delta >= cos_delta_min:
            return u  # Already valid

        # Project thrust direction to satisfy gimbal constraint
        # Keep T_x, scale T_yz to satisfy cos(delta) = cos(delta_max)
        T_x_min = T_mag * cos_delta_min
        T_yz = u[1:3]
        T_yz_mag = np.linalg.norm(T_yz)

        if T_yz_mag < 1e-10:
            return np.array([T_mag, 0.0, 0.0])

        # T_yz_max such that cos(delta_max) = T_x_min / sqrt(T_x_min^2 + T_yz_max^2)
        sin_delta_max = np.sin(self.config.delta_max)
        T_yz_max = T_mag * sin_delta_max

        T_yz_clamped = T_yz * (T_yz_max / T_yz_mag)
        return np.array([T_x_min, T_yz_clamped[0], T_yz_clamped[1]])

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
            (lower_bounds, upper_bounds) each (14,)
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
            x0: Initial state (14,)
            controller: Control law (t, x) -> u
            t_span: (t_start, t_end)
            dt: Timestep, uses config default if None

        Returns:
            t: Time vector (N,)
            x: State trajectory (N, 14)
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
            Dictionary with kinetic, potential, rotational, and total energy
        """
        return self._rocket.energy(x)

    def __repr__(self) -> str:
        return (
            f"Rocket6DoFDynamics("
            f"m_wet={self.config.m_wet}, "
            f"m_dry={self.config.m_dry}, "
            f"T_range=[{self.config.T_min}, {self.config.T_max}])"
        )


def create_szmuk_rocket() -> Rocket6DoFDynamics:
    """
    Create 6-DoF rocket with Szmuk et al. (2018) parameters.

    Returns:
        Configured Rocket6DoFDynamics instance
    """
    return Rocket6DoFDynamics(Rocket6DoFConfig.szmuk_defaults())


def create_rocket_6dof(
    m_wet: float = 2.0,
    m_dry: float = 1.0,
    T_min: float = 1.5,
    T_max: float = 6.5,
    I_sp: float = 30.0,
    **kwargs,
) -> Rocket6DoFDynamics:
    """
    Create 6-DoF rocket with custom parameters.

    Args:
        m_wet: Wet mass [kg]
        m_dry: Dry mass [kg]
        T_min: Minimum thrust [N]
        T_max: Maximum thrust [N]
        I_sp: Specific impulse [s]
        **kwargs: Additional Rocket6DoFConfig parameters

    Returns:
        Configured Rocket6DoFDynamics instance
    """
    config = Rocket6DoFConfig(
        m_wet=m_wet,
        m_dry=m_dry,
        T_min=T_min,
        T_max=T_max,
        I_sp=I_sp,
        **kwargs,
    )
    return Rocket6DoFDynamics(config)
