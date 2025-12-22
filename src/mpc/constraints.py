"""
MPC Constraints for Rocket Landing

Implements all constraints for powered descent guidance:
- Thrust magnitude bounds
- Gimbal angle limits
- Tilt angle constraint (keep rocket mostly upright)
- Glideslope constraint (stay in safe descent cone)
- Angular rate limits
- Terminal constraints

All constraints are formulated for CasADi symbolic expressions.

Reference:
    Szmuk, M., & Açikmeşe, B. (2018). Successive convexification for
    6-DoF Mars rocket powered landing with free-final-time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    import casadi as ca

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class ConstraintParams:
    """Parameters for rocket landing constraints."""

    # Thrust constraints
    T_min: float = 0.5  # Minimum thrust [N]
    T_max: float = 5.0  # Maximum thrust [N]

    # Gimbal constraint
    delta_max: float = 20.0  # Maximum gimbal angle [deg]

    # Tilt constraint
    theta_max: float = 90.0  # Maximum tilt from vertical [deg]

    # Glideslope constraint
    gamma_gs: float = 30.0  # Glideslope angle [deg]

    # Angular rate constraint
    omega_max: float = 60.0  # Maximum angular rate [deg/s]

    # State bounds
    v_max: float = 50.0  # Maximum velocity [m/s]

    # Terminal constraints
    r_tol: float = 0.1  # Position tolerance [m]
    v_tol: float = 0.1  # Velocity tolerance [m/s]
    q_tol: float = 5.0  # Attitude tolerance [deg]
    omega_tol: float = 1.0  # Angular rate tolerance [deg/s]

    def __post_init__(self):
        """Convert degrees to radians."""
        self.delta_max_rad = np.deg2rad(self.delta_max)
        self.theta_max_rad = np.deg2rad(self.theta_max)
        self.gamma_gs_rad = np.deg2rad(self.gamma_gs)
        self.omega_max_rad = np.deg2rad(self.omega_max)
        self.q_tol_rad = np.deg2rad(self.q_tol)
        self.omega_tol_rad = np.deg2rad(self.omega_tol)


# =============================================================================
# NumPy Constraint Evaluation (for simulation/validation)
# =============================================================================


def eval_thrust_magnitude(u: NDArray) -> float:
    """Compute thrust magnitude ||T||."""
    return np.linalg.norm(u)


def eval_gimbal_angle(u: NDArray) -> float:
    """
    Compute gimbal angle (angle of thrust from body z-axis).

    cos(δ) = T_z / ||T||
    """
    T_mag = np.linalg.norm(u)
    if T_mag < 1e-10:
        return 0.0
    cos_delta = u[2] / T_mag  # Assuming thrust primarily in z direction
    cos_delta = np.clip(cos_delta, -1, 1)
    return np.arccos(np.abs(cos_delta))


def eval_tilt_angle(q: NDArray) -> float:
    """
    Compute tilt angle from vertical.

    The tilt angle is the angle between the body z-axis and inertial z-axis.
    For quaternion q = [w, x, y, z]:
        cos(θ) = 1 - 2(x² + y²)
    """
    w, x, y, z = q
    cos_theta = 1 - 2 * (x**2 + y**2)
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)


def eval_glideslope(r: NDArray, gamma: float) -> float:
    """
    Evaluate glideslope constraint.

    The rocket must stay within a cone centered on the landing site:
        ||r_horizontal|| <= r_vertical * tan(gamma)

    Or equivalently:
        r_z * cos(gamma) >= ||r_xy|| * sin(gamma)

    Returns positive value if constraint satisfied.

    Args:
        r: Position [x, y, z] where x is vertical (altitude)
        gamma: Glideslope angle in radians
    """
    altitude = r[0]  # x is up in UEN frame
    horizontal_dist = np.sqrt(r[1] ** 2 + r[2] ** 2)

    # Constraint: altitude * tan(gamma) >= horizontal_dist
    return altitude * np.tan(gamma) - horizontal_dist


def eval_angular_rate(omega: NDArray) -> float:
    """Compute angular rate magnitude."""
    return np.linalg.norm(omega)


def check_all_constraints(
    x: NDArray,
    u: NDArray,
    params: ConstraintParams,
) -> dict:
    """
    Check all constraints and return violation info.

    Args:
        x: State [m, r(3), v(3), q(4), ω(3)]
        u: Control [Tx, Ty, Tz]
        params: Constraint parameters

    Returns:
        Dictionary with constraint values and satisfaction status
    """
    r = x[1:4]
    v = x[4:7]
    q = x[7:11]
    omega = x[11:14]

    T_mag = eval_thrust_magnitude(u)
    gimbal = eval_gimbal_angle(u)
    tilt = eval_tilt_angle(q)
    glideslope = eval_glideslope(r, params.gamma_gs_rad)
    omega_mag = eval_angular_rate(omega)
    v_mag = np.linalg.norm(v)

    return {
        "thrust_magnitude": T_mag,
        "thrust_min_satisfied": T_mag >= params.T_min,
        "thrust_max_satisfied": T_mag <= params.T_max,
        "gimbal_angle_deg": np.rad2deg(gimbal),
        "gimbal_satisfied": gimbal <= params.delta_max_rad,
        "tilt_angle_deg": np.rad2deg(tilt),
        "tilt_satisfied": tilt <= params.theta_max_rad,
        "glideslope_margin": glideslope,
        "glideslope_satisfied": glideslope >= 0,
        "angular_rate_deg_s": np.rad2deg(omega_mag),
        "angular_rate_satisfied": omega_mag <= params.omega_max_rad,
        "velocity_mag": v_mag,
        "velocity_satisfied": v_mag <= params.v_max,
        "all_satisfied": (
            T_mag >= params.T_min
            and T_mag <= params.T_max
            and gimbal <= params.delta_max_rad
            and tilt <= params.theta_max_rad
            and glideslope >= 0
            and omega_mag <= params.omega_max_rad
            and v_mag <= params.v_max
        ),
    }


# =============================================================================
# CasADi Symbolic Constraints (for MPC)
# =============================================================================

if HAS_CASADI:

    class CasADiConstraints:
        """
        CasADi symbolic constraint formulations for MPC.

        All constraints are formulated as g(x, u) >= 0 (inequality)
        or h(x, u) = 0 (equality).
        """

        def __init__(self, params: ConstraintParams):
            """
            Initialize with constraint parameters.

            Args:
                params: Constraint parameters
            """
            self.params = params

        def thrust_magnitude_bounds(
            self,
            u: ca.MX,
        ) -> Tuple[ca.MX, ca.MX]:
            """
            Thrust magnitude constraints.

            T_min <= ||T|| <= T_max

            Returns:
                (lower_bound_constraint, upper_bound_constraint)
                Both should be >= 0
            """
            T_sq = ca.dot(u, u)

            # ||T|| >= T_min  =>  ||T||² >= T_min²
            lb = T_sq - self.params.T_min**2

            # ||T|| <= T_max  =>  T_max² >= ||T||²
            ub = self.params.T_max**2 - T_sq

            return lb, ub

        def thrust_cone_constraint(self, u: ca.MX) -> ca.MX:
            """
            Combined thrust constraint as second-order cone.

            T_min <= ||T|| <= T_max

            Formulated as: T_max² - ||T||² >= 0 (upper bound)
            and ||T||² - T_min² >= 0 (lower bound when engine is on)

            Returns single constraint for upper bound (most critical).
            """
            T_sq = ca.dot(u, u)
            return self.params.T_max**2 - T_sq

        def gimbal_constraint(self, u: ca.MX) -> ca.MX:
            """
            Gimbal angle constraint.

            cos(δ) >= cos(δ_max)
            T_z / ||T|| >= cos(δ_max)
            T_z >= cos(δ_max) * ||T||

            Squared form (for smooth optimization):
            T_z² >= cos²(δ_max) * ||T||²
            """
            T_sq = ca.dot(u, u)
            cos_delta_max_sq = np.cos(self.params.delta_max_rad) ** 2

            # T_z² - cos²(δ_max) * ||T||² >= 0
            return u[2] ** 2 - cos_delta_max_sq * T_sq

        def tilt_constraint(self, q: ca.MX) -> ca.MX:
            """
            Tilt angle constraint (keep rocket upright).

            cos(θ) >= cos(θ_max)
            1 - 2(qx² + qy²) >= cos(θ_max)

            Args:
                q: Quaternion [qw, qx, qy, qz]
            """
            cos_theta = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
            return cos_theta - np.cos(self.params.theta_max_rad)

        def glideslope_constraint(self, r: ca.MX) -> ca.MX:
            """
            Glideslope constraint (stay in landing cone).

            r_x * tan(gamma) >= sqrt(r_y² + r_z²)

            Squared form:
            r_x² * tan²(gamma) >= r_y² + r_z²

            Args:
                r: Position [rx, ry, rz] where rx is altitude
            """
            tan_gamma_sq = np.tan(self.params.gamma_gs_rad) ** 2
            horizontal_sq = r[1] ** 2 + r[2] ** 2

            return r[0] ** 2 * tan_gamma_sq - horizontal_sq

        def angular_rate_constraint(self, omega: ca.MX) -> ca.MX:
            """
            Angular rate constraint.

            ||ω|| <= ω_max
            ω_max² - ||ω||² >= 0
            """
            omega_sq = ca.dot(omega, omega)
            return self.params.omega_max_rad**2 - omega_sq

        def velocity_constraint(self, v: ca.MX) -> ca.MX:
            """
            Velocity magnitude constraint.

            ||v|| <= v_max
            """
            v_sq = ca.dot(v, v)
            return self.params.v_max**2 - v_sq

        def get_path_constraints(
            self,
            x: ca.MX,
            u: ca.MX,
        ) -> List[ca.MX]:
            """
            Get all path constraints for a single (x, u) pair.

            All constraints are formulated as g >= 0.

            Args:
                x: State [m, r(3), v(3), q(4), ω(3)]
                u: Control [Tx, Ty, Tz]

            Returns:
                List of constraint expressions (all should be >= 0)
            """
            r = x[1:4]
            v = x[4:7]
            q = x[7:11]
            omega = x[11:14]

            constraints = []

            # Thrust bounds
            T_lb, T_ub = self.thrust_magnitude_bounds(u)
            constraints.append(T_lb)  # ||T|| >= T_min
            constraints.append(T_ub)  # ||T|| <= T_max

            # Gimbal
            constraints.append(self.gimbal_constraint(u))

            # Tilt
            constraints.append(self.tilt_constraint(q))

            # Glideslope
            constraints.append(self.glideslope_constraint(r))

            # Angular rate
            constraints.append(self.angular_rate_constraint(omega))

            # Velocity
            constraints.append(self.velocity_constraint(v))

            return constraints

        def get_terminal_constraints(
            self,
            x: ca.MX,
            x_target: NDArray,
        ) -> List[ca.MX]:
            """
            Get terminal constraints.

            Args:
                x: Final state
                x_target: Target state

            Returns:
                List of terminal constraint expressions
            """
            r = x[1:4]
            v = x[4:7]
            q = x[7:11]
            omega = x[11:14]

            r_target = x_target[1:4]
            v_target = x_target[4:7]

            constraints = []

            # Position tolerance (soft or hard)
            r_err_sq = ca.dot(r - r_target, r - r_target)
            constraints.append(self.params.r_tol**2 - r_err_sq)

            # Velocity tolerance
            v_err_sq = ca.dot(v - v_target, v - v_target)
            constraints.append(self.params.v_tol**2 - v_err_sq)

            # Attitude: should be upright (qx, qy small)
            # ||[qx, qy]||² <= sin²(θ_tol/2)
            sin_half_tol_sq = np.sin(self.params.q_tol_rad / 2) ** 2
            q_err_sq = q[1] ** 2 + q[2] ** 2
            constraints.append(sin_half_tol_sq - q_err_sq)

            # Angular rate tolerance
            omega_err_sq = ca.dot(omega, omega)
            constraints.append(self.params.omega_tol_rad**2 - omega_err_sq)

            return constraints

        @property
        def n_path_constraints(self) -> int:
            """Number of path constraints per timestep."""
            return 7  # T_lb, T_ub, gimbal, tilt, glideslope, omega, velocity

        @property
        def n_terminal_constraints(self) -> int:
            """Number of terminal constraints."""
            return 4  # r, v, q, omega


# =============================================================================
# Constraint Tightening for Uncertainty
# =============================================================================


@dataclass
class TightenedConstraints:
    """
    Constraint parameters with tightening for uncertainty.

    When we have uncertainty in the state (from GP), we need to
    tighten constraints to maintain probabilistic satisfaction:

    P(g(x) >= 0) >= 1 - ε

    For Gaussian uncertainty:
    g(μ) - κ * sigma_g >= 0

    where κ = Φ^{-1}(1 - ε) ≈ 2.33 for ε = 0.01
    """

    base_params: ConstraintParams
    confidence_level: float = 0.99  # 1 - ε

    def __post_init__(self):
        """Compute confidence multiplier."""
        from scipy.stats import norm  # noqa: PLC0415

        self.kappa = norm.ppf(self.confidence_level)  # ≈ 2.33 for 99%

    def tighten_scalar_constraint(
        self,
        constraint_value: float,
        constraint_std: float,
    ) -> float:
        """
        Tighten a scalar constraint.

        Args:
            constraint_value: g(μ) - nominal constraint value
            constraint_std: sigma_g - standard deviation of constraint

        Returns:
            Tightened constraint value
        """
        return constraint_value - self.kappa * constraint_std

    def get_tightened_params(
        self,
        position_std: float = 0.0,  # noqa: ARG002
        velocity_std: float = 0.0,
        attitude_std: float = 0.0,
        omega_std: float = 0.0,
    ) -> ConstraintParams:
        """
        Get tightened constraint parameters.

        Simple approach: reduce constraint bounds based on uncertainty.

        Args:
            position_std: Position uncertainty (affects glideslope)
            velocity_std: Velocity uncertainty (affects velocity bound)
            attitude_std: Attitude uncertainty (affects tilt)
            omega_std: Angular rate uncertainty

        Returns:
            Tightened constraint parameters
        """
        params = ConstraintParams(
            T_min=self.base_params.T_min,
            T_max=self.base_params.T_max,
            delta_max=self.base_params.delta_max,
            theta_max=self.base_params.theta_max - np.rad2deg(self.kappa * attitude_std),
            gamma_gs=self.base_params.gamma_gs,  # Could tighten based on position_std
            omega_max=self.base_params.omega_max - np.rad2deg(self.kappa * omega_std),
            v_max=self.base_params.v_max - self.kappa * velocity_std,
            r_tol=self.base_params.r_tol,
            v_tol=self.base_params.v_tol,
            q_tol=self.base_params.q_tol,
            omega_tol=self.base_params.omega_tol,
        )

        # Ensure constraints remain feasible
        params.theta_max = max(params.theta_max, 10.0)  # At least 10 deg
        params.omega_max = max(params.omega_max, 10.0)  # At least 10 deg/s
        params.v_max = max(params.v_max, 1.0)  # At least 1 m/s

        return params


# =============================================================================
# Constraint Jacobians (for linearization)
# =============================================================================


def compute_constraint_jacobians(
    x: NDArray,
    u: NDArray,
    params: ConstraintParams,
    eps: float = 1e-6,
) -> Tuple[NDArray, NDArray]:
    """
    Compute Jacobians of constraints w.r.t. state and control.

    Uses finite differences for simplicity.

    Args:
        x: State
        u: Control
        params: Constraint parameters
        eps: Finite difference step

    Returns:
        dg_dx: (n_constraints, n_x)
        dg_du: (n_constraints, n_u)
    """

    def eval_constraints(x, u):
        """Evaluate all constraints as vector."""
        r = x[1:4]
        v = x[4:7]
        q = x[7:11]
        omega = x[11:14]

        T_mag = np.linalg.norm(u)
        gimbal = eval_gimbal_angle(u)
        tilt = eval_tilt_angle(q)
        gs = eval_glideslope(r, params.gamma_gs_rad)
        omega_mag = eval_angular_rate(omega)
        v_mag = np.linalg.norm(v)

        return np.array(
            [
                T_mag**2 - params.T_min**2,  # Thrust lower
                params.T_max**2 - T_mag**2,  # Thrust upper
                np.cos(gimbal) - np.cos(params.delta_max_rad),  # Gimbal
                np.cos(tilt) - np.cos(params.theta_max_rad),  # Tilt
                gs,  # Glideslope
                params.omega_max_rad**2 - omega_mag**2,  # Angular rate
                params.v_max**2 - v_mag**2,  # Velocity
            ]
        )

    g0 = eval_constraints(x, u)
    n_g = len(g0)
    n_x = len(x)
    n_u = len(u)

    # Jacobian w.r.t. x
    dg_dx = np.zeros((n_g, n_x))
    for i in range(n_x):
        x_plus = x.copy()
        x_plus[i] += eps
        dg_dx[:, i] = (eval_constraints(x_plus, u) - g0) / eps

    # Jacobian w.r.t. u
    dg_du = np.zeros((n_g, n_u))
    for i in range(n_u):
        u_plus = u.copy()
        u_plus[i] += eps
        dg_du[:, i] = (eval_constraints(x, u_plus) - g0) / eps

    return dg_dx, dg_du
