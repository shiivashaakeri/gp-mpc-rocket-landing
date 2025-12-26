"""
Demo: MPC Tracking SCVX-style Reference Trajectory

This script demonstrates the MPC tracking a pre-computed reference
trajectory (simulating what SCVX would generate).

Run: python scripts/demo_mpc_tracking.py
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "src")

from dynamics import Rocket3DoFDynamics  # pyright: ignore[reportMissingImports]
from mpc import (  # pyright: ignore[reportMissingImports]
    ConstraintParams,
    MPCConfig,
    NominalMPC3DoF,
)


def generate_scvx_reference(x0, x_target, N_total, dt):
    """
    Generate a reference trajectory simulating SCVX output.

    In practice, SCVX would solve a convex optimization problem.
    Here we generate a smooth polynomial trajectory.
    """
    t_total = N_total * dt
    t = np.linspace(0, t_total, N_total + 1)

    # 3-DoF state: [m, r(3), v(3)]
    X_ref = np.zeros((N_total + 1, 7))
    U_ref = np.zeros((N_total, 3))

    # Mass: linear decrease
    X_ref[:, 0] = np.linspace(x0[0], x_target[0], N_total + 1)

    # Position: 5th order polynomial for smooth trajectory
    # Boundary conditions: r(0)=r0, r(T)=rf, v(0)=v0, v(T)=vf, a(0)=a0, a(T)=af
    for i in range(3):  # x, y, z
        r0, rf = x0[1 + i], x_target[1 + i]
        v0, vf = x0[4 + i], x_target[4 + i]

        # Cubic interpolation for simplicity
        # r(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Solve for coefficients
        a0 = r0
        a1 = v0
        a2 = (3 * (rf - r0) - (2 * v0 + vf) * t_total) / t_total**2
        a3 = (2 * (r0 - rf) + (v0 + vf) * t_total) / t_total**3

        for k, tk in enumerate(t):
            X_ref[k, 1 + i] = a0 + a1 * tk + a2 * tk**2 + a3 * tk**3
            X_ref[k, 4 + i] = a1 + 2 * a2 * tk + 3 * a3 * tk**2

    # Compute reference controls from dynamics
    g = -9.81
    for k in range(N_total):
        m = X_ref[k, 0]
        # Acceleration from trajectory
        a = (X_ref[k + 1, 4:7] - X_ref[k, 4:7]) / dt if k < N_total - 1 else np.zeros(3)

        # T = m * (a - g)
        U_ref[k] = m * (a - np.array([g, 0, 0]))

        # Clamp thrust
        T_mag = np.linalg.norm(U_ref[k])
        if T_mag > 5.0:
            U_ref[k] = U_ref[k] / T_mag * 5.0
        elif T_mag < 0.5:
            U_ref[k] = U_ref[k] / max(T_mag, 1e-6) * 0.5

    return X_ref, U_ref


def run_mpc_tracking_demo():  # noqa: PLR0915
    """Run MPC tracking demonstration."""
    print("=" * 60)
    print("MPC Reference Trajectory Tracking Demo")
    print("=" * 60)

    # Create 3-DoF dynamics (faster for demo)
    dynamics = Rocket3DoFDynamics()

    # Initial and target states
    x0 = np.array([2.0, 15.0, 2.0, 0.0, -3.0, 0.5, 0.0])  # mass, pos, vel
    x_target = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Generate reference trajectory
    N_total = 50
    dt = 0.1
    X_ref, U_ref = generate_scvx_reference(x0, x_target, N_total, dt)

    print(f"\nReference trajectory: {N_total} steps, {N_total * dt:.1f}s duration")
    print(f"Initial: altitude={x0[1]:.1f}m, mass={x0[0]:.2f}kg")
    print(f"Target:  altitude={x_target[1]:.1f}m, mass={x_target[0]:.2f}kg")

    # Create MPC controller
    mpc_config = MPCConfig(N=15, dt=dt, max_iter=50)
    mpc = NominalMPC3DoF(dynamics, mpc_config)

    # Closed-loop simulation tracking the reference
    X_actual = np.zeros((N_total + 1, 7))
    U_actual = np.zeros((N_total, 3))
    tracking_errors = []

    X_actual[0] = x0
    x_current = x0.copy()

    print("\nRunning closed-loop simulation...")

    for k in range(N_total):
        # Get reference window for MPC
        idx_start = k
        idx_end = min(k + mpc_config.N + 1, N_total + 1)

        # Pad if needed
        X_ref_window = X_ref[idx_start:idx_end]
        if len(X_ref_window) < mpc_config.N + 1:
            pad_len = mpc_config.N + 1 - len(X_ref_window)
            X_ref_window = np.vstack([X_ref_window, np.tile(x_target, (pad_len, 1))])

        # Solve MPC with reference as target
        x_ref_k = X_ref_window[-1]  # Target is end of window
        solution = mpc.solve(x_current, x_ref_k)

        # Apply first control
        u_apply = solution.u0
        U_actual[k] = u_apply

        # Simulate one step
        x_next = dynamics.step(x_current, u_apply, dt)
        X_actual[k + 1] = x_next

        # Compute tracking error
        pos_error = np.linalg.norm(x_next[1:4] - X_ref[k + 1, 1:4])
        tracking_errors.append(pos_error)

        x_current = x_next

        # Early termination
        if x_current[1] < 0.1:
            print(f"  Landed at step {k + 1}")
            X_actual = X_actual[: k + 2]
            U_actual = U_actual[: k + 1]
            break

        if k % 10 == 0:
            print(f"  Step {k:3d}: alt={x_current[1]:.2f}m, err={pos_error:.3f}m")

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    final_pos_error = np.linalg.norm(X_actual[-1, 1:4] - x_target[1:4])
    final_vel_error = np.linalg.norm(X_actual[-1, 4:7] - x_target[4:7])
    avg_tracking_error = np.mean(tracking_errors)
    max_tracking_error = np.max(tracking_errors)

    print(f"Final position error: {final_pos_error:.4f} m")
    print(f"Final velocity error: {final_vel_error:.4f} m/s")
    print(f"Avg tracking error:   {avg_tracking_error:.4f} m")
    print(f"Max tracking error:   {max_tracking_error:.4f} m")
    print(f"Landing altitude:     {X_actual[-1, 1]:.4f} m")

    # Check constraints
    params = ConstraintParams()
    violations = 0
    for k in range(len(U_actual)):
        T_mag = np.linalg.norm(U_actual[k])
        if T_mag < params.T_min or T_mag > params.T_max:
            violations += 1

    print(f"Thrust violations:    {violations}/{len(U_actual)}")

    # Success criteria
    success = (
        final_pos_error < 1.0  # Land within 1m
        and final_vel_error < 2.0  # Reasonable terminal velocity
        and X_actual[-1, 1] < 0.5  # Actually landed
    )
    print(f"\nTracking test: {'PASSED ✓' if success else 'FAILED ✗'}")

    # Plot results
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        t_actual = np.arange(len(X_actual)) * dt
        t_ref = np.arange(len(X_ref)) * dt

        # Position
        ax = axes[0, 0]
        ax.plot(t_ref, X_ref[:, 1], "b--", label="Reference", alpha=0.7)
        ax.plot(t_actual, X_actual[:, 1], "b-", label="Actual", linewidth=2)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Altitude Tracking")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3D trajectory
        ax = axes[0, 1]
        ax.plot(X_ref[:, 2], X_ref[:, 1], "g--", label="Reference", alpha=0.7)
        ax.plot(X_actual[:, 2], X_actual[:, 1], "g-", label="Actual", linewidth=2)
        ax.scatter([X_actual[0, 2]], [X_actual[0, 1]], c="green", s=100, marker="o", label="Start")
        ax.scatter([X_actual[-1, 2]], [X_actual[-1, 1]], c="red", s=100, marker="x", label="End")
        ax.set_xlabel("Y Position [m]")
        ax.set_ylabel("Altitude [m]")
        ax.set_title("Trajectory (Y-Z plane)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Tracking error
        ax = axes[1, 0]
        ax.plot(t_actual[1 : len(tracking_errors) + 1], tracking_errors, "r-", linewidth=2)
        ax.axhline(y=avg_tracking_error, color="r", linestyle="--", label=f"Mean={avg_tracking_error:.3f}m")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position Error [m]")
        ax.set_title("Tracking Error")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Thrust
        ax = axes[1, 1]
        t_u = np.arange(len(U_actual)) * dt
        T_mag = np.linalg.norm(U_actual, axis=1)
        ax.plot(t_u, T_mag, "k-", linewidth=2, label="||T||")
        ax.axhline(y=params.T_min, color="r", linestyle="--", label=f"T_min={params.T_min}")
        ax.axhline(y=params.T_max, color="r", linestyle="--", label=f"T_max={params.T_max}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Thrust Magnitude [N]")
        ax.set_title("Thrust Profile")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("outputs/mpc_tracking_demo.png", dpi=150)
        print("\nPlot saved to outputs/mpc_tracking_demo.png")
        plt.close()

    except Exception as e:
        print(f"\nPlotting skipped: {e}")

    return success


if __name__ == "__main__":
    import os

    os.makedirs("outputs", exist_ok=True)

    success = run_mpc_tracking_demo()
    sys.exit(0 if success else 1)
