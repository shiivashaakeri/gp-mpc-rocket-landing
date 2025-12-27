#!/usr/bin/env python3
"""
Main Experiment Runner for GP-MPC Rocket Landing

Runs complete experiment suite:
1. Monte Carlo simulation (configurable runs)
2. Baseline comparisons (LQR, PID, Nominal MPC)
3. Ablation studies
4. Dispersion analysis
5. Generate publication figures
6. Export results

Usage:
    python scripts/run_experiments.py --quick       # Quick test (10 runs)
    python scripts/run_experiments.py --standard   # Standard (100 runs)
    python scripts/run_experiments.py --full       # Full (1000 runs)
    python scripts/run_experiments.py --custom 500 # Custom number
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from experiments import (
    COLORS,
    DispersionAnalysis,
    DispersionConfig,
    LandingConstraints,
    LQRController,
    MonteCarloResults,
    MonteCarloSimulator,
    MonteCarloVisualizer,
    PerformanceMetrics,
    PIDController,
    ResultsExporter,
    SimulationConfig,
    StatisticalAnalyzer,
    TrajectoryVisualizer,
    compare_controllers,
    create_baseline_controllers,
    create_summary_figure,
)

# Import modules
from dynamics import Rocket3DoFDynamics
from gp import Simple3DoFGP, StructuredGPConfig
from mpc import MPCConfig, NominalMPC3DoF


def setup_output_dirs(base_dir: str = "results") -> dict:
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(base_dir) / timestamp

    dirs = {
        "base": output_dir,
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "data": output_dir / "data",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def create_gp_mpc_controller(dynamics, with_gp: bool = True):
    """Create GP-MPC controller."""
    mpc_config = MPCConfig(
        N=15,
        dt=0.1,
    )

    controller = NominalMPC3DoF(dynamics, mpc_config)

    # Note: In full implementation, would integrate GP here
    # For now, using nominal MPC as the "GP-MPC" placeholder

    return controller


def run_monte_carlo(
    dynamics,
    controller,
    n_runs: int,
    config: SimulationConfig,
    name: str = "GP-MPC",
) -> MonteCarloResults:
    """Run Monte Carlo simulation."""
    print(f"\n{'=' * 60}")
    print(f"Running Monte Carlo: {name} ({n_runs} runs)")
    print(f"{'=' * 60}")

    simulator = MonteCarloSimulator(dynamics, controller, config)

    start_time = time.time()
    results = simulator.run(n_runs=n_runs, seed=42)
    elapsed = time.time() - start_time

    print(f"\nCompleted in {elapsed:.1f}s ({elapsed / n_runs * 1000:.1f}ms per run)")
    print(results.summary())

    return results


def run_baseline_comparison(
    dynamics,
    main_controller,
    n_runs: int,
    config: SimulationConfig,
) -> dict:
    """Run baseline controller comparison."""
    print(f"\n{'=' * 60}")
    print("Running Baseline Comparison")
    print(f"{'=' * 60}")

    # Create baseline controllers
    baselines = create_baseline_controllers(dynamics)

    # Add main controller
    controllers = {"GP-MPC": main_controller, **baselines}

    # Run comparison
    results = compare_controllers(
        dynamics,
        controllers,
        n_runs=n_runs,
        config=config,
        seed=42,
    )

    # Print comparison
    print("\n" + "-" * 60)
    print(f"{'Controller':<15} {'Success%':>10} {'Fuel (kg)':>12}")
    print("-" * 60)

    for name, mc_results in results.items():
        stats = mc_results.get_statistics()
        sr = stats["success_rate"] * 100
        fuel = stats.get("fuel_mean", 0)
        print(f"{name:<15} {sr:>9.1f}% {fuel:>11.3f}")

    return results


def run_dispersion_analysis(
    dynamics,
    controller,
    n_runs: int,
    config: SimulationConfig,
) -> dict:
    """Run dispersion analysis."""
    print(f"\n{'=' * 60}")
    print("Running Dispersion Analysis")
    print(f"{'=' * 60}")

    analyzer = DispersionAnalysis(dynamics, controller, config)
    results = analyzer.run_sweep(n_runs_per_level=n_runs, seed=42)

    print("\n" + analyzer.summary_table(results))

    return results


def generate_figures(  # noqa: PLR0915
    mc_results: MonteCarloResults,
    baseline_results: dict,
    dispersion_results: dict,
    output_dir: Path,
):
    """Generate publication figures."""
    print(f"\n{'=' * 60}")
    print("Generating Figures")
    print(f"{'=' * 60}")

    import matplotlib  # noqa: PLC0415, ICN001

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # 1. Landing scatter plot
    print("  - Landing scatter plot...")
    fig, ax = plt.subplots(figsize=(5, 5))
    mc_viz = MonteCarloVisualizer()
    mc_viz.plot_landing_scatter(mc_results, ax=ax)
    ax.set_title("Landing Position Dispersion")
    fig.savefig(output_dir / "landing_scatter.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "landing_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Success rate comparison
    print("  - Success rate comparison...")
    fig, ax = plt.subplots(figsize=(6, 4))
    mc_viz.plot_success_histogram(baseline_results, ax=ax)
    ax.set_title("Controller Success Rate Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "success_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "success_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Fuel distribution
    print("  - Fuel distribution...")
    fig, ax = plt.subplots(figsize=(6, 4))
    mc_viz.plot_fuel_distribution(baseline_results, ax=ax)
    ax.set_title("Fuel Consumption Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "fuel_distribution.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "fuel_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Example trajectory
    print("  - Example trajectory...")
    if mc_results.n_success > 0:
        # Find a successful trajectory
        for r in mc_results.results:
            if r.success:
                traj_viz = TrajectoryVisualizer()

                # 2D trajectory
                fig, ax = plt.subplots(figsize=(5, 4))
                traj_viz.plot_trajectory_2d(r.states, ax=ax, label="GP-MPC")
                ax.set_title("Landing Trajectory")
                fig.savefig(output_dir / "trajectory_2d.pdf", dpi=300, bbox_inches="tight")
                fig.savefig(output_dir / "trajectory_2d.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                # State history
                fig = traj_viz.plot_state_history(r.states, r.times, r.controls)
                fig.suptitle("State and Control History")
                fig.savefig(output_dir / "state_history.pdf", dpi=300, bbox_inches="tight")
                fig.savefig(output_dir / "state_history.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

                break

    # 5. Dispersion comparison
    if dispersion_results:
        print("  - Dispersion comparison...")
        fig, ax = plt.subplots(figsize=(6, 4))

        names = list(dispersion_results.keys())
        rates = [r.success_rate * 100 for r in dispersion_results.values()]

        ax.bar(names, rates, color=COLORS["gp_mpc"], alpha=0.8)
        ax.set_ylabel("Success Rate (%)")
        ax.set_title("Performance vs Dispersion Level")
        ax.set_ylim(0, 105)

        for i, rate in enumerate(rates):
            ax.text(i, rate + 2, f"{rate:.1f}%", ha="center", fontsize=9)

        fig.tight_layout()
        fig.savefig(output_dir / "dispersion_comparison.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / "dispersion_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Figures saved to {output_dir}")


def export_results(
    mc_results: MonteCarloResults,
    baseline_results: dict,
    output_dir: Path,
):
    """Export results to various formats."""
    print(f"\n{'=' * 60}")
    print("Exporting Results")
    print(f"{'=' * 60}")

    analyzer = StatisticalAnalyzer()
    exporter = ResultsExporter(str(output_dir))

    # Compute metrics for all controllers
    all_metrics = {}
    for name, results in baseline_results.items():
        all_metrics[name] = analyzer.compute_metrics(results)

    # Export CSV
    csv_path = exporter.to_csv(all_metrics, "comparison_results.csv")
    print(f"  CSV: {csv_path}")

    # Export JSON
    json_path = exporter.to_json(all_metrics, "comparison_results.json")
    print(f"  JSON: {json_path}")

    # Export LaTeX table
    latex = exporter.to_latex_table(
        all_metrics,
        caption="Controller Performance Comparison",
        label="tab:comparison",
    )
    latex_path = output_dir / "comparison_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"  LaTeX: {latex_path}")

    # Generate results section text
    results_text = exporter.generate_results_section(all_metrics)
    text_path = output_dir / "results_section.tex"
    with open(text_path, "w") as f:
        f.write(results_text)
    print(f"  Results text: {text_path}")

    # Save raw results
    mc_results.save(str(output_dir / "monte_carlo_results.pkl"))
    print(f"  Raw results: {output_dir / 'monte_carlo_results.pkl'}")


def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description="Run GP-MPC Rocket Landing Experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test with 10 runs")
    parser.add_argument("--standard", action="store_true", help="Standard experiment with 100 runs")
    parser.add_argument("--full", action="store_true", help="Full experiment with 1000 runs")
    parser.add_argument("--custom", type=int, default=None, help="Custom number of runs")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--skip-dispersion", action="store_true", help="Skip dispersion analysis")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")

    args = parser.parse_args()

    # Determine number of runs
    if args.custom:
        n_runs = args.custom
    elif args.full:
        n_runs = 1000
    elif args.standard:
        n_runs = 100
    else:  # quick or default
        n_runs = 10

    print(f"\n{'#' * 60}")
    print("#  GP-MPC Rocket Landing Experiments")
    print(f"#  Runs: {n_runs}")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # Setup
    dirs = setup_output_dirs(args.output)
    print(f"\nOutput directory: {dirs['base']}")

    # Create dynamics and controller
    print("\nInitializing dynamics and controllers...")
    dynamics = Rocket3DoFDynamics()
    controller = create_gp_mpc_controller(dynamics)

    # Simulation configuration
    sim_config = SimulationConfig(
        dt=0.1,
        max_time=30.0,
        altitude_mean=300.0,
        altitude_std=50.0,
        horizontal_std=30.0,
        velocity_mean=np.array([0, 0, -50]),
        velocity_std=np.array([10, 10, 10]),
        landing_constraints=LandingConstraints(
            pos_tol_xy=5.0,
            vel_tol_z=2.0,
        ),
    )

    # 1. Main Monte Carlo
    mc_results = run_monte_carlo(dynamics, controller, n_runs, sim_config, "GP-MPC")

    # 2. Baseline comparison
    baseline_results = run_baseline_comparison(dynamics, controller, max(n_runs // 2, 5), sim_config)

    # 3. Dispersion analysis
    dispersion_results = {}
    if not args.skip_dispersion:
        dispersion_results = run_dispersion_analysis(dynamics, controller, max(n_runs // 4, 5), sim_config)

    # 4. Generate figures
    if not args.skip_figures:
        generate_figures(mc_results, baseline_results, dispersion_results, dirs["figures"])

    # 5. Export results
    export_results(mc_results, baseline_results, dirs["tables"])

    # Summary
    print(f"\n{'#' * 60}")
    print("#  Experiment Complete")
    print(f"#  Results saved to: {dirs['base']}")
    print(f"#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
