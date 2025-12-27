"""
Ablation Study Framework

Systematic evaluation of GP-MPC component contributions:
1. GP model ablation (with/without learning)
2. Safety filter ablation
3. LMPC terminal set ablation
4. Kernel choice ablation
5. Horizon length ablation
6. Online learning ablation

Identifies which components provide the most benefit.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class AblationComponent(Enum):
    """Components that can be ablated."""

    GP_MODEL = auto()  # Gaussian Process learning
    SAFETY_FILTER = auto()  # Safety filter
    TERMINAL_SET = auto()  # LMPC terminal set constraints
    ONLINE_LEARNING = auto()  # Online GP updates
    UNCERTAINTY_PROP = auto()  # Uncertainty propagation in MPC
    WARM_START = auto()  # MPC warm starting


@dataclass
class AblationConfig:
    """Configuration for ablation study."""

    # Which components to ablate
    components: List[AblationComponent] = field(default_factory=lambda: list(AblationComponent))

    # Number of runs per configuration
    n_runs: int = 100

    # Random seed
    seed: int = 42

    # Additional configurations to sweep
    horizon_sweep: List[int] = field(default_factory=lambda: [10, 15, 20, 30])
    kernel_sweep: List[str] = field(default_factory=lambda: ["se_ard", "matern32", "matern52"])


@dataclass
class AblationResult:
    """Result from a single ablation configuration."""

    config_name: str
    enabled_components: List[AblationComponent]
    disabled_components: List[AblationComponent]

    # Performance metrics
    success_rate: float
    success_rate_ci: Tuple[float, float]
    fuel_mean: float
    fuel_std: float
    time_mean: float
    compute_mean_ms: float

    # Additional metrics
    constraint_violations: int
    crashes: int

    # Raw results reference
    n_runs: int


@dataclass
class AblationStudyResults:
    """Complete ablation study results."""

    results: List[AblationResult]
    baseline_name: str

    def get_component_impact(
        self,
        component: AblationComponent,
    ) -> Dict[str, float]:
        """
        Compute impact of a single component.

        Compares configurations with and without the component.
        """
        with_component = [r for r in self.results if component in r.enabled_components]
        without_component = [r for r in self.results if component in r.disabled_components]

        if not with_component or not without_component:
            return {"impact": 0.0}

        # Average improvement
        sr_with = np.mean([r.success_rate for r in with_component])
        sr_without = np.mean([r.success_rate for r in without_component])

        fuel_with = np.mean([r.fuel_mean for r in with_component])
        fuel_without = np.mean([r.fuel_mean for r in without_component])

        return {
            "success_rate_improvement": sr_with - sr_without,
            "success_rate_with": sr_with,
            "success_rate_without": sr_without,
            "fuel_reduction": fuel_without - fuel_with,
            "fuel_with": fuel_with,
            "fuel_without": fuel_without,
        }

    def summary(self) -> str:
        """Generate ablation study summary."""
        lines = [
            "=" * 70,
            "Ablation Study Results",
            "=" * 70,
            "",
            f"{'Configuration':<35} {'Success%':>10} {'Fuel':>10} {'Time':>8}",
            "-" * 70,
        ]

        # Sort by success rate
        sorted_results = sorted(self.results, key=lambda r: r.success_rate, reverse=True)

        for r in sorted_results:
            disabled_str = ", ".join(c.name for c in r.disabled_components) or "None"
            if len(disabled_str) > 30:
                disabled_str = disabled_str[:27] + "..."

            lines.append(
                f"w/o {disabled_str:<30} {r.success_rate * 100:>9.1f}% {r.fuel_mean:>9.3f} {r.time_mean:>7.2f}s"
            )

        lines.extend(
            [
                "",
                "-" * 70,
                "Component Impact Analysis:",
                "",
            ]
        )

        for component in AblationComponent:
            impact = self.get_component_impact(component)
            if impact["impact"] != 0.0 or "success_rate_improvement" in impact:
                sr_imp = impact.get("success_rate_improvement", 0) * 100
                fuel_red = impact.get("fuel_reduction", 0)
                lines.append(f"  {component.name:<25}: SR +{sr_imp:+.1f}%, Fuel {fuel_red:+.3f} kg")

        lines.append("=" * 70)

        return "\n".join(lines)

    def to_latex_table(self) -> str:
        """Generate LaTeX table for publication."""
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            r"\caption{Ablation Study Results}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Configuration & Success Rate (\%) & Fuel (kg) & Time (s) \\",
            r"\midrule",
        ]

        for r in self.results:
            disabled = ", ".join(c.name.replace("_", " ") for c in r.disabled_components)
            disabled = "Full System" if not disabled else f"w/o {disabled}"

            lines.append(f"{disabled} & {r.success_rate * 100:.1f} & {r.fuel_mean:.3f} & {r.time_mean:.2f} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)


class AblationStudy:
    """
    Framework for conducting ablation studies.

    Systematically enables/disables components to measure their impact.

    Example:
        >>> study = AblationStudy(dynamics, full_controller, config)
        >>> results = study.run()
        >>> print(results.summary())
    """

    def __init__(
        self,
        dynamics,
        controller_factory: Callable[..., Any],
        monte_carlo_runner,
        config: Optional[AblationConfig] = None,
    ):
        """
        Initialize ablation study.

        Args:
            dynamics: Rocket dynamics
            controller_factory: Function that creates controller with given components
            monte_carlo_runner: Function to run Monte Carlo simulation
            config: Ablation configuration
        """
        self.dynamics = dynamics
        self.controller_factory = controller_factory
        self.mc_runner = monte_carlo_runner
        self.config = config or AblationConfig()

    def _create_controller(
        self,
        enabled_components: List[AblationComponent],
    ) -> Any:
        """Create controller with specified components enabled."""
        kwargs = {
            "use_gp": AblationComponent.GP_MODEL in enabled_components,
            "use_safety": AblationComponent.SAFETY_FILTER in enabled_components,
            "use_terminal_set": AblationComponent.TERMINAL_SET in enabled_components,
            "use_online_learning": AblationComponent.ONLINE_LEARNING in enabled_components,
            "use_uncertainty": AblationComponent.UNCERTAINTY_PROP in enabled_components,
            "use_warm_start": AblationComponent.WARM_START in enabled_components,
        }
        return self.controller_factory(**kwargs)

    def run_single_config(
        self,
        enabled_components: List[AblationComponent],
    ) -> AblationResult:
        """Run ablation for single configuration."""
        disabled = [c for c in AblationComponent if c not in enabled_components]

        config_name = "Full" if not disabled else f"w/o {','.join(c.name for c in disabled)}"
        print(f"  Running: {config_name}")

        # Create controller
        controller = self._create_controller(enabled_components)

        # Run Monte Carlo
        results = self.mc_runner(controller, n_runs=self.config.n_runs)

        stats = results.get_statistics()

        return AblationResult(
            config_name=config_name,
            enabled_components=enabled_components,
            disabled_components=disabled,
            success_rate=stats["success_rate"],
            success_rate_ci=stats["success_rate_ci"],
            fuel_mean=stats.get("fuel_mean", 0.0),
            fuel_std=stats.get("fuel_std", 0.0),
            time_mean=stats.get("time_mean", 0.0),
            compute_mean_ms=stats.get("compute_mean_ms", 0.0),
            constraint_violations=stats["outcomes"].get("CONSTRAINT_VIOLATION", 0),
            crashes=stats["outcomes"].get("CRASH", 0),
            n_runs=stats["n_runs"],
        )

    def run(self) -> AblationStudyResults:
        """
        Run complete ablation study.

        Tests all combinations of component inclusion/exclusion.
        """
        all_components = list(AblationComponent)
        results = []

        print("Running Ablation Study")
        print("=" * 50)

        # Full system (baseline)
        print("\n1. Full System (all components enabled)")
        full_result = self.run_single_config(all_components)
        results.append(full_result)

        # Single component ablation
        print("\n2. Single Component Ablation")
        for component in self.config.components:
            enabled = [c for c in all_components if c != component]
            result = self.run_single_config(enabled)
            results.append(result)

        # Optional: Pairwise ablation (for interaction effects)
        if len(self.config.components) <= 4:
            print("\n3. Pairwise Component Ablation")
            for c1, c2 in itertools.combinations(self.config.components, 2):
                enabled = [c for c in all_components if c not in [c1, c2]]
                result = self.run_single_config(enabled)
                results.append(result)

        # Minimal system
        print("\n4. Minimal System (no learning components)")
        minimal = [
            c
            for c in all_components
            if c not in [AblationComponent.GP_MODEL, AblationComponent.ONLINE_LEARNING, AblationComponent.TERMINAL_SET]
        ]
        result = self.run_single_config(minimal)
        results.append(result)

        return AblationStudyResults(
            results=results,
            baseline_name="Full System",
        )


class HyperparameterSweep:
    """
    Sweep over hyperparameters to find best configuration.
    """

    def __init__(
        self,
        dynamics,
        base_controller_factory: Callable[..., Any],
        monte_carlo_runner,
    ):
        """Initialize hyperparameter sweep."""
        self.dynamics = dynamics
        self.controller_factory = base_controller_factory
        self.mc_runner = monte_carlo_runner

    def sweep_horizon(
        self,
        horizons: List[int],
        n_runs: int = 50,
    ) -> Dict[int, Dict[str, float]]:
        """
        Sweep MPC horizon length.

        Args:
            horizons: List of horizons to test
            n_runs: Runs per configuration

        Returns:
            Results per horizon
        """
        results = {}

        print("Horizon Sweep")
        print("-" * 40)

        for N in horizons:
            print(f"  Testing N={N}...")
            controller = self.controller_factory(horizon=N)
            mc_results = self.mc_runner(controller, n_runs=n_runs)
            stats = mc_results.get_statistics()

            results[N] = {
                "success_rate": stats["success_rate"],
                "fuel_mean": stats.get("fuel_mean", 0.0),
                "compute_mean_ms": stats.get("compute_mean_ms", 0.0),
            }

            print(f"    Success: {stats['success_rate'] * 100:.1f}%, Compute: {stats.get('compute_mean_ms', 0):.1f}ms")

        return results

    def sweep_gp_training_size(
        self,
        sizes: List[int],
        n_runs: int = 50,
    ) -> Dict[int, Dict[str, float]]:
        """
        Sweep GP training data size.

        Args:
            sizes: List of training set sizes to test
            n_runs: Runs per configuration

        Returns:
            Results per size
        """
        results = {}

        print("GP Training Size Sweep")
        print("-" * 40)

        for size in sizes:
            print(f"  Testing size={size}...")
            controller = self.controller_factory(gp_training_size=size)
            mc_results = self.mc_runner(controller, n_runs=n_runs)
            stats = mc_results.get_statistics()

            results[size] = {
                "success_rate": stats["success_rate"],
                "fuel_mean": stats.get("fuel_mean", 0.0),
                "compute_mean_ms": stats.get("compute_mean_ms", 0.0),
            }

        return results

    def sweep_kernel(
        self,
        kernels: List[str],
        n_runs: int = 50,
    ) -> Dict[str, Dict[str, float]]:
        """
        Sweep GP kernel choice.

        Args:
            kernels: List of kernel names to test
            n_runs: Runs per configuration

        Returns:
            Results per kernel
        """
        results = {}

        print("Kernel Sweep")
        print("-" * 40)

        for kernel in kernels:
            print(f"  Testing kernel={kernel}...")
            controller = self.controller_factory(kernel_type=kernel)
            mc_results = self.mc_runner(controller, n_runs=n_runs)
            stats = mc_results.get_statistics()

            results[kernel] = {
                "success_rate": stats["success_rate"],
                "fuel_mean": stats.get("fuel_mean", 0.0),
            }

        return results


def run_simple_ablation(
    dynamics,
    controller_with_gp,
    controller_without_gp,
    mc_config,
    n_runs: int = 100,
) -> Dict[str, Any]:
    """
    Run simplified ablation comparing with/without GP.

    Quick ablation for basic GP benefit measurement.

    Args:
        dynamics: Rocket dynamics
        controller_with_gp: Controller using GP model
        controller_without_gp: Controller without GP
        mc_config: Monte Carlo configuration
        n_runs: Number of runs

    Returns:
        Comparison results
    """
    from .monte_carlo import MonteCarloSimulator  # noqa: PLC0415

    results = {}

    # With GP
    print("Running with GP model...")
    sim_gp = MonteCarloSimulator(dynamics, controller_with_gp, mc_config)
    mc_gp = sim_gp.run(n_runs=n_runs)
    results["with_gp"] = mc_gp.get_statistics()

    # Without GP
    print("Running without GP model...")
    sim_no_gp = MonteCarloSimulator(dynamics, controller_without_gp, mc_config)
    mc_no_gp = sim_no_gp.run(n_runs=n_runs)
    results["without_gp"] = mc_no_gp.get_statistics()

    # Compute improvement
    results["improvement"] = {
        "success_rate": (results["with_gp"]["success_rate"] - results["without_gp"]["success_rate"]),
        "fuel_reduction": (results["without_gp"].get("fuel_mean", 0) - results["with_gp"].get("fuel_mean", 0)),
    }

    print("\nAblation Results:")
    print(f"  With GP:    {results['with_gp']['success_rate'] * 100:.1f}% success")
    print(f"  Without GP: {results['without_gp']['success_rate'] * 100:.1f}% success")
    print(f"  Improvement: {results['improvement']['success_rate'] * 100:+.1f}%")

    return results
