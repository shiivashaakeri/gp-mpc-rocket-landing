"""
Statistical Analysis and Results Module

Comprehensive analysis tools for GP-MPC experiments:
1. Statistical significance testing
2. Confidence interval computation
3. Effect size calculation
4. Results export (LaTeX tables, CSV)
5. Comparison metrics

Generates publication-ready analysis.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class StatisticalTest:
    """Result from statistical hypothesis test."""

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    confidence_level: float = 0.95
    effect_size: Optional[float] = None
    interpretation: str = ""


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    # Primary metrics
    success_rate: float
    success_rate_ci: Tuple[float, float]

    # Fuel metrics
    fuel_mean: float
    fuel_std: float
    fuel_median: float
    fuel_iqr: Tuple[float, float]

    # Timing metrics
    flight_time_mean: float
    flight_time_std: float
    compute_time_mean_ms: float
    compute_time_p95_ms: float

    # Accuracy metrics
    position_error_mean: float
    position_error_std: float
    velocity_error_mean: float
    velocity_error_std: float

    # Safety metrics
    constraint_violations: int
    crashes: int

    # Sample size
    n_runs: int
    n_success: int


class StatisticalAnalyzer:
    """
    Statistical analysis for Monte Carlo results.

    Provides significance testing and confidence intervals.
    """

    def __init__(self, confidence_level: float = 0.95):
        """Initialize analyzer."""
        self.confidence_level = confidence_level

    def compute_metrics(self, results) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.

        Args:
            results: Monte Carlo results

        Returns:
            Performance metrics
        """
        successful = [r for r in results.results if r.success]

        if len(successful) == 0:
            return PerformanceMetrics(
                success_rate=0.0,
                success_rate_ci=(0.0, 0.0),
                fuel_mean=0.0,
                fuel_std=0.0,
                fuel_median=0.0,
                fuel_iqr=(0.0, 0.0),
                flight_time_mean=0.0,
                flight_time_std=0.0,
                compute_time_mean_ms=0.0,
                compute_time_p95_ms=0.0,
                position_error_mean=0.0,
                position_error_std=0.0,
                velocity_error_mean=0.0,
                velocity_error_std=0.0,
                constraint_violations=0,
                crashes=0,
                n_runs=len(results.results),
                n_success=0,
            )

        fuel = np.array([r.fuel_used for r in successful])
        time = np.array([r.flight_time for r in successful])
        compute = np.array([r.compute_time_ms for r in successful])
        pos_err = np.array([r.final_position_error for r in successful])
        vel_err = np.array([r.final_velocity_error for r in successful])

        from .monte_carlo import LandingOutcome  # noqa: PLC0415

        return PerformanceMetrics(
            success_rate=len(successful) / len(results.results),
            success_rate_ci=self._binomial_ci(len(successful), len(results.results)),
            fuel_mean=float(np.mean(fuel)),
            fuel_std=float(np.std(fuel)),
            fuel_median=float(np.median(fuel)),
            fuel_iqr=(float(np.percentile(fuel, 25)), float(np.percentile(fuel, 75))),
            flight_time_mean=float(np.mean(time)),
            flight_time_std=float(np.std(time)),
            compute_time_mean_ms=float(np.mean(compute)),
            compute_time_p95_ms=float(np.percentile(compute, 95)),
            position_error_mean=float(np.mean(pos_err)),
            position_error_std=float(np.std(pos_err)),
            velocity_error_mean=float(np.mean(vel_err)),
            velocity_error_std=float(np.std(vel_err)),
            constraint_violations=sum(1 for r in results.results if r.outcome == LandingOutcome.CONSTRAINT_VIOLATION),
            crashes=sum(1 for r in results.results if r.outcome == LandingOutcome.CRASH),
            n_runs=len(results.results),
            n_success=len(successful),
        )

    def _binomial_ci(
        self,
        successes: int,
        trials: int,
    ) -> Tuple[float, float]:
        """Compute Wilson score confidence interval."""
        from scipy import stats  # noqa: PLC0415

        if trials == 0:
            return (0.0, 0.0)

        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
        p_hat = successes / trials

        denominator = 1 + z**2 / trials
        center = (p_hat + z**2 / (2 * trials)) / denominator
        margin = (z * np.sqrt(p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))) / denominator

        return (max(0, center - margin), min(1, center + margin))

    def compare_success_rates(
        self,
        results_a,
        results_b,
        name_a: str = "A",
        name_b: str = "B",
    ) -> StatisticalTest:
        """
        Compare success rates using Chi-squared test.

        Args:
            results_a: First results
            results_b: Second results
            name_a: Name of first condition
            name_b: Name of second condition

        Returns:
            Statistical test result
        """
        from scipy import stats  # noqa: PLC0415

        n_a = len(results_a.results)
        s_a = sum(1 for r in results_a.results if r.success)

        n_b = len(results_b.results)
        s_b = sum(1 for r in results_b.results if r.success)

        # Contingency table
        table = np.array([[s_a, n_a - s_a], [s_b, n_b - s_b]])

        chi2, p_value, dof, expected = stats.chi2_contingency(table)

        # Effect size (Cramér's V)
        n = n_a + n_b
        effect_size = np.sqrt(chi2 / n)

        significant = p_value < (1 - self.confidence_level)

        if significant:
            rate_a = s_a / n_a
            rate_b = s_b / n_b
            better = name_a if rate_a > rate_b else name_b
            interpretation = f"{better} has significantly higher success rate (p={p_value:.4f}, V={effect_size:.3f})"
        else:
            interpretation = f"No significant difference between {name_a} and {name_b} (p={p_value:.4f})"

        return StatisticalTest(
            test_name="Chi-squared test for success rate",
            statistic=chi2,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
        )

    def compare_means(
        self,
        values_a: NDArray,
        values_b: NDArray,
        name_a: str = "A",
        name_b: str = "B",
        metric_name: str = "metric",
    ) -> StatisticalTest:
        """
        Compare means using Welch's t-test.

        Args:
            values_a: First sample
            values_b: Second sample
            name_a: Name of first condition
            name_b: Name of second condition
            metric_name: Name of metric being compared

        Returns:
            Statistical test result
        """
        from scipy import stats  # noqa: PLC0415

        t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((np.std(values_a) ** 2 + np.std(values_b) ** 2) / 2)
        effect_size = abs(np.mean(values_a) - np.mean(values_b)) / (pooled_std + 1e-10)

        significant = p_value < (1 - self.confidence_level)

        if significant:
            mean_a = np.mean(values_a)
            mean_b = np.mean(values_b)
            better = name_a if mean_a < mean_b else name_b
            interpretation = f"{better} has significantly lower {metric_name} (p={p_value:.4f}, d={effect_size:.3f})"
        else:
            interpretation = f"No significant difference in {metric_name} (p={p_value:.4f})"

        return StatisticalTest(
            test_name=f"Welch's t-test for {metric_name}",
            statistic=t_stat,
            p_value=p_value,
            significant=significant,
            confidence_level=self.confidence_level,
            effect_size=effect_size,
            interpretation=interpretation,
        )


class ResultsExporter:
    """
    Export results to various formats.
    """

    def __init__(self, output_dir: str = "results"):
        """Initialize exporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_latex_table(
        self,
        results_dict: Dict[str, PerformanceMetrics],
        caption: str = "Performance Comparison",
        label: str = "tab:comparison",
    ) -> str:
        """
        Generate LaTeX table from results.

        Args:
            results_dict: Dictionary of {name: metrics}
            caption: Table caption
            label: Table label

        Returns:
            LaTeX table string
        """
        lines = [
            r"\begin{table}[h]",
            r"\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            (
                r"Controller & Success (\%) & Fuel (kg) & Time (s) & "
                r"Pos. Error (m) & Compute (ms) \\"
            ),
            r"\midrule",
        ]

        for name, metrics in results_dict.items():
            sr = f"{metrics.success_rate * 100:.1f}"
            fuel = f"{metrics.fuel_mean:.3f} $\\pm$ {metrics.fuel_std:.3f}"
            time = f"{metrics.flight_time_mean:.2f}"
            pos = f"{metrics.position_error_mean:.2f}"
            comp = f"{metrics.compute_time_mean_ms:.1f}"

            lines.append(f"{name} & {sr} & {fuel} & {time} & {pos} & {comp} \\\\")

        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )

        return "\n".join(lines)

    def to_csv(
        self,
        results_dict: Dict[str, PerformanceMetrics],
        filename: str = "results.csv",
    ) -> str:
        """
        Export results to CSV.

        Args:
            results_dict: Dictionary of {name: metrics}
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename

        headers = [
            "Controller",
            "Success_Rate",
            "Success_Rate_CI_Lower",
            "Success_Rate_CI_Upper",
            "Fuel_Mean",
            "Fuel_Std",
            "Flight_Time_Mean",
            "Compute_Time_Mean_ms",
            "Position_Error_Mean",
            "N_Runs",
            "N_Success",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for name, m in results_dict.items():
                writer.writerow(
                    [
                        name,
                        m.success_rate,
                        m.success_rate_ci[0],
                        m.success_rate_ci[1],
                        m.fuel_mean,
                        m.fuel_std,
                        m.flight_time_mean,
                        m.compute_time_mean_ms,
                        m.position_error_mean,
                        m.n_runs,
                        m.n_success,
                    ]
                )

        return str(filepath)

    def to_json(
        self,
        results_dict: Dict[str, PerformanceMetrics],
        filename: str = "results.json",
    ) -> str:
        """
        Export results to JSON.

        Args:
            results_dict: Dictionary of {name: metrics}
            filename: Output filename

        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename

        data = {}
        for name, m in results_dict.items():
            data[name] = {
                "success_rate": m.success_rate,
                "success_rate_ci": list(m.success_rate_ci),
                "fuel_mean": m.fuel_mean,
                "fuel_std": m.fuel_std,
                "fuel_median": m.fuel_median,
                "fuel_iqr": list(m.fuel_iqr),
                "flight_time_mean": m.flight_time_mean,
                "flight_time_std": m.flight_time_std,
                "compute_time_mean_ms": m.compute_time_mean_ms,
                "compute_time_p95_ms": m.compute_time_p95_ms,
                "position_error_mean": m.position_error_mean,
                "position_error_std": m.position_error_std,
                "velocity_error_mean": m.velocity_error_mean,
                "velocity_error_std": m.velocity_error_std,
                "constraint_violations": m.constraint_violations,
                "crashes": m.crashes,
                "n_runs": m.n_runs,
                "n_success": m.n_success,
            }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        return str(filepath)

    def generate_results_section(
        self,
        results_dict: Dict[str, PerformanceMetrics],
        test_results: Optional[List[StatisticalTest]] = None,
    ) -> str:
        """
        Generate text for results section.

        Args:
            results_dict: Dictionary of {name: metrics}
            test_results: Optional list of statistical tests

        Returns:
            Results text
        """
        lines = []

        # Summary paragraph
        lines.append("\\subsection{Monte Carlo Simulation Results}")
        lines.append("")

        for name, m in results_dict.items():
            lines.append(
                f"The {name} controller achieved a success rate of "
                f"{m.success_rate * 100:.1f}\\% "
                f"(95\\% CI: [{m.success_rate_ci[0] * 100:.1f}\\%, "
                f"{m.success_rate_ci[1] * 100:.1f}\\%]) "
                f"across {m.n_runs} Monte Carlo runs. "
                f"The mean fuel consumption was {m.fuel_mean:.3f} $\\pm$ "
                f"{m.fuel_std:.3f} kg, with an average flight time of "
                f"{m.flight_time_mean:.2f} s. "
                f"Position errors at landing were {m.position_error_mean:.2f} "
                f"$\\pm$ {m.position_error_std:.2f} m."
            )
            lines.append("")

        # Statistical comparisons
        if test_results:
            lines.append("\\subsection{Statistical Comparisons}")
            lines.append("")

            for test in test_results:
                lines.append(test.interpretation)
                lines.append("")

        return "\n".join(lines)


def compute_improvement(
    baseline: PerformanceMetrics,
    improved: PerformanceMetrics,
) -> Dict[str, float]:
    """
    Compute relative improvement from baseline.

    Args:
        baseline: Baseline metrics
        improved: Improved metrics

    Returns:
        Dictionary of improvements
    """
    return {
        "success_rate_improvement": (improved.success_rate - baseline.success_rate),
        "success_rate_relative": (
            (improved.success_rate - baseline.success_rate) / (baseline.success_rate + 1e-10) * 100
        ),
        "fuel_reduction": baseline.fuel_mean - improved.fuel_mean,
        "fuel_reduction_percent": ((baseline.fuel_mean - improved.fuel_mean) / (baseline.fuel_mean + 1e-10) * 100),
        "position_accuracy_improvement": (baseline.position_error_mean - improved.position_error_mean),
        "compute_time_change": (improved.compute_time_mean_ms - baseline.compute_time_mean_ms),
    }


def generate_summary_report(
    experiment_name: str,
    mc_results,
    baseline_results: Optional[Dict] = None,
    ablation_results=None,  # noqa: ARG001
    dispersion_results=None,  # noqa: ARG001
    output_dir: str = "results",
) -> str:
    """
    Generate comprehensive experiment report.

    Args:
        experiment_name: Name of experiment
        mc_results: Main Monte Carlo results
        baseline_results: Optional baseline comparisons
        ablation_results: Optional ablation study results
        dispersion_results: Optional dispersion analysis results
        output_dir: Output directory

    Returns:
        Path to report file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    analyzer = StatisticalAnalyzer()
    exporter = ResultsExporter(output_dir)

    lines = [
        f"# {experiment_name} - Experiment Report",
        "",
        "## Summary",
        "",
    ]

    # Main results
    main_metrics = analyzer.compute_metrics(mc_results)
    lines.extend(
        [
            f"- **Success Rate**: {main_metrics.success_rate * 100:.1f}% "
            f"(CI: [{main_metrics.success_rate_ci[0] * 100:.1f}%, "
            f"{main_metrics.success_rate_ci[1] * 100:.1f}%])",
            f"- **Fuel Used**: {main_metrics.fuel_mean:.3f} ± {main_metrics.fuel_std:.3f} kg",
            f"- **Flight Time**: {main_metrics.flight_time_mean:.2f} ± {main_metrics.flight_time_std:.2f} s",
            f"- **Position Error**: {main_metrics.position_error_mean:.2f} ± {main_metrics.position_error_std:.2f} m",
            f"- **Compute Time**: {main_metrics.compute_time_mean_ms:.2f} ms "
            f"(p95: {main_metrics.compute_time_p95_ms:.2f} ms)",
            "",
        ]
    )

    # Baseline comparisons
    if baseline_results:
        lines.extend(
            [
                "## Baseline Comparisons",
                "",
                "| Controller | Success (%) | Fuel (kg) | Compute (ms) |",
                "|------------|-------------|-----------|--------------|",
            ]
        )

        all_metrics = {"GP-MPC": main_metrics}
        for name, results in baseline_results.items():
            m = analyzer.compute_metrics(results)
            all_metrics[name] = m
            lines.append(f"| {name} | {m.success_rate * 100:.1f} | {m.fuel_mean:.3f} | {m.compute_time_mean_ms:.1f} |")

        lines.append("")

        # Export comparison
        exporter.to_csv(all_metrics, "comparison.csv")
        exporter.to_json(all_metrics, "comparison.json")

    # Save report
    report_path = output_path / f"{experiment_name}_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return str(report_path)
