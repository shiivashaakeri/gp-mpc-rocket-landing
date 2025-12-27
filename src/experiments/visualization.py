"""
Visualization Module for Publication-Quality Figures

Generates publication-ready figures for GP-MPC rocket landing:
1. Trajectory plots (2D, 3D)
2. State and control time histories
3. Monte Carlo scatter plots
4. Dispersion ellipses
5. Learning curves
6. Comparison bar charts
7. GP uncertainty visualization

All figures follow academic publication standards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

# Publication-quality settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
    }
)

# Color scheme (colorblind-friendly)
COLORS = {
    "gp_mpc": "#0072B2",  # Blue
    "nominal": "#D55E00",  # Orange
    "lqr": "#009E73",  # Green
    "tube_mpc": "#CC79A7",  # Pink
    "safety": "#F0E442",  # Yellow
    "reference": "#999999",  # Gray
    "success": "#56B4E9",  # Light blue
    "failure": "#E69F00",  # Amber
}


@dataclass
class FigureConfig:
    """Configuration for figure generation."""

    # Figure sizes (inches)
    single_column_width: float = 3.5
    double_column_width: float = 7.0
    height_ratio: float = 0.75

    # File format
    save_format: str = "pdf"

    # Style
    use_latex: bool = False
    color_scheme: str = "colorblind"

    def get_figsize(self, columns: int = 1) -> Tuple[float, float]:
        """Get figure size."""
        width = self.single_column_width if columns == 1 else self.double_column_width
        return (width, width * self.height_ratio)


class TrajectoryVisualizer:
    """
    Visualize rocket landing trajectories.
    """

    def __init__(self, config: Optional[FigureConfig] = None):
        """Initialize visualizer."""
        self.config = config or FigureConfig()

    def plot_trajectory_2d(
        self,
        states: NDArray,
        ax: Optional[plt.Axes] = None,
        label: str = "",
        color: Optional[str] = None,
        show_velocity: bool = False,
        velocity_scale: float = 0.5,
    ) -> plt.Axes:
        """
        Plot 2D trajectory (altitude vs downrange).

        Args:
            states: State trajectory (T+1, n_x)
            ax: Matplotlib axes
            label: Legend label
            color: Line color
            show_velocity: Show velocity vectors
            velocity_scale: Scale for velocity arrows

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        color = color or COLORS["gp_mpc"]

        # Downrange distance
        downrange = np.sqrt(states[:, 1] ** 2 + states[:, 2] ** 2)
        altitude = states[:, 3]

        ax.plot(downrange, altitude, color=color, label=label)

        # Start and end markers
        ax.plot(downrange[0], altitude[0], "o", color=color, markersize=8)
        ax.plot(downrange[-1], altitude[-1], "s", color=color, markersize=8)

        # Velocity vectors
        if show_velocity:
            step = max(1, len(states) // 10)
            for i in range(0, len(states), step):
                vr = (states[i, 4] * states[i, 1] + states[i, 5] * states[i, 2]) / (downrange[i] + 1e-6)
                vz = states[i, 6]
                ax.arrow(
                    downrange[i],
                    altitude[i],
                    vr * velocity_scale,
                    vz * velocity_scale,
                    head_width=5,
                    head_length=3,
                    fc=color,
                    ec=color,
                    alpha=0.5,
                )

        # Ground line
        ax.axhline(y=0, color="brown", linestyle="--", linewidth=1)

        ax.set_xlabel("Downrange (m)")
        ax.set_ylabel("Altitude (m)")
        ax.set_xlim(left=-10)
        ax.set_ylim(bottom=-10)

        if label:
            ax.legend()

        return ax

    def plot_trajectory_3d(
        self,
        states: NDArray,
        ax: Optional[Axes3D] = None,
        label: str = "",
        color: Optional[str] = None,
        show_landing_zone: bool = True,
    ) -> Axes3D:
        """
        Plot 3D trajectory.

        Args:
            states: State trajectory (T+1, n_x)
            ax: 3D axes
            label: Legend label
            color: Line color
            show_landing_zone: Show landing target zone

        Returns:
            3D axes
        """
        if ax is None:
            fig = plt.figure(figsize=self.config.get_figsize(2))
            ax = fig.add_subplot(111, projection="3d")

        color = color or COLORS["gp_mpc"]

        x, y, z = states[:, 1], states[:, 2], states[:, 3]

        ax.plot(x, y, z, color=color, label=label, linewidth=2)

        # Start and end markers
        ax.scatter([x[0]], [y[0]], [z[0]], color=color, s=100, marker="o")
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color=color, s=100, marker="s")

        # Landing zone
        if show_landing_zone:
            theta = np.linspace(0, 2 * np.pi, 50)
            r = 5.0  # Landing zone radius
            ax.plot(r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta), "g--", alpha=0.5, linewidth=1)

        # Ground plane
        xx, yy = np.meshgrid(np.linspace(-100, 100, 10), np.linspace(-100, 100, 10))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color="brown")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Altitude (m)")

        if label:
            ax.legend()

        return ax

    def plot_state_history(
        self,
        states: NDArray,
        times: NDArray,
        controls: Optional[NDArray] = None,
        fig: Optional[plt.Figure] = None,
    ) -> plt.Figure:
        """
        Plot state and control time histories.

        Args:
            states: State trajectory (T+1, n_x)
            times: Time array (T+1,)
            controls: Control trajectory (T, n_u)
            fig: Matplotlib figure

        Returns:
            Figure
        """
        n_plots = 4 if controls is not None else 3

        if fig is None:
            fig, axes = plt.subplots(
                n_plots,
                1,
                figsize=(self.config.double_column_width, self.config.double_column_width * 0.8),
                sharex=True,
            )
        else:
            axes = fig.axes

        # Mass
        axes[0].plot(times, states[:, 0], color=COLORS["gp_mpc"])
        axes[0].set_ylabel("Mass (kg)")
        axes[0].set_ylim(bottom=0)

        # Position
        axes[1].plot(times, states[:, 1], label="x", color="tab:red")
        axes[1].plot(times, states[:, 2], label="y", color="tab:green")
        axes[1].plot(times, states[:, 3], label="z", color="tab:blue")
        axes[1].set_ylabel("Position (m)")
        axes[1].legend(loc="upper right", ncol=3)

        # Velocity
        axes[2].plot(times, states[:, 4], label="vx", color="tab:red")
        axes[2].plot(times, states[:, 5], label="vy", color="tab:green")
        axes[2].plot(times, states[:, 6], label="vz", color="tab:blue")
        axes[2].set_ylabel("Velocity (m/s)")
        axes[2].legend(loc="upper right", ncol=3)

        # Controls
        if controls is not None and len(controls) > 0:
            t_ctrl = times[:-1]
            axes[3].plot(t_ctrl, controls[:, 0], label="T", color="tab:purple")
            if controls.shape[1] > 1:
                axes[3].plot(t_ctrl, controls[:, 1], label="δx", color="tab:orange")
                axes[3].plot(t_ctrl, controls[:, 2], label="δy", color="tab:cyan")
            axes[3].set_ylabel("Control")
            axes[3].set_xlabel("Time (s)")
            axes[3].legend(loc="upper right", ncol=3)
        else:
            axes[-1].set_xlabel("Time (s)")

        fig.tight_layout()

        return fig


class MonteCarloVisualizer:
    """
    Visualize Monte Carlo simulation results.
    """

    def __init__(self, config: Optional[FigureConfig] = None):
        """Initialize visualizer."""
        self.config = config or FigureConfig()

    def plot_landing_scatter(
        self,
        results,  # MonteCarloResults
        ax: Optional[plt.Axes] = None,
        show_ellipse: bool = True,
        confidence: float = 0.95,
    ) -> plt.Axes:
        """
        Plot landing position scatter.

        Args:
            results: Monte Carlo results
            ax: Matplotlib axes
            show_ellipse: Show dispersion ellipse
            confidence: Confidence level for ellipse

        Returns:
            Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        # Extract final positions
        x_land = []
        y_land = []
        success = []

        for r in results.results:
            x_land.append(r.states[-1, 1])
            y_land.append(r.states[-1, 2])
            success.append(r.success)

        x_land = np.array(x_land)
        y_land = np.array(y_land)
        success = np.array(success)

        # Scatter plot
        ax.scatter(x_land[success], y_land[success], c=COLORS["success"], alpha=0.5, s=20, label="Success")
        ax.scatter(
            x_land[~success], y_land[~success], c=COLORS["failure"], alpha=0.5, s=20, marker="x", label="Failure"
        )

        # Landing zone
        circle = plt.Circle((0, 0), 5, fill=False, color="green", linestyle="--", linewidth=2)
        ax.add_patch(circle)

        # Dispersion ellipse
        if show_ellipse and np.sum(success) > 2:
            self._add_ellipse(ax, x_land[success], y_land[success], confidence, color=COLORS["gp_mpc"])

        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_aspect("equal")
        ax.legend()

        # Set reasonable limits
        max_range = max(np.max(np.abs(x_land)), np.max(np.abs(y_land)), 20)
        ax.set_xlim(-max_range * 1.1, max_range * 1.1)
        ax.set_ylim(-max_range * 1.1, max_range * 1.1)

        return ax

    def _add_ellipse(
        self,
        ax: plt.Axes,
        x: NDArray,
        y: NDArray,
        confidence: float,
        color: str,
    ) -> None:
        """Add confidence ellipse to plot."""
        from scipy import stats  # noqa: PLC0415

        center = np.array([np.mean(x), np.mean(y)])
        cov = np.cov(x, y)

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

        chi2 = stats.chi2.ppf(confidence, 2)
        width = 2 * np.sqrt(eigenvalues[1] * chi2)
        height = 2 * np.sqrt(eigenvalues[0] * chi2)

        ellipse = patches.Ellipse(
            center,
            width,
            height,
            angle=angle,
            fill=False,
            color=color,
            linewidth=2,
            label=f"{confidence * 100:.0f}% confidence",
        )
        ax.add_patch(ellipse)

    def plot_success_histogram(
        self,
        results_dict: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot success rate comparison bar chart.

        Args:
            results_dict: Dictionary of {name: results}
            ax: Matplotlib axes

        Returns:
            Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        names = list(results_dict.keys())
        success_rates = [r.success_rate * 100 for r in results_dict.values()]

        colors = [COLORS.get(n.lower().replace("-", "_"), COLORS["gp_mpc"]) for n in names]

        bars = ax.bar(names, success_rates, color=colors, alpha=0.8)

        # Add value labels
        for bar, rate in zip(bars, success_rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylabel("Success Rate (%)")
        ax.set_ylim(0, 105)
        ax.set_xticklabels(names, rotation=45, ha="right")

        return ax

    def plot_fuel_distribution(
        self,
        results_dict: Dict[str, Any],
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot fuel consumption distribution.

        Args:
            results_dict: Dictionary of {name: results}
            ax: Matplotlib axes

        Returns:
            Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        data = []
        labels = []

        for name, results in results_dict.items():
            fuel = [r.fuel_used for r in results.results if r.success]
            if fuel:
                data.append(fuel)
                labels.append(name)

        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Color boxes
        for patch, name in zip(bp["boxes"], labels):
            color = COLORS.get(name.lower().replace("-", "_"), COLORS["gp_mpc"])
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel("Fuel Used (kg)")
        ax.set_xticklabels(labels, rotation=45, ha="right")

        return ax


class GPVisualizer:
    """
    Visualize GP model predictions and uncertainty.
    """

    def __init__(self, config: Optional[FigureConfig] = None):
        """Initialize visualizer."""
        self.config = config or FigureConfig()

    def plot_prediction_1d(
        self,
        x_train: NDArray,
        y_train: NDArray,
        x_test: NDArray,
        mean: NDArray,
        std: NDArray,
        ax: Optional[plt.Axes] = None,
        dim: int = 0,
    ) -> plt.Axes:
        """
        Plot 1D GP prediction with uncertainty.

        Args:
            x_train: Training inputs
            y_train: Training outputs
            x_test: Test inputs
            mean: Predictive mean
            std: Predictive std
            ax: Matplotlib axes
            dim: Dimension to plot

        Returns:
            Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        # Sort for plotting
        sort_idx = np.argsort(x_test[:, dim])
        x_plot = x_test[sort_idx, dim]
        mean_plot = mean[sort_idx]
        std_plot = std[sort_idx]

        # Prediction
        ax.plot(x_plot, mean_plot, color=COLORS["gp_mpc"], label="Mean")
        ax.fill_between(
            x_plot, mean_plot - 2 * std_plot, mean_plot + 2 * std_plot, color=COLORS["gp_mpc"], alpha=0.2, label="±2σ"  # noqa: RUF001
        )

        # Training data
        ax.scatter(x_train[:, dim], y_train, c="black", s=20, zorder=5, label="Training data")

        ax.set_xlabel(f"Input dimension {dim}")
        ax.set_ylabel("Output")
        ax.legend()

        return ax

    def plot_learning_curve(
        self,
        iterations: NDArray,
        metrics: Dict[str, NDArray],
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot learning curve (performance vs iterations).

        Args:
            iterations: Iteration numbers
            metrics: Dictionary of metric arrays
            ax: Matplotlib axes

        Returns:
            Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.get_figsize())

        for name, values in metrics.items():
            ax.plot(iterations, values, label=name, linewidth=2)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Performance")
        ax.legend()
        ax.set_xlim(left=0)

        return ax


def create_summary_figure(
    trajectory_results,
    mc_results,
    baseline_results: Optional[Dict] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create comprehensive summary figure for publication.

    4-panel figure:
    1. Example trajectory (3D)
    2. Landing scatter with dispersion ellipse
    3. Success rate comparison
    4. State time history

    Args:
        trajectory_results: Example trajectory (states, controls, times)
        mc_results: Monte Carlo results
        baseline_results: Optional baseline comparison
        save_path: Path to save figure

    Returns:
        Figure
    """
    fig = plt.figure(figsize=(7.0, 6.0))

    # 3D trajectory
    ax1 = fig.add_subplot(221, projection="3d")
    traj_viz = TrajectoryVisualizer()
    states, controls, times = trajectory_results
    traj_viz.plot_trajectory_3d(states, ax=ax1, label="GP-MPC")
    ax1.set_title("(a) Landing Trajectory")

    # Landing scatter
    ax2 = fig.add_subplot(222)
    mc_viz = MonteCarloVisualizer()
    mc_viz.plot_landing_scatter(mc_results, ax=ax2)
    ax2.set_title("(b) Landing Dispersion")

    # Success comparison
    ax3 = fig.add_subplot(223)
    all_results = {"GP-MPC": mc_results, **baseline_results} if baseline_results else {"GP-MPC": mc_results}
    mc_viz.plot_success_histogram(all_results, ax=ax3)
    ax3.set_title("(c) Success Rate Comparison")

    # State history
    ax4 = fig.add_subplot(224)
    traj_viz.plot_trajectory_2d(states, ax=ax4, label="GP-MPC")
    ax4.set_title("(d) Altitude Profile")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {save_path}")

    return fig


def create_ablation_figure(
    ablation_results,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create ablation study figure.

    Args:
        ablation_results: Ablation study results
        save_path: Path to save figure

    Returns:
        Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Success rates
    names = [r.config_name[:15] for r in ablation_results.results]
    rates = [r.success_rate * 100 for r in ablation_results.results]

    colors = [
        COLORS["gp_mpc"] if r.success_rate == max(rates) / 100 else COLORS["nominal"] for r in ablation_results.results
    ]

    ax1.barh(names, rates, color=colors, alpha=0.8)
    ax1.set_xlabel("Success Rate (%)")
    ax1.set_xlim(0, 105)
    ax1.set_title("(a) Ablation Study Results")

    # Fuel consumption
    fuels = [r.fuel_mean for r in ablation_results.results]
    fuel_stds = [r.fuel_std for r in ablation_results.results]

    ax2.barh(names, fuels, xerr=fuel_stds, color=colors, alpha=0.8)
    ax2.set_xlabel("Fuel Used (kg)")
    ax2.set_title("(b) Fuel Consumption")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
