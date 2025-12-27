"""
Experiments Module for GP-MPC Rocket Landing

Comprehensive experiment framework for validation:
- Monte Carlo simulation (1000+ runs)
- Baseline comparisons (LQR, PID, Tube-MPC)
- Ablation studies (component impact)
- Dispersion analysis (wind, aero, thrust)
- Publication-quality visualization
- Statistical analysis and reporting

Usage:
    >>> from src.experiments import MonteCarloSimulator, create_baseline_controllers
    >>>
    >>> # Run Monte Carlo
    >>> simulator = MonteCarloSimulator(dynamics, controller)
    >>> results = simulator.run(n_runs=1000)
    >>> print(results.summary())
    >>>
    >>> # Compare with baselines
    >>> baselines = create_baseline_controllers(dynamics)
    >>> comparison = compare_controllers(dynamics, baselines, n_runs=100)
"""

from .ablation import (
    AblationComponent,
    AblationConfig,
    AblationResult,
    AblationStudy,
    AblationStudyResults,
    HyperparameterSweep,
    run_simple_ablation,
)
from .analysis import (
    PerformanceMetrics,
    ResultsExporter,
    StatisticalAnalyzer,
    StatisticalTest,
    compute_improvement,
    generate_summary_report,
)
from .baselines import (
    BaselineComparison,
    LQRConfig,
    LQRController,
    LQRSolution,
    NominalMPCWrapper,
    OpenLoopController,
    PIDConfig,
    PIDController,
    PIDSolution,
    TubeMPCWrapper,
    create_baseline_controllers,
)
from .dispersion import (
    AeroDispersionConfig,
    DispersedDynamics,
    DispersionAnalysis,
    DispersionAnalysisResults,
    DispersionConfig,
    InitialConditionDispersion,
    ThrustDispersionConfig,
    WindConfig,
    WindModel,
)
from .monte_carlo import (
    LandingConstraints,
    LandingOutcome,
    MonteCarloResults,
    MonteCarloSimulator,
    SimulationConfig,
    SimulationResult,
    compare_controllers,
)
from .visualization import (
    COLORS,
    FigureConfig,
    GPVisualizer,
    MonteCarloVisualizer,
    TrajectoryVisualizer,
    create_ablation_figure,
    create_summary_figure,
)

__all__ = [
    "COLORS",
    # Ablation
    "AblationComponent",
    "AblationConfig",
    "AblationResult",
    "AblationStudy",
    "AblationStudyResults",
    "AeroDispersionConfig",
    "BaselineComparison",
    "DispersedDynamics",
    "DispersionAnalysis",
    "DispersionAnalysisResults",
    "DispersionConfig",
    # Visualization
    "FigureConfig",
    "GPVisualizer",
    "HyperparameterSweep",
    "InitialConditionDispersion",
    # Baselines
    "LQRConfig",
    "LQRController",
    "LQRSolution",
    "LandingConstraints",
    # Monte Carlo
    "LandingOutcome",
    "MonteCarloResults",
    "MonteCarloSimulator",
    "MonteCarloVisualizer",
    "NominalMPCWrapper",
    "OpenLoopController",
    "PIDConfig",
    "PIDController",
    "PIDSolution",
    "PerformanceMetrics",
    "ResultsExporter",
    "SimulationConfig",
    "SimulationResult",
    "StatisticalAnalyzer",
    # Analysis
    "StatisticalTest",
    "ThrustDispersionConfig",
    "TrajectoryVisualizer",
    "TubeMPCWrapper",
    "WindConfig",
    # Dispersion
    "WindModel",
    "compare_controllers",
    "compute_improvement",
    "create_ablation_figure",
    "create_baseline_controllers",
    "create_summary_figure",
    "generate_summary_report",
    "run_simple_ablation",
]
