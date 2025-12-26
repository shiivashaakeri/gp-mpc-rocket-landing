"""
Performance Profiler and Benchmark Tools

Tools for profiling and optimizing the GP-MPC control loop:
1. Component timing (GP, MPC, safety filter)
2. Memory usage tracking
3. Bottleneck identification
4. Performance regression testing

Target: 50Hz control loop (20ms total)
- MPC solve: < 10ms
- GP prediction: < 5ms
- Safety filter: < 3ms
- Overhead: < 2ms
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class TimingResult:
    """Result from a timing measurement."""

    name: str
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    n_calls: int
    total_ms: float

    def __str__(self) -> str:
        return (
            f"{self.name}: {self.mean_ms:.2f} ± {self.std_ms:.2f} ms "
            f"(min={self.min_ms:.2f}, max={self.max_ms:.2f}, n={self.n_calls})"
        )


@dataclass
class LoopTiming:
    """Timing breakdown for control loop iteration."""

    gp_predict_ms: float = 0.0
    mpc_prepare_ms: float = 0.0
    mpc_solve_ms: float = 0.0
    safety_filter_ms: float = 0.0
    dynamics_ms: float = 0.0
    overhead_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def achieves_50hz(self) -> bool:
        """Check if loop achieves 50Hz (< 20ms)."""
        return self.total_ms < 20.0

    @property
    def achieves_100hz(self) -> bool:
        """Check if loop achieves 100Hz (< 10ms)."""
        return self.total_ms < 10.0

    def __str__(self) -> str:
        return (
            f"Loop timing: {self.total_ms:.2f}ms total\n"
            f"  GP predict:     {self.gp_predict_ms:.2f}ms\n"
            f"  MPC prepare:    {self.mpc_prepare_ms:.2f}ms\n"
            f"  MPC solve:      {self.mpc_solve_ms:.2f}ms\n"
            f"  Safety filter:  {self.safety_filter_ms:.2f}ms\n"
            f"  Dynamics:       {self.dynamics_ms:.2f}ms\n"
            f"  Overhead:       {self.overhead_ms:.2f}ms\n"
            f"  50Hz feasible:  {self.achieves_50hz}"
        )


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str = ""):
        """Initialize timer."""
        self.name = name
        self.elapsed_ms = 0.0
        self._start = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        """Stop timing."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


class _ProfiledTimer(Timer):
    """Timer that records to profiler."""

    def __init__(self, name: str, profiler: "Profiler"):
        super().__init__(name)
        self._profiler = profiler

    def __exit__(self, *args) -> None:
        """Stop timing and record."""
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        self._profiler._timings[self.name].append(self.elapsed_ms)
        self._profiler._call_counts[self.name] += 1


class Profiler:
    """
    Performance profiler for control loop components.

    Tracks timing statistics and identifies bottlenecks.

    Example:
        >>> profiler = Profiler()
        >>>
        >>> with profiler.time("gp_predict"):
        >>>     mean, var = gp.predict(x)
        >>>
        >>> with profiler.time("mpc_solve"):
        >>>     u = mpc.solve(x)
        >>>
        >>> profiler.report()
    """

    def __init__(self):
        """Initialize profiler."""
        self._timings: Dict[str, List[float]] = defaultdict(list)
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._active_timer: Optional[str] = None
        self._start_time: float = 0.0

    def time(self, name: str) -> _ProfiledTimer:
        """Create timer context for named operation."""
        return _ProfiledTimer(name, self)

    def start(self, name: str) -> None:
        """Start timing named operation."""
        self._active_timer = name
        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop current timer and return elapsed time."""
        if self._active_timer is None:
            return 0.0

        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        self._timings[self._active_timer].append(elapsed_ms)
        self._call_counts[self._active_timer] += 1

        self._active_timer = None

        return elapsed_ms

    def record(self, name: str, elapsed_ms: float) -> None:
        """Record a timing measurement."""
        self._timings[name].append(elapsed_ms)
        self._call_counts[name] += 1

    def get_stats(self, name: str) -> Optional[TimingResult]:
        """Get timing statistics for named operation."""
        if name not in self._timings or len(self._timings[name]) == 0:
            return None

        times = np.array(self._timings[name])

        return TimingResult(
            name=name,
            mean_ms=float(np.mean(times)),
            std_ms=float(np.std(times)),
            min_ms=float(np.min(times)),
            max_ms=float(np.max(times)),
            n_calls=len(times),
            total_ms=float(np.sum(times)),
        )

    def get_all_stats(self) -> Dict[str, TimingResult]:
        """Get statistics for all tracked operations."""
        return {name: self.get_stats(name) for name in self._timings if self.get_stats(name) is not None}

    def report(self) -> str:
        """Generate timing report."""
        lines = ["Performance Report", "=" * 50]

        stats = self.get_all_stats()

        if not stats:
            lines.append("No timing data recorded")
            return "\n".join(lines)

        # Sort by mean time (descending)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].mean_ms, reverse=True)

        total_time = sum(s.total_ms for s in stats.values())

        for name, stat in sorted_stats:
            pct = stat.total_ms / total_time * 100 if total_time > 0 else 0
            lines.append(f"{name:20s}: {stat.mean_ms:8.2f} ms ({pct:5.1f}%) [n={stat.n_calls}]")

        lines.append("=" * 50)
        lines.append(f"Total tracked time: {total_time:.2f} ms")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all timing data."""
        self._timings.clear()
        self._call_counts.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export profiling data as dictionary."""
        return {
            name: {
                "mean_ms": stat.mean_ms,
                "std_ms": stat.std_ms,
                "min_ms": stat.min_ms,
                "max_ms": stat.max_ms,
                "n_calls": stat.n_calls,
            }
            for name, stat in self.get_all_stats().items()
        }


def profile_function(profiler: Profiler, name: Optional[str] = None):
    """Decorator to profile a function."""

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with profiler.time(func_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


class ControlLoopBenchmark:
    """
    Benchmarks the complete control loop.

    Measures timing for each component and overall loop frequency.
    """

    def __init__(
        self,
        dynamics,
        mpc_controller,
        gp_model=None,
        safety_filter=None,
        dt: float = 0.1,
    ):
        """
        Initialize benchmark.

        Args:
            dynamics: Rocket dynamics
            mpc_controller: MPC controller
            gp_model: GP model (optional)
            safety_filter: Safety filter (optional)
            dt: Simulation timestep
        """
        self.dynamics = dynamics
        self.mpc = mpc_controller
        self.gp = gp_model
        self.safety = safety_filter
        self.dt = dt

        self.profiler = Profiler()
        self._loop_timings: List[LoopTiming] = []

    def run(  # noqa: C901
        self,
        x0: NDArray,
        x_target: NDArray,
        n_steps: int = 100,
        warmup: int = 5,
    ) -> "BenchmarkResults":
        """
        Run benchmark simulation.

        Args:
            x0: Initial state
            x_target: Target state
            n_steps: Number of simulation steps
            warmup: Warmup iterations to exclude

        Returns:
            Benchmark results
        """
        self.profiler.reset()
        self._loop_timings.clear()

        x = x0.copy()

        # Initialize MPC
        if hasattr(self.mpc, "initialize"):
            self.mpc.initialize(x0, x_target)

        for step in range(n_steps + warmup):
            loop_start = time.perf_counter()
            timing = LoopTiming()

            # GP prediction
            if self.gp is not None:
                with self.profiler.time("gp_predict"):
                    try:
                        gp_mean, gp_var = self.gp.predict(x.reshape(1, -1))
                        timing.gp_predict_ms = self.profiler._timings["gp_predict"][-1]
                    except Exception:
                        timing.gp_predict_ms = 0.0

            # MPC solve
            with self.profiler.time("mpc_solve"):
                try:
                    if hasattr(self.mpc, "step"):
                        sol = self.mpc.step(x)
                        u = sol.u0
                    elif hasattr(self.mpc, "solve"):
                        sol = self.mpc.solve(x, x_target)
                        u = sol.u0
                    else:
                        u = np.zeros(3)
                except Exception:
                    u = np.zeros(3)
                timing.mpc_solve_ms = self.profiler._timings["mpc_solve"][-1]

            # Safety filter
            if self.safety is not None:
                with self.profiler.time("safety_filter"):
                    try:
                        result = self.safety.filter(x, u)
                        u = result.u_safe
                        timing.safety_filter_ms = self.profiler._timings["safety_filter"][-1]
                    except Exception:
                        timing.safety_filter_ms = 0.0

            # Dynamics simulation
            with self.profiler.time("dynamics"):
                x = self.dynamics.step(x, u, self.dt)
                timing.dynamics_ms = self.profiler._timings["dynamics"][-1]

            # Total loop time
            timing.total_ms = (time.perf_counter() - loop_start) * 1000
            timing.overhead_ms = (
                timing.total_ms
                - timing.gp_predict_ms
                - timing.mpc_solve_ms
                - timing.safety_filter_ms
                - timing.dynamics_ms
            )

            if step >= warmup:
                self._loop_timings.append(timing)

        return BenchmarkResults(
            loop_timings=self._loop_timings,
            profiler=self.profiler,
            n_steps=n_steps,
        )


@dataclass
class BenchmarkResults:
    """Results from control loop benchmark."""

    loop_timings: List[LoopTiming]
    profiler: Profiler
    n_steps: int

    def summary(self) -> str:
        """Generate benchmark summary."""
        if len(self.loop_timings) == 0:
            return "No timing data"

        totals = [t.total_ms for t in self.loop_timings]
        gp_times = [t.gp_predict_ms for t in self.loop_timings]
        mpc_times = [t.mpc_solve_ms for t in self.loop_timings]
        safety_times = [t.safety_filter_ms for t in self.loop_timings]

        lines = [
            "Control Loop Benchmark Results",
            "=" * 50,
            "",
            "Loop Timing Statistics:",
            f"  Mean total:       {np.mean(totals):.2f} ms",
            f"  Std total:        {np.std(totals):.2f} ms",
            f"  Min total:        {np.min(totals):.2f} ms",
            f"  Max total:        {np.max(totals):.2f} ms",
            "",
            "Component Breakdown (mean):",
            f"  GP prediction:    {np.mean(gp_times):.2f} ms",
            f"  MPC solve:        {np.mean(mpc_times):.2f} ms",
            f"  Safety filter:    {np.mean(safety_times):.2f} ms",
            "",
            "Achievable Frequencies:",
            f"  50Hz (< 20ms):    {100 * np.mean([t < 20 for t in totals]):.1f}%",
            f"  100Hz (< 10ms):   {100 * np.mean([t < 10 for t in totals]):.1f}%",
            "",
            "=" * 50,
        ]

        return "\n".join(lines)

    def get_percentile(self, percentile: float) -> float:
        """Get timing percentile."""
        totals = [t.total_ms for t in self.loop_timings]
        return float(np.percentile(totals, percentile))

    @property
    def mean_frequency_hz(self) -> float:
        """Mean achievable frequency."""
        mean_ms = np.mean([t.total_ms for t in self.loop_timings])
        return 1000.0 / mean_ms if mean_ms > 0 else 0.0

    @property
    def achieves_50hz(self) -> bool:
        """Whether 50Hz is achievable (95th percentile)."""
        return self.get_percentile(95) < 20.0


class MemoryProfiler:
    """
    Memory usage profiler.

    Tracks memory consumption of major data structures.
    """

    def __init__(self):
        """Initialize memory profiler."""
        self._tracked: Dict[str, Any] = {}

    def track(self, name: str, obj: Any) -> None:
        """Track an object's memory usage."""
        self._tracked[name] = obj

    def get_size(self, name: str) -> int:
        """Get size of tracked object in bytes."""
        if name not in self._tracked:
            return 0

        obj = self._tracked[name]

        if isinstance(obj, np.ndarray):
            return obj.nbytes
        elif hasattr(obj, "__sizeof__"):
            return obj.__sizeof__()
        else:
            import sys  # noqa: PLC0415

            return sys.getsizeof(obj)

    def report(self) -> str:
        """Generate memory usage report."""
        lines = ["Memory Usage Report", "=" * 50]

        total = 0
        for name in self._tracked:
            size = self.get_size(name)
            total += size

            if size > 1e6:
                size_str = f"{size / 1e6:.2f} MB"
            elif size > 1e3:
                size_str = f"{size / 1e3:.2f} KB"
            else:
                size_str = f"{size} B"

            lines.append(f"{name:30s}: {size_str}")

        lines.append("=" * 50)
        lines.append(f"Total: {total / 1e6:.2f} MB")

        return "\n".join(lines)


def benchmark_gp_prediction(
    gp_predictor,
    n_train: int = 1000,
    n_test: int = 100,
    n_features: int = 17,
    n_outputs: int = 6,
    n_warmup: int = 10,
    n_trials: int = 100,
) -> Dict[str, float]:
    """
    Benchmark GP prediction performance.

    Returns:
        Dictionary with timing statistics
    """
    # Generate synthetic data
    X_train = np.random.randn(n_train, n_features)
    Y_train = np.random.randn(n_train, n_outputs)
    X_test = np.random.randn(n_test, n_features)

    # Fit
    fit_time = gp_predictor.fit(X_train, Y_train) if hasattr(gp_predictor, "fit") else 0.0

    # Warmup
    for _ in range(n_warmup):
        _ = gp_predictor.predict(X_test)

    # Benchmark
    times = []
    for _ in range(n_trials):
        t_start = time.perf_counter()
        _ = gp_predictor.predict(X_test)
        times.append((time.perf_counter() - t_start) * 1000)

    times = np.array(times)

    return {
        "fit_time_ms": fit_time,
        "predict_mean_ms": float(np.mean(times)),
        "predict_std_ms": float(np.std(times)),
        "predict_min_ms": float(np.min(times)),
        "predict_max_ms": float(np.max(times)),
        "per_sample_us": float(np.mean(times) / n_test * 1000),
    }


def benchmark_mpc_solve(
    mpc_controller,
    x0: NDArray,
    x_target: NDArray,
    n_warmup: int = 5,
    n_trials: int = 50,
) -> Dict[str, float]:
    """
    Benchmark MPC solve performance.

    Returns:
        Dictionary with timing statistics
    """
    # Initialize
    if hasattr(mpc_controller, "initialize"):
        mpc_controller.initialize(x0, x_target)

    # Warmup
    for _ in range(n_warmup):
        if hasattr(mpc_controller, "step"):
            mpc_controller.step(x0)
        elif hasattr(mpc_controller, "solve"):
            mpc_controller.solve(x0, x_target)

    # Benchmark
    times = []
    for _ in range(n_trials):
        x = x0 + np.random.randn(len(x0)) * 0.1

        t_start = time.perf_counter()
        sol = mpc_controller.step(x) if hasattr(mpc_controller, "step") else mpc_controller.solve(x, x_target)  # noqa: F841
        times.append((time.perf_counter() - t_start) * 1000)

    times = np.array(times)

    return {
        "solve_mean_ms": float(np.mean(times)),
        "solve_std_ms": float(np.std(times)),
        "solve_min_ms": float(np.min(times)),
        "solve_max_ms": float(np.max(times)),
        "achieves_50hz": bool(np.percentile(times, 95) < 20),
        "achieves_100hz": bool(np.percentile(times, 95) < 10),
    }
