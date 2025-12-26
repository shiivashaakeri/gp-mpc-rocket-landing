"""
Trajectory Library for Reference Trajectory Management

Provides storage, retrieval, and interpolation of reference trajectories:
- Multiple trajectory storage with metadata
- Nearest-neighbor trajectory lookup
- Time-scaled trajectory interpolation
- Trajectory quality metrics

Used for:
- LMPC safe set initialization
- Warm starting MPC
- Baseline comparisons
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryMetadata:
    """Metadata for a stored trajectory."""

    name: str = ""
    source: str = ""  # "scvx", "mpc", "demonstration"
    success: bool = True
    total_cost: float = 0.0
    fuel_used: float = 0.0
    final_time: float = 0.0
    constraints_violated: int = 0
    timestamp: str = ""
    notes: str = ""


@dataclass
class Trajectory:
    """Complete trajectory with states, controls, and metadata."""

    states: NDArray  # (T+1, n_x)
    controls: NDArray  # (T, n_u)
    times: NDArray  # (T+1,)
    dt: float  # Time step
    metadata: TrajectoryMetadata = field(default_factory=TrajectoryMetadata)

    @property
    def length(self) -> int:
        """Number of time steps."""
        return len(self.controls)

    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        return self.times[-1] - self.times[0]

    @property
    def initial_state(self) -> NDArray:
        """Initial state."""
        return self.states[0]

    @property
    def final_state(self) -> NDArray:
        """Final state."""
        return self.states[-1]

    def interpolate(self, t: float) -> Tuple[NDArray, NDArray]:
        """
        Interpolate state and control at time t.

        Args:
            t: Query time

        Returns:
            state: Interpolated state
            control: Interpolated control
        """
        # Find interval
        idx = np.searchsorted(self.times, t) - 1
        idx = np.clip(idx, 0, self.length - 1)

        # Linear interpolation for state
        alpha = (t - self.times[idx]) / (self.times[idx + 1] - self.times[idx] + 1e-10)
        alpha = np.clip(alpha, 0, 1)

        state = (1 - alpha) * self.states[idx] + alpha * self.states[idx + 1]
        control = self.controls[idx]

        return state, control

    def resample(self, n_points: int) -> "Trajectory":
        """
        Resample trajectory to different number of points.

        Args:
            n_points: Number of points in resampled trajectory

        Returns:
            Resampled trajectory
        """
        t_new = np.linspace(self.times[0], self.times[-1], n_points + 1)
        dt_new = (self.times[-1] - self.times[0]) / n_points

        states_new = np.zeros((n_points + 1, self.states.shape[1]))
        controls_new = np.zeros((n_points, self.controls.shape[1]))

        for i, t in enumerate(t_new):
            state, control = self.interpolate(t)
            states_new[i] = state
            if i < n_points:
                controls_new[i] = control

        return Trajectory(
            states=states_new,
            controls=controls_new,
            times=t_new,
            dt=dt_new,
            metadata=self.metadata,
        )

    def compute_cost(
        self,
        Q: Optional[NDArray] = None,
        R: Optional[NDArray] = None,
        x_target: Optional[NDArray] = None,
    ) -> float:
        """Compute trajectory cost."""
        n_x = self.states.shape[1]
        n_u = self.controls.shape[1]

        if Q is None:
            Q = np.eye(n_x)
        if R is None:
            R = np.eye(n_u) * 0.01
        if x_target is None:
            x_target = np.zeros(n_x)

        cost = 0.0
        for k in range(self.length):
            dx = self.states[k] - x_target
            cost += dx @ Q @ dx + self.controls[k] @ R @ self.controls[k]

        dx_f = self.states[-1] - x_target
        cost += dx_f @ Q @ dx_f * 10  # Terminal cost

        return cost


class TrajectoryLibrary:
    """
    Library for storing and managing reference trajectories.

    Provides:
    - Add/remove trajectories
    - Query by initial state
    - Save/load to disk
    - Statistics and analysis

    Example:
        >>> library = TrajectoryLibrary()
        >>> library.add(trajectory, name="landing_1")
        >>>
        >>> # Find best trajectory for given initial state
        >>> traj = library.query_nearest(x0)
        >>> X_ref, U_ref = traj.states, traj.controls
    """

    def __init__(self, name: str = "default"):
        """
        Initialize trajectory library.

        Args:
            name: Library name
        """
        self.name = name
        self._trajectories: Dict[str, Trajectory] = {}
        self._initial_states: List[NDArray] = []
        self._names: List[str] = []

    def add(
        self,
        trajectory: Trajectory,
        name: Optional[str] = None,
    ) -> str:
        """
        Add trajectory to library.

        Args:
            trajectory: Trajectory to add
            name: Optional name (auto-generated if not provided)

        Returns:
            Assigned trajectory name
        """
        if name is None:
            name = f"traj_{len(self._trajectories)}"

        trajectory.metadata.name = name
        self._trajectories[name] = trajectory
        self._initial_states.append(trajectory.initial_state)
        self._names.append(name)

        return name

    def add_from_arrays(
        self,
        states: NDArray,
        controls: NDArray,
        dt: float,
        name: Optional[str] = None,
        **metadata_kwargs,
    ) -> str:
        """
        Add trajectory from numpy arrays.

        Args:
            states: State trajectory
            controls: Control trajectory
            dt: Time step
            name: Trajectory name
            **metadata_kwargs: Additional metadata

        Returns:
            Assigned name
        """
        T = len(controls)
        times = np.arange(T + 1) * dt

        metadata = TrajectoryMetadata(**metadata_kwargs)
        metadata.final_time = times[-1]

        trajectory = Trajectory(
            states=states,
            controls=controls,
            times=times,
            dt=dt,
            metadata=metadata,
        )

        return self.add(trajectory, name)

    def remove(self, name: str) -> bool:
        """
        Remove trajectory from library.

        Args:
            name: Trajectory name

        Returns:
            True if removed, False if not found
        """
        if name not in self._trajectories:
            return False

        idx = self._names.index(name)
        del self._trajectories[name]
        del self._initial_states[idx]
        del self._names[idx]

        return True

    def get(self, name: str) -> Optional[Trajectory]:
        """Get trajectory by name."""
        return self._trajectories.get(name)

    def query_nearest(
        self,
        x0: NDArray,
        k: int = 1,
        success_only: bool = True,
    ) -> List[Trajectory]:
        """
        Find k nearest trajectories by initial state.

        Args:
            x0: Query initial state
            k: Number of trajectories to return
            success_only: Only return successful trajectories

        Returns:
            List of nearest trajectories
        """
        if len(self._trajectories) == 0:
            return []

        # Filter by success
        candidates = []
        for name, traj in self._trajectories.items():
            if success_only and not traj.metadata.success:
                continue
            candidates.append((name, traj))

        if len(candidates) == 0:
            return []

        # Compute distances
        distances = []
        for name, traj in candidates:
            dist = np.linalg.norm(traj.initial_state - x0)
            distances.append((dist, name, traj))

        # Sort by distance
        distances.sort(key=lambda x: x[0])

        # Return top k
        return [traj for _, _, traj in distances[:k]]

    def query_best(
        self,
        x0: NDArray,
        radius: float = 100.0,
        metric: str = "cost",
    ) -> Optional[Trajectory]:
        """
        Find best trajectory within radius of initial state.

        Args:
            x0: Query initial state
            radius: Search radius
            metric: Ranking metric ("cost", "fuel", "time")

        Returns:
            Best trajectory or None
        """
        # Find trajectories within radius
        candidates = []
        for name, traj in self._trajectories.items():
            dist = np.linalg.norm(traj.initial_state - x0)
            if dist <= radius:
                candidates.append(traj)

        if len(candidates) == 0:
            return None

        # Rank by metric
        if metric == "cost":
            candidates.sort(key=lambda t: t.metadata.total_cost)
        elif metric == "fuel":
            candidates.sort(key=lambda t: t.metadata.fuel_used)
        elif metric == "time":
            candidates.sort(key=lambda t: t.metadata.final_time)

        return candidates[0]

    def get_all_successful(self) -> List[Trajectory]:
        """Get all successful trajectories."""
        return [traj for traj in self._trajectories.values() if traj.metadata.success]

    def get_statistics(self) -> Dict[str, Any]:
        """Get library statistics."""
        if len(self._trajectories) == 0:
            return {"count": 0}

        trajectories = list(self._trajectories.values())
        successful = [t for t in trajectories if t.metadata.success]

        costs = [t.metadata.total_cost for t in trajectories]
        fuels = [t.metadata.fuel_used for t in trajectories]
        times = [t.metadata.final_time for t in trajectories]

        return {
            "count": len(trajectories),
            "successful": len(successful),
            "success_rate": len(successful) / len(trajectories),
            "cost_mean": np.mean(costs),
            "cost_std": np.std(costs),
            "fuel_mean": np.mean(fuels),
            "time_mean": np.mean(times),
        }

    def save(self, filepath: str) -> None:
        """
        Save library to file.

        Args:
            filepath: Path to save file
        """
        data = {
            "name": self.name,
            "trajectories": {
                name: {
                    "states": traj.states,
                    "controls": traj.controls,
                    "times": traj.times,
                    "dt": traj.dt,
                    "metadata": {
                        "name": traj.metadata.name,
                        "source": traj.metadata.source,
                        "success": traj.metadata.success,
                        "total_cost": traj.metadata.total_cost,
                        "fuel_used": traj.metadata.fuel_used,
                        "final_time": traj.metadata.final_time,
                        "notes": traj.metadata.notes,
                    },
                }
                for name, traj in self._trajectories.items()
            },
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """
        Load library from file.

        Args:
            filepath: Path to load file
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.name = data["name"]
        self._trajectories.clear()
        self._initial_states.clear()
        self._names.clear()

        for name, traj_data in data["trajectories"].items():
            metadata = TrajectoryMetadata(**traj_data["metadata"])

            trajectory = Trajectory(
                states=traj_data["states"],
                controls=traj_data["controls"],
                times=traj_data["times"],
                dt=traj_data["dt"],
                metadata=metadata,
            )

            self.add(trajectory, name)

    def __len__(self) -> int:
        """Number of trajectories."""
        return len(self._trajectories)

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self._trajectories.values())


def generate_trajectory_library(
    dynamics,
    n_trajectories: int = 10,
    initial_state_sampler=None,
    scvx_solver=None,
) -> TrajectoryLibrary:
    """
    Generate a library of reference trajectories.

    Args:
        dynamics: Rocket dynamics
        n_trajectories: Number of trajectories to generate
        initial_state_sampler: Function to sample initial states
        scvx_solver: SCVX solver (or None to use simple generation)

    Returns:
        Populated trajectory library
    """
    library = TrajectoryLibrary("generated")

    # Default initial state sampler
    if initial_state_sampler is None:

        def initial_state_sampler():
            # Random initial conditions within reasonable bounds
            m0 = 2.0 + np.random.rand() * 0.5
            pos = np.random.randn(3) * np.array([50, 50, 200])
            pos[0] = abs(pos[0]) + 100  # Ensure positive altitude
            vel = np.random.randn(3) * np.array([20, 20, 30])
            vel[0] = -abs(vel[0])  # Descending
            return np.concatenate([[m0], pos, vel])

    # Target state (landed)
    n_x = getattr(dynamics, "n_x", 7)
    x_target = np.zeros(n_x)
    x_target[0] = 1.5  # Final mass

    for i in range(n_trajectories):
        x0 = initial_state_sampler()

        try:
            if scvx_solver is not None:
                solution = scvx_solver.solve(x0, x_target)
                states = solution.X
                controls = solution.U
                dt = solution.dt
                success = solution.converged
            else:
                # Simple forward simulation
                from .scvx_interface import SimpleSCVX  # noqa: PLC0415

                simple_scvx = SimpleSCVX(dynamics)
                states, controls, dt = simple_scvx.generate_reference(x0, x_target, t_f=15.0)
                success = True

            # Compute metrics
            fuel_used = float(x0[0] - states[-1, 0])
            final_time = len(controls) * dt

            library.add_from_arrays(
                states=states,
                controls=controls,
                dt=dt,
                name=f"traj_{i}",
                source="scvx" if scvx_solver else "simple",
                success=success,
                fuel_used=fuel_used,
                final_time=final_time,
            )

        except Exception as e:
            print(f"Failed to generate trajectory {i}: {e}")
            continue

    return library
