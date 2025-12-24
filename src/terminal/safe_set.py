"""
Sampled Safe Set Data Structure for LMPC

The safe set SS contains states from which we have demonstrated
successful trajectories to the target. It grows iteratively as
we collect more successful landings.

Key concepts:
- SS^j = Safe set at iteration j
- SS^j = SS^{j-1} U {x_0^j, x_1^j, ..., x_T^j} after successful trajectory j
- Q^j(x) = Cost-to-go from state x (stored with each point)

Reference:
    Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control
    for Iterative Tasks: A Data-Driven Control Framework.
    IEEE Transactions on Automatic Control.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrajectoryData:
    """
    Data from a single successful trajectory.

    Attributes:
        iteration: Iteration number when this trajectory was recorded
        states: State trajectory (T+1, n_x)
        controls: Control trajectory (T, n_u)
        costs: Stage costs at each timestep (T,)
        cost_to_go: Cost-to-go from each state (T+1,)
        total_cost: Total trajectory cost
        metadata: Additional info (e.g., initial conditions, parameters)
    """

    iteration: int
    states: NDArray
    controls: NDArray
    costs: NDArray
    cost_to_go: NDArray
    total_cost: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate trajectory data."""
        T = len(self.controls)
        assert len(self.states) == T + 1, "States should have T+1 points"
        assert len(self.costs) == T, "Costs should have T points"
        assert len(self.cost_to_go) == T + 1, "Cost-to-go should have T+1 points"

    @property
    def T(self) -> int:
        """Trajectory length."""
        return len(self.controls)

    @property
    def n_x(self) -> int:
        """State dimension."""
        return self.states.shape[1]

    @property
    def n_u(self) -> int:
        """Control dimension."""
        return self.controls.shape[1]


class SampledSafeSet:
    """
    Sampled Safe Set for LMPC.

    Stores states from successful trajectories along with their
    cost-to-go values. The safe set grows as more trajectories
    are added.

    Features:
    - Efficient storage and retrieval of safe states
    - Cost-to-go (Q-function) values for each state
    - Iteration tracking for convergence analysis
    - Fuel-aware filtering (states with sufficient fuel)

    Example:
        >>> ss = SampledSafeSet(n_x=14, n_u=3)
        >>>
        >>> # Add successful trajectory
        >>> ss.add_trajectory(X, U, costs, iteration=0)
        >>>
        >>> # Query safe set
        >>> safe_states = ss.get_all_states()
        >>> q_values = ss.get_all_q_values()
    """

    def __init__(
        self,
        n_x: int,
        n_u: int,
        target_state: Optional[NDArray] = None,
        fuel_index: int = 0,  # Index of mass/fuel in state
    ):
        """
        Initialize empty safe set.

        Args:
            n_x: State dimension
            n_u: Control dimension
            target_state: Target state (for distance computations)
            fuel_index: Index of fuel/mass in state vector
        """
        self.n_x = n_x
        self.n_u = n_u
        self.target_state = target_state
        self.fuel_index = fuel_index

        # Storage
        self._trajectories: List[TrajectoryData] = []
        self._iteration_count = 0

        # Cached arrays for fast access
        self._all_states: Optional[NDArray] = None
        self._all_q_values: Optional[NDArray] = None
        self._all_controls: Optional[NDArray] = None
        self._state_iterations: Optional[NDArray] = None
        self._cache_valid = False

    def add_trajectory(
        self,
        states: NDArray,
        controls: NDArray,
        stage_costs: NDArray,
        iteration: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """
        Add a successful trajectory to the safe set.

        Args:
            states: State trajectory (T+1, n_x)
            controls: Control trajectory (T, n_u)
            stage_costs: Stage costs l(x_k, u_k) at each step (T,)
            iteration: Iteration number (auto-incremented if None)
            metadata: Additional trajectory information
        """
        if iteration is None:
            iteration = self._iteration_count

        # Compute cost-to-go for each state
        # Q(x_k) = sum_{i=k}^{T-1} l(x_i, u_i)
        T = len(controls)
        cost_to_go = np.zeros(T + 1)
        cost_to_go[T] = 0  # Terminal state has zero cost-to-go

        for k in range(T - 1, -1, -1):
            cost_to_go[k] = stage_costs[k] + cost_to_go[k + 1]

        total_cost = cost_to_go[0]

        traj = TrajectoryData(
            iteration=iteration,
            states=states.copy(),
            controls=controls.copy(),
            costs=stage_costs.copy(),
            cost_to_go=cost_to_go,
            total_cost=total_cost,
            metadata=metadata or {},
        )

        self._trajectories.append(traj)
        self._iteration_count = max(self._iteration_count, iteration + 1)
        self._cache_valid = False

    def _rebuild_cache(self) -> None:
        """Rebuild cached arrays from trajectories."""
        if self._cache_valid:
            return

        if len(self._trajectories) == 0:
            self._all_states = np.zeros((0, self.n_x))
            self._all_q_values = np.zeros(0)
            self._all_controls = np.zeros((0, self.n_u))
            self._state_iterations = np.zeros(0, dtype=int)
            self._cache_valid = True
            return

        # Collect all states and Q-values
        states_list = []
        q_values_list = []
        controls_list = []
        iterations_list = []

        for traj in self._trajectories:
            # Add all states except terminal (which has Q=0)
            # Or include terminal? Let's include all.
            states_list.append(traj.states)
            q_values_list.append(traj.cost_to_go)

            # For controls, pad last with zeros (no control at terminal)
            ctrl_padded = np.vstack([traj.controls, np.zeros((1, self.n_u))])
            controls_list.append(ctrl_padded)

            iterations_list.append(np.full(len(traj.states), traj.iteration))

        self._all_states = np.vstack(states_list)
        self._all_q_values = np.concatenate(q_values_list)
        self._all_controls = np.vstack(controls_list)
        self._state_iterations = np.concatenate(iterations_list)
        self._cache_valid = True

    def get_all_states(self) -> NDArray:
        """Get all states in safe set (N, n_x)."""
        self._rebuild_cache()
        return self._all_states

    def get_all_q_values(self) -> NDArray:
        """Get Q-values for all states (N,)."""
        self._rebuild_cache()
        return self._all_q_values

    def get_all_controls(self) -> NDArray:
        """Get controls associated with each state (N, n_u)."""
        self._rebuild_cache()
        return self._all_controls

    def get_states_with_min_fuel(self, min_fuel: float) -> Tuple[NDArray, NDArray]:
        """
        Get states with at least min_fuel remaining.

        Args:
            min_fuel: Minimum fuel/mass required

        Returns:
            states: Filtered states (M, n_x)
            q_values: Corresponding Q-values (M,)
        """
        self._rebuild_cache()

        fuel = self._all_states[:, self.fuel_index]
        mask = fuel >= min_fuel

        return self._all_states[mask], self._all_q_values[mask]

    def get_states_from_iteration(self, iteration: int) -> Tuple[NDArray, NDArray]:
        """
        Get states from a specific iteration.

        Args:
            iteration: Iteration number

        Returns:
            states: States from that iteration
            q_values: Corresponding Q-values
        """
        self._rebuild_cache()

        mask = self._state_iterations == iteration
        return self._all_states[mask], self._all_q_values[mask]

    def get_best_trajectory(self) -> Optional[TrajectoryData]:
        """Get trajectory with lowest total cost."""
        if len(self._trajectories) == 0:
            return None

        best_idx = np.argmin([t.total_cost for t in self._trajectories])
        return self._trajectories[best_idx]

    def get_trajectory(self, iteration: int) -> Optional[TrajectoryData]:
        """Get trajectory from specific iteration."""
        for traj in self._trajectories:
            if traj.iteration == iteration:
                return traj
        return None

    @property
    def num_trajectories(self) -> int:
        """Number of trajectories in safe set."""
        return len(self._trajectories)

    @property
    def num_states(self) -> int:
        """Total number of states in safe set."""
        self._rebuild_cache()
        return len(self._all_states)

    @property
    def num_iterations(self) -> int:
        """Number of iterations completed."""
        return self._iteration_count

    def get_statistics(self) -> Dict[str, Any]:
        """Get safe set statistics."""
        self._rebuild_cache()

        if len(self._trajectories) == 0:
            return {
                "num_trajectories": 0,
                "num_states": 0,
                "num_iterations": 0,
            }

        costs = [t.total_cost for t in self._trajectories]

        return {
            "num_trajectories": len(self._trajectories),
            "num_states": len(self._all_states),
            "num_iterations": self._iteration_count,
            "best_cost": min(costs),
            "worst_cost": max(costs),
            "mean_cost": np.mean(costs),
            "cost_improvement": costs[0] - min(costs) if len(costs) > 1 else 0,
        }

    def save(self, filepath: str) -> None:
        """Save safe set to file."""
        data = {
            "n_x": self.n_x,
            "n_u": self.n_u,
            "target_state": self.target_state,
            "fuel_index": self.fuel_index,
            "trajectories": self._trajectories,
            "iteration_count": self._iteration_count,
        }

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, filepath: str) -> "SampledSafeSet":
        """Load safe set from file."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        ss = cls(
            n_x=data["n_x"],
            n_u=data["n_u"],
            target_state=data["target_state"],
            fuel_index=data["fuel_index"],
        )
        ss._trajectories = data["trajectories"]
        ss._iteration_count = data["iteration_count"]

        return ss

    def clear(self) -> None:
        """Clear all trajectories."""
        self._trajectories = []
        self._iteration_count = 0
        self._cache_valid = False


class FuelAwareSafeSet(SampledSafeSet):
    """
    Safe set with fuel-aware filtering.

    The safe set shrinks based on available fuel - states requiring
    more fuel than available are excluded from the terminal set.

    Key insight: A state x with fuel m is only safe if:
    1. We have a demonstrated trajectory from x to target
    2. That trajectory requires fuel <= m

    This ensures recursive feasibility under fuel constraints.
    """

    def __init__(
        self,
        n_x: int,
        n_u: int,
        target_state: Optional[NDArray] = None,
        fuel_index: int = 0,
        fuel_margin: float = 0.1,  # Safety margin
    ):
        """
        Initialize fuel-aware safe set.

        Args:
            n_x: State dimension
            n_u: Control dimension
            target_state: Target state
            fuel_index: Index of fuel/mass in state
            fuel_margin: Additional fuel margin for safety
        """
        super().__init__(n_x, n_u, target_state, fuel_index)
        self.fuel_margin = fuel_margin

        # Store fuel required for each state (to reach target)
        self._fuel_required: Optional[NDArray] = None

    def add_trajectory(
        self,
        states: NDArray,
        controls: NDArray,
        stage_costs: NDArray,
        iteration: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Add trajectory and compute fuel requirements."""
        super().add_trajectory(states, controls, stage_costs, iteration, metadata)
        self._fuel_required = None  # Invalidate

    def _compute_fuel_required(self) -> None:
        """Compute fuel required from each state to target."""
        if self._fuel_required is not None:
            return

        self._rebuild_cache()

        if len(self._all_states) == 0:
            self._fuel_required = np.zeros(0)
            return

        # Fuel required = initial fuel - final fuel for trajectory segment
        fuel_required = []

        for traj in self._trajectories:
            initial_fuel = traj.states[:, self.fuel_index]
            final_fuel = traj.states[-1, self.fuel_index]
            fuel_req = initial_fuel - final_fuel + self.fuel_margin
            fuel_required.append(fuel_req)

        self._fuel_required = np.concatenate(fuel_required)

    def get_feasible_states(
        self,
        available_fuel: float,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Get states feasible with given fuel budget.

        Args:
            available_fuel: Available fuel/mass

        Returns:
            states: Feasible states
            q_values: Corresponding Q-values
            indices: Indices in full safe set
        """
        self._rebuild_cache()
        self._compute_fuel_required()

        mask = self._fuel_required <= available_fuel
        indices = np.where(mask)[0]

        return (
            self._all_states[mask],
            self._all_q_values[mask],
            indices,
        )

    def get_fuel_required(self) -> NDArray:
        """Get fuel required for each state in safe set."""
        self._compute_fuel_required()
        return self._fuel_required
