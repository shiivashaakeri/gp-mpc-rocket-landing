# GP-MPC Rocket Landing

A Gaussian Process enhanced Model Predictive Control framework for autonomous rocket landing, implementing safe learning-based control with formal safety guarantees.

## Overview

This project implements a complete GP-MPC pipeline for rocket powered descent guidance:

- **6-DoF and 3-DoF Rocket Dynamics**: Full attitude dynamics with quaternion representation
- **Gaussian Process Learning**: Online learning of model residuals with structured kernels
- **Model Predictive Control**: Nominal MPC, GP-MPC, and Real-Time Iteration schemes
- **Learning MPC (LMPC)**: Iterative improvement with safe set expansion
- **Safety Filter**: Control barrier function-based safety guarantees
- **Online Learning**: Novelty-based data selection and hyperparameter adaptation

## Features

| Module | Description |
|--------|-------------|
| `dynamics` | 6-DoF/3-DoF rocket dynamics with RK4 integration |
| `gp` | Exact/Sparse GP with SE-ARD, Matérn kernels |
| `mpc` | Nominal MPC, GP-MPC, RTI-MPC with OSQP |
| `terminal` | Safe set management, convex hull constraints |
| `lmpc` | Learning MPC with Q-function approximation |
| `safety` | Predictive safety filter, backup controllers |
| `learning` | Online data collection, novelty selection |

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/gp-mpc-rocket-landing.git
cd gp-mpc-rocket-landing

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
from src.dynamics import Rocket3DoFDynamics
from src.mpc import NominalMPC3DoF, MPCConfig
from src.gp import Simple3DoFGP, StructuredGPConfig

# Create dynamics
rocket = Rocket3DoFDynamics()

# Setup MPC controller
mpc = NominalMPC3DoF(rocket, MPCConfig(N=20, dt=0.1))

# Initial and target states
x0 = np.array([2.0, 500.0, 50.0, 0.0, -75.0, 10.0, 0.0])  # mass, pos, vel
x_target = np.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # landed

# Solve MPC
solution = mpc.solve(x0, x_target)
u_apply = solution.u0  # First control to apply
```

### With GP Learning

```python
from src.gp import Simple3DoFGP, StructuredGPConfig
from src.mpc import GPMPC, GPMPCConfig

# Create GP model
gp_config = StructuredGPConfig(n_inducing=100)
gp = Simple3DoFGP(gp_config)

# Add training data (collected from real/simulated flights)
gp.add_data(X_train, U_train, residuals)
gp.fit()

# Create GP-MPC
gp_mpc = GPMPC(rocket, gp, GPMPCConfig(N=20))
solution = gp_mpc.solve(x0, x_target)
```

### With Safety Filter

```python
from src.safety import PredictiveSafetyFilter, SafetyConfig

# Create safety filter
safety = PredictiveSafetyFilter(rocket, SafetyConfig())
safety.initialize(x0)

# Filter potentially unsafe control
result = safety.filter(x_current, u_nominal)
u_safe = result.u_safe  # Safe control to apply
```

## Project Structure

```
gp-mpc-rocket-landing/
├── config/                 # YAML configuration files
│   ├── rocket_params.yaml
│   ├── mpc_params.yaml
│   ├── gp_params.yaml
│   └── safety_params.yaml
├── src/
│   ├── dynamics/          # Rocket dynamics models
│   ├── gp/                # Gaussian Process learning
│   ├── mpc/               # MPC implementations
│   ├── terminal/          # Terminal set management
│   ├── lmpc/              # Learning MPC
│   ├── safety/            # Safety filter
│   ├── learning/          # Online learning loop
│   ├── reference/         # Reference trajectory generation
│   └── utils/             # Utilities (quaternions, profiling)
├── tests/                 # Unit tests (215 tests)
├── scripts/               # Simulation scripts
├── notebooks/             # Jupyter notebooks
└── docs/                  # Documentation
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_dynamics.py -v
pytest tests/test_mpc.py -v
pytest tests/test_gp.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance

The framework achieves real-time performance suitable for onboard execution:

| Component | Typical Time | Target |
|-----------|-------------|--------|
| MPC solve (N=20) | 5-15 ms | < 20 ms |
| GP prediction (1000 pts) | 2-5 ms | < 5 ms |
| Safety filter | 1-3 ms | < 5 ms |
| **Total loop** | **10-15 ms** | **< 20 ms (50Hz)** |

## Configuration

Key parameters can be configured via YAML files in `config/`:

```yaml
# mpc_params.yaml
horizon: 20
dt: 0.1
max_iter: 100

# Cost weights
Q_pos: 10.0
Q_vel: 1.0
R_thrust: 0.01

# Constraints
thrust_min: 0.3
thrust_max: 5.0
gimbal_max: 0.35  # 20 degrees
```

## References

This implementation is based on:

1. Szmuk, M., & Açıkmeşe, B. (2018). Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time. *AIAA Guidance, Navigation, and Control Conference*.

2. Hewing, L., et al. (2020). Learning-Based Model Predictive Control: Toward Safe Learning in Control. *Annual Review of Control, Robotics, and Autonomous Systems*.

3. Rosolia, U., & Borrelli, F. (2017). Learning Model Predictive Control for Iterative Tasks. *IEEE Conference on Decision and Control*.

4. Wabersich, K. P., & Zeilinger, M. N. (2021). A Predictive Safety Filter for Learning-Based Control of Constrained Nonlinear Dynamical Systems. *Automatica*.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Authors

GP-MPC Rocket Landing Project Team
