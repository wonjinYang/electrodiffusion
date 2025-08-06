
# Electrodiffusion Library

**Electrodiffusion** is a comprehensive Python library for modeling ion channel dynamics using stochastic differential equations (SDEs), score-based diffusion models, and multi-scale simulation techniques. It integrates physical principles with machine learning approaches to simulate electrodiffusion processes, predict IV curves, and analyze complex biophysical systems. The library is designed for researchers in computational biophysics, providing modular components for custom extensions while ensuring numerical stability and physical consistency.

This library builds upon concepts from Claude's code artifacts, particularly the integration of physics-informed SDEs (e.g., IonChannelSDE with autograd-based forces), score-based generative modeling (e.g., PhysicsInformedScoreNetwork with attention mechanisms), and multi-scale solvers (e.g., Mori-Zwanzig projections in MultiScaleSDE). It emphasizes reproducibility with default configurations, extensive testing, and example scripts.

## Key Features

- **Coarse-Grained Modeling**: Efficient representations separating protein backbone and side chains, with ion state handling (e.g., Cartesian to conical conversions for channel geometry).
- **SDE Framework**: Physics-informed stochastic differential equations for electrodiffusion, including custom drift/diffusion terms, variance-preserving adjustments, and extensions for protein dynamics.
- **Score-Based Diffusion Models**: Generative modeling of equilibrium distributions using VP-SDE, with conditional generation for external fields (e.g., voltage).
- **Multi-Scale Simulations**: Hierarchical integration of fast (e.g., ion fluctuations) and slow (e.g., conformational changes) dynamics, including memory kernels and implicit solvent effects.
- **IV Curve Prediction**: Computation of current-voltage relationships, conductances, and open probabilities from sampled states, with Nernst-Planck flux calculations.
- **Utilities and Validation**: Data loading/preprocessing, mathematical tools (e.g., PMF, gradients), topology constraints via persistent homology, visualization functions, and built-in checks for detailed balance and thermodynamic consistency.
- **Testing and Examples**: Comprehensive unit tests covering models, simulations, and utils; executable examples for quick starts and advanced demos.


## Installation

To install the library, clone the repository and use pip:

```bash
git clone https://github.com/your-repo/electrodiffusion.git  # Replace with actual repo URL
cd electrodiffusion
pip install -e .  # Editable install for development
```


### Dependencies

Core requirements (automatically installed):

- torch (>=2.0.0)
- numpy (>=1.20.0)
- matplotlib (>=3.5.0)
- seaborn (>=0.11.0)
- scipy (>=1.7.0)

Optional (for full functionality):

- mdanalysis (>=2.0.0): For loading real MD trajectories in utils.data
- gudhi (>=3.4.0): For persistent homology in utils.topology
- pytest (>=7.0.0): For running tests

Install optionals with:

```bash
pip install -e ".[md,topology,dev]"
```


## Quick Start

Here's a minimal example to simulate a simple SDE trajectory and visualize it:

```python
import torch
from electrodiffusion.models.sde import IonChannelSDE
from electrodiffusion.simulations.integrators import SDEIntegrator
from electrodiffusion.utils.viz import plot_trajectory

# Mock system (replace with real for production)
class MockSystem:
    def potential_of_mean_force(self, z, protein_state): return torch.zeros_like(z)
    def electrostatic_potential(self, z, E_ext): return torch.zeros_like(z)
    def position_dependent_diffusion(self, z): return torch.ones_like(z)

noise_schedule = lambda t: 0.1 + 10.0 * t
sde = IonChannelSDE(MockSystem(), noise_schedule)

initial_state = torch.randn(1, 6)  # [batch=1, dim=6: pos3, vel3]
integrator = SDEIntegrator(sde, method='euler')
trajectory = integrator.integrate(initial_state, T=1.0, time_steps=1000)

plot_trajectory(trajectory)
```

For more advanced usage, see the examples directory.

## Directory Structure

- **config/**: Configuration management with defaults and physical constants.
- **models/**: Core model definitions (coarse-grained, SDE, diffusion, multi-scale, score networks, noise schedules).
- **simulations/**: Simulation logic (integrators, sampling, IV prediction, adaptive solvers, memory kernels).
- **utils/**: Utilities (data processing, math functions, topology constraints, visualization, embeddings).
- **tests/**: Unit tests for models, simulations, utils, and solvers.
- **examples/**: Executable scripts demonstrating key features.
- **setup.py**: Installation script.
- **README.md**: This file.


## Examples

The `examples/` directory contains self-contained scripts:

- **simple_sde_simulation.py**: Basic SDE integration and trajectory visualization.
- **diffusion_training.py**: Training a score-based diffusion model on synthetic data.
- **iv_curve_prediction.py**: Predicting and plotting IV curves using conditional sampling.
- **multi_scale_demo.py**: Demonstrating multi-scale dynamics with fast/slow separations.
- **convergence_analysis.py**: Analyzing solver convergence rates on a test problem.

Run any example with `python examples/<script>.py --help` for options.

## Testing

The library includes comprehensive unit tests. Run them with:

```bash
pytest tests/
```

Tests cover initialization, computations, shapes, and physical consistency (e.g., positive energies, decaying correlations).

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Create a Pull Request.

Ensure new code includes tests and follows PEP8 style (use black and flake8).

## License

This library is licensed under the MIT License. See LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```
@software{electrodiffusion2024,
  author = {Wonjin Yang},
  title = {Electrodiffusion: A Library for Ion Channel Dynamics Modeling},
  month = aug,
  year = 2024,
  url = {https://github.com/WonjinYang/electrodiffusion},
  version = {0.1.0}
}
```

For questions or issues, open a GitHub issue or contact the maintainers.

Happy modeling! 🚀