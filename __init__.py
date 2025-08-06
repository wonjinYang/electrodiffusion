# Define package version
__version__ = '0.1.0'  # Initial version; update with releases

# Import from config subpackage
from .config import DefaultConfig, PhysicalConstants

# Import from models subpackage (aggregated in models/__init__.py)
from .models import (
    CoarseGrainedModel,  # Coarse-grained protein representation
    IonChannelSDE, ProteinDynamicsSDE, PhysicsInformedSDE,  # SDE frameworks
    ScoreBasedDiffusion, ConditionalDiffusion,  # Diffusion models
    MultiScaleSDE, ImplicitSolvent,  # Multi-scale extensions
    PhysicsInformedScoreNetwork,  # Score estimation network
    NoiseSchedule  # Noise scheduling
)

# Import from simulations subpackage (aggregated in simulations/__init__.py)
from .simulations import (
    SDEIntegrator, EulerMaruyamaSolver, MilsteinSolver,  # SDE integrators
    DiffusionSampler,  # Score-based sampling
    IVCurvePredictor, NernstPlanckSolver,  # IV curve prediction
    AdaptiveSDESolver,  # Adaptive timestepping
    MemoryKernelSDE  # Non-Markovian SDE with memory
)

# Import from utils subpackage (individual utilities for direct access)
from .utils.data import DataLoader  # Data loading and preprocessing
from .utils.math import (
    compute_pmf, fokker_planck_residual, compute_autocorrelation,
    compute_gradient, estimate_diffusion_coeff  # Mathematical utilities
)
from .utils.topology import TopologyConstraint  # Persistent homology constraints
from .utils.viz import (
    plot_iv_curve, visualize_distribution, plot_trajectory, plot_convergence  # Visualization functions
)
from .utils.embeddings import PositionalEncoding  # Positional embeddings

# Define __all__ to specify the public API for wildcard imports (*)
__all__ = [
    # Config
    'DefaultConfig',
    'PhysicalConstants',
    
    # Models
    'CoarseGrainedModel',
    'IonChannelSDE',
    'ProteinDynamicsSDE',
    'PhysicsInformedSDE',
    'ScoreBasedDiffusion',
    'ConditionalDiffusion',
    'MultiScaleSDE',
    'ImplicitSolvent',
    'PhysicsInformedScoreNetwork',
    'NoiseSchedule',
    
    # Simulations
    'SDEIntegrator',
    'EulerMaruyamaSolver',
    'MilsteinSolver',
    'DiffusionSampler',
    'IVCurvePredictor',
    'NernstPlanckSolver',
    'AdaptiveSDESolver',
    'MemoryKernelSDE',
    
    # Utils
    'DataLoader',
    'compute_pmf',
    'fokker_planck_residual',
    'compute_autocorrelation',
    'compute_gradient',
    'estimate_diffusion_coeff',
    'TopologyConstraint',
    'plot_iv_curve',
    'visualize_distribution',
    'plot_trajectory',
    'plot_convergence',
    'PositionalEncoding',
]

# Optional: Package-level initialization or logging
print(f"Electrodiffusion library initialized (version {__version__}).")  # Debug message; remove in production
