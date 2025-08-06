# Import from submodules: Aggregate key classes and functions for easy access
from .integrators import (
    SDESolver,  # Abstract base for SDE solvers
    EulerMaruyamaSolver,  # First-order Euler-Maruyama method
    MilsteinSolver,  # Second-order Milstein method
    SDEIntegrator  # High-level trajectory integrator
)
from .sampling import DiffusionSampler  # Sampler for score-based generation and ensemble averaging
from .iv_curve import NernstPlanckSolver, IVCurvePredictor  # Flux computation and IV curve prediction
from .adaptive_solvers import AdaptiveSDESolver  # Adaptive timestepping solver
from .memory_kernel import MemoryKernelSDE  # Non-Markovian GLE solver with memory kernels

# Define __all__ to control the public interface for wildcard imports
__all__ = [
    # From integrators.py
    'SDESolver',
    'EulerMaruyamaSolver',
    'MilsteinSolver',
    'SDEIntegrator',
    
    # From sampling.py
    'DiffusionSampler',
    
    # From iv_curve.py
    'NernstPlanckSolver',
    'IVCurvePredictor',
    
    # From adaptive_solvers.py
    'AdaptiveSDESolver',
    
    # From memory_kernel.py
    'MemoryKernelSDE',
]

# Convenience function: Create a default SDE integrator with adaptive method
def create_adaptive_integrator(sde_model, tolerance: float = 1e-4) -> SDEIntegrator:
    """
    Factory function to create an SDEIntegrator with adaptive timestepping.
    
    Args:
        sde_model: The SDE model (e.g., IonChannelSDE instance).
        tolerance (float): Error tolerance for adaptive steps (default: 1e-4).
    
    Returns:
        SDEIntegrator: Configured integrator ready for use.
    
    Note: Inspired by Claude's adaptive solver usage; simplifies setup for examples like simple_sde_simulation.py.
    """
    adaptive_solver = AdaptiveSDESolver(tolerance=tolerance)
    return SDEIntegrator(sde_model, method="adaptive")  # Assumes SDEIntegrator accepts solver instances in extensions

# Additional helpers can be added, e.g., a quick sampler initializer or validation runner
