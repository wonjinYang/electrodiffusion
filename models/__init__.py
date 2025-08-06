# Import standard libraries if needed (none required here)

# Import from submodules: Aggregate key classes and functions
from .coarse_grained import CoarseGrainedModel, IonState  # Core classes for protein and ion state management
from .sde import IonChannelSDE, ProteinDynamicsSDE, PhysicsInformedSDE  # SDE frameworks for electrodiffusion and physics-informed dynamics
from .diffusion import ScoreBasedDiffusion, ConditionalDiffusion  # Diffusion model classes for forward/reverse processes and conditional generation
from .multi_scale import MultiScaleSDE, ImplicitSolvent  # Multi-scale handling and solvent effects integration
from .score_network import PositionalEncoding, AttentionBlock, PhysicsInformedScoreNetwork  # Neural network components for score estimation with attention and embeddings
from .noise_schedule import NoiseSchedule  # Noise scheduling utilities for VP-SDE

# Define __all__ to specify the public interface (controls wildcard imports)
__all__ = [
    # From coarse_grained.py
    'CoarseGrainedModel',
    'IonState',
    
    # From sde.py
    'IonChannelSDE',
    'ProteinDynamicsSDE',
    'PhysicsInformedSDE',
    
    # From diffusion.py
    'ScoreBasedDiffusion',
    'ConditionalDiffusion',
    
    # From multi_scale.py
    'MultiScaleSDE',
    'ImplicitSolvent',
    
    # From score_network.py
    'PositionalEncoding',
    'AttentionBlock',
    'PhysicsInformedScoreNetwork',
    
    # From noise_schedule.py
    'NoiseSchedule',
]

def get_default_score_network(input_dim: int = 128, hidden_dim: int = 256) -> PhysicsInformedScoreNetwork:
    """
    Convenience function to instantiate a default PhysicsInformedScoreNetwork.
    
    Args:
        input_dim (int): Dimensionality of the input state (default: 128 for typical coarse-grained states).
        hidden_dim (int): Hidden layer size (default: 256).
    
    Returns:
        PhysicsInformedScoreNetwork: Instantiated network ready for use in diffusion models.
    
    Note: This helper simplifies setup in examples or tests, drawing from Claude's default network initializations.
    """
    return PhysicsInformedScoreNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

# Additional helpers can be added here, e.g., factory functions for SDEs or diffusion models
