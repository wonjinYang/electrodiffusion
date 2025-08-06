import pytest
import torch
import numpy as np
from electrodiffusion.models import (
    CoarseGrainedModel, IonState,  # From coarse_grained
    IonChannelSDE, ProteinDynamicsSDE, PhysicsInformedSDE,  # From sde
    ScoreBasedDiffusion, ConditionalDiffusion,  # From diffusion
    MultiScaleSDE, ImplicitSolvent,  # From multi_scale
    PhysicsInformedScoreNetwork,  # From score_network
    NoiseSchedule  # From noise_schedule
)
from electrodiffusion.config import PhysicalConstants, DefaultConfig  # For params and constants

# Fixture for default constants and config
@pytest.fixture
def default_constants():
    return PhysicalConstants()

@pytest.fixture
def default_config():
    return DefaultConfig()

# Fixture for synthetic state data
@pytest.fixture
def synthetic_state(batch_size=2, num_residues=10):
    backbone_dim = num_residues * 9
    sidechain_dim = num_residues * 7
    ion_dim = 4  # coords:3, charge:1
    full_dim = backbone_dim + sidechain_dim + ion_dim
    return torch.randn(batch_size, full_dim)

# Fixture for small SDE model
@pytest.fixture
def small_sde(default_constants):
    class MockSystem:
        def potential_of_mean_force(self, z, protein_state):
            return torch.zeros_like(z)
        def electrostatic_potential(self, z, E_ext):
            return torch.zeros_like(z)
        def position_dependent_diffusion(self, z):
            return torch.ones_like(z)

    noise_schedule = lambda t: 0.1 + 10.0 * t
    return IonChannelSDE(MockSystem(), noise_schedule, constants=default_constants)

@pytest.mark.parametrize("num_residues", [5, 10])
def test_coarse_grained_init(default_constants, num_residues):
    """Test initialization and dimensions of CoarseGrainedModel."""
    model = CoarseGrainedModel(num_residues, default_constants)
    assert model.num_residues == num_residues
    assert model.backbone_dim == num_residues * 9
    assert model.sidechain_dim == num_residues * 7
    assert isinstance(model.constants, PhysicalConstants)

def test_coarse_grained_separate_states(synthetic_state):
    """Test state separation in CoarseGrainedModel."""
    batch_size, full_dim = synthetic_state.shape
    num_residues = 10  # Matches fixture
    model = CoarseGrainedModel(num_residues, PhysicalConstants())
    backbone, sidechain = model.separate_states(synthetic_state)
    assert backbone.shape == (batch_size, model.backbone_dim)
    assert sidechain.shape == (batch_size, model.sidechain_dim)
    assert torch.allclose(torch.cat([backbone, sidechain], dim=1), synthetic_state)

def test_ion_state_conversions():
    """Test coordinate conversions in IonState."""
    state = IonState(channel_length=50e-10, channel_radius=5e-10)
    cartesian = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    conical = state.cartesian_to_conical(cartesian)
    assert conical.shape == cartesian.shape
    # Check r = sqrt(x^2 + y^2), theta = atan2(y,x), z unchanged
    assert torch.allclose(conical[:, 0], torch.sqrt(cartesian[:, 0]**2 + cartesian[:, 1]**2))
    assert torch.allclose(conical[:, 2], cartesian[:, 2])

    recon_cart = state.conical_to_cartesian(conical)
    assert torch.allclose(recon_cart, cartesian, atol=1e-6)  # Round-trip check with tolerance

def test_ion_state_inside_channel():
    """Test boundary checks in IonState."""
    state = IonState(channel_length=10.0, channel_radius=2.0)
    positions = torch.tensor([[0.0, 0.0, 5.0], [1.0, 1.0, 5.0], [3.0, 0.0, 5.0], [0.0, 0.0, 11.0]])
    inside = state.is_inside_channel(positions)
    expected = torch.tensor([True, True, False, False])  # First two inside radius and length, third out radius, fourth out length
    assert torch.all(inside == expected)

def test_ion_channel_sde_init(small_sde):
    """Test initialization of IonChannelSDE."""
    assert small_sde.noise_type == "diagonal"
    assert small_sde.sde_type == "ito"
    assert hasattr(small_sde, 'beta_fn')
    assert small_sde.E_ext == 0.0  # Default

def test_ion_channel_sde_drift_diffusion(small_sde, synthetic_state):
    """Test drift f and diffusion g computations."""
    t = torch.tensor([0.5])
    y = synthetic_state
    drift = small_sde.f(t, y)
    assert drift.shape == y.shape
    # Check VP term presence (negative)
    assert torch.all(drift < 0).item()  # Simplified check; depends on beta

    diffusion = small_sde.g(t, y)
    assert diffusion.shape == y.shape
    assert torch.all(diffusion > 0).item()  # Positive sqrt(beta)

@pytest.mark.parametrize("schedule_type", ["linear", "cosine"])
def test_noise_schedule(schedule_type):
    """Test NoiseSchedule computations."""
    schedule = NoiseSchedule(schedule_type=schedule_type)
    t = torch.tensor([0.0, 0.5, 1.0])
    beta = schedule.beta(t)
    assert beta.shape == t.shape
    assert beta[0] >= 0 and beta[-1] > beta[0]  # Increasing

    alpha = schedule.alpha(t)
    assert torch.all(alpha <= 1.0) and torch.all(alpha >= 0.0)
    assert alpha[0].item() == pytest.approx(1.0, abs=1e-3)  # alpha(0) ~1

    sigma = schedule.sigma(t)
    assert sigma[0].item() == pytest.approx(0.0, abs=1e-3)  # sigma(0) ~0

def test_score_network_forward():
    """Test PhysicsInformedScoreNetwork forward pass."""
    input_dim = 10
    net = PhysicsInformedScoreNetwork(input_dim)
    x = torch.randn(2, input_dim)
    t = torch.tensor([0.5, 0.6])
    outputs = net(x, t)
    assert 'score' in outputs and outputs['score'].shape == x.shape
    assert 'energy' in outputs and outputs['energy'].shape == (2, 1)
    assert 'force' in outputs and outputs['force'].shape == x.shape

def test_diffusion_forward_reverse():
    """Test ScoreBasedDiffusion processes."""
    net = PhysicsInformedScoreNetwork(4)
    schedule = NoiseSchedule()
    diffusion = ScoreBasedDiffusion(net, schedule)
    x0 = torch.randn(2, 4)
    t = torch.tensor([0.5, 0.5])
    x_t, noise = diffusion.forward_process(x0, t)
    assert x_t.shape == x0.shape
    assert torch.allclose(x_t, x0 * diffusion.noise_schedule.alpha(t).unsqueeze(-1) + 
                          noise * diffusion.noise_schedule.sigma(t).unsqueeze(-1), atol=1e-5)

    # Reverse step (dummy check for shape)
    rev = diffusion.reverse_step(x_t, t, 0.01)
    assert rev.shape == x_t.shape

def test_multi_scale_projection():
    """Test MultiScaleSDE projection."""
    fast_sde = IonChannelSDE(...)  # Use small_sde fixture or mock
    slow_sde = IonChannelSDE(...)
    multi = MultiScaleSDE(fast_sde, slow_sde)
    fast_vars = torch.randn(5, 2, 3)  # [steps, batch, fast_dim]
    slow_vars = torch.randn(2, 4)     # [batch, slow_dim]
    projected = multi.project_slow(fast_vars, slow_vars)
    assert projected.shape == slow_vars.shape

@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for performance test")
def test_performance():
    """Performance test for large batch integration."""
    # Example: Integrate with large batch and check time
    pass  # Implement if needed
