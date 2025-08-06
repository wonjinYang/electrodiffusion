import pytest
import torch
import numpy as np
from electrodiffusion.simulations import (
    EulerMaruyamaSolver, MilsteinSolver, SDEIntegrator,  # From integrators
    DiffusionSampler,  # From sampling
    IVCurvePredictor, NernstPlanckSolver,  # From iv_curve
    AdaptiveSDESolver,  # From adaptive_solvers
    MemoryKernelSDE  # From memory_kernel
)
from electrodiffusion.models import IonChannelSDE, ScoreBasedDiffusion  # For mocking
from electrodiffusion.config import PhysicalConstants

# Fixture for mock SDE
@pytest.fixture
def mock_sde():
    class MockSDE(IonChannelSDE):
        def f(self, t, y):
            return -0.1 * y  # Simple damping
        def g(self, t, y):
            return torch.ones_like(y) * 0.5  # Constant diffusion
    return MockSDE(None, lambda t: 1.0)  # Dummy args

# Fixture for mock diffusion model
@pytest.fixture
def mock_diffusion():
    class MockDiffusion(ScoreBasedDiffusion):
        def reverse_process(self, xt, t):
            return -xt * 0.1  # Mock denoising
    return MockDiffusion(None, None)  # Dummy

# Fixture for synthetic samples
@pytest.fixture
def synthetic_samples():
    return torch.randn(100, 8)  # [num_samples, state_dim=8 as in examples]

@pytest.mark.parametrize("method", ["euler", "milstein"])
def test_sde_integrator(mock_sde, method):
    """Test SDEIntegrator trajectory generation."""
    integrator = SDEIntegrator(mock_sde, method=method)
    initial = torch.zeros(2, 4)  # [batch=2, dim=4]
    traj = integrator.integrate(initial, T=1.0, time_steps=10)
    assert traj.shape == (11, 2, 4)  # Includes initial
    assert torch.all(traj[0] == initial)  # Initial preserved
    # Check damping: values decrease over time
    assert torch.mean(traj[-1]) < torch.mean(traj[0])

def test_adaptive_solver(mock_sde):
    """Test AdaptiveSDESolver step adjustment."""
    solver = AdaptiveSDESolver(tolerance=1e-3)
    t = 0.0
    x = torch.ones(2, 3)
    x_new, dt_used = solver.adaptive_step(t, x, mock_sde.f, mock_sde.g)
    assert x_new.shape == x.shape
    assert 0 < dt_used <= solver.max_dt
    # Check if step reduces error (simplified; compare norms)
    assert torch.norm(x_new - x).item() > 0  # Some change occurred

def test_diffusion_sampler(mock_diffusion):
    """Test DiffusionSampler generation."""
    sampler = DiffusionSampler(mock_diffusion, num_steps=50)
    samples = sampler.sample(num_samples=5, dim=4, device='cpu')
    assert samples.shape == (5, 4)
    # Check if denoised (mock returns damped values)
    assert torch.mean(samples) < 0  # From -xt *0.1 iterated

def test_iv_curve_predictor(mock_diffusion, synthetic_samples):
    """Test IVCurvePredictor computations."""
    sampler = DiffusionSampler(mock_diffusion)
    predictor = IVCurvePredictor(sampler, PhysicalConstants())
    voltages = np.linspace(-0.1, 0.1, 3)
    iv_data = predictor.compute_iv(voltages, num_samples=10, state_dim=8)
    assert 'currents' in iv_data and len(iv_data['currents']) == len(voltages)
    assert np.all(iv_data['open_probabilities'] >= 0) and np.all(iv_data['open_probabilities'] <= 1)

    validation = predictor.validate(num_samples=10, state_dim=8)
    assert 'detailed_balance_pass' in validation  # Check keys present

def test_memory_kernel_sde():
    """Test MemoryKernelSDE solving."""
    kernel_sde = MemoryKernelSDE(gamma=1.0, memory_time=0.1)
    T = 1.0
    x0 = torch.zeros(2, 3)  # [batch=2, dim=3]
    v0 = torch.ones(2, 3)
    def force(t, x): return -x * 0.5  # Simple harmonic
    pos, vel = kernel_sde.solve_with_memory(T, x0, v0, force, dt=0.01)
    assert pos.shape == vel.shape  # Matching shapes
    assert pos.shape[0] == int(T / 0.01) + 1  # Correct number of steps
    # Check damping: final velocity smaller
    assert torch.mean(torch.abs(vel[-1])) < torch.mean(torch.abs(v0))

@pytest.mark.parametrize("dt", [0.01, 0.001])
def test_solver_convergence(mock_sde, dt):
    """Test convergence by comparing to analytical (e.g., Ornstein-Uhlenbeck)."""
    # Simplified OU: dx = -x dt + dW; mean ->0, var->0.5
    integrator = SDEIntegrator(mock_sde)
    initial = torch.ones(1, 1)
    T = 1.0
    steps = int(T / dt)
    traj = integrator.integrate(initial, T, steps)
    final_mean = torch.mean(traj[-1]).item()
    assert abs(final_mean) < 0.1  # Converges to 0
