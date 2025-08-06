import pytest
import torch
import numpy as np
from electrodiffusion.simulations.integrators import (
    EulerMaruyamaSolver, MilsteinSolver, SDEIntegrator
)
from electrodiffusion.simulations.adaptive_solvers import AdaptiveSDESolver
from electrodiffusion.models.sde import IonChannelSDE  # For mocking

# Fixture for Ornstein-Uhlenbeck SDE (analytical test case)
@pytest.fixture
def ou_sde():
    theta = 1.0  # Decay rate
    sigma = np.sqrt(2 * theta)  # For var=1 at equilibrium
    class OUSDE(IonChannelSDE):
        def f(self, t, y):
            return -theta * y
        def g(self, t, y):
            return sigma * torch.ones_like(y)
    return OUSDE(None, None)  # Dummy args

# Fixture for dt values
@pytest.fixture
def dt_values():
    return [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

@pytest.mark.parametrize("solver_class", [EulerMaruyamaSolver, MilsteinSolver])
def test_solver_step(ou_sde, solver_class):
    """Test single step of solvers."""
    solver = solver_class()
    t = 0.0
    dt = 0.01
    x = torch.tensor([[1.0], [2.0]])  # [batch=2, dim=1]
    x_new = solver.step(t, dt, x, ou_sde.f, ou_sde.g)
    assert x_new.shape == x.shape
    # Check approximate decrease (drift negative)
    assert torch.all(x_new < x)

def test_sde_integrator(ou_sde):
    """Test full integration and shape."""
    integrator = SDEIntegrator(ou_sde, method="euler")
    initial = torch.tensor([[1.0]])
    T = 1.0
    steps = 100
    traj = integrator.integrate(initial, T, steps)
    assert traj.shape == (steps + 1, 1, 1)
    # Check decay towards 0
    assert abs(traj[-1].item()) < 0.5  # Rough check for damping

@pytest.mark.parametrize("tolerance", [1e-3, 1e-4])
def test_adaptive_solver(ou_sde, tolerance):
    """Test adaptive step size adjustment."""
    solver = AdaptiveSDESolver(tolerance=tolerance)
    t = 0.0
    x = torch.tensor([[1.0]])
    x_new, dt_used = solver.adaptive_step(t, x, ou_sde.f, ou_sde.g)
    assert x_new.shape == x.shape
    assert dt_used <= solver.max_dt
    # Tighter tolerance should yield smaller dt
    if tolerance == 1e-4:
        assert dt_used < 0.01  # Empirical

def test_convergence_analysis(ou_sde, dt_values):
    """Test convergence rates for mean and variance."""
    num_paths = 50
    T = 1.0
    x0 = torch.tensor([[1.0]])
    
    solvers = {
        'Euler': EulerMaruyamaSolver(),
        'Milstein': MilsteinSolver()
    }
    
    results = {}
    for name, solver in solvers.items():
        mean_errors = []
        var_errors = []
        for dt in dt_values:
            steps = int(T / dt)
            finals = []
            for _ in range(num_paths):
                integrator = SDEIntegrator(ou_sde)  # New for each path
                traj = integrator.integrate(x0, T, steps)
                finals.append(traj[-1].item())
            
            finals = np.array(finals)
            num_mean = np.mean(finals)
            num_var = np.var(finals)
            
            # Analytical OU: mean = x0 exp(-θ T), var = (σ²/2θ)(1 - exp(-2θ T))
            anal_mean = x0.item() * np.exp(-ou_sde.theta * T)  # Assuming theta=1 from fixture
            anal_var = (ou_sde.sigma**2 / (2 * ou_sde.theta)) * (1 - np.exp(-2 * ou_sde.theta * T))
            
            mean_errors.append(abs(num_mean - anal_mean))
            var_errors.append(abs(num_var - anal_var))
        
        results[name] = {'mean_errors': mean_errors, 'var_errors': var_errors}
    
    # Check decreasing errors (convergence)
    for name in results:
        assert all(np.diff(results[name]['mean_errors']) < 0)  # Strictly decreasing
        assert results['Milstein']['mean_errors'][-1] < results['Euler']['mean_errors'][-1]  # Milstein better

@pytest.mark.skipif(torch.cuda.is_available() == False, reason="Requires GPU")
def test_gpu_performance(ou_sde):
    """Test performance on GPU with large batch."""
    ou_sde = ou_sde.to('cuda')  # Assuming to method for device
    integrator = SDEIntegrator(ou_sde, method="milstein")
    initial = torch.ones(1000, 10, device='cuda')  # Large batch
    traj = integrator.integrate(initial, T=1.0, time_steps=100)
    assert traj.device.type == 'cuda'
    assert traj.shape == (101, 1000, 10)
