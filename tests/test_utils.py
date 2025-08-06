import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO  # For capturing plot bytes without display
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False

from electrodiffusion.utils.data import DataLoader  # Data loading and processing
from electrodiffusion.utils.math import compute_pmf, fokker_planck_residual, compute_autocorrelation, compute_gradient, estimate_diffusion_coeff  # Math utils
from electrodiffusion.utils.topology import TopologyConstraint  # Topology utils
from electrodiffusion.utils.viz import plot_iv_curve, visualize_distribution, plot_trajectory, plot_convergence  # Viz utils
from electrodiffusion.utils.embeddings import PositionalEncoding  # Embeddings
from electrodiffusion.config import PhysicalConstants

# Fixture for DataLoader
@pytest.fixture
def data_loader():
    return DataLoader()

# Fixture for synthetic data
@pytest.fixture
def synthetic_data():
    return torch.randn(100, 20)  # [frames=100, features=20]

# Fixture for positions
@pytest.fixture
def positions():
    return torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]])  # [batch=1, particles=2, dim=3]

def test_data_load_synthetic(data_loader, tmp_path):
    """Test synthetic data loading when MDA unavailable."""
    traj = data_loader.load_trajectory("dummy.pdb")  # Triggers synthetic
    assert traj.shape[0] > 0  # At least some frames
    assert traj.dim() == 2  # [frames, features]

def test_data_preprocess(data_loader, synthetic_data):
    """Test preprocessing with normalization and centering."""
    preprocessed = data_loader.preprocess(synthetic_data)
    assert preprocessed.shape == synthetic_data.shape
    assert torch.allclose(torch.mean(preprocessed, dim=0), torch.zeros_like(preprocessed[0]), atol=1e-5)  # Centered
    assert torch.allclose(torch.std(preprocessed, dim=0), torch.ones_like(preprocessed[0]), atol=1e-5)  # Normalized

def test_data_augment(data_loader, synthetic_data):
    """Test data augmentation with noise."""
    augmented = data_loader.augment_data(synthetic_data, noise_level=0.1)
    assert augmented.shape == synthetic_data.shape
    diff = augmented - synthetic_data
    assert torch.mean(torch.abs(diff)) > 0  # Noise added
    assert pytest.approx(torch.std(diff).item(), abs=0.05) == 0.1  # Std matches level (approx)

def test_compute_pmf(positions):
    """Test PMF computation for different types."""
    pmf_harmonic = compute_pmf(positions, potential_type='harmonic', params={'center': 0.5, 'k_spring': 1.0})
    assert pmf_harmonic.shape == (1,)  # Batch size
    assert pmf_harmonic.item() > 0  # Positive energy

    pmf_barrier = compute_pmf(positions, potential_type='gaussian_barrier')
    assert pmf_barrier.item() > 0

@pytest.mark.parametrize("drift_val, diffusion_val, score_val", [(1.0, 1.0, 1.0), (0.0, 0.0, 0.0)])
def test_fokker_planck_residual(drift_val, diffusion_val, score_val):
    """Test Fokker-Planck residual."""
    drift = torch.full((2, 3), drift_val)
    diffusion = torch.full((2, 3), diffusion_val)
    score = torch.full((2, 3), score_val)
    residual = fokker_planck_residual(drift, diffusion, score)
    assert residual.item() >= 0  # Non-negative (MSE)
    if drift_val == 0 and diffusion_val == 0:
        assert residual.item() == 0  # Zero at equilibrium

def test_compute_autocorrelation(synthetic_data):
    """Test autocorrelation computation."""
    autocorr = compute_autocorrelation(synthetic_data[:, 0])  # First column
    assert autocorr.shape[0] > 0
    assert autocorr[0].item() == pytest.approx(1.0, abs=1e-5)  # ACF(0) =1
    assert torch.all(autocorr <= 1.0)  # Normalized

def test_compute_gradient():
    """Test gradient computation."""
    def func(x): return torch.sum(x**2)
    input_t = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    grad = compute_gradient(func, input_t)
    expected = 2 * input_t  # grad(x^2) = 2x
    assert torch.allclose(grad, expected)

@pytest.mark.skipif(not GUDHI_AVAILABLE, reason="Gudhi required")
def test_topology_barcode(positions):
    """Test barcode computation."""
    constraint = TopologyConstraint()
    barcode = constraint.compute_barcode(positions)
    assert isinstance(barcode, list) and len(barcode) == 1  # Batch size
    assert all(isinstance(interval, tuple) for interval in barcode[0])  # Valid intervals

def test_viz_iv_curve():
    """Test IV curve plotting without display."""
    iv_data = {'voltages': np.linspace(-0.1, 0.1, 10), 'currents': np.random.randn(10)}
    # Use BytesIO to capture output without showing
    with BytesIO() as buf:
        plt.switch_backend('agg')  # Non-interactive
        plot_iv_curve(iv_data)
        assert plt.gcf() is not None  # Figure created

def test_embeddings():
    """Test PositionalEncoding."""
    enc = PositionalEncoding(d_model=16, max_len=100)
    t = torch.tensor([0.0, 0.5, 1.0])
    embed = enc(t)
    assert embed.shape == (3, 16)
    assert torch.all(embed >= -1) and torch.all(embed <= 1)  # Sinusoidal range
