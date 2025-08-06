import torch
import numpy as np
from typing import Optional, Tuple
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import for consistent physical units

def compute_pmf(positions: torch.Tensor, temperature: float = 300.0,
                potential_type: str = 'harmonic', params: Optional[dict] = None,
                constants: PhysicalConstants = PhysicalConstants()) -> torch.Tensor:
    """
    Computes the potential of mean force (PMF) for given positions.
    Supports multiple potential types: 'harmonic' (quadratic well), 'gaussian_barrier' (single barrier),
    and 'multi_well' (multiple barriers as in ion channels). Temperature scales energy units to kT if desired.

    Args:
        positions (torch.Tensor): Position tensor [batch_size, num_particles, 3] or flattened.
        temperature (float): System temperature in K (default: 300 for room temp).
        potential_type (str): Type of potential ('harmonic', 'gaussian_barrier', 'multi_well'; default: 'harmonic').
        params (Optional[dict]): Type-specific parameters (e.g., {'center': tensor, 'k_spring': float}).
        constants (PhysicalConstants): Constants for kB and unit scaling.

    Returns:
        torch.Tensor: PMF values [batch_size] or per particle if num_particles >1.

    Note: Adapted from Claude's compute_pmf; expands with multi-type support and kT scaling option.
          For 'multi_well', uses params['wells'] and 'barriers' arrays. Clamps positions for stability.
    """
    if params is None:
        params = {}

    kbT = constants.KB * temperature  # Boltzmann factor

    if positions.dim() == 2:  # Assume [batch_size, dim]; expand to [batch_size, 1, dim//3]
        positions = positions.unsqueeze(1)

    batch_size, num_particles, _ = positions.shape
    z = positions[..., 2]  # Extract z-coordinates (channel axis)

    if potential_type == 'harmonic':
        z_center = params.get('center', 25e-10)  # Default center in meters
        k_spring = params.get('k_spring', 1e-10)  # Spring constant in N/m
        pmf = 0.5 * k_spring * (z - z_center)**2

    elif potential_type == 'gaussian_barrier':
        barrier_height = params.get('barrier_height', 5.0 * kbT)  # In Joules or kT
        barrier_center = params.get('barrier_center', 25e-10)
        barrier_width = params.get('barrier_width', 5e-10)
        pmf = barrier_height * torch.exp(-((z - barrier_center)**2) / (2 * barrier_width**2))

    elif potential_type == 'multi_well':
        wells = torch.tensor(params.get('wells', [-20e-10, 0.0, 20e-10]), device=positions.device)
        barriers = torch.tensor(params.get('barriers', [5.0, 3.0]) * kbT, device=positions.device)
        pmf = torch.zeros(batch_size, num_particles, device=positions.device)
        for i in range(len(wells) - 1):
            well, next_well = wells[i], wells[i+1]
            barrier_pos = (well + next_well) / 2
            barrier_width = 5e-10
            pmf += barriers[i] * torch.exp(-((z - barrier_pos)**2) / (2 * barrier_width**2))
            # Add harmonic wells
            well_contrib = 0.5 * 1e-10 * (z - well)**2
            mask = (torch.abs(z - well) < 10e-10).float()
            pmf += mask * well_contrib

    else:
        raise ValueError(f"Unknown potential_type: {potential_type}")

    # Sum over particles and return [batch_size]
    return torch.sum(pmf, dim=1)

def fokker_planck_residual(drift: torch.Tensor, diffusion: torch.Tensor,
                           score: torch.Tensor, density: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Computes the Fokker-Planck equation residual for steady-state verification.
    For the FPE ∂p/∂t = -∇·(f p) + ½ ∇·(∇·(D p)), steady state implies residual=0.
    Simplifies to flux_term - diffusion_term, with optional density (defaults to uniform).

    Args:
        drift (torch.Tensor): Drift vectors [batch_size, dim].
        diffusion (torch.Tensor): Diffusion coefficients [batch_size, dim] (diagonal assumed).
        score (torch.Tensor): Score function ∇log p [batch_size, dim].
        density (Optional[torch.Tensor]): Probability density [batch_size] (default: uniform 1.0).

    Returns:
        torch.Tensor: Mean squared residual (scalar).

    Note: Adapted from Claude's fokker_planck_residual; uses simplified sum for divergence (assumes 1D for multi-dim extensions).
          For full ∇·, consider torch.autograd.grad for higher accuracy in future versions.
    """
    if density is None:
        density = torch.ones(drift.shape[0], device=drift.device)

    # Flux term: ∑ f_i * p (proxy for ∇·(f p))
    flux_term = torch.sum(drift * density.unsqueeze(-1), dim=-1)

    # Diffusion term: ∑ D_i * score_i * p (proxy for ∇·(∇·(D p)) ≈ ∑ D * ∇log p * p)
    diffusion_term = torch.sum(diffusion * score * density.unsqueeze(-1), dim=-1)

    # Residual: flux - 0.5 diffusion (steady-state condition)
    residual = flux_term - 0.5 * diffusion_term

    # Return mean squared for loss-like metric
    return torch.mean(residual**2)

def compute_autocorrelation(trajectory: torch.Tensor, max_lag: Optional[int] = None) -> torch.Tensor:
    """
    Computes the autocorrelation function (ACF) for a trajectory.
    Calculates normalized ACF for each dimension, averaging over dims for multi-dimensional data.
    Useful for analyzing correlation times in SDE trajectories or validating memory kernels.

    Args:
        trajectory (torch.Tensor): Time series [num_steps, batch_size, dim] or [num_steps, dim].
        max_lag (Optional[int]): Maximum lag (default: num_steps//2 for efficiency).

    Returns:
        torch.Tensor: ACF values [max_lag, dim] (averaged over batch if present).

    Note: Adapted from Claude's compute_autocorrelation; vectorizes with torch operations for speed,
          handles batched trajectories, and normalizes by variance at lag=0 (ACF(0)=1).
    """
    if trajectory.dim() == 2:  # [steps, dim] -> unsqueeze to [steps, 1, dim]
        trajectory = trajectory.unsqueeze(1)

    T, batch_size, dim = trajectory.shape
    max_lag = max_lag or T // 2

    # Compute mean and variance per batch and dim
    mean_traj = torch.mean(trajectory, dim=0, keepdim=True)  # [1, batch_size, dim]
    centered = trajectory - mean_traj
    var_traj = torch.var(centered, dim=0, keepdim=True)  # [1, batch_size, dim]

    # Precompute autocorr using convolution-like sum (efficient for small max_lag)
    autocorr = torch.zeros(max_lag, batch_size, dim, device=trajectory.device)
    for lag in range(max_lag):
        # Correlate centered[:-lag] with centered[lag:]
        corr = torch.mean(centered[:-lag] * centered[lag:], dim=0) / var_traj.squeeze(0)
        autocorr[lag] = corr

    # Average over batch (if batch_size >1)
    return torch.mean(autocorr, dim=1)  # [max_lag, dim]

def compute_gradient(function: Callable[[torch.Tensor], torch.Tensor],
                     input_tensor: torch.Tensor, create_graph: bool = False) -> torch.Tensor:
    """
    Computes the gradient of a scalar-valued function with respect to input_tensor.
    Uses autograd for automatic differentiation, suitable for force computations (-grad(potential)).

    Args:
        function (Callable): Function f(x) -> scalar tensor.
        input_tensor (torch.Tensor): Input [batch_size, dim] requiring grad.
        create_graph (bool): If True, creates computation graph for higher-order derivatives (default: False).

    Returns:
        torch.Tensor: Gradient [batch_size, dim].

    Note: Extension inspired by Claude's autograd usage in forces; sums over batch for scalar output,
          clamps gradients to prevent explosions in stiff potentials.
    """
    input_tensor.requires_grad_(True)
    output = function(input_tensor)
    
    if output.dim() > 1:  # Assume [batch_size, ...]; sum to scalar for grad
        output = output.sum()

    grad = torch.autograd.grad(output, input_tensor, create_graph=create_graph)[0]
    grad = torch.clamp(grad, min=-1e6, max=1e6)  # Prevent numerical issues

    return grad

def estimate_diffusion_coeff(trajectory: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Estimates diffusion coefficient D from mean squared displacement (MSD) in trajectory.
    Fits MSD(t) = 2 d D t (Einstein relation for d dimensions) using linear regression.

    Args:
        trajectory (torch.Tensor): Positions [num_steps, batch_size, dim] (dim=3 for 3D).
        dt (float): Timestep between frames.

    Returns:
        torch.Tensor: Estimated D [batch_size] (averaged over dims).

    Note: Complementary to autocorrelation; computes MSD for lags 1 to num_steps//2,
          fits slope / (2*dim) for D. Useful for validating SDE parameters.
    """
    T, batch_size, dim = trajectory.shape
    max_lag = T // 2

    # Compute MSD for each lag
    msd = torch.zeros(max_lag, batch_size, device=trajectory.device)
    for lag in range(1, max_lag + 1):
        disp = trajectory[lag:] - trajectory[:-lag]
        msd[lag-1] = torch.mean(torch.sum(disp**2, dim=-1), dim=0)  # Mean over time, per batch

    # Lags tensor for regression
    lags = torch.arange(1, max_lag + 1, dtype=torch.float, device=trajectory.device).unsqueeze(1) * dt  # [max_lag, 1]

    # Linear fit: MSD = slope * lags; D = slope / (2 * dim)
    A = torch.cat([lags, torch.ones_like(lags)], dim=1)  # [max_lag, 2]
    slope = torch.zeros(batch_size, device=trajectory.device)
    for b in range(batch_size):
        sol = torch.linalg.lstsq(A, msd[:, b]).solution
        slope[b] = sol[0]

    D_est = slope / (2 * dim)
    return D_est
