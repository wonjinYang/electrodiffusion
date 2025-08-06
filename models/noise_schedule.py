import torch
import numpy as np
from typing import Optional

class NoiseSchedule:
    """
    Noise scheduling class for VP-SDE in diffusion models.
    This class manages the time-dependent noise level β(t), cumulative scaling α(t),
    and noise standard deviation σ(t). It supports various schedule types to control
    the diffusion process, ensuring smooth transition from data to noise.

    Args:
        schedule_type (str): Type of noise schedule ('linear', 'cosine', 'exponential').
        beta_min (float): Minimum beta value at t=0 (default: 0.1 for mild initial noise).
        beta_max (float): Maximum beta value at t=1 (default: 20.0 for strong terminal noise).
        cosine_s (float): Smoothing factor for cosine schedule (default: 0.008, as in improved DDPM).
    """
    def __init__(self, schedule_type: str = 'linear', beta_min: float = 0.1, beta_max: float = 20.0,
                 cosine_s: float = 0.008):
        self.schedule_type = schedule_type.lower()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.cosine_s = cosine_s  # Smoothing parameter for cosine schedule to avoid abrupt starts

        # Validate schedule type
        valid_types = ['linear', 'cosine', 'exponential']
        if self.schedule_type not in valid_types:
            raise ValueError(f"Invalid schedule_type '{self.schedule_type}'. Choose from {valid_types}.")

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise schedule β(t), which controls the diffusion strength at time t.

        Args:
            t (torch.Tensor): Time tensor, typically in [0,1] (scalar or batched [batch_size]).

        Returns:
            torch.Tensor: β(t) values, same shape as t.

        Note: For linear: simple interpolation; cosine: smoother curve mimicking annealed variance;
              exponential: rapid growth suitable for systems needing quick noise escalation.
              Clamps t to [0,1] for safety.
        """
        t = torch.clamp(t, min=0.0, max=1.0)  # Ensure t is within valid range

        if self.schedule_type == 'linear':
            # Linear interpolation: β(t) = β_min + t * (β_max - β_min)
            return self.beta_min + t * (self.beta_max - self.beta_min)

        elif self.schedule_type == 'cosine':
            # Cosine schedule: β(t) = β_max * [cos((t + s)/(1 + s) * π/2)]^2
            # This provides a smoother increase, reducing artifacts in early diffusion steps
            angle = (t + self.cosine_s) / (1 + self.cosine_s) * (np.pi / 2)
            return self.beta_max * torch.cos(angle) ** 2

        elif self.schedule_type == 'exponential':
            # Exponential: β(t) = β_min * exp(t * log(β_max / β_min))
            # Useful for scenarios where noise needs to ramp up quickly
            log_ratio = torch.log(torch.tensor(self.beta_max / self.beta_min))
            return self.beta_min * torch.exp(t * log_ratio)

    def alpha(self, t: torch.Tensor, integration_steps: Optional[int] = 1000) -> torch.Tensor:
        """
        Computes the cumulative scaling α(t) = exp(-1/2 ∫_0^t β(s) ds).
        For linear schedules, closed-form; for others, numerical integration via trapezoidal rule.

        Args:
            t (torch.Tensor): Time tensor [batch_size] or scalar.
            integration_steps (Optional[int]): Number of steps for numerical integration (default: 1000 for accuracy).

        Returns:
            torch.Tensor: α(t) values, same shape as t.

        Note: Numerical integration ensures precision for complex schedules. Uses torch.trapz for efficiency.
              If t is batched, computes integral per batch element.
        """
        t = torch.clamp(t, min=0.0, max=1.0)

        if self.schedule_type == 'linear':
            # Closed-form for linear: ∫ β(s) ds = β_min t + 0.5 (β_max - β_min) t^2
            integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2
        else:
            # Numerical integration for non-linear schedules
            # Create a grid from 0 to t with 'integration_steps' points
            device = t.device
            if t.dim() == 0:  # Scalar t
                t_grid = torch.linspace(0, t.item(), integration_steps, device=device)
                beta_vals = self.beta(t_grid)
                integral = torch.trapz(beta_vals, t_grid)  # Trapezoidal rule
            else:  # Batched t
                # For efficiency, compute per batch element (vectorized where possible)
                integral = torch.zeros_like(t)
                for i in range(t.shape[0]):
                    t_grid = torch.linspace(0, t[i].item(), integration_steps, device=device)
                    beta_vals = self.beta(t_grid)
                    integral[i] = torch.trapz(beta_vals, t_grid)

        # α(t) = exp(-0.5 * integral)
        return torch.exp(-0.5 * integral)

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the noise standard deviation σ(t) = √(1 - α(t)^2).
        This represents the cumulative noise added up to time t in the forward process.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: σ(t) values.

        Note: Directly derived from α(t); used in forward sampling to add Gaussian noise.
              Ensures σ(0) ≈ 0 and σ(1) ≈ 1 for pure noise at t=1.
        """
        alpha_t = self.alpha(t)
        return torch.sqrt(1 - alpha_t**2)

    def cumulative_variance(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the cumulative variance 1 - α(t)^2, equivalent to σ(t)^2.
        Useful for variance-preserving properties in VP-SDE.

        Args:
            t (torch.Tensor): Time tensor.

        Returns:
            torch.Tensor: Cumulative variance.

        Note: This is a convenience method; directly calls sigma(t)^2 for efficiency.
        """
        return self.sigma(t)**2
