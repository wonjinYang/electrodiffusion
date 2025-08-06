import torch
from typing import Dict, Optional, Callable
from electrodiffusion.models.diffusion import ScoreBasedDiffusion  # Import base diffusion model for reverse processes

class DiffusionSampler:
    """
    Sampler class for generating states using score-based diffusion models.
    This class initializes with a diffusion model and performs reverse SDE sampling to denoise from Gaussian noise
    back to the data distribution. It supports conditional generation and ensemble computations for observables.
    The sampling process is stochastic, allowing for diverse generations, but can be made deterministic by omitting noise.

    Args:
        diffusion_model (ScoreBasedDiffusion): The underlying diffusion model providing reverse_process.
        num_steps (int): Number of denoising steps (default: 1000 for fine-grained sampling; lower for speed).
    """
    def __init__(self, diffusion_model: ScoreBasedDiffusion, num_steps: int = 1000):
        self.diffusion_model = diffusion_model
        self.num_steps = num_steps
        if self.num_steps < 10:
            print(f"Warning: Low num_steps ({self.num_steps}); sampling may be inaccurate.")

    def sample(self, num_samples: int, dim: int,
               condition: Optional[Dict] = None, device: str = 'cpu',
               deterministic: bool = False) -> torch.Tensor:
        """
        Generates samples by running the reverse diffusion process from pure noise.
        Starts from x_T ~ N(0, I) and iteratively applies reverse steps until t=0, producing samples from the learned distribution.

        Args:
            num_samples (int): Number of samples to generate.
            dim (int): Dimensionality of each sample (must match model's input_dim).
            condition (Optional[Dict]): Conditioning parameters (e.g., {'voltage': tensor[num_samples]}).
            device (str): Target device ('cpu' or 'cuda'; default: 'cpu').
            deterministic (bool): If True, omits stochastic noise for ODE-like sampling (default: False).

        Returns:
            torch.Tensor: Generated samples [num_samples, dim].

        Note: Adapted from Claude's sample method; computes dt = 1/num_steps, iterates from t=1 to t=0.
              For conditions, assumes they are tensors matching num_samples; broadcasts scalars.
              Progressive output every 100 steps for monitoring large runs (commented out for brevity).
        """
        dt = 1.0 / self.num_steps  # Fixed timestep for reverse process (t decreases from 1 to 0)

        # Initialize from Gaussian noise at t=1
        x = torch.randn(num_samples, dim, device=device)

        # Prepare t tensor for batching
        t = torch.ones(num_samples, device=device)  # Start at t=1

        with torch.no_grad():  # Disable gradients for sampling efficiency
            for i in range(self.num_steps):
                # Update t for current step (decreases towards 0)
                t = torch.full((num_samples,), 1.0 - i * dt, device=device)

                # Compute reverse drift (includes score)
                drift = self.diffusion_model.reverse_process(x, t)  # Assumes reverse_process handles condition internally

                # Diffusion coefficient for noise term
                beta_t = self.diffusion_model.noise_schedule.beta(t).unsqueeze(-1)  # [num_samples, 1]
                diffusion = torch.sqrt(beta_t * dt)  # Scalar per sample

                # Stochastic noise (backward Wiener)
                noise = torch.randn_like(x) if not deterministic else torch.zeros_like(x)

                # Update x: x_{t-dt} = x_t + drift * (-dt) + diffusion * noise (reverse direction)
                x = x + drift * (-dt) + diffusion * noise

                # Optional: Progressive monitoring (e.g., print norms every 100 steps)
                # if i % 100 == 0:
                #     print(f"Step {i}: Mean norm {torch.mean(torch.norm(x, dim=1)).item()}")

        return x

    def ensemble_average(self, samples: torch.Tensor,
                         observable_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Computes the ensemble average of a given observable over the samples.
        This is useful for estimating macroscopic properties like average flux, energy, or probability densities
        from a set of generated microscopic states.

        Args:
            samples (torch.Tensor): Generated samples [num_samples, dim].
            observable_fn (Callable): Function that computes the observable for a single sample (input: [dim], output: scalar or tensor).

        Returns:
            torch.Tensor: Averaged observable (scalar or tensor, depending on observable_fn).

        Note: From Claude's ensemble_average; vectorized for efficiency using torch operations instead of loops.
              Handles cases where observable_fn returns tensors (e.g., multi-dimensional observables) by stacking and mean.
        """
        # Apply observable_fn to each sample (vectorized if possible; fallback to loop for complex fns)
        if observable_fn is not None:
            # Attempt vectorized application (assumes fn handles batched input)
            try:
                observables = observable_fn(samples)  # Should return [num_samples, obs_dim]
            except TypeError:
                # Fallback to loop if fn expects single sample
                observables = torch.stack([observable_fn(sample) for sample in samples])
        else:
            raise ValueError("observable_fn must be provided")

        # Compute mean over samples
        return torch.mean(observables, dim=0)

    def guided_sample(self, num_samples: int, dim: int,
                      guide_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      strength: float = 1.0, **kwargs) -> torch.Tensor:
        """
        Performs guided sampling by adding a guidance term (e.g., from a classifier) to the score.
        This extends standard sampling for tasks like conditional generation with additional priors.

        Args:
            num_samples (int): Number of samples.
            dim (int): Sample dimension.
            guide_fn (Callable): Guidance function (t, x) -> guidance vector [num_samples, dim].
            strength (float): Strength of guidance (default: 1.0; higher biases towards guide).
            **kwargs: Passed to sample (e.g., condition, device).

        Returns:
            torch.Tensor: Guided samples [num_samples, dim].

        Note: Extension not in Claude's direct code but inspired by score-based guidance techniques;
              modifies drift by adding strength * guide_fn(t, x) during reverse steps.
        """
        dt = 1.0 / self.num_steps
        x = torch.randn(num_samples, dim, device=kwargs.get('device', 'cpu'))

        with torch.no_grad():
            for i in range(self.num_steps):
                t = torch.full((num_samples,), 1.0 - i * dt, device=x.device)

                # Base reverse drift
                drift = self.diffusion_model.reverse_process(x, t)

                # Add guidance
                guidance = guide_fn(t, x)
                drift += strength * guidance

                beta_t = self.diffusion_model.noise_schedule.beta(t).unsqueeze(-1)
                diffusion = torch.sqrt(beta_t * dt)

                noise = torch.randn_like(x) if not kwargs.get('deterministic', False) else torch.zeros_like(x)

                x = x + drift * (-dt) + diffusion * noise

        return x

    def estimate_variance(self, num_samples: int, dim: int, num_repeats: int = 10, **kwargs) -> torch.Tensor:
        """
        Estimates sampling variance by generating multiple sets and computing per-dimension std.

        Args:
            num_samples (int): Samples per repeat.
            dim (int): Dimension.
            num_repeats (int): Number of independent sampling runs.
            **kwargs: Passed to sample.

        Returns:
            torch.Tensor: Variance estimate [dim].

        Note: Useful for uncertainty quantification; computes std over means of each repeat.
        """
        means = []
        for _ in range(num_repeats):
            samples = self.sample(num_samples, dim, **kwargs)
            means.append(torch.mean(samples, dim=0))
        
        means_stack = torch.stack(means)
        variance = torch.var(means_stack, dim=0)
        return variance
