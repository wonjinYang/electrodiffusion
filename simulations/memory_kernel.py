import torch
import numpy as np
from typing import Tuple, Callable
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import for thermodynamic constants

class MemoryKernelSDE:
    """
    Memory kernel SDE class for solving generalized Langevin equations (GLE).
    This class models non-Markovian dynamics with memory effects, incorporating friction, memory kernels,
    and colored noise consistent with the fluctuation-dissipation theorem. It is designed for projected
    dynamics in multi-scale systems, where fast variables are integrated out, leaving memory terms.

    Args:
        gamma (float): Friction coefficient (default: 1e12 for typical ionic damping in s^-1).
        memory_time (float): Timescale of memory decay τ (default: 1e-6 s for ps-ns correlations).
        constants (PhysicalConstants): Physical constants for noise scaling and temperature.
    """
    def __init__(self, gamma: float = 1e12, memory_time: float = 1e-6,
                 constants: PhysicalConstants = PhysicalConstants()):
        self.gamma = gamma
        self.tau_memory = memory_time
        self.constants = constants

    def memory_kernel(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the memory kernel K(t) = (γ / τ) exp(-t / τ) for exponential decay.
        This form arises from Ornstein-Uhlenbeck approximations of projected dynamics.

        Args:
            t (torch.Tensor): Time differences [batch_size] or scalar (non-negative).

        Returns:
            torch.Tensor: Kernel values, same shape as t.

        Note: From Claude's memory_kernel; clamps t >=0 and adds small epsilon to τ to avoid division by zero.
              For t=0, K(0) = γ / τ, representing instantaneous friction.
        """
        t = torch.clamp(t, min=0.0)  # Ensure non-negative for decay
        K0 = self.gamma / (self.tau_memory + 1e-10)  # Avoid division by zero
        return K0 * torch.exp(-t / self.tau_memory)

    def generate_colored_noise(self, num_steps: int, batch_size: int, dim: int,
                               dt: float, device: torch.device) -> torch.Tensor:
        """
        Generates colored noise ξ(t) satisfying <ξ(t) ξ(s)> = K(|t-s|) kT (fluctuation-dissipation).
        Uses an autoregressive process to approximate correlated Gaussian noise.

        Args:
            num_steps (int): Number of timesteps.
            batch_size (int): Number of trajectories.
            dim (int): Dimensionality per state.
            dt (float): Timestep.
            device (torch.device): Target device.

        Returns:
            torch.Tensor: Noise tensor [num_steps, batch_size, dim].

        Note: Extension to Claude's xi_t; uses AR(1) model with correlation exp(-dt/τ) and variance scaling.
              Ensures zero mean and correct variance for thermodynamic consistency.
        """
        alpha = np.exp(-dt / self.tau_memory)  # Autoregressive coefficient
        sigma = np.sqrt(2 * self.gamma * self.constants.KB * self.constants.T * dt) * np.sqrt(1 - alpha**2)  # Innovation std

        noise = torch.zeros(num_steps, batch_size, dim, device=device)
        innovation = torch.randn(num_steps, batch_size, dim, device=device) * sigma

        noise[0] = innovation[0]
        for step in range(1, num_steps):
            noise[step] = alpha * noise[step-1] + innovation[step]

        return noise

    def solve_with_memory(self, T: float, x0: torch.Tensor, v0: torch.Tensor,
                          force_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                          dt: float = 1e-7, max_history: int = 1000, mass: float = 1e-26) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solves the GLE: m dv/dt = F(t, x) - ∫_0^t K(t-s) v(s) ds + ξ(t), with x update via velocity Verlet-like.
        Uses discrete history for memory integral approximation and generates colored noise on-the-fly.

        Args:
            T (float): Total simulation time.
            x0 (torch.Tensor): Initial positions [batch_size, dim].
            v0 (torch.Tensor): Initial velocities [batch_size, dim].
            force_func (Callable): Force function (t, x) -> force [batch_size, dim].
            dt (float): Timestep (default: 1e-7 s for fine resolution).
            max_history (int): Maximum steps to keep in history for integral (default: 1000 to limit memory use).
            mass (float): Particle mass (default: 1e-26 kg for typical ions).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Position trajectory [steps, batch_size, dim],
                                               Velocity trajectory [steps, batch_size, dim].

        Note: From Claude's solve_with_memory; enhances with batched operations, history truncation (FIFO for long sims),
              and velocity Verlet integration for stability. Warns if dt > τ/10 for accuracy.
        """
        if dt > self.tau_memory / 10:
            print(f"Warning: dt ({dt}) large relative to memory_time ({self.tau_memory}); may lose accuracy.")

        steps = int(T / dt) + 1  # Include initial state
        batch_size, dim = x0.shape
        device = x0.device

        # Storage for trajectories
        positions = torch.zeros(steps, batch_size, dim, device=device)
        velocities = torch.zeros(steps, batch_size, dim, device=device)
        positions[0] = x0.clone()
        velocities[0] = v0.clone()

        # Precompute times and colored noise
        times = torch.arange(0, T + dt, dt, device=device)[:steps]  # Truncate to steps
        xi = self.generate_colored_noise(steps, batch_size, dim, dt, device)  # Colored noise

        # History buffers for v and force (FIFO if exceeds max_history)
        v_history = [velocities[0].clone()]
        force_history = [force_func(times[0], positions[0])]

        for i in range(1, steps):
            t = times[i]

            # Compute current force
            F_t = force_func(t, positions[i-1])
            force_history.append(F_t)
            if len(force_history) > max_history:
                force_history.pop(0)  # Truncate oldest

            # Memory integral approximation: ∑ K(t - t_past) * v_past * dt (trapezoidal)
            memory_integral = torch.zeros(batch_size, dim, device=device)
            hist_len = len(v_history)
            for j in range(hist_len):
                t_past = times[max(0, i - j - 1)]  # Adjust index for history
                kernel_val = self.memory_kernel(t - t_past)
                memory_integral += kernel_val.unsqueeze(0).unsqueeze(1) * v_history[hist_len - j - 1] * dt

            # GLE update for velocity: dv = (F - memory + xi) / m * dt
            dv = (F_t - memory_integral + xi[i]) / mass * dt
            velocities[i] = velocities[i-1] + dv

            # Position update (Verlet-like): dx = v * dt + 0.5 dv * dt (for stability)
            positions[i] = positions[i-1] + velocities[i-1] * dt + 0.5 * dv * dt

            # Update history
            v_history.append(velocities[i].clone())
            if len(v_history) > max_history:
                v_history.pop(0)

        return positions, velocities

    def estimate_memory_time(self, autocorrelation: torch.Tensor, dt: float) -> float:
        """
        Estimates memory timescale τ from velocity autocorrelation function (VACF).
        Fits exponential decay to autocorrelation for use in kernel parameterization.

        Args:
            autocorrelation (torch.Tensor): VACF values [num_lags].
            dt (float): Timestep between lags.

        Returns:
            float: Estimated τ (negative log slope of log(ACF)).

        Note: Extension not in Claude's code; useful for data-driven kernel fitting from MD trajectories.
              Assumes ACF decays exponentially; uses linear regression on log scale.
        """
        lags = torch.arange(autocorrelation.shape[0], dtype=torch.float, device=autocorrelation.device) * dt
        log_acf = torch.log(autocorrelation + 1e-8)  # Avoid log(0)

        # Linear fit: log(ACF) = -lags / τ + const
        A = torch.stack([lags, torch.ones_like(lags)], dim=1)
        slope, _ = torch.linalg.lstsq(A, log_acf).solution[:2]
        tau_est = -1 / slope.item()

        return tau_est
