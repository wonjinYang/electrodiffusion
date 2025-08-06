import torch
import numpy as np
from typing import Tuple, Optional, Callable
from electrodiffusion.models.sde import IonChannelSDE  # Import base SDE class for fast/slow instances
from electrodiffusion.config.physics_constants import PhysicalConstants  # For solvation calculations

class MultiScaleSDE:
    """
    Multi-scale SDE class for handling timescale separation in ion channel dynamics.
    This class separates the system into fast and slow variables, using Mori-Zwanzig projection
    to derive effective equations for slow dynamics while averaging over fast fluctuations.
    It supports memory kernels for non-Markovian effects and hierarchical integration for efficiency.

    Args:
        fast_sde (IonChannelSDE): SDE for fast variables (e.g., side chains, ions).
        slow_sde (IonChannelSDE): SDE for slow variables (e.g., backbone).
        fast_dt (float): Timestep for fast integration (default: 1e-6 for ns scale).
        slow_dt (float): Timestep for slow integration (default: 1e-4 for μs scale).
    """
    def __init__(self, fast_sde: IonChannelSDE, slow_sde: IonChannelSDE,
                 fast_dt: float = 1e-6, slow_dt: float = 1e-4):
        if fast_dt >= slow_dt:
            raise ValueError("fast_dt must be smaller than slow_dt for timescale separation")
        self.fast_sde = fast_sde
        self.slow_sde = slow_sde
        self.fast_dt = fast_dt
        self.slow_dt = slow_dt

    def project_slow(self, fast_vars: torch.Tensor, slow_vars: torch.Tensor,
                     memory_kernel: Optional[Callable] = None) -> torch.Tensor:
        """
        Projects fast variables onto slow dynamics using Mori-Zwanzig formalism.
        Computes the conditional average of slow drift and adds memory kernel contributions.

        Args:
            fast_vars (torch.Tensor): Fast variable trajectories [num_steps, batch_size, fast_dim].
            slow_vars (torch.Tensor): Current slow variables [batch_size, slow_dim].
            memory_kernel (Optional[Callable]): Function K(t) for memory effects (default: None, no memory).

        Returns:
            torch.Tensor: Projected slow drift [batch_size, slow_dim].

        Note: Averages over recent fast steps; memory integral approximated via trapezoidal rule if kernel provided.
        """
        # Conditional average of fast variables (mean over recent trajectory)
        fast_avg = torch.mean(fast_vars[-int(self.slow_dt / self.fast_dt):], dim=0)  # Average over one slow step

        # Compute base slow drift with fast coupling
        t_dummy = torch.zeros(1, device=slow_vars.device)  # Time not used in base drift here
        slow_drift = self.slow_sde.f(t_dummy, torch.cat([slow_vars, fast_avg], dim=-1))[:, :slow_vars.shape[-1]]

        # Add memory kernel contribution if provided
        if memory_kernel is not None:
            # Approximate integral ∫ K(t-s) δf(s) ds over discrete steps
            num_steps = fast_vars.shape[0]
            memory_contrib = torch.zeros_like(slow_drift)
            for s in range(1, num_steps):
                delta_t = s * self.fast_dt
                kernel_val = memory_kernel(delta_t)
                # δf(s) ≈ fast_sde.f(t_dummy, fast_vars[-s]) - average drift
                delta_f = self.fast_sde.f(t_dummy, fast_vars[-s]) - self.fast_sde.f(t_dummy, fast_avg)
                memory_contrib += kernel_val * delta_f * self.fast_dt  # Trapezoidal approximation
            slow_drift += 0.1 * memory_contrib  # Scaling factor for contribution strength

        return slow_drift

    def integrate_multiscale(self, x0: torch.Tensor, T: float,
                             memory_kernel: Optional[Callable] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multi-scale integration over time T, alternating fast and slow updates.
        Fast variables are integrated adiabatically, then projected to update slow variables.

        Args:
            x0 (torch.Tensor): Initial state [batch_size, total_dim] (fast + slow concatenated).
            T (float): Total simulation time.
            memory_kernel (Optional[Callable]): Memory function for projection.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Fast trajectory [num_slow_steps + 1, batch_size, fast_dim],
                                               Slow trajectory [num_slow_steps + 1, batch_size, slow_dim].

        Note: Adapted from Claude's MultiScaleSolver.solve; uses Euler-Maruyama for steps, with Wiener noise.
              Assumes total_dim = fast_dim + slow_dim; splits accordingly.
        """
        batch_size = x0.shape[0]
        dim_fast = x0.shape[-1] // 2  # Assume even split; adjust if needed
        fast_vars = x0[:, :dim_fast].clone()
        slow_vars = x0[:, dim_fast:].clone()

        # Compute number of steps
        slow_steps = int(T / self.slow_dt)
        fast_steps_per_slow = int(self.slow_dt / self.fast_dt)

        # Storage for trajectories
        fast_trajectory = [fast_vars.clone()]
        slow_trajectory = [slow_vars.clone()]

        for i in range(slow_steps):
            t_slow = i * self.slow_dt

            # Integrate fast variables over one slow timestep
            fast_temp = fast_vars.clone()
            fast_local_traj = [fast_temp.clone()]  # Local traj for averaging
            for j in range(fast_steps_per_slow):
                t_fast = t_slow + j * self.fast_dt
                t_tensor = torch.tensor([t_fast], device=fast_temp.device).expand(batch_size, 1)

                # Compute fast drift and diffusion
                fast_drift = self.fast_sde.f(t_tensor, fast_temp)
                fast_diffusion = self.fast_sde.g(t_tensor, fast_temp)

                # Wiener increment
                dW = torch.randn_like(fast_temp) * np.sqrt(self.fast_dt)

                # Euler-Maruyama step
                fast_temp = fast_temp + fast_drift * self.fast_dt + fast_diffusion * dW

                fast_local_traj.append(fast_temp.clone())

            fast_vars = fast_temp
            fast_trajectory.append(fast_vars.clone())

            # Project and update slow variables
            projected_drift = self.project_slow(torch.stack(fast_local_traj), slow_vars, memory_kernel)

            t_slow_tensor = torch.tensor([t_slow], device=slow_vars.device).expand(batch_size, 1)
            slow_diffusion = self.slow_sde.g(t_slow_tensor, slow_vars)

            dW_slow = torch.randn_like(slow_vars) * np.sqrt(self.slow_dt)

            # Update slow vars with projected drift
            slow_vars = slow_vars + projected_drift * self.slow_dt + slow_diffusion * dW_slow
            slow_trajectory.append(slow_vars.clone())

        return torch.stack(fast_trajectory), torch.stack(slow_trajectory)

class ImplicitSolvent:
    """
    Implicit solvent model for incorporating solvation effects without explicit water molecules.
    This class computes solvation free energy using approximations like generalized Born and modifies
    SDE drifts accordingly, enhancing electrostatic accuracy in ion-protein interactions.

    Args:
        constants (PhysicalConstants): Physical constants for dielectric and charge scaling.
    """
    def __init__(self, constants: PhysicalConstants):
        self.constants = constants

    def compute_solvation_energy(self, state: torch.Tensor) -> torch.Tensor:
        """
        Computes solvation free energy using a simplified generalized Born model.
        Approximates Poisson-Boltzmann effects for efficiency in SDE integrations.

        Args:
            state (torch.Tensor): Full state [batch_size, state_dim] (protein + ion).

        Returns:
            torch.Tensor: Solvation energy [batch_size].

        Note: Adapted from Claude's compute_solvation_energy; uses Born self-energy with dielectric factor.
              Assumes last 6 dims are ion position (3) + charge (1) + extra (2); adjust as needed.
        """
        batch_size = state.shape[0]
        # Extract protein state and ion details (assuming concatenated state)
        protein_state = state[:, :-6]  # Protein coordinates/charges
        ion_position = state[:, -6:-3]  # Ion position [batch_size, 3]
        ion_charge = state[:, -3] if state.shape[1] > 6 else torch.ones(batch_size, device=state.device)  # Ion charge

        # Simplified Born radius (average over protein; configurable)
        a_born = 2e-10  # Typical Born radius in meters

        # Dielectric factor tau = 1 - 1/eps_r (eps_r ~80 for water)
        tau = 1 - 1 / self.constants.EPS_R  # EPS_R from constants (relative permittivity)

        # Self-energy term (simplified, ignoring pairwise for speed)
        self_energy = -tau * (ion_charge * self.constants.E)**2 / (8 * np.pi * self.constants.EPS0 * a_born)

        # Optional: Add reaction field from protein (mean-field approximation)
        # For full PB, use external solver; here approximate as constant shift
        reaction_field = torch.zeros(batch_size, device=state.device)  # Placeholder for extensions

        return self_energy + reaction_field

    def modify_drift(self, original_drift: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Modifies the original SDE drift by adding solvation force (negative gradient of solvation energy).

        Args:
            original_drift (torch.Tensor): Base drift [batch_size, state_dim].
            state (torch.Tensor): Current state [batch_size, state_dim].

        Returns:
            torch.Tensor: Modified drift [batch_size, state_dim].

        Note: Uses autograd for gradient computation; scales by factor (e.g., 0.1) to balance contributions.
              From Claude's modify_drift; ensures create_graph for higher-order derivatives if needed.
        """
        state.requires_grad_(True)  # Enable gradient tracking
        solvation_energy = self.compute_solvation_energy(state)

        # Compute solvation force as -grad(energy)
        solvation_force = -torch.autograd.grad(
            solvation_energy.sum(), state,
            create_graph=True, retain_graph=True
        )[0]

        # Add to original drift with scaling
        return original_drift + 0.1 * solvation_force
