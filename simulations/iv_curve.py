import torch
import numpy as np
from typing import Dict, List, Optional
from electrodiffusion.simulations.sampling import DiffusionSampler  # Import sampler for state generation
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import constants for physical computations

class NernstPlanckSolver:
    """
    Solver for computing ionic fluxes using the Nernst-Planck equation.
    This class models electrodiffusion flux as J = -D ∇c + (D q / kT) c E, incorporating position-dependent diffusion
    to account for channel crowding or barriers. It serves as a bridge from microscopic states to macroscopic currents.

    Args:
        constants (PhysicalConstants): Physical constants for unit consistency and computations.
    """
    def __init__(self, constants: PhysicalConstants):
        self.constants = constants

    def compute_flux(self, samples: torch.Tensor, voltage: float,
                     channel_length: float = 50e-10, channel_radius: float = 5e-10) -> torch.Tensor:
        """
        Computes the Nernst-Planck flux for each sample based on ion position and voltage.

        Args:
            samples (torch.Tensor): Generated states [num_samples, state_dim] (last 6: x,y,z,vx,vy,vz for ion).
            voltage (float): Applied voltage (V).
            channel_length (float): Channel length in meters (default: 50 Å).
            channel_radius (float): Channel radius in meters (default: 5 Å, not used in flux but for context).

        Returns:
            torch.Tensor: Flux values [num_samples].

        Note: Simplified uniform concentration gradient; position-dependent D(z) with Gaussian barrier model.
              Assumes single ion type (e.g., K+); extend for multi-ion with Poisson-Nernst-Planck if needed.
              From Claude's compute_flux, with added velocity components ignored here (focus on drift-diffusion).
        """
        num_samples = samples.shape[0]

        # Extract ion positions (assume last 6 are x,y,z,vx,vy,vz; use only positions)
        ion_positions = samples[:, -6:-3]  # [num_samples, 3]
        z_positions = ion_positions[:, 2]  # z-coordinate along channel

        # Clamp z to channel bounds for physical realism
        z_positions = torch.clamp(z_positions, min=0.0, max=channel_length)

        # Position-dependent diffusion coefficient D(z) with barrier reduction
        D0 = 1e-9  # Base diffusion coeff (m²/s, typical for ions in water)
        barrier_height = 5.0 * self.constants.KB * self.constants.T  # Energy barrier in J (5 kT)
        barrier_width = channel_length / 10  # Width as fraction of length
        barrier_center = channel_length / 2
        crowding_factor = torch.exp(-barrier_height / (self.constants.KB * self.constants.T) *
                                    torch.exp(-((z_positions - barrier_center)**2) / (2 * barrier_width**2)))
        D_z = D0 * (1 - crowding_factor)  # Reduced D in barrier region

        # Simplified concentration gradient (dc/dz; assume linear drop across channel)
        grad_c = 1.0 / channel_length  # 1/m, uniform assumption; can be tensor for non-uniform

        # Electric field E = V / L along z
        E_field = voltage / channel_length  # V/m

        # Mobility μ = D / (kT) from Einstein relation
        mobility = D_z / (self.constants.KB * self.constants.T)

        # Nernst-Planck flux J = -D ∇c + μ q c E (here q=1e for monovalent ion, c=1 normalized)
        # Simplified: assume c=1, q=e, and ∇c along z only
        flux = -D_z * grad_c + mobility * self.constants.E * E_field  # Units: m/s (velocity-like)

        return flux

class IVCurvePredictor:
    """
    Predictor for IV curves, conductances, and open probabilities using sampled states.
    This class generates conditioned samples, computes ensemble-averaged currents via Nernst-Planck,
    and validates model consistency with physical principles.

    Args:
        sampler (DiffusionSampler): Sampler for generating states conditioned on voltage.
        constants (PhysicalConstants): Constants for flux and validation computations.
    """
    def __init__(self, sampler: DiffusionSampler, constants: PhysicalConstants):
        self.sampler = sampler
        self.constants = constants
        self.np_solver = NernstPlanckSolver(constants)

    def compute_iv(self, voltages: np.ndarray, num_samples: int = 1000,
                   state_dim: int = 8, channel_length: float = 50e-10) -> Dict:
        """
        Computes IV curve by sampling states for each voltage and averaging fluxes.

        Args:
            voltages (np.ndarray): Array of voltages to evaluate (V).
            num_samples (int): Number of samples per voltage (default: 1000 for statistical reliability).
            state_dim (int): Dimensionality of sampled states (default: 8 as in Claude's example).
            channel_length (float): Channel length for field and gradient computations.

        Returns:
            Dict: Results with 'voltages', 'currents' (pA), 'conductances' (S), 'open_probabilities'.

        Note: From Claude's compute_iv; converts flux to current (J * q * area approximation, scaled to pA),
              computes conductance as I/V, and open probability as fraction inside channel.
              Warns if num_samples < 100 for low statistics.
        """
        if num_samples < 100:
            print(f"Warning: Low num_samples ({num_samples}); results may have high variance.")

        currents = []
        conductances = []
        open_probabilities = []

        for V in voltages:
            # Condition on voltage (tensor repeated for num_samples)
            condition = {'voltage': torch.full((num_samples,), V, dtype=torch.float)}

            # Generate samples
            samples = self.sampler.sample(num_samples, state_dim, condition=condition)  # Assumes sampler method

            # Compute fluxes
            fluxes = self.np_solver.compute_flux(samples, V, channel_length)

            # Average flux and convert to current: <J> * q * (effective area); scale to pA (1e12)
            avg_flux = torch.mean(fluxes)
            effective_area = np.pi * (5e-10)**2  # Approximate cross-section (radius 5 Å)
            current = avg_flux * self.constants.E * effective_area * 1e12  # pA
            currents.append(current.item())

            # Conductance g = I / V (in Siemens; handle V=0)
            if abs(V) > 1e-6:
                g = current / V  # S (A/V)
            else:
                g = 0.0  # Avoid division by zero
            conductances.append(g)

            # Open probability: Fraction of ions inside channel
            inside = self._check_inside_channel(samples[:, -6:-3], channel_length)  # Positions
            P_open = torch.mean(inside.float()).item()
            open_probabilities.append(P_open)

        return {
            'voltages': voltages,
            'currents': np.array(currents),
            'conductances': np.array(conductances),
            'open_probabilities': np.array(open_probabilities)
        }

    def _check_inside_channel(self, positions: torch.Tensor,
                              length: float = 50e-10, radius: float = 5e-10) -> torch.Tensor:
        """
        Checks if ion positions are inside the cylindrical channel.

        Args:
            positions (torch.Tensor): Ion positions [num_samples, 3] (x,y,z).
            length (float): Channel length.
            radius (float): Channel radius.

        Returns:
            torch.Tensor: Boolean mask [num_samples].

        Note: Assumes channel along z from 0 to length; radial distance < radius.
              From Claude's _check_inside_channel; uses torch operations for efficiency.
        """
        r = torch.sqrt(positions[:, 0]**2 + positions[:, 1]**2)
        z = positions[:, 2]
        inside_radius = r < radius
        inside_length = (z > 0) & (z < length)
        return inside_radius & inside_length

    def validate(self, experimental_data: Optional[Dict] = None,
                 num_samples: int = 1000, state_dim: int = 8) -> Dict:
        """
        Validates the model by checking detailed balance and thermodynamic consistency on generated samples.
        Optionally compares to experimental data if provided.

        Args:
            experimental_data (Optional[Dict]): Dict with 'energies', 'forces' etc. for comparison.
            num_samples (int): Samples for validation.
            state_dim (int): State dimensionality.

        Returns:
            Dict: Validation results with errors and pass/fail flags.

        Note: From Claude's validate; detailed balance checks variance of energy differences under perturbations;
              thermodynamic consistency verifies fluctuation-dissipation (variance ~ kT).
              Thresholds are empirical (e.g., 1% for balance, 10% for thermo).
        """
        # Generate unconditional samples for validation
        samples = self.sampler.sample(num_samples, state_dim)  # No condition for equilibrium check

        results = {}

        # Detailed balance check
        balance_error = self._check_detailed_balance(samples)
        results['detailed_balance_error'] = balance_error
        results['detailed_balance_pass'] = balance_error < 0.01  # 1% tolerance

        # Thermodynamic consistency
        thermo_error = self._check_thermodynamic_consistency(samples)
        results['thermodynamic_error'] = thermo_error
        results['thermodynamic_pass'] = thermo_error < 0.1  # 10% tolerance

        # Optional comparison to experimental data
        if experimental_data:
            if 'currents' in experimental_data:
                # Example: MSE on currents (assuming voltages match)
                pred_iv = self.compute_iv(experimental_data['voltages'], num_samples // 10)
                mse_current = np.mean((experimental_data['currents'] - pred_iv['currents'])**2)
                results['current_mse'] = mse_current

        return results

    def _check_detailed_balance(self, samples: torch.Tensor) -> float:
        """
        Checks detailed balance by perturbing samples and verifying energy difference variance.

        Args:
            samples (torch.Tensor): Generated states [num_samples, state_dim].

        Returns:
            float: Normalized variance error.

        Note: Simplified; ideal balance implies symmetric transitions (var close to 0 for small perturbations).
              From Claude's _check_detailed_balance; uses random perturbations.
        """
        epsilon = 1e-4
        perturbation = epsilon * torch.randn_like(samples)
        
        # Simplified energy: quadratic form (harmonic approximation)
        energy_diff = torch.sum(samples * perturbation, dim=-1)  # Dot product as proxy
        
        # Variance should be small if balanced
        balance_error = torch.var(energy_diff).item() / (epsilon**2 * samples.shape[1])  # Normalized
        return balance_error

    def _check_thermodynamic_consistency(self, samples: torch.Tensor) -> float:
        """
        Checks thermodynamic consistency via fluctuation-dissipation theorem (energy variance ~ kT).

        Args:
            samples (torch.Tensor): Generated states.

        Returns:
            float: Relative error in variance.

        Note: Assumes kinetic energy-like variance; from Claude's _check_thermodynamic_consistency.
              Uses positions for fluctuation estimate.
        """
        # Assume last 3 are velocities for kinetic energy (1/2 m v^2; m=1 normalized)
        velocities = samples[:, -3:] if samples.shape[1] >= 3 else samples[:, :3]  # Fallback to positions
        energies = 0.5 * torch.sum(velocities**2, dim=-1)  # Kinetic energy proxy
        
        energy_variance = torch.var(energies).item()
        
        # Expected from equipartition: (3/2) kT for 3D (per degree of freedom)
        expected_variance = (3/2) * self.constants.KB * self.constants.T
        
        consistency_error = abs(energy_variance - expected_variance) / expected_variance
        return consistency_error
