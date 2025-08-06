import torch
import torch.nn as nn
import torchsde
from typing import Callable, Tuple, Optional, Dict
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import physical constants
from electrodiffusion.models.coarse_grained import CoarseGrainedModel, IonState  # For state separation and ion handling

class IonChannelSDE(torchsde.SDEIto):
    """
    Base SDE class for ion channel electrodiffusion dynamics.
    This class models the stochastic evolution of ion and protein states using Itô SDE formulation.
    It integrates noise schedules for VP-SDE compatibility and custom drift/diffusion based on physical potentials.
    
    Args:
        system (object): System object containing potential functions (e.g., from CoarseGrainedModel).
        noise_schedule (Callable): Function beta(t) for time-dependent noise (from noise_schedule.py).
        external_field (float): Initial external electric field strength (V/m).
        constants (PhysicalConstants): Physical constants for unit consistency.
    """
    def __init__(self, system: object, noise_schedule: Callable[[torch.Tensor], torch.Tensor],
                 external_field: float = 0.0, constants: PhysicalConstants = PhysicalConstants()):
        super().__init__(noise_type="diagonal", sde_type="ito")
        self.system = system
        self.beta_fn = noise_schedule  # Noise schedule β(t) from Claude's NoiseSchedule
        self.E_ext = external_field  # External electric field
        self.constants = constants

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Drift coefficient f(t, y) for the SDE.
        Computes the deterministic part of the dynamics, including physics-based forces and VP-SDE adjustments.
        
        Args:
            t (torch.Tensor): Time tensor (scalar or batched).
            y (torch.Tensor): State tensor [batch_size, state_dim] (combined protein and ion states).
        
        Returns:
            torch.Tensor: Drift vector [batch_size, state_dim].
        
        Note: Uses autograd for gradient computation of potentials, as in Claude's IonChannelSDE.
              Splits state into protein and ion components for modular force calculations.
        """
        batch_size = y.shape[0]
        
        # Split state: Assume last 'ion_dim' dimensions are ion state (e.g., 4: position + charge)
        # Adjust ion_dim based on IonState; here assuming 4 for simplicity (coords:3, charge:1)
        ion_dim = 4
        protein_state = y[:, :-ion_dim]  # Protein states (backbone + side chain)
        ion_state = y[:, -ion_dim:]      # Ion state (position:3, charge:1)
        
        # Compute physics-based drift for ion (from Claude's implementation)
        pmf = self.system.potential_of_mean_force(ion_state[:, 2:3], protein_state)  # PMF A(z, R), z is ion_state[:,2]
        electrostatic = self.system.electrostatic_potential(ion_state[:, 2:3], self.E_ext)  # V(z)
        total_energy = pmf + self.constants.e * ion_state[:, 3] * electrostatic  # q_ion * V (Claude's q_ion)
        
        # Compute force as negative gradient of energy (using autograd for differentiability)
        ion_pos = ion_state[:, :3]  # Positions for gradient
        ion_pos.requires_grad_(True)  # Enable grad tracking
        force = -torch.autograd.grad(total_energy.sum(), ion_pos, create_graph=True, retain_graph=True)[0]
        
        # Position-dependent diffusion coefficient D(z) from system (Claude's position_dependent_diffusion)
        D_z = self.system.position_dependent_diffusion(ion_state[:, 2:3])
        
        # Ion drift: D / (k_B T) * force (Nernst-Einstein relation, inverted from Claude's -D/(kBT)*dE/dz)
        ion_drift = D_z * force / (self.constants.kB * self.constants.T)
        
        # Simplified protein drift: Restoring force to equilibrium (from Claude's -0.1 * protein_state)
        protein_drift = -0.1 * protein_state
        
        # VP-SDE adjustment term: -0.5 β(t) y (variance-preserving)
        beta_t = self.beta_fn(t)
        vp_drift = -0.5 * beta_t.unsqueeze(-1) * y  # Broadcast beta_t to state_dim
        
        # Combine all drifts with scaling (0.1 for physical contribution as in Claude)
        physical_drift = torch.cat([protein_drift, ion_drift], dim=-1)
        total_drift = vp_drift + 0.1 * physical_drift
        
        return total_drift

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient g(t, y) for the SDE.
        Models thermal fluctuations and stochastic noise, with diagonal noise type.
        
        Args:
            t (torch.Tensor): Time tensor.
            y (torch.Tensor): State tensor [batch_size, state_dim].
        
        Returns:
            torch.Tensor: Diffusion matrix [batch_size, state_dim] (diagonal elements).
        
        Note: Based on Claude's g: sqrt(β(t)) * ones_like(y), extended for position-dependence.
        """
        beta_t = self.beta_fn(t).unsqueeze(-1)  # Broadcast to state_dim
        # Optional: Modulate by position-dependent factor (e.g., reduced in crowded regions)
        # For now, uniform as in Claude
        return torch.sqrt(beta_t) * torch.ones_like(y)

class ProteinDynamicsSDE(IonChannelSDE):
    """
    Extended SDE for protein dynamics, separating backbone and side chain SDEs.
    This class builds on IonChannelSDE to handle hierarchical dynamics, using CoarseGrainedModel for state separation.
    
    Args:
        system (CoarseGrainedModel): Coarse-grained model for state handling.
        noise_schedule (Callable): Noise function β(t).
        external_field (float): External field.
        constants (PhysicalConstants): Physical constants.
    """
    def __init__(self, system: CoarseGrainedModel, noise_schedule: Callable[[torch.Tensor], torch.Tensor],
                 external_field: float = 0.0, constants: PhysicalConstants = PhysicalConstants()):
        super().__init__(system, noise_schedule, external_field, constants)

    def separate_dynamics(self, backbone: torch.Tensor, side_chain: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separates and computes drifts for backbone and side chain dynamics independently.
        
        Args:
            backbone (torch.Tensor): Backbone state [batch_size, backbone_dim].
            side_chain (torch.Tensor): Side chain state [batch_size, sidechain_dim].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Drifts for backbone and side chain.
        
        Note: Implements timescale separation; backbone is slower, side chain faster (adiabatic approximation).
        """
        # Compute backbone drift (slow, structural changes)
        backbone_drift = -0.05 * backbone  # Weaker restoring force for slow dynamics
        
        # Compute side chain drift (fast, responsive to fields)
        # Use system's compute_potential for side chain-ion interactions
        # Assume ion_state is externally provided; here dummy for illustration
        dummy_ion_state = torch.zeros(backbone.shape[0], 4, device=backbone.device)
        side_potential = self.system.compute_potential(side_chain, dummy_ion_state)
        side_chain.requires_grad_(True)
        side_force = -torch.autograd.grad(side_potential.sum(), side_chain, create_graph=True)[0]
        side_chain_drift = side_force / (self.constants.kB * self.constants.T)  # Mobility factor omitted for simplicity
        
        return backbone_drift, side_chain_drift

class PhysicsInformedSDE(IonChannelSDE):
    """
    Physics-informed SDE incorporating explicit force terms (conservative, electrostatic, thermal).
    This directly adapts Claude's PhysicsInformedSDE, adding drift and diffusion coefficients.
    
    Args:
        system_params (Dict): Parameters like temperature, ion_charge, diffusion_coeff.
        noise_schedule (Callable): β(t).
        external_field (float): External field.
    """
    def __init__(self, system_params: Dict, noise_schedule: Callable[[torch.Tensor], torch.Tensor],
                 external_field: float = 0.0):
        super().__init__(None, noise_schedule, external_field)  # System not used; params direct
        self.params = system_params
        self.beta = 1.0 / (self.constants.kB * self.params.get('temperature', 300.0))

    def conservative_force(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes conservative force -∇U(x) using multi-well potential (from Claude's PhysicsInformedSDE).
        
        Args:
            x (torch.Tensor): State [batch_size, dim].
        
        Returns:
            torch.Tensor: Force vector [batch_size, dim].
        """
        x.requires_grad_(True)
        # Multi-well potential (simplified ion channel barriers)
        wells = torch.tensor([-20e-10, 0.0, 20e-10], device=x.device)  # Three wells in meters
        barriers = torch.tensor([5.0, 3.0], device=x.device) * self.constants.kB * self.params['temperature']
        
        potential = torch.zeros(x.shape[0], device=x.device)
        for i in range(len(wells) - 1):
            well, next_well = wells[i], wells[i+1]
            barrier_pos = (well + next_well) / 2
            barrier_width = 5e-10  # 5 Angstrom
            barrier_contrib = barriers[i] * torch.exp(-((x[:, 0] - barrier_pos)**2) / (2 * barrier_width**2))
            potential += barrier_contrib
            
            # Harmonic well contribution
            well_contrib = 0.5 * 1e-10 * (x[:, 0] - well)**2
            mask = (torch.abs(x[:, 0] - well) < 10e-10).float()
            potential += mask * well_contrib
        
        # Negative gradient of potential
        force = -torch.autograd.grad(potential.sum(), x, create_graph=True)[0]
        return force

    def electrostatic_force(self, x: torch.Tensor, external_field: float = 0.0) -> torch.Tensor:
        """
        Computes electrostatic force q * E_ext (from Claude's PhysicsInformedSDE).
        
        Args:
            x (torch.Tensor): State [batch_size, dim].
            external_field (float): Field strength.
        
        Returns:
            torch.Tensor: Force vector [batch_size, dim].
        """
        q = self.params.get('ion_charge', 1.0) * self.constants.e
        E_field = torch.tensor([external_field, 0.0, 0.0], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        return q * E_field

    def drift_coefficient(self, t: float, x: torch.Tensor, external_field: float = 0.0) -> torch.Tensor:
        """
        Full drift coefficient combining forces and mobility (from Claude's drift_coefficient).
        
        Args:
            t (float): Time.
            x (torch.Tensor): State.
            external_field (float): Field.
        
        Returns:
            torch.Tensor: Drift [batch_size, dim].
        """
        conservative = self.conservative_force(x)
        electrostatic = self.electrostatic_force(x, external_field)
        D = self.params.get('diffusion_coeff', 1e-9)
        mobility = D / (self.constants.kB * self.params['temperature'])
        physical_drift = mobility * (conservative + electrostatic)
        
        # Add VP-SDE term
        beta_t = self.beta_fn(torch.tensor(t, device=x.device))
        vp_drift = -0.5 * beta_t.unsqueeze(-1) * x
        return vp_drift + physical_drift

    def diffusion_coefficient(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Diffusion coefficient sqrt(2 D) (from Claude's diffusion_coefficient).
        
        Args:
            t (float): Time.
            x (torch.Tensor): State.
        
        Returns:
            torch.Tensor: Diffusion [batch_size, dim].
        """
        D = self.params.get('diffusion_coeff', 1e-9)
        return torch.sqrt(2 * D) * torch.ones_like(x)
