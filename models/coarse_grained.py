import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import physical constants from config

class CoarseGrainedModel(nn.Module):
    """
    Coarse-grained model for protein representation in ion channel dynamics.
    This class handles the separation of backbone and side chain states, computes dihedral angles,
    and calculates effective potentials for side chain-ion interactions.
    
    Args:
        num_residues (int): Number of residues in the protein chain.
        constants (PhysicalConstants): Instance of physical constants for consistent unit handling.
    """
    def __init__(self, num_residues: int, constants: PhysicalConstants):
        super().__init__()
        self.num_residues = num_residues
        self.constants = constants
        
        # Dimensionality for backbone: 9 per residue (CA:3, CB:3, normal:3)
        self.backbone_dim = num_residues * 9
        
        # Dimensionality for side chain: 7 per residue (position:3, charge:1, dipole:3)
        # Polarizability could be added as a 3x3 tensor if needed, but simplified here.
        self.sidechain_dim = num_residues * 7

    def separate_states(self, full_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Separates the full protein state tensor into backbone and side chain components.
        
        Args:
            full_state (torch.Tensor): Full state tensor of shape [batch_size, backbone_dim + sidechain_dim].
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Backbone and side chain tensors.
        
        Raises:
            ValueError: If input dimensions do not match expected sizes.
        """
        batch_size = full_state.shape[0]
        expected_dim = self.backbone_dim + self.sidechain_dim
        if full_state.shape[1] != expected_dim:
            raise ValueError(f"Expected full_state dim {expected_dim}, got {full_state.shape[1]}")
        
        backbone = full_state[:, :self.backbone_dim]  # First part: backbone states
        sidechain = full_state[:, self.backbone_dim:]  # Remaining: side chain states
        return backbone, sidechain

    def compute_dihedral_angles(self, backbone: torch.Tensor) -> torch.Tensor:
        """
        Computes dihedral angles (phi, psi) from backbone coordinates.
        This is crucial for capturing torsional dynamics in coarse-grained models.
        
        Args:
            backbone (torch.Tensor): Backbone tensor of shape [batch_size, backbone_dim].
        
        Returns:
            torch.Tensor: Dihedral angles tensor of shape [batch_size, num_residues, 2] (phi, psi).
        
        Note: Simplified calculation using vector differences; for accuracy, consider more robust methods like those in BioPython.
        """
        batch_size = backbone.shape[0]
        # Reshape to [batch_size, num_residues, 9] for easier access
        backbone_reshaped = backbone.view(batch_size, self.num_residues, 9)
        
        # Extract components: CA positions [..., :3], CB [..., 3:6], normals [..., 6:9]
        ca_positions = backbone_reshaped[:, :, :3]
        cb_positions = backbone_reshaped[:, :, 3:6]  # Not directly used here, but available for extensions
        normals = backbone_reshaped[:, :, 6:9]      # Peptide plane normals
        
        # Initialize phi and psi angles
        phi_angles = torch.zeros(batch_size, self.num_residues, device=backbone.device)
        psi_angles = torch.zeros(batch_size, self.num_residues, device=backbone.device)
        
        # Compute dihedrals for internal residues (skip terminals for simplicity)
        for i in range(1, self.num_residues - 1):
            # Vectors for phi: between CA(i-1) to CA(i) and CA(i) to CA(i+1)
            v1 = ca_positions[:, i] - ca_positions[:, i-1]
            v2 = ca_positions[:, i+1] - ca_positions[:, i]
            
            # Cross product and atan2 for angle
            cross = torch.cross(v1, v2, dim=1)
            phi_angles[:, i] = torch.atan2(
                torch.norm(cross, dim=1),
                torch.sum(v1 * v2, dim=1)
            )
            
            # Similar for psi, using normals or adjacent vectors (simplified)
            # In full implementation, use four atoms: N(i), CA(i), C(i), N(i+1)
            # Here, approximate with CA vectors and normals
            v3 = normals[:, i]  # Using normal as proxy for psi
            psi_angles[:, i] = torch.atan2(
                torch.norm(torch.cross(v2, v3, dim=1), dim=1),
                torch.sum(v2 * v3, dim=1)
            )
        
        # Stack phi and psi along the last dimension
        return torch.stack([phi_angles, psi_angles], dim=-1)

    def compute_potential(self, sidechain: torch.Tensor, ion_state: torch.Tensor,
                           external_fields: Optional[Dict] = None) -> torch.Tensor:
        """
        Computes the effective potential V_sc for side chain-ion interactions.
        Includes Coulomb (screened), dipole, and external field terms.
        
        Args:
            sidechain (torch.Tensor): Side chain tensor [batch_size, sidechain_dim].
            ion_state (torch.Tensor): Ion state tensor [batch_size, ion_dim] (e.g., position:3, charge:1, etc.).
            external_fields (Optional[Dict]): Dictionary of external fields (e.g., {'electric_field': tensor}).
        
        Returns:
            torch.Tensor: Total potential energy [batch_size].
        """
        batch_size = sidechain.shape[0]
        # Reshape sidechain to [batch_size, num_residues, 7]
        sidechain_reshaped = sidechain.view(batch_size, self.num_residues, 7)
        
        # Extract components
        positions = sidechain_reshaped[:, :, :3]   # Mass centers [batch_size, num_residues, 3]
        charges = sidechain_reshaped[:, :, 3]      # Net charges [batch_size, num_residues]
        dipoles = sidechain_reshaped[:, :, 4:7]    # Dipole moments [batch_size, num_residues, 3]
        
        # Assume ion_state last 3 dims are position, and charge is at index 3 (adjust based on IonState)
        ion_pos = ion_state[:, :3]  # Assuming first 3 are positions; adjust if conical
        ion_charge = ion_state[:, 3] if ion_state.shape[1] > 3 else torch.ones(batch_size, device=sidechain.device)
        
        # Initialize total potential
        total_potential = torch.zeros(batch_size, device=sidechain.device)
        
        # Coulomb interactions (Debye-Hückel screened)
        for i in range(self.num_residues):
            # Distances between side chain i and ion
            distances = torch.norm(positions[:, i] - ion_pos, dim=1) + 1e-10  # Avoid zero division
            
            # Debye length (example: for 150mM ionic strength)
            debye_length = self.constants.debye_length(ionic_strength=0.15)  # Method assumed in PhysicalConstants
            
            # Screening factor
            screening = torch.exp(-distances / debye_length)
            
            # Coulomb energy: q_i * q_ion / (4 pi eps0 eps_r r) * screening
            coulomb_energy = (self.constants.E * charges[:, i] * ion_charge * screening) / \
                             (4 * np.pi * self.constants.EPS0 * self.constants.EPS_R * distances)
            total_potential += coulomb_energy
        
        # Dipole-field interactions if external fields provided
        if external_fields and 'electric_field' in external_fields:
            E_ext = external_fields['electric_field']  # [batch_size, 3] or broadcastable
            if E_ext.shape != (batch_size, 3):
                E_ext = E_ext.expand(batch_size, 3)  # Broadcast if needed
            
            for i in range(self.num_residues):
                # Dipole energy: - mu_i • E_ext
                dipole_energy = -torch.sum(dipoles[:, i] * E_ext, dim=1)
                total_potential += dipole_energy
        
        # Additional terms (e.g., polarizability) can be added here for extensions
        
        return total_potential

class IonState:
    """
    Represents the state of an ion in conical coordinates, suitable for cylindrical channel geometry.
    Handles conversions between Cartesian and conical coordinates, and boundary checks.
    
    Args:
        channel_length (float): Length of the ion channel (default: 50 Angstroms = 50e-10 m).
        channel_radius (float): Radius of the channel (default: 5 Angstroms = 5e-10 m).
    """
    def __init__(self, channel_length: float = 50e-10, channel_radius: float = 5e-10):
        self.L = channel_length
        self.R = channel_radius

    def cartesian_to_conical(self, cartesian: torch.Tensor) -> torch.Tensor:
        """
        Converts Cartesian coordinates (x, y, z) to conical (r, theta, z).
        
        Args:
            cartesian (torch.Tensor): [batch_size, 3] tensor.
        
        Returns:
            torch.Tensor: [batch_size, 3] conical coordinates.
        """
        x, y, z = cartesian[:, 0], cartesian[:, 1], cartesian[:, 2]
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return torch.stack([r, theta, z], dim=1)

    def conical_to_cartesian(self, conical: torch.Tensor) -> torch.Tensor:
        """
        Converts conical (r, theta, z) to Cartesian (x, y, z).
        
        Args:
            conical (torch.Tensor): [batch_size, 3] tensor.
        
        Returns:
            torch.Tensor: [batch_size, 3] Cartesian coordinates.
        """
        r, theta, z = conical[:, 0], conical[:, 1], conical[:, 2]
        x = r * torch.cos(theta)
        y = r * torch.sin(theta)
        return torch.stack([x, y, z], dim=1)

    def is_inside_channel(self, position: torch.Tensor) -> torch.Tensor:
        """
        Checks if the ion position is inside the channel boundaries.
        
        Args:
            position (torch.Tensor): [batch_size, 3] tensor (either Cartesian or conical).
        
        Returns:
            torch.Tensor: Boolean tensor [batch_size] indicating if inside.
        """
        if position.shape[-1] == 3:
            # Assume Cartesian if 3 dims
            r = torch.sqrt(position[:, 0]**2 + position[:, 1]**2)
            z = position[:, 2]
        else:
            # Assume conical
            r = position[:, 0]
            z = position[:, 2]
        
        inside_radius = r < self.R
        inside_length = (z > 0) & (z < self.L)
        return inside_radius & inside_length
