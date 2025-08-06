import torch
import numpy as np
from typing import Optional, Tuple
try:
    import MDAnalysis as mda
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False
    print("Warning: MDAnalysis not installed. Falling back to synthetic data generation.")

from electrodiffusion.config.physics_constants import PhysicalConstants  # Import for physics-based augmentations

class DataLoader:
    """
    Data loading and preprocessing class for MD trajectories and synthetic data.
    This class handles loading from files using MDAnalysis, generating synthetic trajectories if needed,
    and applying preprocessing/augmentation steps to prepare data for models like CoarseGrainedModel or diffusion training.

    Args:
        constants (PhysicalConstants): Instance for scaling augmentations (default: new instance).
    """
    def __init__(self, constants: Optional[PhysicalConstants] = None):
        self.constants = constants or PhysicalConstants()
        if not MDA_AVAILABLE:
            print("MDAnalysis unavailable; all loads will use synthetic data.")

    def load_trajectory(self, topology_file: str, trajectory_file: Optional[str] = None,
                        num_residues: int = 50) -> torch.Tensor:
        """
        Loads MD trajectory data from files, extracting coarse-grained features.
        If MDAnalysis is available and files are valid, parses protein CA atoms for backbone and generates mock side chain/ion data.
        Falls back to synthetic generation on failure or if MDA unavailable.

        Args:
            topology_file (str): Path to topology file (e.g., PDB or GRO).
            trajectory_file (Optional[str]): Path to trajectory file (e.g., XTC); if None, uses static topology.
            num_residues (int): Expected number of residues for synthetic fallback (default: 50).

        Returns:
            torch.Tensor: Trajectory tensor [num_frames, feature_dim] (flattened coordinates per frame).

        Note: Feature_dim = backbone (9 per residue: CA3, CB3, normal3) + sidechain (7 per: pos3, charge1, dipole3) + ion (6: pos3, vel3).
              From attachment's load_trajectory; adds residue count check and mock ion velocities for dynamics.
        """
        if MDA_AVAILABLE:
            try:
                if trajectory_file:
                    u = mda.Universe(topology_file, trajectory_file)
                else:
                    u = mda.Universe(topology_file)
                
                # Select protein atoms
                protein = u.select_atoms("protein")
                ca_atoms = u.select_atoms("name CA")
                
                if len(ca_atoms) != num_residues:
                    print(f"Warning: Expected {num_residues} residues, found {len(ca_atoms)}; proceeding with actual.")
                
                trajectories = []
                for ts in u.trajectory:
                    # Backbone: CA positions, mock CB (shifted), mock normals (unit vectors)
                    ca_pos = ca_atoms.positions.flatten()
                    cb_pos = ca_pos + np.random.randn(len(ca_pos)) * 0.1  # Mock CB with small offset
                    normals = np.random.randn(len(ca_pos))  # Mock normals; normalize
                    normals /= np.linalg.norm(normals) + 1e-8
                    backbone = np.concatenate([ca_pos, cb_pos, normals])
                    
                    # Side chains: mock positions, charges (-1 to 1), dipoles
                    side_pos = np.random.randn(len(ca_atoms) * 3) * 0.5
                    charges = np.random.uniform(-1, 1, len(ca_atoms))
                    dipoles = np.random.randn(len(ca_atoms) * 3)
                    sidechain = np.concatenate([side_pos, charges, dipoles])
                    
                    # Ion: mock position and velocity in 3D
                    ion_pos = np.random.randn(3) * channel_scale  # Assume channel scale ~10 Å
                    ion_vel = np.random.randn(3) * velocity_scale  # Mock thermal velocity
                    ion = np.concatenate([ion_pos, ion_vel])
                    
                    frame_data = np.concatenate([backbone, sidechain, ion])
                    trajectories.append(frame_data)
                
                return torch.tensor(np.array(trajectories), dtype=torch.float32)
            
            except Exception as e:
                print(f"Error loading trajectory: {e}. Falling back to synthetic data.")
                return self._generate_synthetic_data(num_frames=1000, num_residues=num_residues)
        else:
            return self._generate_synthetic_data(num_frames=1000, num_residues=num_residues)

    def _generate_synthetic_data(self, num_frames: int = 1000, num_residues: int = 50) -> torch.Tensor:
        """
        Generates synthetic trajectory data mimicking MD outputs.
        Uses autoregressive Gaussian processes for temporal correlations, simulating Brownian-like motion.

        Args:
            num_frames (int): Number of synthetic frames (default: 1000).
            num_residues (int): Number of residues for dimension calculation (default: 50).

        Returns:
            torch.Tensor: Synthetic trajectories [num_frames, feature_dim].

        Note: From attachment's _generate_synthetic_data; adds AR(1) correlation (alpha=0.9) for realism,
              with dimensions matching load_trajectory (backbone + sidechain + ion).
        """
        backbone_dim = num_residues * 9  # CA3, CB3, normal3
        sidechain_dim = num_residues * 7  # pos3, charge1, dipole3
        ion_dim = 6  # pos3, vel3
        total_dim = backbone_dim + sidechain_dim + ion_dim

        # Generate base data with normal distribution
        data = torch.randn(num_frames, total_dim)

        # Apply temporal correlation (AR(1) process)
        alpha = 0.9  # Correlation strength (close to 1 for smooth trajectories)
        for i in range(1, num_frames):
            data[i] = alpha * data[i-1] + np.sqrt(1 - alpha**2) * data[i]  # Maintain variance

        return data

    def preprocess(self, raw_data: torch.Tensor, normalize: bool = True,
                   center: bool = True) -> torch.Tensor:
        """
        Preprocesses raw trajectory data with normalization and centering.

        Args:
            raw_data (torch.Tensor): Input data [num_frames, feature_dim].
            normalize (bool): If True, apply z-score normalization (default: True).
            center (bool): If True, subtract mean (default: True; separate from normalize for flexibility).

        Returns:
            torch.Tensor: Preprocessed data [num_frames, feature_dim].

        Note: From attachment's preprocess; expands with optional centering and per-dimension std check (avoids div by zero).
        """
        processed = raw_data.clone()

        if center:
            mean = torch.mean(processed, dim=0, keepdim=True)
            processed -= mean

        if normalize:
            std = torch.std(processed, dim=0, keepdim=True)
            std = torch.clamp(std, min=1e-8)  # Avoid division by zero
            processed /= std

        return processed

    def augment_data(self, data: torch.Tensor, noise_level: float = 0.01,
                     temperature_scale: bool = True) -> torch.Tensor:
        """
        Augments data by adding Gaussian noise, optionally scaled by temperature for physical realism.

        Args:
            data (torch.Tensor): Input data [num_frames, feature_dim].
            noise_level (float): Base standard deviation of noise (default: 0.01).
            temperature_scale (bool): If True, scale noise by sqrt(kT) for positions/velocities (default: True).

        Returns:
            torch.Tensor: Augmented data [num_frames, feature_dim].

        Note: From attachment's augment_data; enhances with physics-based scaling using constants (e.g., noise ~ sqrt(2 D dt) proxy via kT).
              Assumes last ion_dim=6 are sensitive to thermal noise; scales accordingly.
        """
        if temperature_scale:
            # Scale noise by sqrt(kT / m) proxy (velocity dispersion); assume m=1
            thermal_scale = np.sqrt(self.constants.KB * self.constants.T)
            noise = torch.randn_like(data) * noise_level * thermal_scale
        else:
            noise = torch.randn_like(data) * noise_level

        return data + noise
