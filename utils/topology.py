import torch
import numpy as np
from typing import List, Optional, Tuple
try:
    import gudhi as gd
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: Gudhi library not installed. Topology features are disabled. Install with 'pip install gudhi'.")

from electrodiffusion.config.physics_constants import PhysicalConstants  # Import for scaling thresholds physically

class TopologyConstraint:
    """
    Class for applying topological constraints using persistent homology.
    This class computes persistence barcodes from point cloud data (e.g., atomic coordinates) and calculates
    losses based on barcode distances to enforce topological consistency in generative models or simulations.
    It supports batch processing for efficiency in ensemble methods and falls back gracefully if Gudhi is unavailable.

    Args:
        max_dimension (int): Maximum homology dimension to compute (default: 2 for loops and voids).
        constants (PhysicalConstants): Constants for physical scaling (e.g., Debye length in thresholds).
    """
    def __init__(self, max_dimension: int = 2, constants: PhysicalConstants = PhysicalConstants()):
        self.max_dimension = max_dimension
        self.constants = constants
        if not GUDHI_AVAILABLE:
            print("TopologyConstraint initialized without Gudhi; methods will return defaults (e.g., zero loss).")

    def compute_barcode(self, state: torch.Tensor, threshold: Optional[float] = None,
                        dimension: int = 1) -> Optional[List[List[Tuple[float, float]]]]:
        """
        Computes the persistence barcode for the given state using Rips complex.
        The barcode represents topological features (e.g., connected components, holes) with birth-death intervals.

        Args:
            state (torch.Tensor): Input state [batch_size, num_points, 3] or [num_points, 3] (flattened coordinates).
            threshold (Optional[float]): Max edge length for Rips complex (default: Debye length scaled by factor).
            dimension (int): Homology dimension to extract (default: 1 for loops; use 0 for components, 2 for voids).

        Returns:
            Optional[List[List[Tuple[float, float]]]]: Barcodes per batch item, or None if Gudhi unavailable.

        Note: From attachment's compute_barcode; expands to batch support (list of barcodes) and auto-threshold
              based on Debye length (physical screening distance). Clamps points for numerical stability.
        """
        if not GUDHI_AVAILABLE:
            return None

        if state.dim() == 2:  # [num_points, 3] -> unsqueeze to [1, num_points, 3]
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        barcodes = []

        # Auto-threshold: Use Debye length if not provided (typical atomic scale ~1-10 Å)
        if threshold is None:
            ionic_strength = 0.15  # Default 150mM
            debye_length = np.sqrt(self.constants.EPS0 * self.constants.EPS_R * self.constants.KB * self.constants.T /
                                   (2 * ionic_strength * (self.constants.E)**2))  # Debye formula
            threshold = float(debye_length * 5)  # Scale by factor for Rips

        for b in range(batch_size):
            try:
                coords = state[b].detach().cpu().numpy()  # Gudhi requires numpy; move to CPU
                # Clamp coordinates to prevent extreme values
                coords = np.clip(coords, -1e-6, 1e-6)

                # Build Rips complex
                rips_complex = gd.RipsComplex(points=coords, max_edge_length=threshold)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)

                # Compute persistence
                simplex_tree.persistence()

                # Extract intervals for specified dimension
                intervals = simplex_tree.persistence_intervals_in_dimension(dimension)
                barcode = [(birth, death) for birth, death in intervals if death < np.inf]  # Filter infinite deaths

                barcodes.append(barcode)
            except Exception as e:
                print(f"Error computing barcode for batch {b}: {e}. Returning empty barcode.")
                barcodes.append([])

        return barcodes

    def loss(self, gen_barcode: List[List[Tuple[float, float]]],
             ref_barcode: List[List[Tuple[float, float]]],
             p: int = 2, epsilon: float = 1e-5) -> torch.Tensor:
        """
        Computes the Wasserstein distance-based loss between generated and reference barcodes.
        Aggregates over batch by mean distance; supports p-norm (default: 2 for Earth Mover's Distance).

        Args:
            gen_barcode (List[List[Tuple[float, float]]]): Generated barcodes per batch item.
            ref_barcode (List[List[Tuple[float, float]]]): Reference barcodes per batch item.
            p (int): Order of Wasserstein distance (default: 2).
            epsilon (float): Small value for numerical stability in distances (default: 1e-5).

        Returns:
            torch.Tensor: Mean loss scalar.

        Note: From attachment's loss; expands to batch mean, handles unequal lengths by padding with (0,0),
              and uses Gudhi's wasserstein_distance with ground distance p.
        """
        if not GUDHI_AVAILABLE:
            return torch.tensor(0.0)

        if len(gen_barcode) != len(ref_barcode):
            raise ValueError("Generated and reference barcodes must have the same batch size.")

        distances = []
        for gen, ref in zip(gen_barcode, ref_barcode):
            try:
                # Convert to numpy arrays [num_features, 2]
                gen_array = np.array(gen) if gen else np.empty((0, 2))
                ref_array = np.array(ref) if ref else np.empty((0, 2))

                # Pad shorter array with (0, epsilon) to match lengths for distance computation
                max_len = max(len(gen_array), len(ref_array))
                if len(gen_array) < max_len:
                    pad = np.zeros((max_len - len(gen_array), 2))
                    pad[:, 1] = epsilon  # Small death for padding
                    gen_array = np.vstack([gen_array, pad])
                if len(ref_array) < max_len:
                    pad = np.zeros((max_len - len(ref_array), 2))
                    pad[:, 1] = epsilon
                    ref_array = np.vstack([ref_array, pad])

                # Compute Wasserstein distance
                dist = gd.wasserstein.wasserstein_distance(gen_array, ref_array, order=p)
                distances.append(dist)
            except Exception as e:
                print(f"Error computing loss: {e}. Using zero distance.")
                distances.append(0.0)

        # Mean over batch
        return torch.tensor(np.mean(distances), dtype=torch.float32)

    def normalize_barcode(self, barcode: List[Tuple[float, float]],
                          scale_factor: float = 1e-10) -> List[Tuple[float, float]]:
        """
        Normalizes barcode intervals by scaling (e.g., by atomic units) for consistent comparisons.

        Args:
            barcode (List[Tuple[float, float]]): Input barcode.
            scale_factor (float): Scaling divisor (default: 1 Å = 1e-10 m).

        Returns:
            List[Tuple[float, float]]: Scaled barcode.

        Note: Utility not in attachment; added for handling varying scales in physical simulations.
              Clamps to non-negative and handles infinite deaths by setting to large finite value.
        """
        normalized = []
        for birth, death in barcode:
            birth_scaled = max(birth / scale_factor, 0.0)
            death_scaled = death / scale_factor if death < np.inf else 1e6  # Cap infinite
            normalized.append((birth_scaled, death_scaled))
        return normalized
