import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional
from electrodiffusion.config.physics_constants import PhysicalConstants  # Import physical constants for physics-informed computations

class PositionalEncoding(nn.Module):
    """
    Positional encoding module for embedding time t into a high-dimensional vector.
    This uses sinusoidal functions to create position-aware embeddings, allowing the network
    to differentiate between different diffusion timesteps. Based on the transformer architecture,
    it helps in conditioning the score estimation on the noise level.

    Args:
        d_model (int): Dimensionality of the embedding vector.
        max_len (int): Maximum length for precomputing the encoding matrix (default: 10000 for t in [0,1] scaled).

    Note: t is expected to be normalized in [0,1]; scaling maps it to integer positions.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        # Precompute the positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # Divisor terms for sine/cosine frequencies
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer for non-trainable persistence
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Embeds the time t into the positional encoding space.

        Args:
            t (torch.Tensor): Time tensor [batch_size] or scalar, normalized in [0,1].

        Returns:
            torch.Tensor: Embedded tensor [batch_size, d_model].

        Note: Clamps scaled t to valid range to prevent index errors.
        """
        # Scale t to integer positions within max_len
        t_scaled = (t * (self.pe.size(0) - 1)).long().clamp(0, self.pe.size(0) - 1)
        return self.pe[t_scaled]

class AttentionBlock(nn.Module):
    """
    Self-attention block for modeling long-range interactions in molecular states.
    This block uses multi-head attention to capture dependencies between different parts of the input,
    such as interactions between protein residues, side chains, and ion positions. It includes residual
    connections and feed-forward layers for deeper representations.

    Args:
        d_model (int): Dimensionality of the input and output features.
        n_heads (int): Number of attention heads (default: 8 for parallel processing).

    Note: Supports optional masks for handling padded or variable-length sequences.
    """
    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        # Normalization layers for stability
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # Feed-forward network with expansion and GELU activation
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # Expansion for richer representations
            nn.GELU(),                        # Non-linearity
            nn.Linear(4 * d_model, d_model)   # Projection back to d_model
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the attention block.

        Args:
            x (torch.Tensor): Input tensor [batch_size, seq_len, d_model] (seq_len=1 if not tokenized).
            mask (Optional[torch.Tensor]): Attention mask [batch_size, seq_len, seq_len] (optional).

        Returns:
            torch.Tensor: Output tensor [batch_size, seq_len, d_model].

        Note: Applies residual connections to preserve information flow and prevent vanishing gradients.
        """
        # Self-attention with optional mask
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        # Residual connection and normalization
        x = self.norm1(x + attn_out)
        # Feed-forward processing
        ff_out = self.ff(x)
        # Another residual and normalization
        x = self.norm2(x + ff_out)
        return x

class PhysicsInformedScoreNetwork(nn.Module):
    """
    Physics-informed score network for estimating the score function while incorporating domain knowledge.
    This network predicts the score (primary output) along with auxiliary physics-based quantities like energy and force.
    It uses time embeddings, physics-aware layers, and attention blocks to process molecular states conditioned on time and external factors.
    Adapted from Claude's implementation, with enhancements for ion channel specifics (e.g., electric field embeddings tied to physical constants).

    Args:
        input_dim (int): Dimensionality of the input state (e.g., coarse-grained protein + ion features).
        hidden_dim (int): Hidden layer dimensionality (default: 256 for balance between capacity and efficiency).
        num_layers (int): Total number of layers (split between physics and attention; default: 6).
        num_heads (int): Attention heads (default: 8).
        constants (Optional[PhysicalConstants]): Physical constants for embedding scaling (default: new instance).

    Note: Outputs a dictionary for multi-task learning; can be extended with more heads (e.g., for flux prediction).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 num_layers: int = 6, num_heads: int = 8,
                 constants: Optional[PhysicalConstants] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.constants = constants or PhysicalConstants()  # Ensure constants are available for physics scaling

        # Time embedding components
        self.time_encoder = PositionalEncoding(hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)  # Project embeddings to match hidden_dim

        # Input projection layer
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Physics-aware layers: Sequential blocks with residuals for gradient flow
        self.physics_layers = nn.ModuleList([
            self._make_physics_layer(hidden_dim) for _ in range(num_layers // 2)
        ])

        # Attention layers for capturing interactions
        self.attention_layers = nn.ModuleList([
            AttentionBlock(hidden_dim, num_heads) for _ in range(num_layers // 2)
        ])

        # Output heads: Separate projections for score, energy, and force
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)  # Matches input_dim for score output
        )
        self.energy_head = nn.Linear(hidden_dim, 1)  # Scalar energy prediction
        self.force_head = nn.Linear(hidden_dim, input_dim)  # Force vector matching input_dim

    def _make_physics_layer(self, dim: int) -> nn.Module:
        """
        Creates a physics-aware layer with linear transformations, normalization, and activation.
        These layers incorporate inductive biases like residual connections for stable training.

        Args:
            dim (int): Input/output dimensionality.

        Returns:
            nn.Module: Sequential layer block.
        """
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),  # Stabilizes activations
            nn.GELU(),          # Smooth non-linearity
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                external_conditions: Optional[Dict] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing score, energy, and force predictions.

        Args:
            x (torch.Tensor): Input state [batch_size, input_dim].
            t (torch.Tensor): Time [batch_size] or scalar.
            external_conditions (Optional[Dict]): Dictionary with keys like 'electric_field' [batch_size, 3] or 'concentration_gradient' [batch_size, 3].

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'score' [batch_size, input_dim], 'energy' [batch_size, 1], 'force' [batch_size, input_dim].

        Note: Modulates hidden states with external conditions using physics-informed embeddings. Reshapes for attention if needed.
        """
        batch_size = x.shape[0]

        # Time embedding: Encode t and project
        t_embed = self.time_encoder(t)
        t_embed = self.time_proj(t_embed)

        # Project input and add time embedding
        h = self.input_proj(x)
        h = h + t_embed  # Broadcast addition for conditioning

        # Process through physics layers with residuals
        for layer in self.physics_layers:
            h = h + layer(h)  # Residual connection for information preservation

        # Reshape for attention: Treat as sequence of length 1 (can be extended for tokenized inputs)
        h_reshaped = h.unsqueeze(1)  # [batch_size, 1, hidden_dim]

        # Process through attention layers
        for attn_layer in self.attention_layers:
            h_reshaped = attn_layer(h_reshaped)

        # Squeeze back to [batch_size, hidden_dim]
        h = h_reshaped.squeeze(1)

        # Modulate with external conditions if provided
        if external_conditions is not None:
            if 'electric_field' in external_conditions:
                E_field = external_conditions['electric_field']
                E_embed = self._embed_electric_field(E_field, h.device)
                h = h + E_embed  # Add embedding to hidden state

            if 'concentration_gradient' in external_conditions:
                grad_c = external_conditions['concentration_gradient']
                grad_embed = self._embed_concentration(grad_c, h.device)
                h = h + grad_embed

        # Compute outputs via heads
        score = self.score_head(h)
        energy = self.energy_head(h)
        force = self.force_head(h)

        return {
            'score': score,
            'energy': energy,
            'force': force
        }

    def _embed_electric_field(self, E_field: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Embeds electric field vector into hidden space, scaled by physical constants.

        Args:
            E_field (torch.Tensor): Electric field [batch_size, 3].
            device (torch.device): Target device.

        Returns:
            torch.Tensor: Embedding [batch_size, hidden_dim].

        Note: Incorporates magnitude and vector components; scales by elementary charge for physical relevance.
        """
        E_mag = torch.norm(E_field, dim=-1, keepdim=True)  # Magnitude
        E_embed = torch.zeros(E_field.shape[0], self.hidden_dim, device=device)
        E_embed[:, :3] = E_field * self.constants.e  # Scale vector by elementary charge
        E_embed[:, 3] = E_mag.squeeze()  # Add magnitude
        return E_embed

    def _embed_concentration(self, grad_c: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Embeds concentration gradient into hidden space.

        Args:
            grad_c (torch.Tensor): Gradient [batch_size, 3].
            device (torch.device): Target device.

        Returns:
            torch.Tensor: Embedding [batch_size, hidden_dim].

        Note: Places gradient in specific positions; can be extended with normalization.
        """
        grad_embed = torch.zeros(grad_c.shape[0], self.hidden_dim, device=device)
        grad_embed[:, 4:7] = grad_c  # Embed in positions 4-6
        return grad_embed

    def physics_regularization_loss(self, energy_pred: torch.Tensor, force_pred: torch.Tensor,
                                    true_energy: Optional[torch.Tensor] = None,
                                    true_force: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes regularization loss to enforce physical consistency (e.g., force as negative gradient of energy).

        Args:
            energy_pred (torch.Tensor): Predicted energy [batch_size, 1].
            force_pred (torch.Tensor): Predicted force [batch_size, input_dim].
            true_energy (Optional[torch.Tensor]): Ground truth energy (if available).
            true_force (Optional[torch.Tensor]): Ground truth force.

        Returns:
            torch.Tensor: Scalar loss value.

        Note: This method is optional for training; encourages consistency with physics laws.
        """
        loss = 0.0
        if true_energy is not None:
            loss += F.mse_loss(energy_pred, true_energy)  # MSE on energy
        if true_force is not None:
            loss += F.mse_loss(force_pred, true_force)  # MSE on force
        # Consistency: Force should approximate -grad(energy), but requires autograd; omitted for simplicity
        return loss
