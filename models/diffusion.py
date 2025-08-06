import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Dict, Optional
from electrodiffusion.models.noise_schedule import NoiseSchedule  # Import for noise scheduling
from electrodiffusion.models.score_network import PhysicsInformedScoreNetwork  # Import for score predictions
from electrodiffusion.config.physics_constants import PhysicalConstants  # For physical scaling in conditions

class ScoreBasedDiffusion:
    """
    Base class for score-based diffusion models using VP-SDE.
    This class manages the forward diffusion process (adding noise to data) and provides utilities
    for reverse process steps (denoising). It integrates a score network to estimate the gradient of the log-density
    and a noise schedule to control the diffusion strength over time.

    Args:
        score_network (nn.Module): Neural network to predict scores (e.g., PhysicsInformedScoreNetwork).
        noise_schedule (NoiseSchedule): Scheduler for beta(t), alpha(t), sigma(t).
    """
    def __init__(self, score_network: nn.Module, noise_schedule: NoiseSchedule):
        self.score_net = score_network
        self.noise_schedule = noise_schedule

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the forward diffusion process to add noise to the initial state x0 at time t.
        This corresponds to q(x_t | x_0) = N(sqrt(alpha_t) x0, sigma_t^2 I), where noise is sampled from N(0, I).

        Args:
            x0 (torch.Tensor): Initial state [batch_size, state_dim].
            t (torch.Tensor): Time [batch_size] or scalar, in [0,1].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Noisy state x_t and the added noise.

        Note: Clamps t to [0,1] for safety; uses unsqueeze for broadcasting to state_dim.
        """
        t = torch.clamp(t, min=0.0, max=1.0)  # Ensure valid time range
        alpha_t = self.noise_schedule.alpha(t).unsqueeze(-1)  # [batch_size, 1]
        sigma_t = self.noise_schedule.sigma(t).unsqueeze(-1)  # [batch_size, 1]
        
        noise = torch.randn_like(x0)  # Sample Gaussian noise matching x0's shape and device
        x_t = alpha_t * x0 + sigma_t * noise  # Affine transformation for forward process
        
        return x_t, noise

    def reverse_step(self, x_t: torch.Tensor, t: torch.Tensor, dt: float,
                     external_conditions: Optional[Dict] = None) -> torch.Tensor:
        """
        Computes a single step of the reverse SDE using the learned score.
        The reverse SDE is dX = [0.5 beta(t) X + beta(t) score] dt + sqrt(beta(t)) dW_backward.

        Args:
            x_t (torch.Tensor): Current noisy state [batch_size, state_dim].
            t (torch.Tensor): Current time [batch_size].
            dt (float): Time step size (negative for reverse).
            external_conditions (Optional[Dict]): External params for conditional scoring.

        Returns:
            torch.Tensor: Updated state after one reverse step.

        Note: From Claude's reverse_sde_step; adds noise term for stochastic sampling. For deterministic ODE, set noise to zero.
        """
        t = torch.clamp(t, min=0.0, max=1.0)
        beta_t = self.noise_schedule.beta(t).unsqueeze(-1)  # [batch_size, 1]
        
        # Get predictions from score network (includes score, energy, force)
        predictions = self.score_net(x_t, t, external_conditions)
        score = predictions['score']  # [batch_size, state_dim]
        
        # Reverse drift term
        drift = 0.5 * beta_t * x_t + beta_t * score
        
        # Stochastic noise term (backward Wiener process)
        noise = torch.randn_like(x_t) * torch.sqrt(beta_t * torch.abs(torch.tensor(dt)))  # Abs for reverse direction
        
        # Euler-Maruyama step
        return x_t + drift * dt + noise

    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
                   external_conditions: Optional[Dict] = None) -> float:
        """
        Performs a single training step using denoising score matching loss.
        Samples random t, applies forward process, computes predicted vs. true score, and updates parameters.

        Args:
            batch (torch.Tensor): Training data batch [batch_size, state_dim].
            optimizer (torch.optim.Optimizer): Optimizer (e.g., Adam).
            external_conditions (Optional[Dict]): Conditions for conditional training.

        Returns:
            float: Loss value for the step.

        Note: Adapted from Claude's train_step; adds optional physics regularization using energy/force predictions.
        """
        batch_size = batch.shape[0]
        device = batch.device
        
        # Sample random times uniformly in [0,1]
        t = torch.rand(batch_size, device=device)
        
        # Forward process to get x_t and noise
        x_t, noise = self.forward_process(batch, t)
        
        # Predict scores and auxiliary outputs
        predictions = self.score_net(x_t, t, external_conditions)
        predicted_score = predictions['score']
        
        # True score: -noise / sigma_t (denoising objective)
        sigma_t = self.noise_schedule.sigma(t).unsqueeze(-1) + 1e-8  # Epsilon for stability
        true_score = -noise / sigma_t
        
        # Denoising score matching loss
        score_loss = torch.mean((predicted_score - true_score) ** 2)
        
        # Optional physics regularization (e.g., force consistency; assumes true_energy/force available in extended training)
        phys_loss = 0.0  # Placeholder; can compute using predictions['energy'], predictions['force']
        
        # Total loss
        total_loss = score_loss + 0.1 * phys_loss  # Weighted combination
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()

class ConditionalDiffusion(ScoreBasedDiffusion):
    """
    Extended diffusion model for conditional generation based on external physical parameters.
    This class maps conditions like voltage or ionic strength to embeddings that modulate the score network.

    Args:
        score_network (nn.Module): Score network supporting conditional inputs.
        noise_schedule (NoiseSchedule): Noise scheduler.
        constants (PhysicalConstants): For physical unit conversions in conditioning.
    """
    def __init__(self, score_network: nn.Module, noise_schedule: NoiseSchedule,
                 constants: PhysicalConstants = PhysicalConstants()):
        super().__init__(score_network, noise_schedule)
        self.constants = constants

    def condition_on_external(self, ext_params: Dict) -> Dict:
        """
        Prepares external conditions as a dictionary of tensors for score network input.
        Converts physical parameters (e.g., voltage to electric field) using constants.

        Args:
            ext_params (Dict): Raw parameters like {'voltage': tensor, 'ionic_strength': tensor}.

        Returns:
            Dict: Processed conditions ready for score_net (e.g., {'electric_field': tensor[batch_size, 3]}).

        Note: Adapted from Claude's condition_on_external; adds unit conversions and clamping for physical realism.
        """
        conditions = {}
        batch_size = next(iter(ext_params.values())).shape[0] if ext_params else 1  # Infer batch size
        
        if 'voltage' in ext_params:
            V = ext_params['voltage']  # [batch_size] or scalar
            if V.dim() == 0:
                V = V.expand(batch_size)  # Broadcast if scalar
            L = 50e-10  # Channel length in meters (configurable)
            E_field = torch.zeros(batch_size, 3, device=V.device)
            E_field[:, 2] = V / L  # Electric field in z-direction (V/m)
            conditions['electric_field'] = E_field
        
        if 'ionic_strength' in ext_params:
            I = ext_params['ionic_strength']  # [batch_size]
            if I.dim() == 0:
                I = I.expand(batch_size)
            # Simplified gradient: Assume uniform gradient along z
            grad_c = torch.zeros(batch_size, 3, device=I.device)
            grad_c[:, 2] = I * self.constants.e / (self.constants.kB * self.constants.T)  # Scaled by physical constants
            conditions['concentration_gradient'] = grad_c
        
        # Additional conditions can be added here (e.g., magnetic field, temperature)
        
        return conditions

    def conditional_forward_process(self, x0: torch.Tensor, t: torch.Tensor,
                                    ext_params: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conditional version of forward_process, modulating noise based on external params.

        Args:
            x0 (torch.Tensor): Initial state.
            t (torch.Tensor): Time.
            ext_params (Dict): External parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Noisy state and noise.

        Note: Optionally scales sigma_t by a factor derived from ext_params for condition-dependent diffusion.
        """
        conditions = self.condition_on_external(ext_params)
        # Example modulation: Increase noise in high ionic strength
        modulation_factor = 1.0
        if 'ionic_strength' in ext_params:
            modulation_factor += 0.1 * ext_params['ionic_strength'].mean()  # Simple scaling
        
        alpha_t = self.noise_schedule.alpha(t).unsqueeze(-1)
        sigma_t = self.noise_schedule.sigma(t).unsqueeze(-1) * modulation_factor
        
        noise = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * noise
        
        return x_t, noise

    def train_step(self, batch: torch.Tensor, optimizer: torch.optim.Optimizer,
                   ext_params: Optional[Dict] = None) -> float:
        """
        Conditional training step, incorporating ext_params in forward and score prediction.

        Args:
            batch (torch.Tensor): Data batch.
            optimizer (torch.optim.Optimizer): Optimizer.
            ext_params (Optional[Dict]): External parameters for conditioning.

        Returns:
            float: Loss value.

        Note: Overrides base train_step to use conditional_forward_process and pass conditions to score_net.
        """
        batch_size = batch.shape[0]
        device = batch.device
        
        t = torch.rand(batch_size, device=device)
        
        # Use conditional forward if params provided
        if ext_params:
            x_t, noise = self.conditional_forward_process(batch, t, ext_params)
            conditions = self.condition_on_external(ext_params)
        else:
            x_t, noise = self.forward_process(batch, t)
            conditions = None
        
        # Predict with conditions
        predictions = self.score_net(x_t, t, conditions)
        predicted_score = predictions['score']
        
        sigma_t = self.noise_schedule.sigma(t).unsqueeze(-1) + 1e-8
        true_score = -noise / sigma_t
        
        loss = torch.mean((predicted_score - true_score) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
