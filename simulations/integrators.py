import torch
import numpy as np
from typing import Tuple, Callable, Optional
from abc import ABC, abstractmethod
from electrodiffusion.models.sde import IonChannelSDE  # Import for type hinting in integrators

class SDESolver(ABC):
    """
    Abstract base class for SDE solvers.
    Defines the interface for taking a single step in solving an SDE of the form dX = drift(t, X) dt + diffusion(t, X) dW.
    Subclasses must implement the step method, which advances the state by dt using the provided drift and diffusion functions.

    Note: Assumes Itô interpretation and diagonal noise; extensions for Stratonovich or non-diagonal can be added in subclasses.
    """
    @abstractmethod
    def step(self, t: float, dt: float, x: torch.Tensor,
             drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """
        Advances the state x by one timestep dt using the solver's method.

        Args:
            t (float): Current time.
            dt (float): Timestep size.
            x (torch.Tensor): Current state [batch_size, state_dim].
            drift (Callable): Drift function f(t, x) -> [batch_size, state_dim].
            diffusion (Callable): Diffusion function g(t, x) -> [batch_size, state_dim].

        Returns:
            torch.Tensor: Updated state [batch_size, state_dim].
        """
        pass

class EulerMaruyamaSolver(SDESolver):
    """
    Euler-Maruyama solver for SDEs.
    This is a first-order method that approximates the SDE solution using simple Euler steps plus Wiener noise.
    Suitable for quick prototyping but may require small dt for accuracy, especially with stiff drifts.

    Note: From Claude's EulerMaruyamaSolver; ensures tensor conversion for t and broadcasting for batched operations.
    """
    def step(self, t: float, dt: float, x: torch.Tensor,
             drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        # Convert t to tensor and expand to batch size for consistency
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.float, device=x.device)
        
        # Compute drift and diffusion terms
        drift_term = drift(t_tensor, x) * dt
        
        # Wiener increment (dW ~ N(0, dt))
        dW = torch.randn_like(x) * torch.sqrt(torch.tensor(dt, device=x.device))
        diffusion_term = diffusion(t_tensor, x) * dW
        
        # Update state
        return x + drift_term + diffusion_term

class MilsteinSolver(SDESolver):
    """
    Milstein solver for SDEs with higher-order accuracy.
    Extends Euler-Maruyama by adding a correction term involving the derivative of the diffusion coefficient,
    improving convergence for SDEs with non-constant diffusion (e.g., position-dependent in ion channels).

    Args:
        h (float): Small perturbation for numerical derivative computation (default: 1e-6 for balance between accuracy and stability).

    Note: From Claude's MilsteinSolver; adds clamping to prevent NaN in dg_dx and supports vectorized perturbations.
    """
    def __init__(self, h: float = 1e-6):
        self.h = h

    def step(self, t: float, dt: float, x: torch.Tensor,
             drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        batch_size = x.shape[0]
        t_tensor = torch.full((batch_size,), t, dtype=torch.float, device=x.device)
        
        # Compute base Euler term
        drift_term = drift(t_tensor, x) * dt
        g = diffusion(t_tensor, x)
        dW = torch.randn_like(x) * torch.sqrt(torch.tensor(dt, device=x.device))
        euler_term = g * dW
        x_new = x + drift_term + euler_term
        
        # Compute numerical derivative dg/dx
        x_pert = x + self.h  # Perturb entire batch
        g_pert = diffusion(t_tensor, x_pert)
        dg_dx = (g_pert - g) / self.h  # Element-wise derivative approximation
        dg_dx = torch.clamp(dg_dx, min=-1e6, max=1e6)  # Clamp to prevent overflow
        
        # Milstein correction: 0.5 * g * (dg/dx) * (dW^2 - dt)
        correction = 0.5 * g * dg_dx * (dW**2 - dt)
        
        return x_new + correction

class AdaptiveSDESolver:
    """
    Adaptive timestep SDE solver that adjusts dt based on local error estimates.
    Uses a base solver (e.g., Milstein) and compares full-step vs. two half-steps to estimate truncation error,
    reducing dt if error exceeds tolerance. This is crucial for stiff SDEs in electrodiffusion where forces vary rapidly.

    Args:
        base_solver (SDESolver): Underlying solver for steps (default: Milstein for accuracy).
        tolerance (float): Maximum allowed error per step (default: 1e-4).
        max_dt (float): Initial and maximum timestep (default: 1e-4).
        min_dt (float): Minimum timestep before accepting with warning (default: 1e-8).

    Note: From Claude's AdaptiveSDESolver; adds min_dt clamping and returns actual dt used for logging.
    """
    def __init__(self, base_solver: Optional[SDESolver] = None,
                 tolerance: float = 1e-4, max_dt: float = 1e-4, min_dt: float = 1e-8):
        self.base_solver = base_solver or MilsteinSolver()
        self.tol = tolerance
        self.max_dt = max_dt
        self.min_dt = min_dt

    def adaptive_step(self, t: float, x: torch.Tensor,
                      drift: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                      diffusion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Performs an adaptive step, adjusting dt until error is below tolerance.

        Args:
            t (float): Current time.
            x (torch.Tensor): Current state.
            drift (Callable): Drift function.
            diffusion (Callable): Diffusion function.

        Returns:
            Tuple[torch.Tensor, float]: Updated state and the dt used.

        Note: Uses L2 norm for error; reduces dt by 0.8 factor on failure, accepts half-step result for conservatism.
        """
        dt = self.max_dt
        while True:
            # Full step with dt
            x_full = self.base_solver.step(t, dt, x, drift, diffusion)
            
            # Two half steps
            x_half1 = self.base_solver.step(t, dt/2, x, drift, diffusion)
            x_half2 = self.base_solver.step(t + dt/2, dt/2, x_half1, drift, diffusion)
            
            # Error estimate (L2 norm)
            error = torch.norm(x_full - x_half2).item()
            
            if error < self.tol:
                return x_half2, dt  # Accept the more accurate half-step result
            else:
                dt *= 0.8  # Reduce timestep
                if dt < self.min_dt:
                    print(f"Warning: Minimum dt reached at t={t}, error={error}; accepting step.")
                    return x_full, dt

class SDEIntegrator:
    """
    High-level integrator for generating SDE trajectories.
    Wraps a solver (fixed or adaptive) to simulate over total time T with specified steps or adaptively.
    Stores full trajectory for analysis, with optional downsampling.

    Args:
        sde_model (IonChannelSDE): The SDE model providing f and g.
        method (str): Solver method ('euler', 'milstein', 'adaptive'; default: 'euler').

    Note: From Claude's SDEIntegrator; expands with adaptive support, trajectory stacking, and optional callback for monitoring (e.g., energy computation).
    """
    def __init__(self, sde_model: IonChannelSDE, method: str = "euler"):
        self.sde_model = sde_model
        if method == "euler":
            self.solver = EulerMaruyamaSolver()
        elif method == "milstein":
            self.solver = MilsteinSolver()
        elif method == "adaptive":
            self.solver = AdaptiveSDESolver()
        else:
            raise ValueError(f"Unknown method: {method}")

    def integrate(self, initial_state: torch.Tensor, T: float,
                  time_steps: Optional[int] = None,
                  callback: Optional[Callable[[torch.Tensor, float], None]] = None) -> torch.Tensor:
        """
        Integrates the SDE from initial_state over time T.

        Args:
            initial_state (torch.Tensor): Starting state [batch_size, state_dim].
            T (float): Total integration time.
            time_steps (Optional[int]): Number of fixed steps (ignored if adaptive).
            callback (Optional[Callable]): Function called after each step with (current_x, current_t).

        Returns:
            torch.Tensor: Trajectory [num_steps + 1, batch_size, state_dim].

        Note: For adaptive, time_steps is ignored; steps are determined dynamically. Trajectory includes initial state.
        """
        trajectory = [initial_state.clone()]
        x = initial_state.clone()
        t_current = 0.0
        
        if isinstance(self.solver, AdaptiveSDESolver):
            # Adaptive integration: Loop until t_current >= T
            while t_current < T:
                x, dt_used = self.solver.adaptive_step(t_current, x, self.sde_model.f, self.sde_model.g)
                t_current += dt_used
                trajectory.append(x.clone())
                if callback:
                    callback(x, t_current)
        else:
            # Fixed-step integration
            if time_steps is None:
                raise ValueError("time_steps required for non-adaptive methods")
            dt = T / time_steps
            for i in range(time_steps):
                x = self.solver.step(t_current, dt, x, self.sde_model.f, self.sde_model.g)
                t_current += dt
                trajectory.append(x.clone())
                if callback:
                    callback(x, t_current)
        
        return torch.stack(trajectory)
