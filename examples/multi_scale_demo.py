import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch  # For power spectral density
from electrodiffusion.models import CoarseGrainedModel, IonChannelSDE  # Core models
from electrodiffusion.models.multi_scale import MultiScaleSDE  # Multi-scale integrator
from electrodiffusion.utils.viz import plot_trajectory  # Reuse for custom plotting if needed
from electrodiffusion.config import PhysicalConstants, DefaultConfig  # Configurations

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the multi-scale demo.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Multi-Scale Simulation Demo for Electrodiffusion Library")
    parser.add_argument('--batch_size', type=int, default=1, help="Number of parallel simulations (default: 1)")
    parser.add_argument('--T', type=float, default=1e-3, help="Total simulation time in seconds (default: 1e-3)")
    parser.add_argument('--fast_dt', type=float, default=1e-6, help="Fast timestep in seconds (default: 1e-6)")
    parser.add_argument('--slow_dt', type=float, default=1e-4, help="Slow timestep in seconds (default: 1e-4)")
    parser.add_argument('--use_memory', action='store_true', help="Enable memory kernel in projection (default: False)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available (default: False)")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save figure (default: None, show interactively)")
    return parser.parse_args()

def create_mock_sde(cg_model, constants, drift_scale=0.1, diffusion_scale=0.5):
    """
    Creates a mock IonChannelSDE with simple drift and diffusion.

    Args:
        cg_model (CoarseGrainedModel): Coarse-grained model for state dims.
        constants (PhysicalConstants): Physical constants.
        drift_scale (float): Scaling for damping drift (default: 0.1).
        diffusion_scale (float): Scaling for constant diffusion (default: 0.5).

    Returns:
        IonChannelSDE: Mock SDE instance.
    """
    class MockSystem:
        def potential_of_mean_force(self, z, protein_state):
            return torch.zeros_like(z)  # Flat PMF for simplicity
        def electrostatic_potential(self, z, E_ext):
            return torch.zeros_like(z)
        def position_dependent_diffusion(self, z):
            return torch.ones_like(z) * 1e-9

    noise_schedule = lambda t: 1.0 + 10.0 * t  # Linear increasing noise

    return IonChannelSDE(system=MockSystem(), noise_schedule=noise_schedule,
                         external_field=0.0, constants=constants)

def exponential_kernel(t: torch.Tensor, gamma: float = 1.0, tau: float = 1e-5) -> torch.Tensor:
    """
    Exponential memory kernel K(t) = (gamma / tau) exp(-t / tau).

    Args:
        t (torch.Tensor): Time differences.
        gamma (float): Friction coefficient (default: 1.0).
        tau (float): Memory timescale (default: 1e-5 s).

    Returns:
        torch.Tensor: Kernel values.
    """
    return (gamma / tau) * torch.exp(-t / tau)

def main(args: argparse.Namespace):
    """
    Main function to run the multi-scale demo.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DefaultConfig()
    constants = PhysicalConstants()

    # Initialize coarse-grained model
    cg_model = CoarseGrainedModel(config.NUM_RESIDUES, constants)

    # Create fast and slow SDEs with different scales
    fast_sde = create_mock_sde(cg_model, constants, drift_scale=1.0, diffusion_scale=1.0).to(device)  # Faster dynamics
    slow_sde = create_mock_sde(cg_model, constants, drift_scale=0.1, diffusion_scale=0.5).to(device)   # Slower

    # Multi-scale SDE
    multi_sde = MultiScaleSDE(fast_sde=fast_sde, slow_sde=slow_sde,
                              fast_dt=args.fast_dt, slow_dt=args.slow_dt)

    # Initial state (backbone + sidechain + ion; simplified dim for demo)
    state_dim = 10  # Mock small dim
    x0 = torch.randn(args.batch_size, state_dim, device=device) * 0.1

    # Optional memory kernel
    memory_kernel = exponential_kernel if args.use_memory else None

    # Integrate
    print(f"Integrating multi-scale SDE for T={args.T}...")
    fast_traj, slow_traj = multi_sde.integrate_multiscale(x0=x0, T=args.T, memory_kernel=memory_kernel)

    # Convert to ms for plotting
    time_fast = np.linspace(0, args.T * 1e3, fast_traj.shape[0])
    time_slow = np.linspace(0, args.T * 1e3, slow_traj.shape[0])

    # Compute PSD
    fs_fast = 1 / args.fast_dt
    f_fast, psd_fast = welch(fast_traj[:, 0, 0].cpu().detach().numpy(), fs=fs_fast)
    fs_slow = 1 / args.slow_dt
    f_slow, psd_slow = welch(slow_traj[:, 0, 0].cpu().detach().numpy(), fs=fs_slow)

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Multi-Scale Dynamics Demo")

    # Fast trajectory
    axs[0, 0].plot(time_fast, fast_traj[:, 0, 0].cpu().detach().numpy(), color='blue')
    axs[0, 0].set_xlabel('Time (ms)')
    axs[0, 0].set_ylabel('Fast Variable')
    axs[0, 0].set_title('Fast Dynamics Trajectory')
    axs[0, 0].grid(True, alpha=0.3)

    # Slow trajectory
    axs[0, 1].plot(time_slow, slow_traj[:, 0, 0].cpu().detach().numpy(), color='green')
    axs[0, 1].set_xlabel('Time (ms)')
    axs[0, 1].set_ylabel('Slow Variable')
    axs[0, 1].set_title('Slow Dynamics Trajectory')
    axs[0, 1].grid(True, alpha=0.3)

    # Fast PSD
    axs[1, 0].loglog(f_fast, psd_fast, color='blue')
    axs[1, 0].set_xlabel('Frequency (Hz)')
    axs[1, 0].set_ylabel('PSD')
    axs[1, 0].set_title('Fast Dynamics Spectrum')
    axs[1, 0].grid(True, alpha=0.3)

    # Slow PSD
    axs[1, 1].loglog(f_slow, psd_slow, color='green')
    axs[1, 1].set_xlabel('Frequency (Hz)')
    axs[1, 1].set_ylabel('PSD')
    axs[1, 1].set_title('Slow Dynamics Spectrum')
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if args.save_path:
        plt.savefig(args.save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    print("Multi-scale demo completed!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
