import argparse
import torch
from electrodiffusion.models import CoarseGrainedModel, IonChannelSDE  # Core models
from electrodiffusion.simulations import SDEIntegrator  # Integrator for SDE solving
from electrodiffusion.utils.viz import plot_trajectory, visualize_distribution  # Visualization tools
from electrodiffusion.config import PhysicalConstants, DefaultConfig  # Configuration and constants

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the simulation.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Simple SDE Simulation Example for Electrodiffusion Library")
    parser.add_argument('--batch_size', type=int, default=10, help="Number of parallel trajectories to simulate (default: 10)")
    parser.add_argument('--T', type=float, default=1.0, help="Total simulation time in seconds (default: 1.0)")
    parser.add_argument('--steps', type=int, default=1000, help="Number of integration steps (default: 1000)")
    parser.add_argument('--method', type=str, default='euler', choices=['euler', 'milstein'], help="Integration method (default: euler)")
    parser.add_argument('--external_field', type=float, default=0.0, help="External electric field in V/m (default: 0.0)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available (default: False)")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save trajectory plot (default: None, show interactively)")
    return parser.parse_args()

def create_mock_system() -> object:
    """
    Creates a mock system object with simple potential and diffusion functions.
    This mimics a basic ion channel environment for testing purposes.

    Returns:
        object: Mock system with required methods.
    """
    class MockSystem:
        def potential_of_mean_force(self, z, protein_state):
            # Simple harmonic PMF along z: A(z) = 5.0 * z^2 (in arbitrary units)
            return 5.0 * z**2
        
        def electrostatic_potential(self, z, E_ext):
            # Linear potential: V(z) = E_ext * z
            return E_ext * z
        
        def position_dependent_diffusion(self, z):
            # Constant D(z) = 1e-9 m²/s, typical for ions
            return torch.ones_like(z) * 1e-9

    return MockSystem()

def main(args: argparse.Namespace):
    """
    Main function to run the simple SDE simulation.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    # Determine device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load default config and constants
    config = DefaultConfig()
    constants = PhysicalConstants()

    # Initialize coarse-grained model
    cg_model = CoarseGrainedModel(num_residues=config.NUM_RESIDUES, constants=constants)

    # Create mock system
    mock_system = create_mock_system()

    # Define a simple noise schedule (linear increase)
    def noise_schedule(t: torch.Tensor) -> torch.Tensor:
        return config.BETA_MIN + t * (config.BETA_MAX - config.BETA_MIN)

    # Initialize IonChannelSDE
    sde = IonChannelSDE(system=mock_system, noise_schedule=noise_schedule,
                         external_field=args.external_field, constants=constants).to(device)

    # Compute state dimension: backbone + sidechain + ion (pos3 + vel3 =6 for simplicity)
    state_dim = cg_model.backbone_dim + cg_model.sidechain_dim + 6

    # Initial state: Small random fluctuations around zero
    initial_state = torch.randn(args.batch_size, state_dim, device=device) * 0.1

    # Set up integrator
    integrator = SDEIntegrator(sde_model=sde, method=args.method)

    # Run integration
    print(f"Integrating SDE with {args.method} method for T={args.T} over {args.steps} steps...")
    trajectory = integrator.integrate(initial_state=initial_state, T=args.T, time_steps=args.steps)

    # Basic analysis: Final state statistics
    final_state = trajectory[-1]
    mean_final = torch.mean(final_state, dim=0)
    var_final = torch.var(final_state, dim=0)
    print(f"Final state mean (first 5 dims): {mean_final[:5]}")
    print(f"Final state variance (first 5 dims): {var_final[:5]}")

    # Visualize trajectory (time series of first few dimensions)
    plot_trajectory(trajectory, save_path=args.save_path)

    # Visualize final state distribution
    visualize_distribution(final_state, title="Final State Distribution")

    print("Simulation completed!")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
