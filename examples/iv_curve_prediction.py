import argparse
import torch
import numpy as np
from electrodiffusion.models import PhysicsInformedScoreNetwork, NoiseSchedule, ScoreBasedDiffusion  # Core models
from electrodiffusion.simulations import DiffusionSampler, IVCurvePredictor  # Sampler and predictor
from electrodiffusion.utils.viz import plot_iv_curve  # Visualization
from electrodiffusion.config import PhysicalConstants, DefaultConfig  # Configurations

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the IV curve prediction example.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="IV Curve Prediction Example for Electrodiffusion Library")
    parser.add_argument('--num_samples', type=int, default=500, help="Number of samples per voltage (default: 500)")
    parser.add_argument('--state_dim', type=int, default=100, help="State dimension for sampling (default: 100)")
    parser.add_argument('--voltage_min', type=float, default=-0.1, help="Minimum voltage in V (default: -0.1)")
    parser.add_argument('--voltage_max', type=float, default=0.1, help="Maximum voltage in V (default: 0.1)")
    parser.add_argument('--points', type=int, default=21, help="Number of voltage points (default: 21)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available (default: False)")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save IV plot (default: None, show interactively)")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to trained score network checkpoint (default: None, use mock)")
    return parser.parse_args()

def load_or_create_model(state_dim: int, checkpoint: Optional[str], device: torch.device) -> ScoreBasedDiffusion:
    """
    Loads a trained score network from checkpoint or creates a mock for demonstration.

    Args:
        state_dim (int): Input dimension for the network.
        checkpoint (Optional[str]): Path to checkpoint file.
        device (torch.device): Target device.

    Returns:
        ScoreBasedDiffusion: Initialized diffusion model.
    """
    constants = PhysicalConstants()
    config = DefaultConfig()

    if checkpoint:
        # In practice, load from file (mocked here)
        print(f"Loading trained model from {checkpoint}...")
        score_network = PhysicsInformedScoreNetwork(input_dim=state_dim, constants=constants).to(device)
        # score_network.load_state_dict(torch.load(checkpoint))  # Uncomment for real loading
    else:
        print("Using mock untrained model for demonstration...")
        score_network = PhysicsInformedScoreNetwork(input_dim=state_dim,
                                                    hidden_dim=config.HIDDEN_DIM or 256,
                                                    num_layers=4, num_heads=4,
                                                    constants=constants).to(device)

    noise_schedule = NoiseSchedule(schedule_type='linear',
                                   beta_min=config.BETA_MIN,
                                   beta_max=config.BETA_MAX)

    return ScoreBasedDiffusion(score_network=score_network, noise_schedule=noise_schedule)

def main(args: argparse.Namespace):
    """
    Main function to run IV curve prediction.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load or create diffusion model
    diffusion_model = load_or_create_model(args.state_dim, args.checkpoint, device)

    # Create sampler
    sampler = DiffusionSampler(diffusion_model, num_steps=500)  # Adjust steps as needed

    # Create IV predictor
    predictor = IVCurvePredictor(sampler=sampler, constants=PhysicalConstants())

    # Voltage range
    voltages = np.linspace(args.voltage_min, args.voltage_max, args.points)
    print(f"Predicting IV curve for voltages from {args.voltage_min*1000:.0f} to {args.voltage_max*1000:.0f} mV...")

    # Compute IV data
    iv_data = predictor.compute_iv(voltages=voltages, num_samples=args.num_samples, state_dim=args.state_dim)

    print("IV prediction completed!")
    print(f"Sample currents: {iv_data['currents'][:5]} pA")  # First few for logging

    # Validate
    validation = predictor.validate(num_samples=100, state_dim=args.state_dim)  # Small sample for quick validation
    print(f"Detailed balance: {'PASS' if validation['detailed_balance_pass'] else 'FAIL'} (error: {validation['detailed_balance_error']:.4f})")
    print(f"Thermodynamic consistency: {'PASS' if validation['thermodynamic_pass'] else 'FAIL'} (error: {validation['thermodynamic_error']:.4f})")

    # Visualize
    plot_iv_curve(iv_data, save_path=args.save_path, title="Predicted IV Curve")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
