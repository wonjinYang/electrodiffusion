import argparse
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from electrodiffusion.models import PhysicsInformedScoreNetwork, NoiseSchedule, ScoreBasedDiffusion  # Model components
from electrodiffusion.utils.data import DataLoader  # Data handling
from electrodiffusion.utils.viz import plot_convergence  # For loss plotting (adapted for metrics)
from electrodiffusion.config import DefaultConfig, PhysicalConstants  # Configurations

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the diffusion training example.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Diffusion Model Training Example for Electrodiffusion Library")
    parser.add_argument('--epochs', type=int, default=500, help="Number of training epochs (default: 500)")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument('--state_dim', type=int, default=100, help="State dimension for data and model (default: 100)")
    parser.add_argument('--num_samples', type=int, default=5000, help="Number of synthetic training samples (default: 5000)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available (default: False)")
    parser.add_argument('--checkpoint_interval', type=int, default=100, help="Save checkpoint every N epochs (default: 100)")
    parser.add_argument('--early_stop_patience', type=int, default=50, help="Epochs without improvement before stopping (default: 50)")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save metrics plot (default: None, show interactively)")
    return parser.parse_args()

def plot_training_metrics(losses: list, save_path: Optional[str] = None):
    """
    Plots the training loss curve.

    Args:
        losses (list): List of loss values per epoch.
        save_path (Optional[str]): Path to save the plot.
    """
    epochs = np.arange(1, len(losses) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main(args: argparse.Namespace):
    """
    Main function to run the diffusion model training.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = DefaultConfig()
    constants = PhysicalConstants()

    # Data setup
    data_loader = DataLoader(constants=constants)
    print("Generating synthetic training data...")
    training_data = data_loader._generate_synthetic_data(num_frames=args.num_samples, num_residues=config.NUM_RESIDUES // 10)  # Adjust for dim
    training_data = data_loader.preprocess(training_data, normalize=True, center=True).to(device)
    print(f"Training data shape: {training_data.shape}")

    # Split for validation (10% holdout)
    val_split = int(0.1 * len(training_data))
    train_data = training_data[:-val_split]
    val_data = training_data[-val_split:]

    # Model setup
    score_network = PhysicsInformedScoreNetwork(input_dim=args.state_dim,
                                                hidden_dim=128,  # Smaller for demo
                                                num_layers=4, num_heads=4,
                                                constants=constants).to(device)

    noise_schedule = NoiseSchedule(schedule_type='linear',
                                   beta_min=config.BETA_MIN,
                                   beta_max=config.BETA_MAX)

    diffusion_model = ScoreBasedDiffusion(score_network=score_network, noise_schedule=noise_schedule)

    # Optimizer
    optimizer = optim.AdamW(score_network.parameters(), lr=args.lr)

    # Training loop
    print("Starting training...")
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        # Train step
        train_loss = diffusion_model.train_step(train_data, optimizer)
        train_losses.append(train_loss)

        # Validation loss (simple forward on val_data)
        with torch.no_grad():
            val_t = torch.rand(len(val_data), device=device)
            val_xt, _ = diffusion_model.forward_process(val_data, val_t)
            predictions = score_network(val_xt, val_t)
            sigma_t = noise_schedule.sigma(val_t).unsqueeze(-1)
            true_score = -_ / (sigma_t + 1e-8)  # From noise in forward
            val_loss = torch.mean((predictions['score'] - true_score)**2).item()
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Checkpoint saving
        if epoch % args.checkpoint_interval == 0:
            torch.save(score_network.state_dict(), f"checkpoint_epoch_{epoch}.pth")
            print(f"Checkpoint saved at epoch {epoch}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print("Training completed!")

    # Visualize metrics
    metrics = {'train_loss': train_losses, 'val_loss': val_losses}
    plot_convergence(metrics, save_path=args.save_path)  # Assuming plot_convergence handles dict of lists

    # Generate and visualize samples post-training
    sampler = DiffusionSampler(diffusion_model, num_steps=1000)
    samples = sampler.sample(num_samples=100, dim=args.state_dim, device=device)
    visualize_distribution(samples, title="Generated Samples After Training")

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
