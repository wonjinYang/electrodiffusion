import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from electrodiffusion.simulations.integrators import SDEIntegrator, EulerMaruyamaSolver, MilsteinSolver  # Solvers and integrator
from electrodiffusion.models.sde import IonChannelSDE  # Base SDE class for mocking OU

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments for the convergence analysis.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Convergence Analysis Example for Electrodiffusion Solvers")
    parser.add_argument('--num_paths', type=int, default=1000, help="Number of Monte Carlo paths per dt (default: 1000)")
    parser.add_argument('--T', type=float, default=1.0, help="Total integration time (default: 1.0)")
    parser.add_argument('--dt_values', type=str, default="1e-2,5e-3,1e-3,5e-4,1e-4", help="Comma-separated dt values (default: 1e-2,5e-3,1e-3,5e-4,1e-4)")
    parser.add_argument('--solvers', type=str, default="euler,milstein", help="Comma-separated solvers to test (default: euler,milstein)")
    parser.add_argument('--gpu', action='store_true', help="Use GPU if available (default: False)")
    parser.add_argument('--save_path', type=str, default=None, help="Path to save convergence plot (default: None, show interactively)")
    return parser.parse_args()

class OUSDE(IonChannelSDE):
    """
    Mock SDE for Ornstein-Uhlenbeck process: dx = -θ x dt + σ dW.
    Used as a test problem with analytical solutions.
    """
    def __init__(self, theta: float = 1.0, sigma: float = np.sqrt(2.0)):
        super().__init__(None, None)  # Dummy system and schedule
        self.theta = theta
        self.sigma = sigma

    def f(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -self.theta * y

    def g(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.sigma * torch.ones_like(y)

def analytical_mean(x0: float, t: float, theta: float) -> float:
    """
    Analytical mean for OU process: E[x(t)] = x0 exp(-θ t).

    Args:
        x0 (float): Initial value.
        t (float): Time.
        theta (float): Decay rate.

    Returns:
        float: Expected mean.
    """
    return x0 * np.exp(-theta * t)

def analytical_variance(t: float, theta: float, sigma: float) -> float:
    """
    Analytical variance for OU process: Var[x(t)] = (σ² / (2θ)) (1 - exp(-2θ t)).

    Args:
        t (float): Time.
        theta (float): Decay rate.
        sigma (float): Diffusion coefficient.

    Returns:
        float: Expected variance.
    """
    return (sigma**2 / (2 * theta)) * (1 - np.exp(-2 * theta * t))

def run_analysis(args: argparse.Namespace, device: torch.device) -> Dict[str, Dict[str, List[float]]]:
    """
    Runs the convergence analysis for specified solvers and dt values.

    Args:
        args (argparse.Namespace): Parsed arguments.
        device (torch.device): Computation device.

    Returns:
        Dict[str, Dict[str, List[float]]]: Results with errors per solver and dt.
    """
    ou = OUSDE().to(device)
    x0 = torch.tensor([[1.0]], device=device)
    
    dt_values = [float(dt) for dt in args.dt_values.split(',')]
    solver_map = {
        'euler': EulerMaruyamaSolver,
        'milstein': MilsteinSolver
    }
    solvers = {name: solver_map[name]() for name in args.solvers.split(',') if name in solver_map}
    
    results = {}
    for name, solver in solvers.items():
        results[name] = {'dt': [], 'mean_error': [], 'var_error': []}
        for dt in dt_values:
            steps = int(args.T / dt)
            finals = torch.zeros(args.num_paths, device=device)
            for i in range(args.num_paths):
                integrator = SDEIntegrator(ou, method=name)  # New integrator per path
                traj = integrator.integrate(x0, args.T, steps)
                finals[i] = traj[-1].item()
            
            num_mean = torch.mean(finals).item()
            num_var = torch.var(finals).item()
            
            anal_mean = analytical_mean(x0.item(), args.T, ou.theta)
            anal_var = analytical_variance(args.T, ou.theta, ou.sigma)
            
            results[name]['dt'].append(dt)
            results[name]['mean_error'].append(abs(num_mean - anal_mean))
            results[name]['var_error'].append(abs(num_var - anal_var))
    
    return results

def plot_results(results: Dict[str, Dict[str, List[float]]], save_path: Optional[str]):
    """
    Plots convergence results with log-log scales and reference lines.

    Args:
        results (Dict): Analysis results.
        save_path (Optional[str]): Path to save figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("SDE Solver Convergence Analysis")

    dt_ref = np.logspace(-4, -1, 10)
    for ax, error_type in zip([ax1, ax2], ['mean_error', 'var_error']):
        for name in results:
            ax.loglog(results[name]['dt'], results[name][error_type], 'o-', label=name)
        ax.loglog(dt_ref, dt_ref, 'k--', alpha=0.5, label='O(dt)')
        ax.loglog(dt_ref, dt_ref**2, 'k:', alpha=0.5, label='O(dt²)')
        ax.set_xlabel('Time Step (dt)')
        ax.set_ylabel(f'{error_type.capitalize()} Error')
        ax.set_title(f'Convergence of {error_type.capitalize()}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main(args: argparse.Namespace):
    """
    Main function to run the convergence analysis example.

    Args:
        args (argparse.Namespace): Parsed arguments.
    """
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Running SDE Solver Convergence Analysis...")
    results = run_analysis(args, device)

    print("Analysis completed! Plotting results...")
    plot_results(results, args.save_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
