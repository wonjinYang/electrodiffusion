import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import Dict, Optional, List

# Set default Seaborn style for consistent, publication-quality plots
sns.set(style="whitegrid", palette="deep", font_scale=1.2)

def plot_iv_curve(iv_data: Dict[str, np.ndarray], save_path: Optional[str] = None,
                  title: str = 'IV Curve Analysis', figsize: Tuple[int, int] = (14, 10)):
    """
    Visualizes IV curve data with multiple panels for comprehensive analysis.
    Includes main IV plot, conductance, open probability, and log-scale current.
    Data should be a dict with keys 'voltages' (V), 'currents' (pA), optionally 'conductances' (pS) and 'open_probabilities'.

    Args:
        iv_data (Dict[str, np.ndarray]): Dictionary of IV data arrays.
        save_path (Optional[str]): Path to save figure (default: None, shows interactively).
        title (str): Main figure title (default: 'IV Curve Analysis').
        figsize (Tuple[int, int]): Figure size (default: (14, 10) for detailed views).

    Returns:
        None: Displays or saves the figure.

    Note: Adapted from attachment's plot_iv_curve; enhances with Seaborn lines, dynamic subplots (skips missing data),
          voltage conversion to mV, and confidence bands if std data provided (future extension).
    """
    if 'voltages' not in iv_data or 'currents' not in iv_data:
        raise ValueError("iv_data must include 'voltages' and 'currents' keys.")

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    voltages_mv = iv_data['voltages'] * 1000  # Convert V to mV
    currents = iv_data['currents']

    # Main IV plot
    sns.lineplot(x=voltages_mv, y=currents, ax=axs[0, 0], marker='o', color='blue', linewidth=2)
    axs[0, 0].set_xlabel('Voltage (mV)')
    axs[0, 0].set_ylabel('Current (pA)')
    axs[0, 0].set_title('Current-Voltage Relationship')

    # Conductance plot
    if 'conductances' in iv_data:
        sns.lineplot(x=voltages_mv, y=iv_data['conductances'], ax=axs[0, 1], marker='o', color='green', linewidth=2)
        axs[0, 1].set_xlabel('Voltage (mV)')
        axs[0, 1].set_ylabel('Conductance (pS)')
        axs[0, 1].set_title('Conductance vs Voltage')
    else:
        axs[0, 1].axis('off')  # Hide if no data

    # Open probability plot
    if 'open_probabilities' in iv_data:
        sns.lineplot(x=voltages_mv, y=iv_data['open_probabilities'], ax=axs[1, 0], marker='o', color='red', linewidth=2)
        axs[1, 0].set_xlabel('Voltage (mV)')
        axs[1, 0].set_ylabel('Open Probability')
        axs[1, 0].set_title('Channel Gating Probability')
        axs[1, 0].set_ylim(0, 1.05)  # Slightly above 1 for visibility
    else:
        axs[1, 0].axis('off')

    # Log-scale current plot
    pos_currents = np.abs(currents)  # Absolute for log
    pos_mask = pos_currents > 0
    if np.any(pos_mask):
        axs[1, 1].semilogy(voltages_mv[pos_mask], pos_currents[pos_mask], 'co-', markersize=6)
        axs[1, 1].set_xlabel('Voltage (mV)')
        axs[1, 1].set_ylabel('|Current| (pA)')
        axs[1, 1].set_title('Log-Scale IV Curve')
    else:
        axs[1, 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_distribution(samples: torch.Tensor, save_path: Optional[str] = None,
                           title: str = 'State Distribution Analysis', figsize: Tuple[int, int] = (15, 10)):
    """
    Visualizes distributions and correlations in sampled states.
    Includes histograms for key dimensions, scatter plots for positions/correlations, energy distributions,
    and autocorrelation for temporal analysis. Handles tensor inputs, converting to numpy for plotting.

    Args:
        samples (torch.Tensor): Samples [num_samples, state_dim] (e.g., protein coords + ion pos/vel).
        save_path (Optional[str]): Path to save figure.
        title (str): Main figure title.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        None: Displays or saves the figure.

    Note: Adapted from attachment's visualize_distribution; enhances with Seaborn for histograms/scatters,
          dynamic subplots based on dim (e.g., assumes dim >=6 for ion-specific plots), and autocorrelation computation.
          Warns if num_samples <50 for poor statistics.
    """
    if samples.shape[0] < 50:
        print("Warning: Low number of samples; distributions may not be representative.")

    samples_np = samples.detach().cpu().numpy()  # Convert to numpy for plotting
    num_samples, state_dim = samples_np.shape

    fig, axs = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16)

    # Protein coordinate histogram (first dimension as example)
    sns.histplot(samples_np[:, 0], bins=50, kde=True, ax=axs[0, 0], color='blue', alpha=0.7)
    axs[0, 0].set_xlabel('Protein Coordinate 1')
    axs[0, 0].set_ylabel('Density')
    axs[0, 0].set_title('Protein State Distribution')

    # Ion radial position scatter (if dim >=6, assuming last 6: x,y,z,vx,vy,vz)
    if state_dim >= 6:
        ion_x = samples_np[:, -6]
        ion_y = samples_np[:, -5]
        sns.scatterplot(x=ion_x, y=ion_y, ax=axs[0, 1], alpha=0.6, s=50, color='green')
        axs[0, 1].set_xlabel('Ion X (m)')
        axs[0, 1].set_ylabel('Ion Y (m)')
        axs[0, 1].set_title('Ion Radial Position')

        # Ion axial histogram
        ion_z = samples_np[:, -4]
        sns.histplot(ion_z, bins=50, kde=True, ax=axs[0, 2], color='red', alpha=0.7)
        axs[0, 2].set_xlabel('Ion Z (m)')
        axs[0, 2].set_ylabel('Density')
        axs[0, 2].set_title('Ion Axial Distribution')

    # State correlation scatter (first two dimensions)
    if state_dim >= 2:
        sns.scatterplot(x=samples_np[:, 0], y=samples_np[:, 1], ax=axs[1, 0], alpha=0.6, s=50, color='purple')
        axs[1, 0].set_xlabel('Dimension 1')
        axs[1, 0].set_ylabel('Dimension 2')
        axs[1, 0].set_title('State Correlation')

    # Energy distribution (quadratic norm as proxy)
    energies = np.sum(samples_np**2, axis=1)
    sns.histplot(energies, bins=50, kde=True, ax=axs[1, 1], color='orange', alpha=0.7)
    axs[1, 1].set_xlabel('Energy (a.u.)')
    axs[1, 1].set_ylabel('Density')
    axs[1, 1].set_title('Energy Distribution')

    # Autocorrelation plot (for first dimension)
    if num_samples > 100:
        from electrodiffusion.utils.math import compute_autocorrelation  # Assume available in math.py
        autocorr = compute_autocorrelation(torch.tensor(samples_np[:, 0]), max_lag=50)
        sns.lineplot(x=np.arange(len(autocorr)), y=autocorr, ax=axs[1, 2], color='cyan')
        axs[1, 2].set_xlabel('Lag')
        axs[1, 2].set_ylabel('Autocorrelation')
        axs[1, 2].set_title('Temporal Correlation')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_trajectory(trajectory: torch.Tensor, save_path: Optional[str] = None,
                    labels: Optional[List[str]] = None, figsize: Tuple[int, int] = (12, 8)):
    """
    Plots time series trajectories for selected dimensions.
    Useful for visualizing SDE integrations or sampled paths over time.

    Args:
        trajectory (torch.Tensor): Trajectories [num_steps, batch_size, dim] or [num_steps, dim].
        save_path (Optional[str]): Save path.
        labels (Optional[List[str]]): Dimension labels (default: 'Dim 0', etc.).
        figsize (Tuple[int, int]): Figure size.

    Returns:
        None.

    Note: Extension not in attachment; plots first few dims or all if small dim, with legend and grid.
    """
    if trajectory.dim() == 2:  # [steps, dim] -> unsqueeze to [steps, 1, dim]
        trajectory = trajectory.unsqueeze(1)

    steps, batch_size, dim = trajectory.shape
    traj_np = trajectory.detach().cpu().numpy()

    plt.figure(figsize=figsize)
    for d in range(min(dim, 5)):  # Limit to 5 dims for clarity
        label = labels[d] if labels and d < len(labels) else f'Dim {d}'
        for b in range(min(batch_size, 3)):  # Plot up to 3 batches
            plt.plot(traj_np[:, b, d], label=f'{label} (Batch {b})', alpha=0.7)

    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Trajectory Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_convergence(errors: Dict[str, List[float]], save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (10, 6)):
    """
    Plots convergence curves (e.g., error vs. timestep) for solver analysis.

    Args:
        errors (Dict[str, List[float]]): Dict with method names as keys and error lists as values.
        save_path (Optional[str]): Save path.
        figsize (Tuple[int, int]): Size.

    Returns:
        None.

    Note: Inspired by Claude's convergence plots; uses log-log with reference lines (O(dt), O(dt²)).
    """
    plt.figure(figsize=figsize)
    for method, err_list in errors.items():
        dt_values = np.logspace(-4, -1, len(err_list))  # Example dt range; adjust as needed
        plt.loglog(dt_values, err_list, marker='o', label=method)

    # Reference lines
    dt_ref = np.logspace(-4, -1, 10)
    plt.loglog(dt_ref, dt_ref, 'k--', label='O(dt)')
    plt.loglog(dt_ref, dt_ref**2, 'k:', label='O(dt²)')

    plt.xlabel('Time Step (dt)')
    plt.ylabel('Error')
    plt.title('Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
