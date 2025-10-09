"""
Ablation study with Advecting Gaussian Wave.
"""

#%% Load modules
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse
from QM.quadmani import quadmani_greedy, _cubic_mapping_jax
import sparsemodesnet as smn

def quadratic_mapping_numpy(x):
    """
    Numpy version - must match the torch version exactly!
    """
    if x.ndim == 1:
        n = x.shape[0]
        i_indices, j_indices = np.tril_indices(n)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        _, n = x.shape
        i_indices, j_indices = np.tril_indices(n)
        result = x[:, i_indices] * x[:, j_indices]
        return result

def _cubic_mapping_numpy(x):
    """
    Fast vectorized computation of unique cubic terms x ⊗ x ⊗ x (NumPy version).
    Uses meshgrid for efficient index generation.
    
    Args:
        x: np.ndarray of shape (batch_size, n) or (n,)
        
    Returns:
        np.ndarray of shape (batch_size, n*(n+1)*(n+2)//6) or (n*(n+1)*(n+2)//6,)
    """
    if x.ndim == 1:
        n = x.shape[0]
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products
        result = x[i_indices] * x[j_indices] * x[k_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products for all batches
        result = x[:, i_indices] * x[:, j_indices] * x[:, k_indices]
        return result
    


#%% %============================= Main Script ================================%
if __name__ == "__main__":
    # # Device selection: CUDA > MPS (Apple Silicon) > CPU
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    # else:
    #     device = 'cpu'
    device = 'cpu'
    print("Using device:", device)

    # For reproducibility
    torch.manual_seed(42)
    
    # number of grids
    n_grids = 2**10
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Advecting Pulse ----------
    X, xspan, tspan = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    d, n = X.shape
    s = min(d, n)
    s = 100
    r = 15
    p = int(r**2)
    
    # Create 3D surface plot for Advecting Pulse (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan, tspan)
        Z_mesh = X.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(
            X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Advecting Gaussian Pulse')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
        plt.close(fig)

    os.makedirs('results/ablation', exist_ok=True)

#%% %=============== Configuration of SparseModesNet (LassoNet) ===============%

    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': p,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [r, 500, p],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 2,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.0005,
        'lasso_epochs': 100,
        'M': 12.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'lasso_bias': True,
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
        'l1_only': False,
        'full_z': False,
        # Decoder Phase
        'decoder_lr': 1.0e-2,
        'decoder_lr_patience': 30,
        'decoder_epochs': 2000,
        'decoder_batch_size': 200,
        'decoder_optimizer': 'Adam',
        'decoder_momentum': 0.9,
        'decoder_bias': True,
        # General training
        'skip_sparse': False,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'device': device,
        # Experiment Setup
        'label': "Advecting Pulse",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    I_nn_standard = []
    re_standard = []
    for i in range(100):
        model_2, I_nn_2, omegas_2, path_history, rel_error = smn.fit(X, config)
        I_nn_standard.append(I_nn_2)
        re_standard.append(rel_error)
    np.savez(
        "results/ablation/ablation_results_standard.npz", 
        I_nn=I_nn_standard, re=re_standard
    )


#%% %================ Configuration of SparseModesNet (l1-only) ===============%

    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': p,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [r, 500, p],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 2,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 1.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.0005,
        'lasso_epochs': 100,
        'M': 12.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'lasso_bias': True,
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
        'l1_only': True,
        'full_z': False,
        # Decoder Phase
        'decoder_lr': 1.0e-2,
        'decoder_lr_patience': 30,
        'decoder_epochs': 2000,
        'decoder_batch_size': 200,
        'decoder_optimizer': 'Adam',
        'decoder_momentum': 0.9,
        'decoder_bias': True,
        # General training
        'skip_sparse': False,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'device': device,
        # Experiment Setup
        'label': "Advecting Pulse",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    I_nn_l1_only = []
    re_l1_only = []
    for i in range(100):
        model_2, I_nn_2, omegas_2, path_history, rel_error = smn.fit(X, config)
        I_nn_l1_only.append(I_nn_2)
        re_l1_only.append(rel_error)
    np.savez(
        "results/ablation/ablation_results_l1_only.npz", 
        I_nn=I_nn_l1_only, re=re_l1_only
    )


#%% %================ Configuration of SparseModesNet (z_batch) ===============%

    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': p,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [r, 500, p],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 2,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.0005,
        'lasso_epochs': 100,
        'M': 12.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'lasso_bias': True,
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
        'l1_only': False,
        'full_z': True,
        # Decoder Phase
        'decoder_lr': 1.0e-2,
        'decoder_lr_patience': 30,
        'decoder_epochs': 2000,
        'decoder_batch_size': 200,
        'decoder_optimizer': 'Adam',
        'decoder_momentum': 0.9,
        'decoder_bias': True,
        # General training
        'skip_sparse': False,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'device': device,
        # Experiment Setup
        'label': "Advecting Pulse",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    I_nn_full_z = []
    re_full_z = []
    for i in range(100):
        model_2, I_nn_2, omegas_2, path_history, rel_error = smn.fit(X, config)
        I_nn_full_z.append(I_nn_2)
        re_full_z.append(rel_error)
    np.savez(
        "results/ablation/ablation_results_full_z.npz", 
        I_nn=I_nn_full_z, re=re_full_z
    )

#%% %======================= Plot the results of study ========================%
    # Load the saved results
    foo = np.load("results/ablation/ablation_results_standard.npz")
    I_nn_standard = foo['I_nn']
    re_standard = foo['re']

    foo  = np.load("results/ablation/ablation_results_l1_only.npz")
    I_nn_l1_only  = foo['I_nn']
    re_l1_only  = foo['re']

    foo  = np.load("results/ablation/ablation_results_full_z.npz")
    I_nn_full_z  = foo['I_nn']
    re_full_z  = foo['re']

    #%% Create figures directory if it doesn't exist
    import os
    os.makedirs('figures/ablation', exist_ok=True)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": "Ubuntu",
        "font.monospace": "Ubuntu Mono",
        "axes.labelweight": "bold",
    })

#%% Plot 1: Box plot of relative errors
    fig, ax = plt.subplots(figsize=(12, 6))

    # Prepare data for box plot
    data = [re_standard, re_l1_only, re_full_z]
    labels = [
        r'SparseModesNet (Proposed)', 
        r'$\ell_1$ penalty only', 
        r'LassoNet with $h_{\mathrm{NN}}(\mathbf{z})$',
    ]

    # Create horizontal box plot
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, 
                    showmeans=True, meanline=True, vert=False)

    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add individual points (jittered for visibility)
    for i, (values, label) in enumerate(zip(data, labels)):
        y = np.random.normal(i+1, 0.04, size=len(values))
        ax.scatter(values, y, alpha=0.65, s=20, color='darkgray')

    # Set x-axis to log scale
    ax.set_xscale('log')

    # Set x-limits
    ax.set_xlim(1e-9, 1e-4)


    # Customize the plot
    ax.set_xlabel('Relative Reconstruction Error', fontsize=20)
    ax.set_ylabel('', fontsize=20)
    ax.set_title('Reconstruction Error Comparison Across Mode Selection Variants',
                 fontsize=22)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(True, alpha=0.3)

    # Add statistics as text above each box
    for i, (values, label) in enumerate(zip(data, labels)):
        mean_val = np.mean(values)
        std_val = np.std(values)
        # Position text above each box
        ax.text(np.median(values), i+1+0.25, 
                rf'$\mu={mean_val:.2e}$' + '\n' + rf'$\sigma={std_val:.2e}$', 
                ha='center', va='bottom', fontsize=13, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/ablation/relative_error_boxplot.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()

    #%% Plot 2: Bar plots of index selection frequency
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Function to count index frequencies
    imax = 60
    def count_index_frequencies(I_nn_list, max_index=imax):
        counts = np.zeros(max_index)
        for I_nn in I_nn_list:
            for idx in I_nn:
                if idx < max_index:
                    counts[idx] += 1
        return counts

    # Calculate frequencies for each case
    freq_standard = count_index_frequencies(I_nn_standard)
    freq_l1_only = count_index_frequencies(I_nn_l1_only)
    freq_full_z = count_index_frequencies(I_nn_full_z)

    # Create bar plots
    x_indices = np.arange(1, imax+1)  # 1-100 on x-axis

    # Standard LassoNet
    axes[0].bar(x_indices, freq_standard, color='lightblue', alpha=0.9)
    axes[0].set_title(
        r'SparseModesNet (Proposed)', 
        fontsize=30)
    axes[0].set_xlabel('Mode Index', fontsize=24)
    axes[0].set_ylabel('Number of Times Selected', fontsize=24)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, imax+1)

    # L1-only
    axes[1].bar(x_indices, freq_l1_only, color='lightgreen', alpha=0.9)
    axes[1].set_title(r'$\ell_1$ penalty only', fontsize=30)
    axes[1].set_xlabel('Mode Index', fontsize=24)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, imax+1)

    # Z-batch (full_z)
    axes[2].bar(x_indices, freq_full_z, color='lightcoral', alpha=0.9)
    axes[2].set_title(r'LassoNet with $h_{\mathrm{NN}}(\mathbf{z})$', fontsize=30)
    axes[2].set_xlabel('Mode Index', fontsize=24)
    axes[2].tick_params(axis='both', which='major', labelsize=20)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, imax+1)

    plt.tight_layout()
    plt.savefig('figures/ablation/index_selection_frequency.pdf', dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("Summary Statistics:")
    print("-" * 50)
    for name, re_vals in zip(['LassoNet', 'l1-only', 'z-batch'], 
                            [re_standard, re_l1_only, re_full_z]):
        print(f"{name}:")
        print(f"  Mean relative error: {np.mean(re_vals):.4e}")
        print(f"  Std relative error:  {np.std(re_vals):.4e}")
        print(f"  Min relative error:  {np.min(re_vals):.4e}")
        print(f"  Max relative error:  {np.max(re_vals):.4e}")
        print()

    print("Index Selection Statistics:")
    print("-" * 50)
    for name, freq in zip(['LassoNet', 'l1-only', 'z-batch'], 
                        [freq_standard, freq_l1_only, freq_full_z]):
        selected_indices = np.where(freq > 0)[0]
        print(f"{name}:")
        print(f"  Number of unique indices selected: {len(selected_indices)}")
        print(f"  Most frequently selected index: {np.argmax(freq) + 1} ({int(np.max(freq))} times)")
        print(f"  Average selection frequency: {np.mean(freq[freq > 0]):.2f}")
        print()
# %%
