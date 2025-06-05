import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.burgers import generate_burgers_data
from sparsemodesnet import run_sparsemodesnet_with_lambda_selection

if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)
    
    # Regularization parameter selection method
    lambda_method = 'cv'  # 'path', 'cv', or 'stability'

    # Common hyperparameters
    hidden_units_burg = [128, 64, 32]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    lambdas_ss = np.logspace(-6, 0, 12)     # 12 values from 1e-6 to 1e0
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Burgers' Equation ----------
    X_burgers, xspan_b, tspan_b = generate_burgers_data(nx=2**7, nt=1000, nu=0.01, x_max=1.0, t_max=1.0)
    d_b, n_b = X_burgers.shape
    s_b = min(d_b, n_b)
    
    # Create 3D surface plot for Burgers' Equation (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_b, tspan_b)
        Z_mesh = X_burgers.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title("Burgers' Equation Solution")
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('../figures/burgers_data.png', dpi=300)
        # plt.show()
        plt.close(fig)

    model_burg, info_burg, selected_b, freq_tab = run_sparsemodesnet_with_lambda_selection(
        X_np            = X_burgers,
        s               = s_b,
        hidden_units    = hidden_units_burg,
        M               = 10.0,
        lambda_method   = lambda_method,
        lam0            = 1e-6,
        epsilon         = 0.10,
        B_path          = 20,
        max_iters       = 100,
        lambdas_cv      = lambdas_cv,
        k_folds         = 5,
        num_epochs_cv   = 20,
        lambdas_ss      = lambdas_ss,
        B_ss            = 50,
        pi_thresh       = 0.6,
        num_epochs_sub  = 20,
        lr              = 1e-3,
        batch_size      = 16,
        device          = device,
        label           = "Burgers' Equation"
    )
