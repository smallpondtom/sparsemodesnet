import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.heat1d import generate_heat_data
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
    hidden_units_heat = [256, 128, 64, 32]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    lambdas_ss = np.logspace(-6, 0, 12)     # 12 values from 1e-6 to 1e0
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Heat Equation ----------
    X_heat, xspan_h, tspan_h = generate_heat_data(nx=2**7, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    
    # Create 3D surface plot for Heat Equation (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_h, tspan_h)
        Z_mesh = X_heat.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Heat Equation Solution')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('../figures/heat_data.png', dpi=300)
        # plt.show()
        plt.close(fig)

    model_heat, info_heat, selected_h, freq_tab = run_sparsemodesnet_with_lambda_selection(
        X_np            = X_heat,
        s               = s_h,
        hidden_units    = hidden_units_heat,
        M               = 0.1,
        lambda_method   = lambda_method,
        lam0            = 1e-6,         # only used if path
        epsilon         = 0.10,         # only used if path
        B_path          = 20,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 20,           # for cv
        lambdas_ss      = lambdas_ss,   # only used if stability
        B_ss            = 10,           # subsamples per λ for stability
        pi_thresh       = 0.6,          # threshold for stability
        num_epochs_sub  = 20,           # epochs per subsample for stability
        lr              = 1e-3,
        batch_size      = 16,
        device          = device,
        label           = "Heat Equation"
    )