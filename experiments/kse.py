import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.kse import generate_kse_data
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
    hidden_units_ks   = [256, 128, 64, 32]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    lambdas_ss = np.logspace(-6, 0, 12)     # 12 values from 1e-6 to 1e0
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Kuramoto–Sivashinsky Equation ----------
    # Note: smaller nt for speed, adjust as desired
    X_ks, xspan_ks, tspan_ks = generate_kse_data(nx=2**10, nt=1000, L=100.0, t_max=100.0)
    d_ks, n_ks = X_ks.shape
    s_ks = min(d_ks, n_ks)
    
    # Create flow-field for Kuramoto-Sivashinsky Equation
    if sanity_check:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            X_ks, aspect='auto', cmap='viridis', origin='lower',
            extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
        ax.set_xlabel('Time')
        ax.set_ylabel('Space (x)')
        ax.set_title('Kuramoto-Sivashinsky Equation Solution')
        plt.colorbar(im, ax=ax, label='u(x,t)')
        plt.tight_layout()
        plt.savefig('../figures/kse_data.png', dpi=300)
        plt.show()
        plt.close(fig)

    model_ks, info_ks, selected_ks, freq_tab = run_sparsemodesnet_with_lambda_selection(
        X_np            = X_ks,
        s               = s_ks,
        hidden_units    = hidden_units_ks,
        M               = 1.0,
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
        label           = "Kuramoto-Sivashinsky Equation"
    )