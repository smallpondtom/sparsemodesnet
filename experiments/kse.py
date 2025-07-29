#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.kse import generate_kse_data
from sparsemodesnet import run_sparsemodesnet
from sparsemodesnet.linalg.pod import compute_pod_basis

#%%
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
    reg_path = 'dense2sparse'
    
    # Number of spatial grids
    n_grids = 2**10  

    # Reduced dimension 
    s = 100
    
    # Common hyperparameters
    hidden_units_ks   = [s, int(s * (s + 1) / 2)]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Kuramoto–Sivashinsky Equation ----------
    # Note: smaller nt for speed, adjust as desired
    X_ks, xspan_ks, tspan_ks = generate_kse_data(
        nx=n_grids, nt=5000, L=32*np.pi, t_max=150.0)
    d_ks, n_ks = X_ks.shape
    s_ks = min(d_ks, n_ks, s)
    
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

    #%%
    model_ks, info_ks, selected_ks, freq_tab = run_sparsemodesnet(
        X_np            = X_ks,
        s               = s_ks,
        hidden_units    = hidden_units_ks,
        M               = 10.0,
        reg_path        = reg_path,
        lr              = 1e-3,
        batch_size      = 256,
        knee_method     = 'zmethod',
        optimizer       = 'Adam',
        nonzero_thresh  = 1e-14,
        r_max           = 25,           # max modes for constraint stopping
        lam0            = 1e-3,         # only used if path
        epsilon         = 0.20,         # only used if path
        B_path          = 80,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 80,           # for cv
        device          = device,
        label           = "Kuramoto-Sivashinsky Equation"
    )
    
    #%% Plot the first 20 modes of the POD b# recompute just the first 20 POD modes
    U_s20, _, _ = compute_pod_basis(X_ks, s=s_ks)
    fig, axes = plt.subplots(4, 5, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(xspan_ks, U_s20[:, i])
        ax.set_title(f"Mode {i+1}")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('../figures/kse_pod_modes.png', dpi=300)
    plt.show()
    
    #%% Plot the POD modes vs the reconstruction error 
    U, S, _ = np.linalg.svd(X_ks, full_matrices=False)
    Us_20 = U[:, :500].astype(np.float64)  
    fig, ax = plt.subplots(figsize=(8, 6))
    proj_err = []
    X_heat_f64 = X_ks.astype(np.float64)
    Us_20_f64 = Us_20.astype(np.float64)
    for i in range(500):
        proj_err.append(
            np.linalg.norm(
                X_heat_f64 - Us_20_f64[:, :i+1] 
                @ (Us_20_f64[:, :i+1].T @ X_heat_f64), 'fro') 
            / np.linalg.norm(X_heat_f64, 'fro')
        )
    ax.semilogy(range(1, 500+1), proj_err)
    ax.set_xlabel('Number of POD Modes')
    ax.set_ylabel('projection error (relative)')
    ax.set_title(f'POD Mode {i+1} vs Projection Errors')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figures/kse_pod_mode_vs_recon.png', dpi=300)
    plt.show()
    
    
    #%% Plot the λ vs selected modes and λ vs relative error
    if reg_path == 'dense2sparse':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        # Extract data
        lambdas    = [freq['lambda']        for freq in freq_tab]
        num_modes  = [freq['nonzero_count'] for freq in freq_tab]
        rel_errors = [freq['error']         for freq in freq_tab]
        # Plot 1: λ vs relative error
        ax1.loglog(
            lambdas, rel_errors, 'o-', markersize=8, linewidth=2, color='red')
        ax1.set_xlabel('Regularization Parameter (λ)', fontsize=16)
        ax1.set_ylabel('Relative Error', fontsize=16)
        ax1.set_title('λ vs Relative Error', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.grid(True, alpha=0.3)
        # Plot 2: λ vs selected modes
        ax2.semilogx(
            lambdas, num_modes, 'o-', markersize=8, linewidth=2, color='blue')
        ax2.set_xlabel('Regularization Parameter (λ)', fontsize=16)
        ax2.set_ylabel('Number of POD Modes', fontsize=16)
        ax2.set_title('λ vs # Modes', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.grid(True, alpha=0.3)
        # Plot 3: # Modes vs Relative Error
        ax3.semilogy(
            num_modes, rel_errors, 'o-', markersize=8, linewidth=2, color='green')
        ax3.set_xlabel('Number of POD Modes', fontsize=16)
        ax3.set_ylabel('Relative Error', fontsize=16)
        ax3.set_title('# Modes vs Relative Error', fontsize=18)
        ax3.tick_params(axis='both', which='major', labelsize=14)
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('../figures/kse_path_summary.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    
    #%% Plot L-curve
    if reg_path == 'dense2sparse':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('L1 Regularization Term (||ω||₁)')
        ax.set_ylabel('Relative Error')
        ax.set_title('L-curve for Heat Equation')
        ax.grid(True, alpha=0.3)
        for freq in freq_tab:
            ax.loglog(freq['l1_b'], freq['error'], 'o-', markersize=6, linewidth=2)
        plt.tight_layout()
        plt.savefig('../figures/kse_lcurve.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    #%% Plot the λ vs selected modes and λ vs relative error
    if reg_path == 'dense2sparse':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        # Extract data
        lambdas = [freq['lambda'] for freq in freq_tab]
        num_modes = [freq['nonzero_count'] for freq in freq_tab]
        rel_errors = [freq['error'] for freq in freq_tab]
        # Plot 1: λ vs relative error
        ax1.loglog(lambdas, rel_errors, 'o-', markersize=8, linewidth=2, color='red')
        ax1.set_xlabel('Regularization Parameter (λ)', fontsize=16)
        ax1.set_ylabel('Relative Error', fontsize=16)
        ax1.set_title('λ vs Relative Error', fontsize=18)
        ax1.tick_params(axis='both', which='major', labelsize=14)
        ax1.grid(True, alpha=0.3)
        # Plot 2: λ vs selected modes
        ax2.semilogx(lambdas, num_modes, 'o-', markersize=8, linewidth=2, color='blue')
        ax2.set_xlabel('Regularization Parameter (λ)', fontsize=16)
        ax2.set_ylabel('Number of Selected Modes', fontsize=16)
        ax2.set_title('λ vs Number of Selected Modes', fontsize=18)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        ax2.grid(True, alpha=0.3)
        # Plot 3: Superimposed plot with dual y-axes
        color1 = 'blue'
        color2 = 'red'
        ax3.set_xlabel('Regularization Parameter (λ)', fontsize=16)
        ax3.set_ylabel('Number of Selected Modes', color=color1, fontsize=16)
        line1 = ax3.semilogx(lambdas, num_modes, 'o-', markersize=8, 
                             linewidth=2, color=color1, label='Selected Modes')
        ax3.tick_params(axis='both', which='major', 
                        labelsize=14, labelcolor='black')
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, alpha=0.3)
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Relative Error', color=color2, fontsize=16)
        line2 = ax3_twin.loglog(lambdas, rel_errors, 's-', markersize=8, 
                                linewidth=2, color=color2, label='Relative Error')
        ax3_twin.tick_params(axis='y', labelcolor=color2, labelsize=14)
        ax3.set_title('λ vs Selected Modes & Relative Error', fontsize=18)
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, loc='center left', fontsize=16)
        plt.tight_layout()
        plt.savefig('../figures/kse_lambda_analysis.png', dpi=300)
        plt.show()
        plt.close(fig)
        

    #%% Plot the reconstructed flow fields (heatmap)
    V, _, _ = np.linalg.svd(X_ks, full_matrices=False)
    V_selected = V[:, selected_ks]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_ks
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, :s_ks].T @ X_ks).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_ks.eval()
        _, X_sparse_recon_tensor = model_ks(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
    # Calculate errors
    pod_error = X_ks - X_pod_recon
    sparse_error = X_ks - X_sparse_recon
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')
    
    # (1,2) POD error
    im2 = axes[0,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Space (x)')
    axes[0,1].set_title('POD Error')
    plt.colorbar(im2, ax=axes[0,1], label='Error')
    
    # (2,1) Sparse reconstruction
    im3 = axes[1,0].imshow(
        X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('Sparse Reconstruction')
    plt.colorbar(im3, ax=axes[1,0], label='u(x,t)')
    
    # (2,2) Sparse error
    im4 = axes[1,1].imshow(
        sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Space (x)')
    axes[1,1].set_title('Sparse Error')
    plt.colorbar(im4, ax=axes[1,1], label='Error')
    
    plt.tight_layout()
    plt.savefig('../figures/kse_comparison.png', dpi=300)
    plt.show()
    plt.close(fig)

    #%% Test the model on a new sample
    # Generate new test data with different parameters
    X_test, xspan_test, tspan_test = generate_kse_data(
        nx=n_grids, nt=5000, L=32*np.pi, t_max=150.0)

    # Project test data onto the learned POD basis
    V_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
    Z_test = torch.from_numpy(
        (V[:, :s_ks].T @ X_test).T.astype(np.float32)).to(device)

    # POD reconstruction using selected modes
    X_test_pod_recon = V_selected @ V_selected.T @ X_test

    # Test the model
    with torch.no_grad():
        model_ks.eval()
        _, X_test_recon_tensor = model_ks(Z_test)
        X_test_recon = X_test_recon_tensor.cpu().numpy().T

    # Calculate reconstruction errors
    test_error_sparse = X_test - X_test_recon
    test_error_pod = X_test - X_test_pod_recon
    relative_error_sparse = np.linalg.norm(test_error_sparse, 'fro') / np.linalg.norm(X_test, 'fro')
    relative_error_pod = np.linalg.norm(test_error_pod, 'fro') / np.linalg.norm(X_test, 'fro')
    print(f"Test reconstruction relative error (Sparse): {relative_error_sparse:.4f}")
    print(f"Test reconstruction relative error (POD): {relative_error_pod:.4f}")

    # Plot test results
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Original test data
    im1 = axes[0,0].imshow(
        X_test, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('Test Data (Original)')
    plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')

    # POD reconstruction
    im2 = axes[0,1].imshow(
        X_test_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Space (x)')
    axes[0,1].set_title('Test Data (POD Reconstruction)')
    plt.colorbar(im2, ax=axes[0,1], label='u(x,t)')

    # POD error
    im3 = axes[0,2].imshow(
        test_error_pod, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
    axes[0,2].set_xlabel('Time')
    axes[0,2].set_ylabel('Space (x)')
    axes[0,2].set_title('POD Test Error')
    plt.colorbar(im3, ax=axes[0,2], label='Error')

    # Sparse reconstruction
    im4 = axes[1,1].imshow(
        X_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Space (x)')
    axes[1,1].set_title('Test Data (Sparse Reconstruction)')
    plt.colorbar(im4, ax=axes[1,1], label='u(x,t)')

    # Sparse error
    im5 = axes[1,2].imshow(
        test_error_sparse, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_ylabel('Space (x)')
    axes[1,2].set_title('Sparse Test Error')
    plt.colorbar(im5, ax=axes[1,2], label='Error')

    # Hide empty subplot
    axes[1,0].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/kse_test_results.png', dpi=300)
    plt.show()
    plt.close(fig)
# %%
