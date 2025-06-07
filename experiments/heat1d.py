#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.heat1d import generate_heat_data
from sparsemodesnet import run_sparsemodesnet_with_lambda_selection

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
    lambda_method = 'stability'  # 'path', 'cv', or 'stability'
    
    # Stopping criterion for the regularization path
    stop_method = 'constraint'

    # Common hyperparameters
    hidden_units_heat = [128, 8256]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)      # 10 values from 1e-6 to 1e-2
    lambdas_ss = np.logspace(-2.1, -1.2, 12)  # 12 values from 1e-6 to 1e0
    
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
        plt.show()
        plt.close(fig)

    model_heat, info_heat, selected_h, freq_tab = run_sparsemodesnet_with_lambda_selection(
        X_np            = X_heat,
        s               = s_h,
        hidden_units    = hidden_units_heat,
        M               = 2.0,
        lambda_method   = lambda_method,
        lam0            = 1e-6,         # only used if path
        epsilon         = 0.10,         # only used if path
        B_path          = 100,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 20,           # for cv
        lambdas_ss      = lambdas_ss,   # only used if stability
        B_ss            = 10,           # subsamples per λ for stability
        pi_thresh       = 0.7,          # threshold for stability
        num_epochs_sub  = 80,           # epochs per subsample for stability
        stop_method     = stop_method,
        aic_alpha       = 0.5,          # significance level for AIC
        lr              = 1e-3,
        optimizer       = 'Adam',
        batch_size      = 32,
        device          = device,
        label           = "Heat Equation"
    )

    #%% Plot the reconstructed flow fields (heatmap)
    V, _, _ = np.linalg.svd(X_heat, full_matrices=False)
    V_selected = V[:, selected_h]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_heat
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy((V[:, :s_h].T @ X_heat).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_heat.eval()
        _, X_sparse_recon_tensor = model_heat(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
    # Calculate errors
    pod_error = X_heat - X_pod_recon
    sparse_error = X_heat - X_sparse_recon
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')
    
    # (1,2) POD error
    im2 = axes[0,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Space (x)')
    axes[0,1].set_title('POD Error')
    plt.colorbar(im2, ax=axes[0,1], label='Error')
    
    # (2,1) Sparse reconstruction
    im3 = axes[1,0].imshow(
        X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('Sparse Reconstruction')
    plt.colorbar(im3, ax=axes[1,0], label='u(x,t)')
    
    # (2,2) Sparse error
    im4 = axes[1,1].imshow(
        sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Space (x)')
    axes[1,1].set_title('Sparse Error')
    plt.colorbar(im4, ax=axes[1,1], label='Error')
    
    plt.tight_layout()
    plt.savefig('../figures/heat_comparison.png', dpi=300)
    plt.show()
    plt.close(fig)

    #%% Test the model on a new sample
    # Generate new test data with different parameters
    X_test, xspan_test, tspan_test = generate_heat_data(nx=2**7, nt=800, alpha=0.02, x_max=1.0, t_max=0.8)

    # Project test data onto the learned POD basis
    V_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
    Z_test = torch.from_numpy((V[:, :s_h].T @ X_test).T.astype(np.float32)).to(device)

    # POD reconstruction using selected modes
    X_test_pod_recon = V_selected @ V_selected.T @ X_test

    # Test the model
    with torch.no_grad():
        model_heat.eval()
        _, X_test_recon_tensor = model_heat(Z_test)
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
    plt.savefig('../figures/heat_test_results.png', dpi=300)
    plt.show()
    plt.close(fig)
# %%
