#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse
from sparsemodesnet import run_sparsemodesnet
from sparsemodesnet.pod import compute_pod_basis

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
    reg_path = 'dense2sparse'  # 'dense2sparse' or 'cv'
    
    # Common hyperparameters
    # hidden_units_pulse = [128, 8256]
    hidden_units_pulse = [100, 2000, 5050]
    
    # Parameter‐grid for CV
    lambdas_cv = np.logspace(-2.1, -0.8, 12)  
    
    # number of grids
    n_grids = 2**10
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Advecting Pulse ----------
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=2.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    d_p, n_p = X_pulse.shape
    s_p = min(d_p, n_p)
    s_p = 100
    
    ## Create 3D surface plot for Advecting Pulse (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_p, tspan_p)
        Z_mesh = X_pulse.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(
            X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Advecting Gaussian Pulse')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('../figures/pulse_data.png', dpi=300)
        plt.show()
        plt.close(fig)

    #%% Train
    model_pulse, info_pulse, selected_p, freq_tab = run_sparsemodesnet(
        X_np            = X_pulse,
        s               = s_p,
        hidden_units    = hidden_units_pulse,
        M               = 10.0,
        reg_path        = reg_path,
        lr              = 1e-3,
        batch_size      = 256,
        knee_method     = 'zmethod',
        optimizer       = 'Adam',
        nonzero_thresh  = 1e-8,
        r_max           = 20,          # max modes for constraint stopping
        lam0            = 1e-1,         # only used if path
        epsilon         = 0.05,         # only used if path
        B_path          = 160,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 80,           # for cv
        device          = device,
        label           = "Advecting Pulse"
    )
    
    #%% Plot the first 20 modes of the POD basis
    U_s20, _, _ = compute_pod_basis(X_pulse, s=20)
    fig, axes = plt.subplots(4, 5, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(xspan_p, U_s20[:, i])
        ax.set_title(f"Mode {i+1}")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('../figures/pulse_pod_modes.png', dpi=300)
    plt.show()
    
    #%% Plot the POD modes vs the reconstruction error 
    U, S, _ = np.linalg.svd(X_pulse, full_matrices=False)
    Us_20 = U[:, :s_p].astype(np.float64)  # First s_p POD modes
    fig, ax = plt.subplots(figsize=(8, 6))
    proj_err = []
    X_pulse_f64 = X_pulse.astype(np.float64)
    Us_20_f64 = Us_20.astype(np.float64)
    for i in range(s_p):
        proj_err.append(
            np.linalg.norm(
                X_pulse_f64 - Us_20_f64[:, :i+1] 
                @ (Us_20_f64[:, :i+1].T @ X_pulse_f64), 'fro') 
            / np.linalg.norm(X_pulse_f64, 'fro')
        )
    ax.semilogy(range(1, s_p+1), proj_err)
    ax.set_xlabel('Number of POD Modes')
    ax.set_ylabel('Projection Error (Relative)')
    ax.set_title('POD Modes vs Projection Errors')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('../figures/pulse_pod_mode_vs_recon.png', dpi=300)
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
        plt.savefig('../figures/pulse_path_summary.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    
    #%% Plot L-curve
    if reg_path == 'dense2sparse':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('L1 Regularization Term (||ω||₁)')
        ax.set_ylabel('Relative Error')
        ax.set_title('L-curve for Advecting Pulse')
        ax.grid(True, alpha=0.3)
        for freq in freq_tab:
            ax.loglog(freq['l1_b'], freq['error'], 'o-', markersize=6, linewidth=2)
        plt.tight_layout()
        plt.savefig('../figures/pulse_lcurve.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    #%% Plot the λ vs selected modes and λ vs relative error with dual axes
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
        plt.savefig('../figures/pulse_lambda_analysis.png', dpi=300)
        plt.show()
        plt.close(fig)

    #%% Plot the reconstructed flow fields (heatmap)
    V, _, _ = np.linalg.svd(X_pulse, full_matrices=False)
    V_selected = V[:, selected_p]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_pulse
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, :s_p].T @ X_pulse).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_pulse.eval()
        _, X_sparse_recon_tensor = model_pulse(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
    # Calculate errors
    pod_error = X_pulse - X_pod_recon
    sparse_error = X_pulse - X_sparse_recon
    
    # Calculate common error scale
    error_max = max(np.abs(pod_error).max(), np.abs(sparse_error).max())
    error_min = -error_max
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')
    
    # (1,2) POD error
    im2 = axes[0,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Space (x)')
    axes[0,1].set_title('POD Error')
    plt.colorbar(im2, ax=axes[0,1], label='Error')
    
    # (2,1) Sparse reconstruction
    im3 = axes[1,0].imshow(
        X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('Sparse Reconstruction')
    plt.colorbar(im3, ax=axes[1,0], label='u(x,t)')
    
    # (2,2) Sparse error
    im4 = axes[1,1].imshow(
        sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Space (x)')
    axes[1,1].set_title('Sparse Error')
    plt.colorbar(im4, ax=axes[1,1], label='Error')
    
    plt.tight_layout()
    plt.savefig('../figures/pulse_comparison.png', dpi=300)
    plt.show()
    plt.close(fig)
    
    #%% Plot waves at specific time points
    # Select 3 equally spaced time points
    n_times = len(tspan_p)
    time_indices = [n_times//4, n_times//2, 3*n_times//4]
    time_points = [tspan_p[i] for i in time_indices]

    # Get reconstructions
    V, _, _ = np.linalg.svd(X_pulse, full_matrices=False)
    V_selected = V[:, selected_p]
    X_pod_recon = V_selected @ V_selected.T @ X_pulse

    Z_input = torch.from_numpy(
        (V[:, :s_p].T @ X_pulse).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_pulse.eval()
        _, X_sparse_recon_tensor = model_pulse(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices, time_points)):
        # Plot original data
        ax.plot(xspan_p, X_pulse[:, t_idx], 'k-', linewidth=2, label='Original', alpha=0.8)
        # Plot POD reconstruction
        ax.plot(xspan_p, X_pod_recon[:, t_idx], 'b--', linewidth=2, label='POD', alpha=0.8)
        # Plot sparse reconstruction
        ax.plot(xspan_p, X_sparse_recon[:, t_idx], 'r:', linewidth=2, label='SparseModesNet', alpha=0.8)
        
        ax.set_xlabel('Space (x)', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f't = {t_val:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('../figures/pulse_waves_timepoints.png', dpi=300)
    plt.show()
    plt.close(fig)
    
    #%% Detailed reconstruction comparison with linear and nonlinear components
    V, _, _ = np.linalg.svd(X_pulse, full_matrices=False)
    V_selected = V[:, selected_p]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_pulse
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, :s_p].T @ X_pulse).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_pulse.eval()
        
        # Linear part
        omega = np.diag(model_pulse.omega.cpu().numpy())
        X_sparse_lin = V[:, :s_p] @ omega @ V[:, :s_p].T @ X_pulse
        omega_tensor = torch.from_numpy(omega).to(device)
        
        # Nonlinear part 
        nonlin_part = model_pulse.net(Z_input @ omega_tensor)
        X_sparse_nonlin = nonlin_part.cpu().numpy().T
        
        # Together 
        _, X_sparse_recon_tensor = model_pulse(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
    # Calculate errors
    pod_error = X_pulse - X_pod_recon
    sparse_error = X_pulse - X_sparse_recon
    sparse_lin_error = X_pulse - X_sparse_lin
    sparse_nonlin_error = X_pulse - X_sparse_nonlin
    
    # Calculate common error scale for all error plots
    error_max = max(np.abs(pod_error).max(), np.abs(sparse_error).max(), 
                    np.abs(sparse_lin_error).max(), np.abs(sparse_nonlin_error).max())
    error_min = -error_max
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')
    
    # (1,2) POD error
    im2 = axes[0,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Space (x)')
    axes[0,1].set_title('POD Error')
    plt.colorbar(im2, ax=axes[0,1], label='Error')
    
    # (2,1) Sparse reconstruction
    im3 = axes[1,0].imshow(
        X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('Sparse Reconstruction')
    plt.colorbar(im3, ax=axes[1,0], label='u(x,t)')
    
    # (2,2) Sparse error
    im4 = axes[1,1].imshow(
        sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Space (x)')
    axes[1,1].set_title('Sparse Error')
    plt.colorbar(im4, ax=axes[1,1], label='Error')
    
    # (3,1) Sparse linear reconstruction
    im5 = axes[2,0].imshow(
        X_sparse_lin, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[2,0].set_xlabel('Time')
    axes[2,0].set_ylabel('Space (x)')
    axes[2,0].set_title('Sparse Linear Reconstruction')
    plt.colorbar(im5, ax=axes[2,0], label='u(x,t)')
    
    # (3,2) Sparse linear error
    im6 = axes[2,1].imshow(
        sparse_lin_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[2,1].set_xlabel('Time')
    axes[2,1].set_ylabel('Space (x)')
    axes[2,1].set_title('Sparse Linear Error')
    plt.colorbar(im6, ax=axes[2,1], label='Error')
    
    # (4,1) Sparse nonlinear reconstruction
    im7 = axes[3,0].imshow(
        X_sparse_nonlin, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]])
    axes[3,0].set_xlabel('Time')
    axes[3,0].set_ylabel('Space (x)')
    axes[3,0].set_title('Sparse Nonlinear Reconstruction')
    plt.colorbar(im7, ax=axes[3,0], label='u(x,t)')
    
    # (4,2) Sparse nonlinear error
    im8 = axes[3,1].imshow(
        sparse_nonlin_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_min, vmax=error_max)
    axes[3,1].set_xlabel('Time')
    axes[3,1].set_ylabel('Space (x)')
    axes[3,1].set_title('Sparse Nonlinear Error')
    plt.colorbar(im8, ax=axes[3,1], label='Error')
    
    plt.tight_layout()
    plt.savefig('../figures/pulse_comparison_separated.png', dpi=300)
    plt.show()
    plt.close(fig)
    
    # Plot wave at specific time points with linear and nonlinear components
    # Select 3 equally spaced time points
    n_times = len(tspan_p)
    time_indices = [n_times//4, n_times//2, 3*n_times//4]
    time_points = [tspan_p[i] for i in time_indices]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices, time_points)):
        # Plot original data
        ax.plot(xspan_p, X_pulse[:, t_idx], 'k-', linewidth=2, label='Original', alpha=0.8)
        # Plot POD reconstruction
        ax.plot(xspan_p, X_pod_recon[:, t_idx], 'b--', linewidth=2, label='POD', alpha=0.8)
        # Plot sparse linear reconstruction
        ax.plot(xspan_p, X_sparse_lin[:, t_idx], 'g:', linewidth=2, label='Sparse Linear', alpha=0.8)
        # Plot sparse nonlinear reconstruction
        ax.plot(xspan_p, X_sparse_nonlin[:, t_idx], 'r:', linewidth=2, label='Sparse Nonlinear', alpha=0.8)
        # Plot full sparse reconstruction
        ax.plot(xspan_p, X_sparse_recon[:, t_idx], 'm-.', linewidth=2, label='Sparse Full', alpha=0.8)
        
        ax.set_xlabel('Space (x)', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f't = {t_val:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('../figures/pulse_waves_timepoints_separated.png', dpi=300)
    plt.show()
    plt.close(fig)
    
    
    #%% Test the model on a new sample
    # Generate new test data with different parameters (different speed)
    X_test, xspan_test, tspan_test = generate_advecting_pulse(
        pulse_width=1.5e-4,
        pulse_shift=0.15,
        speed=7.0,
        final_time=0.12,
        n_time_samples=800,
        n_space_samples=n_grids
    )

    # Project test data onto the learned POD basis
    V_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
    Z_test = torch.from_numpy(
        (V[:, :s_p].T @ X_test).T.astype(np.float32)).to(device)

    # POD reconstruction using selected modes
    X_test_pod_recon = V_selected @ V_selected.T @ X_test

    # Test the model
    with torch.no_grad():
        model_pulse.eval()
        _, X_test_recon_tensor = model_pulse(Z_test)
        X_test_recon = X_test_recon_tensor.cpu().numpy().T

    # Calculate reconstruction errors
    test_error_sparse = X_test - X_test_recon
    test_error_pod = X_test - X_test_pod_recon
    relative_error_sparse = np.linalg.norm(test_error_sparse, 'fro') / np.linalg.norm(X_test, 'fro')
    relative_error_pod = np.linalg.norm(test_error_pod, 'fro') / np.linalg.norm(X_test, 'fro')
    print(f"Test reconstruction relative error (Sparse): {relative_error_sparse:.4f}")
    print(f"Test reconstruction relative error (POD): {relative_error_pod:.4f}")

    # Calculate common error scale for consistent color mapping
    test_error_max = max(np.abs(test_error_pod).max(), np.abs(test_error_sparse).max())
    test_error_min = -test_error_max

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
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_min, vmax=test_error_max)
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
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_min, vmax=test_error_max)
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_ylabel('Space (x)')
    axes[1,2].set_title('Sparse Test Error')
    plt.colorbar(im5, ax=axes[1,2], label='Error')

    # Hide empty subplot
    axes[1,0].set_visible(False)

    plt.tight_layout()
    plt.savefig('../figures/pulse_test_results.png', dpi=300)
    plt.show()
    plt.close(fig)
    
#%% Plot wave at specific time points for test data
# Select 3 equally spaced time points for test data
n_times_test = len(tspan_test)
time_indices_test = [n_times_test//4, n_times_test//2, 3*n_times_test//4]
time_points_test = [tspan_test[i] for i in time_indices_test]

# Create subplots for test data waves
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices_test, time_points_test)):
    # Plot original test data
    ax.plot(xspan_test, X_test[:, t_idx], 'k-', linewidth=2, label='Original', alpha=0.8)
    # Plot POD reconstruction
    ax.plot(xspan_test, X_test_pod_recon[:, t_idx], 'b--', linewidth=2, label='POD', alpha=0.8)
    # Plot sparse reconstruction
    ax.plot(xspan_test, X_test_recon[:, t_idx], 'r:', linewidth=2, label='SparseModesNet', alpha=0.8)
    
    ax.set_xlabel('Space (x)', fontsize=12)
    ax.set_ylabel('u(x,t)', fontsize=12)
    ax.set_title(f't = {t_val:.3f}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('../figures/pulse_test_waves_timepoints.png', dpi=300)
plt.show()
plt.close(fig)
# %%
