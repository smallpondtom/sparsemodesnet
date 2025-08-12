#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.heat1d import generate_heat_data

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
    hidden_units_heat = [128, 8256, 2048]
    # hidden_units_heat = [20, 210, 400] 

    # Parameter‐grid for CV
    lambdas_cv = np.logspace(-3.0, -1.0, 15)  
    
    # number of grids
    n_grids = 2**7
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Heat Equation ----------
    X_heat, xspan_h, tspan_h = generate_heat_data(
        nx=n_grids, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    # s_h = 20
    
    ## Create 3D surface plot for Heat Equation (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_h, tspan_h)
        Z_mesh = X_heat.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(
            X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Heat Equation Solution')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('../figures/heat_data.png', dpi=300)
        plt.show()
        plt.close(fig)

    # Plot the singular value vs the retained energy
    U, S, _ = np.linalg.svd(X_heat, full_matrices=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative_energy = np.cumsum(S**2) / np.sum(S**2)
    # Plot up to s modes
    mode_range = np.arange(1, len(S)+1)
    ax.plot(mode_range, cumulative_energy, 'b-o', linewidth=3, markersize=8)
    # Add horizontal lines for common energy thresholds
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.7, label='99% Energy')
    ax.axhline(y=0.999, color='purple', linestyle='--', alpha=0.7, label='99.9% Energy')
    # Find modes corresponding to energy thresholds
    modes_99 = np.argmax(cumulative_energy >= 0.99) + 1
    modes_999 = np.argmax(cumulative_energy >= 0.999) + 1
    # Add vertical lines at these points
    ax.axvline(x=modes_99, color='green', linestyle=':', alpha=0.5)
    ax.axvline(x=modes_999, color='purple', linestyle=':', alpha=0.5)
    # Add text annotations
    ax.text(modes_999 + 2, 0.9976, f'{modes_999} modes', fontsize=12, color='purple')
    ax.set_xlabel('Number of POD Modes', fontsize=14)
    ax.set_ylabel('Cumulative Energy Fraction', fontsize=14)
    ax.set_title('POD Energy Content vs Number of Modes', fontsize=16)
    plt.yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim([0, 10])
    ax.set_ylim([0.985, 1.005])
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.savefig('figures/heat/energy_vs_modes.png', dpi=200)
    plt.show()
    plt.close(fig)

    #%% Train
    model_heat, info_heat, selected_h, freq_tab = run_sparsemodesnet(
        X_np            = X_heat,
        s               = s_h,
        hidden_units    = hidden_units_heat,
        M               = 4.0,
        reg_path        = reg_path,
        lr              = 1e-3,
        batch_size      = 64,
        knee_method     = 'zmethod',
        optimizer       = 'Adam',
        nonzero_thresh  = 1e-14,
        r_max           = 20,           # max modes for constraint stopping
        lam0            = 1e-3,         # only used if path
        epsilon         = 0.20,         # only used if path
        network_type    = 'PiNetCCP',   # 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip'
        poly_order      = 2,            # order of polynomial
        num_polys       = 1,            # number of polynomials
        drop_linear     = True,         # whether to drop linear term
        B_path          = 80,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 80,           # for cv
        device          = device,
        label           = "Heat Equation",
        enable_logging=True,  
        logs_dir="./logs"     
    )
    
    
    #%% === Plot the first 20 modes of the POD b# recompute just the 
    # first 20 POD modes ===
    U_s20, _, _ = compute_pod_basis(X_heat, s=s_h)
    fig, axes = plt.subplots(4, 5, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(xspan_h, U_s20[:, i])
        ax.set_title(f"Mode {i+1}")
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('../figures/heat_pod_modes.png', dpi=300)
    plt.show()
    
    #%% === Plot the POD modes vs the reconstruction error ===
    U, S, _ = np.linalg.svd(X_heat, full_matrices=False)
    Us_20 = U[:, :s_h].astype(np.float64)  # First s_h POD modes
    fig, ax = plt.subplots(figsize=(8, 6))
    proj_err = []
    X_heat_f64 = X_heat.astype(np.float64)
    Us_20_f64 = Us_20.astype(np.float64)
    for i in range(s_h):
        proj_err.append(
            np.linalg.norm(
                X_heat_f64 - Us_20_f64[:, :i+1] 
                @ (Us_20_f64[:, :i+1].T @ X_heat_f64), 'fro') 
            / np.linalg.norm(X_heat_f64, 'fro')
        )
    ax.semilogy(range(1, s_h+1), proj_err)
    ax.set_xlabel('Number of POD Modes')
    ax.set_ylabel('projection error (relative)')
    ax.set_title(f'POD Mode {i+1} vs Projection Errors')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'../figures/heat_pod_mode_vs_recon.png', dpi=300)
    plt.show()
    
    #%% === Plot the λ vs selected modes and λ vs relative error ===
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
        plt.savefig('../figures/heat_path_summary.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    
    #%% === Plot L-curve ====
    if reg_path == 'dense2sparse':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('L1 Regularization Term (||ω||₁)')
        ax.set_ylabel('Relative Error')
        ax.set_title('L-curve for Heat Equation')
        ax.grid(True, alpha=0.3)
        for freq in freq_tab:
            ax.loglog(freq['l1_b'], freq['error'], 'o-', markersize=6, linewidth=2)
        plt.tight_layout()
        plt.savefig('../figures/heat_lcurve.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    #%% === Plot the λ vs selected modes and λ vs relative error ===
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
        plt.savefig('../figures/heat_lambda_analysis.png', dpi=300)
        plt.show()
        plt.close(fig)

    #%% === Plot the reconstructed flow fields (heatmap) ===
    V, _, _ = np.linalg.svd(X_heat, full_matrices=False)
    V_selected = V[:, selected_h]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_heat
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, :s_h].T @ X_heat).T.astype(np.float32)).to(device)
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
    
    #%% === Plot the reconstructed flow fields (heatmap) with  ====
    # linear and nonlinear parts
    V, _, _ = np.linalg.svd(X_heat, full_matrices=False)
    V_selected = V[:, selected_h]
    fig, ax = plt.subplots(figsize=(12, 6))
    X_pod_recon = V_selected @ V_selected.T @ X_heat
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, :s_h].T @ X_heat).T.astype(np.float32)).to(device)
    with torch.no_grad():
        model_heat.eval()
        
        # # If you enforce omega to be ones 
        # omega_ones = np.zeros((s_h, s_h), dtype=np.float32)
        # for i in range(len(model_heat.omega)):
        #     if model_heat.omega[i].abs().item() > 0:
        #         omega_ones[i, i] = 1.0 
        # X_sparse_lin = V[:, :s_h] @ omega_ones @ V[:, :s_h].T @ X_heat
        # omega_ones_tensor = torch.from_numpy(omega_ones).to(device)
        # nonlin_part = model_heat.net(Z_input @ omega_ones_tensor)
        
        # Linear part
        omega = np.diag(model_heat.omega.cpu().numpy())
        X_sparse_lin = V[:, :s_h] @ omega @ V[:, :s_h].T @ X_heat
        omega_tensor = torch.from_numpy(omega).to(device)
        
        # # Nonlinear part 
        # nonlin_part = model_heat.net(Z_input @ omega_tensor)
        # X_sparse_nonlin = nonlin_part.cpu().numpy().T
        # Nonlinear part 
        if model_heat.network_type == 'FF':
            # For feedforward networks, use the full net
            nonlin_part = model_heat.net(Z_input @ omega_tensor)
        else:
            # For Pi-Net models (PiNetCCP, PiNetNCP, PiNetNCPSkip)
            z_sparse = Z_input @ omega_tensor  # Apply sparsity
            h = model_heat.first_layer(z_sparse)  # First layer: (batch, in_dim)
            h_poly = model_heat.pinet(h)  # Pi-Net: (batch, out_dim)
            nonlin_part = model_heat.C(h_poly)  # Final layer: (batch, d)

        X_sparse_nonlin = nonlin_part.cpu().numpy().T
        
        # Together 
        _, X_sparse_recon_tensor = model_heat(Z_input)
        X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
    # Calculate errors
    pod_error = X_heat - X_pod_recon
    sparse_error = X_heat - X_sparse_recon
    sparse_lin_error = X_heat - X_sparse_lin
    sparse_nonlin_error = X_heat - X_sparse_nonlin
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
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
    
    # (3,1) Sparse linear reconstruction
    im5 = axes[2,0].imshow(
        X_sparse_lin, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[2,0].set_xlabel('Time')
    axes[2,0].set_ylabel('Space (x)')
    axes[2,0].set_title('Sparse Linear Reconstruction')
    plt.colorbar(im5, ax=axes[2,0], label='u(x,t)')
    
    # (3,2) Sparse linear error
    im6 = axes[2,1].imshow(
        sparse_lin_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[2,1].set_xlabel('Time')
    axes[2,1].set_ylabel('Space (x)')
    axes[2,1].set_title('Sparse Linear Error')
    plt.colorbar(im6, ax=axes[2,1], label='Error')
    
    # (4,1) Sparse nonlinear reconstruction
    im7 = axes[3,0].imshow(
        X_sparse_nonlin, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[3,0].set_xlabel('Time')
    axes[3,0].set_ylabel('Space (x)')
    axes[3,0].set_title('Sparse Nonlinear Reconstruction')
    plt.colorbar(im7, ax=axes[3,0], label='u(x,t)')
    
    # (4,2) Sparse nonlinear error
    im8 = axes[3,1].imshow(
        sparse_nonlin_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]])
    axes[3,1].set_xlabel('Time')
    axes[3,1].set_ylabel('Space (x)')
    axes[3,1].set_title('Sparse Nonlinear Error')
    plt.colorbar(im8, ax=axes[3,1], label='Error')
    
    plt.tight_layout()
    plt.savefig('../figures/heat_comparison_separated.png', dpi=300)
    plt.show()
    plt.close(fig)

    #%% Test the model on a new sample
    # Generate new test data with different parameters
    X_test, xspan_test, tspan_test = generate_heat_data(
        nx=n_grids, nt=800, alpha=0.02, x_max=1.0, t_max=0.8)

    # Project test data onto the learned POD basis
    V_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
    Z_test = torch.from_numpy(
        (V[:, :s_h].T @ X_test).T.astype(np.float32)).to(device)

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
