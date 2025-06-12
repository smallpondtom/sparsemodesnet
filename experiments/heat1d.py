#%%
import numpy as np
import torch
import matplotlib.pyplot as plt
import kneed
import kneeliverse

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
    lambda_method = 'cv'  # 'path', 'cv', or 'stability'
    
    # Common hyperparameters
    hidden_units_heat = [128, 8256]
    # hidden_units_heat = [20, 120, 400]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-2.1, -0.8, 12)  
    lambdas_ss = np.logspace(-2.1, -1.0, 50)  
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Heat Equation ----------
    X_heat, xspan_h, tspan_h = generate_heat_data(nx=2**7, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    # s_h = 20
    
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
        M               = 5.0,
        lambda_method   = lambda_method,
        knee_method     = 'dfdt',
        network_type    = 'feedforward',    # Specify network type
        kernel_size     = 5,                  # Conv-specific parameters
        num_channels    = [64, 128, 64],      # Conv-specific parameters
        padding         = 'same',             # Conv-specific parameters
        optimizer       = 'Adam',
        nonzero_thresh  = 1e-6,
        r_max           = 10,           # max modes for constraint stopping
        lam0            = 1e-3,         # only used if path
        epsilon         = 0.20,         # only used if path
        B_path          = 80,          # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 80,           # for cv
        lambdas_ss      = lambdas_ss,   # only used if stability
        B_ss            = 2,            # subsamples per λ for stability
        pi_thresh       = 0.75,         # threshold for stability
        num_epochs_ss   = 20,           # epochs per subsample for stability
        final_epochs_ss = 100,          # epochs for final fit after stability
        lr              = 1e-3,
        batch_size      = 64,
        device          = device,
        label           = "Heat Equation"
    )
    
    # #%% Get the data to compute knees 
    # lambdas = np.array([freq['lambda'] for freq in freq_tab])
    # loglam = np.log(lambdas)
    # loglam_max, loglam_min = loglam.max(), loglam.min()
    # rs = np.array([freq['nonzero_count'] for freq in freq_tab])
    # rs_max, rs_min = max(rs), min(rs)
    # llam_norm = []
    # rs_norm = []
    # for lam, r in zip(loglam, rs):
    #     llam_norm.append((lam - loglam_min) / (loglam_max - loglam_min))
    #     rs_norm.append((r - rs_min) / (rs_max - rs_min))
    # llam_norm = np.array(llam_norm)
    # rs_norm = np.array(rs_norm)
    # data = np.stack((llam_norm, rs_norm), axis=1)
    
    # #%% curvature 
    # knee_curv_idx = kneeliverse.curvature.multi_knee(data)
    # knee_curv = llam_norm[knee_curv_idx]
    # lam_knee_curv = np.exp(knee_curv * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by curvature at λ ≃", lam_knee_curv)
    # print("Number of selected modes at knee:", rs[knee_curv_idx])
    
    # #%% dfdt
    # knee_dfdt_idx = kneeliverse.dfdt.multi_knee(data)
    # knee_dfdt = llam_norm[knee_dfdt_idx]
    # lam_knee_dfdt = np.exp(knee_dfdt * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by dfdt at λ ≃", lam_knee_dfdt)
    # print("Number of selected modes at knee:", rs[knee_dfdt_idx])    
    # knee_ = llam_norm[knee_dfdt_idx]
    # lam_stars = np.exp(knee_ * (loglam_max - loglam_min) + loglam_min)
    # r_stars = rs[knee_dfdt_idx]
    # # Pick the first one less than or equal to the budget r_max
    # mask = np.where(r_stars <= 10, 1, 0)  # mask for r <= rmax
    # i_star = np.nonzero(lam_stars * mask)[0][0]
    # print(i_star)
    # lam_star = lam_stars[i_star]
    # r_star = r_stars[i_star]
    # print("Optimal λ* at r ≤ 10 is λ* ≃", lam_star)
    
    # #%% kneedle
    # knee_kneedle_idx = kneeliverse.kneedle.multi_knee(data, t1=0.1, t2=10)
    # knee_kneedle = llam_norm[knee_kneedle_idx]
    # lam_knee_kneedle = np.exp(knee_kneedle * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by Kneedle at λ ≃", lam_knee_kneedle)
    # print("Number of selected modes at knee:", rs[knee_kneedle_idx])
    
    # #%% lmethod 
    # knee_lmethod_idx = kneeliverse.lmethod.multi_knee(data, t1=0.001, t2=5)
    # knee_lmethod = llam_norm[knee_lmethod_idx]
    # lam_knee_lmethod = np.exp(knee_lmethod * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by Lmethod at λ ≃", lam_knee_lmethod)
    # print("Number of selected modes at knee:", rs[knee_lmethod_idx])
    
    # #%% menger
    # knee_menger_idx = kneeliverse.menger.multi_knee(data, t1=0.001, t2=5)
    # knee_menger = llam_norm[knee_menger_idx]
    # lam_knee_menger = np.exp(knee_menger * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by Menger at λ ≃", lam_knee_menger)
    # print("Number of selected modes at knee:", rs[knee_menger_idx])
    
    # #%% z-method
    # knee_zmethod_idx = kneeliverse.zmethod.knees2(data)
    # knee_zmethod = llam_norm[knee_zmethod_idx]
    # lam_knee_zmethod = np.exp(knee_zmethod * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by Z-method at λ ≃", lam_knee_zmethod)
    # print("Number of selected modes at knee:", rs[knee_zmethod_idx])
    
    # #%% Plot the knee plot using Kneed
    # # normalize log‐λ and k to [0,1]
    # lambdas = [freq['lambda'] for freq in freq_tab]
    # loglam = np.log(lambdas)
    # loglam_max, loglam_min = loglam.max(), loglam.min()
    # rs = [freq['l1_b'] for freq in freq_tab]
    # rs_max, rs_min = max(rs), min(rs)
    # llam_norm = []
    # rs_norm = []
    # for lam, r in zip(loglam, rs):
    #     llam_norm.append((lam - loglam_min) / (loglam_max - loglam_min))
    #     rs_norm.append((r - rs_min) / (rs_max - rs_min))
    # kl = kneed.KneeLocator(
    #     rs_norm, llam_norm, S=10, curve='convex', direction='decreasing',
    #     interp_method='polynomial', polynomial_degree=9
    # )
    # # kl.knee is in normalized x; map back to λ:
    # lam_knee = np.exp(kl.knee_y * (loglam_max - loglam_min) + loglam_min)
    # print("Elbow by Kneedle at λ ≃", lam_knee) 
    # # Plotting
    # kl.plot_knee_normalized()
    
    # #%%
    # rs = np.array([freq['nonzero_count'] for freq in freq_tab])
    # dks   = np.diff(rs)
    # dlogl = np.diff(loglam)
    # slopes = dks / dlogl
    # rmax = 10
    # mask = np.where(rs[1:] <= rmax, 1, 0)  # mask for r <= rmax
    # i_elbow = np.argmin(slopes * mask)     # most negative slope
    # lambda_elbow = lambdas[i_elbow+1]
    # print("Elbow at λ ≃", lambda_elbow)
    
    #%% Plot L-curve
    if lambda_method == 'path':
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel('L1 Regularization Term (||ω||₁)')
        ax.set_ylabel('Relative Error')
        ax.set_title('L-curve for Heat Equation')
        ax.grid(True, alpha=0.3)
        for freq in freq_tab:
            ax.loglog(freq['l1_b'], freq['rel_error'], 'o-', markersize=6, linewidth=2)
        plt.tight_layout()
        plt.savefig('../figures/heat_lcurve.png', dpi=300)
        plt.show()
        plt.close(fig)
    
    #%% Plot the λ vs selected modes and λ vs relative error
    if lambda_method == 'path':
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
        # Extract data
        lambdas = [freq['lambda'] for freq in freq_tab]
        num_modes = [freq['nonzero_count'] for freq in freq_tab]
        rel_errors = [freq['rel_error'] for freq in freq_tab]
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
        line1 = ax3.semilogx(lambdas, num_modes, 'o-', markersize=8, linewidth=2, color=color1, label='Selected Modes')
        ax3.tick_params(axis='both', which='major', labelsize=14, labelcolor='black')
        ax3.tick_params(axis='y', labelcolor=color1)
        ax3.grid(True, alpha=0.3)
        ax3_twin = ax3.twinx()
        ax3_twin.set_ylabel('Relative Error', color=color2, fontsize=16)
        line2 = ax3_twin.loglog(lambdas, rel_errors, 's-', markersize=8, linewidth=2, color=color2, label='Relative Error')
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
