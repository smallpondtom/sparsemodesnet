
"""
Kuramoto-Sivashinksy experiment using SparseModesNet.
"""

#%% Load modules
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.kse import generate_kse_data
from QM.quadmani import quadmani_greedy
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

#%% %============================= Main Script ================================%
if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = 'cpu'
    print("Using device:", device)

    # For reproducibility
    torch.manual_seed(42)
    
    # number of grids
    n_grids = 2**10
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- KSE ----------
    X, xspan, tspan = generate_kse_data(
        nx=n_grids, nt=2000, L=32*np.pi, t_max=150.0
    )
    X = X.astype(np.float64) 
    d, n = X.shape
    s = min(d, n)
    s = 100
    r = 20    

    # Create 3D surface plot for KSE (sanity check)
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
        ax.set_title('Kuramoto-Sivashinsky Equation Solution')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('figures/kse/kse_data.png', dpi=200)
        plt.show()
        plt.close(fig)


#%% %===================== Additional Plotting and Analysis ===================%
    # Plot the first 20 modes of the POD basis
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    U_20 = U[:, :20] # First 20 POD modes
    fig, axes = plt.subplots(4, 5, figsize=(15, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.plot(xspan, U_20[:, i])
        ax.set_title(f"Mode {i+1}")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig('figures/kse/kse_pod_modes.png', dpi=200)
    plt.show()
    
    # Plot the POD modes vs the reconstruction error 
    U_s = U[:, :s].astype(np.float64)  # First s POD modes
    fig, ax = plt.subplots(figsize=(8, 6))
    proj_err = []
    for i in range(s):
        proj_err.append(
            np.linalg.norm(X - U_s[:, :i+1] @ (U_s[:, :i+1].T @ X), 'fro') 
            / np.linalg.norm(X, 'fro')
        )
    ax.semilogy(range(1, s+1), proj_err)
    ax.set_xlabel('Number of POD Modes')
    ax.set_ylabel('Projection Error (Relative)')
    ax.set_title('POD Modes vs Projection Errors')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('figures/kse/kse_pod_mode_vs_recon.png', dpi=200)
    plt.show()


#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V_qm, W_qm, mu_qm, I_qm = quadmani_greedy(
        X, r, s, 1e-15, np.array([], dtype=int))
    mu_qm = mu_qm.reshape(-1, 1)  

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())


#%% %==================== Configuration of SparseModesNet =====================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        # Architecture
        # 'hidden_units': [200, 500, 800, 1200, 1200, 1024, 1024],
        # 'hidden_units': [32, 7, 64, 128, 256, 512],
        # 'hidden_units': [32, 128],
        'hidden_units': [20, 512, 1024],
        'network_type': 'PiNetCCP',
        'poly_order': 3,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 10.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 100000,
        'epsilon': 0.001,
        'lasso_epochs': 100,
        'M': 5.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
        # Decoder Phase
        'decoder_lr': 1.0e-3,
        'decoder_lr_patience': 100,
        'decoder_epochs': 5000,
        'decoder_batch_size': 200,
        'decoder_optimizer': 'SGD',
        'decoder_momentum': 0.99,
        # General training
        'skip_sparse': True,
        'gamma': 1e-6,
        'I_nn': [0, 1, 3, 2, 5, 4, 8, 7, 9, 12, 6, 11, 10, 13, 16, 15, 14, 19, 17, 23],
        'device': device,
        'analytical': False,
        # Experiment Setup
        'label': "KSE",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model, I_nn, omegas, path_history = smn.fit(X, config)


#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas, I_nn, config.s, save=False, 
                     filename='figures/kse/omega_evolution.png')

#%% #===================== Plot Reconstruction Errors =========================#
    # Collect reconstruction errors for different numbers of modes
    mode_counts = []
    qm_errors = []
    pod_errors = []
    sparse_errors = []

    V_all = np.linalg.svd(
        config.preprocessing.forward(X), full_matrices=False
    )[0][:, :s]
    
    # Test different numbers of modes
    for r_test in range(1, min(r + 1, 21)):  # Test up to 20 modes or r
        # Quadratic Manifold with r_test modes
        V_test, W_test, shift_test, I_qm_test = quadmani_greedy(
            X, r_test, s, 1e-15, np.array([], dtype=int))
        shift_test = shift_test.reshape(-1, 1)
        
        Z_qm_test = V_test.T @ (X - shift_test)
        Z_quad_qm_test = quadratic_mapping_numpy(Z_qm_test.T).T
        recon_error_qm_test = np.linalg.norm(
            X - (V_test @ Z_qm_test + W_test @ Z_quad_qm_test + shift_test), ord='fro')
        rel_recon_error_qm_test = recon_error_qm_test / np.linalg.norm(X, ord='fro')
        
        # Leading-r POD reconstruction
        X_proc_test = config.preprocessing.forward(X)
        V_leading_test = np.linalg.svd(X_proc_test, full_matrices=False)[0][:, :r_test]
        X_pod_recon_test = V_leading_test @ V_leading_test.T @ X_proc_test
        X_pod_recon_test = config.preprocessing.backward(X_pod_recon_test)
        recon_error_pod_test = np.linalg.norm(X - X_pod_recon_test, ord='fro')
        rel_recon_error_pod_test = recon_error_pod_test / np.linalg.norm(X, ord='fro')
        
        # SparseModesNet with first r_test modes from selected modes
        if len(I_nn) >= r_test:
            V_tmp = np.zeros((d, r))
            V_tmp[:, :r_test] = V_all[:, I_nn[:r_test]]
            Z_input_test = torch.from_numpy(
                (V_tmp.T @ X_proc_test).T.astype(np.float32)).to(device)
            with torch.no_grad():
                if  config.network.network_type == 'QM':
                    # Use analytical decoder
                    X_sparse_recon_test = model(V_all[:, I_nn[:r_test]].T @ X_proc_test)
                else:
                    model.eval()
                    # Create a temporary model with fewer output features for this test
                    X_sparse_recon_tensor_test = model(Z_input_test)
                    X_sparse_recon_test = X_sparse_recon_tensor_test.cpu().numpy().T 
                    X_sparse_recon_test = config.preprocessing.backward(X_sparse_recon_test)
            
            recon_error_sparse_test = np.linalg.norm(X - X_sparse_recon_test, ord='fro')
            rel_recon_error_sparse_test = recon_error_sparse_test / np.linalg.norm(X, ord='fro')
        else:
            rel_recon_error_sparse_test = np.nan
        
        mode_counts.append(r_test)
        qm_errors.append(rel_recon_error_qm_test)
        pod_errors.append(rel_recon_error_pod_test)
        sparse_errors.append(rel_recon_error_sparse_test)
    
    # Plot reconstruction errors
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.semilogy(mode_counts, qm_errors, 'g-^', label='Quadratic Manifold', 
                markersize=10, linewidth=4)
    ax.semilogy(mode_counts, pod_errors, 'b-o', label='POD (leading-r)', 
                markersize=10, linewidth=4)
    
    # Only plot valid SparseModesNet errors
    valid_sparse_errors = [err for err in sparse_errors if not np.isnan(err)]
    valid_mode_counts = [mode_counts[i] for i, err in enumerate(sparse_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts, valid_sparse_errors, 'r-s', label='SparseModesNet', 
                markersize=10, linewidth=4)
    
    ax.set_xlabel('Number of Modes', fontsize=16)
    ax.set_ylabel('Relative Reconstruction Error', fontsize=16)
    ax.set_title('Reconstruction Error vs Number of Modes', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=14)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig('figures/kse/reconstruction_errors.png', dpi=300)
    plt.show()
    plt.close(fig)
    

#%% %============ Plot the reconstructed flow fields (heatmap) ================%
    X_proc = config.preprocessing.forward(X)
    V, _, _ = np.linalg.svd(X_proc, full_matrices=False)
    V_selected = V[:, 1:len(I_nn)]

    # Quadratic Manifold reconstruction
    # First get the linear coefficients
    Z_qm = V_qm.T @ (X - mu_qm)
    # Then compute quadratic terms from these coefficients
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    # Reconstruct using both linear and quadratic terms
    X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Leading-r POD reconstruction (using top r modes)
    r = len(I_nn)  # or whatever r value you want to use
    V_leading_r = V[:, :r]
    X_pod_recon = V_leading_r @ V_leading_r.T @ X_proc
    X_pod_recon = config.preprocessing.backward(X_pod_recon)
    
    # Fix: Convert numpy array to tensor and move to correct device
    Z_input = torch.from_numpy(
        (V[:, I_nn].T @ X_proc).T.astype(np.float32)).to(device)
    with torch.no_grad():
        if  config.network.network_type == 'QM':
            # Use analytical decoder
            X_sparse_recon_test = model(V[:, I_nn].T @ X_proc)
        else:
            model.eval()
            X_sparse_recon_tensor = model(Z_input)
            X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
            X_sparse_recon = config.preprocessing.backward(X_sparse_recon)
    
    # Calculate errors
    pod_error = X - X_pod_recon
    qm_error = X - X_qm_recon
    sparse_error = X - X_sparse_recon
    
    # Set consistent color scales for reconstructions
    recon_vmin = min(X.min(), X_pod_recon.min(), X_qm_recon.min(), X_sparse_recon.min())
    recon_vmax = max(X.max(), X_pod_recon.max(), X_qm_recon.max(), X_sparse_recon.max())
    
    # Set consistent color scales for errors
    error_vmax = max(np.abs(pod_error).max(), np.abs(qm_error).max(), np.abs(sparse_error).max())
    error_vmin = -error_vmax
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Reconstructions
    # (1,1) Original data
    im1 = axes[0,0].imshow(
        X, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,0].set_ylabel('Space (x)', fontsize=14)
    axes[0,0].set_title('Original Data', fontsize=15)
    
    # (1,2) Leading-r POD reconstruction
    im2 = axes[0,1].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,1].set_title(f'POD Reconstruction (r={r})', fontsize=15)
    
    # (1,3) Quadratic Manifold reconstruction
    im3 = axes[0,2].imshow(
        X_qm_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,2].set_title('Quadratic Manifold Reconstruction', fontsize=15)
    
    # (1,4) SparseModesNet reconstruction
    im4 = axes[0,3].imshow(
        X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,3].set_title('SparseModesNet Reconstruction', fontsize=15)
    
    # Row 2: Errors
    # (2,1) Empty - no error for original data
    axes[1,0].axis('off')
    
    # (2,2) POD error
    im5 = axes[1,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,1].set_xlabel('Time', fontsize=14)
    axes[1,1].set_ylabel('Space (x)', fontsize=14)
    axes[1,1].set_title('POD Error', fontsize=15)
    
    # (2,3) Quadratic Manifold error
    im6 = axes[1,2].imshow(
        qm_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,2].set_xlabel('Time', fontsize=14)
    axes[1,2].set_title('Quadratic Manifold Error', fontsize=15)
    
    # (2,4) SparseModesNet error
    im7 = axes[1,3].imshow(
        sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,3].set_xlabel('Time', fontsize=14)
    axes[1,3].set_title('SparseModesNet Error', fontsize=15)
    
    # Add unified colorbars
    cax1 = fig.add_axes([0.92, 0.57, 0.02, 0.35])
    cbar1 = plt.colorbar(im4, cax=cax1)
    cbar1.set_label('u(x,t)', fontsize=14)
    
    cax2 = fig.add_axes([0.92, 0.11, 0.02, 0.35])
    cbar2 = plt.colorbar(im7, cax=cax2)
    cbar2.set_label('Error', fontsize=14)

    plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.1, 
                        wspace=0.3, hspace=0.3)
    plt.suptitle('Reconstruction Comparison', fontsize=19, y=0.98)
    plt.savefig('figures/kse/kse_comparison.png', dpi=300)
    plt.show()
    plt.close(fig)


    
#%% %================= Plot waves at specific time points =====================%
    # Select 3 equally spaced time points
    n_times = len(tspan)
    time_indices = [n_times//4, n_times//2, 3*n_times//4]
    time_points = [tspan[i] for i in time_indices]

    # Get reconstructions
    X_proc = config.preprocessing.forward(X)
    V, _, _ = np.linalg.svd(X_proc, full_matrices=False)
    
    # Leading-r POD reconstruction (using top r modes)
    r = len(I_nn)  # or whatever r value you want to use
    V_leading_r = V[:, :r]
    X_pod_recon = V_leading_r @ V_leading_r.T @ X_proc
    X_pod_recon = config.preprocessing.backward(X_pod_recon)

    # Quadratic Manifold reconstruction
    # First get the linear coefficients
    Z_qm = V_qm.T @ (X - mu_qm)
    # Then compute quadratic terms from these coefficients
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    # Reconstruct using both linear and quadratic terms
    X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # SparseModesNet reconstruction
    Z_input = torch.from_numpy(
        (V[:, I_nn].T @ X_proc).T.astype(np.float32)).to(device)
    with torch.no_grad():
        if  config.network.network_type == 'QM':
            # Use analytical decoder
            X_sparse_recon_test = model(V[:, I_nn].T @ X_proc)
        else:
            model.eval()
            X_sparse_recon_tensor = model(Z_input)
            X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T
            X_sparse_recon = config.preprocessing.backward(X_sparse_recon)

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices, time_points)):
        # Plot original data
        ax.plot(xspan, X[:, t_idx], 'k-', linewidth=3, 
                label='Original', alpha=0.9)
        # Plot leading-r POD reconstruction
        ax.plot(xspan, X_pod_recon[:, t_idx], 'b--', 
                linewidth=2, label=f'POD (r={r})', alpha=0.8)
        # Plot Quadratic Manifold reconstruction
        ax.plot(xspan, X_qm_recon[:, t_idx], 'g-.', 
                linewidth=2, label='Quadratic Manifold', alpha=0.8)
        # Plot SparseModesNet reconstruction
        ax.plot(xspan, X_sparse_recon[:, t_idx], 'r:', 
                linewidth=2, label='SparseModesNet', alpha=0.8)
        
        ax.set_xlabel('Space (x)', fontsize=14)
        ax.set_ylabel('u(x,t)', fontsize=14)
        ax.set_title(f't = {t_val:.3f}', fontsize=16)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=16)

    plt.tight_layout()
    plt.suptitle('Wave Profiles at Different Time Points', fontsize=19, y=1.02)
    plt.savefig('figures/kse/wave_profiles_comparison.png', dpi=300)
    plt.show()
    plt.close(fig)

# #%%
# if __name__ == "__main__":
#     # Device selection: CUDA > MPS (Apple Silicon) > CPU
#     if torch.cuda.is_available():
#         device = 'cuda'
#     elif torch.backends.mps.is_available():
#         device = 'mps'
#     else:
#         device = 'cpu'
#     print("Using device:", device)
    
#     # Regularization parameter selection method
#     reg_path = 'dense2sparse'
    
#     # Number of spatial grids
#     n_grids = 2**10  

#     # Reduced dimension 
#     s = 100
    
#     # Common hyperparameters
#     hidden_units_ks   = [s, int(s * (s + 1) / 2)]

#     # Parameter‐grid for CV or SS (you can customize)
#     lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    
#     # Sanity check flag (plotting)
#     sanity_check = True

#     # ---------- Kuramoto–Sivashinsky Equation ----------
#     # Note: smaller nt for speed, adjust as desired
#     X_ks, xspan_ks, tspan_ks = generate_kse_data(
#         nx=n_grids, nt=5000, L=32*np.pi, t_max=150.0)
#     d_ks, n_ks = X_ks.shape
#     s_ks = min(d_ks, n_ks, s)
    
#     # Create flow-field for Kuramoto-Sivashinsky Equation
#     if sanity_check:
#         fig, ax = plt.subplots(figsize=(12, 8))
#         im = ax.imshow(
#             X_ks, aspect='auto', cmap='viridis', origin='lower',
#             extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
#         ax.set_xlabel('Time')
#         ax.set_ylabel('Space (x)')
#         ax.set_title('Kuramoto-Sivashinsky Equation Solution')
#         plt.colorbar(im, ax=ax, label='u(x,t)')
#         plt.tight_layout()
#         plt.savefig('../figures/kse_data.png', dpi=300)
#         plt.show()
#         plt.close(fig)

#     #%%
#     model_ks, info_ks, selected_ks, freq_tab = run_sparsemodesnet(
#         X_np            = X_ks,
#         s               = s_ks,
#         hidden_units    = hidden_units_ks,
#         M               = 10.0,
#         reg_path        = reg_path,
#         lr              = 1e-3,
#         batch_size      = 256,
#         knee_method     = 'zmethod',
#         optimizer       = 'Adam',
#         nonzero_thresh  = 1e-14,
#         r_max           = 25,           # max modes for constraint stopping
#         lam0            = 1e-3,         # only used if path
#         epsilon         = 0.20,         # only used if path
#         B_path          = 80,           # epochs per λ for path or final fit
#         max_iters       = 100,          # max iterations for path
#         lambdas_cv      = lambdas_cv,   # only used if cv
#         k_folds         = 5,            # for cv
#         num_epochs_cv   = 80,           # for cv
#         device          = device,
#         label           = "Kuramoto-Sivashinsky Equation"
#     )
    
#     #%% Plot the first 20 modes of the POD b# recompute just the first 20 POD modes
#     U_s20, _, _ = compute_pod_basis(X_ks, s=s_ks)
#     fig, axes = plt.subplots(4, 5, figsize=(15, 8))
#     for i, ax in enumerate(axes.flatten()):
#         ax.plot(xspan_ks, U_s20[:, i])
#         ax.set_title(f"Mode {i+1}")
#         ax.grid(True)

#     plt.tight_layout()
#     plt.savefig('../figures/kse_pod_modes.png', dpi=300)
#     plt.show()
    
#     #%% Plot the POD modes vs the reconstruction error 
#     U, S, _ = np.linalg.svd(X_ks, full_matrices=False)
#     Us_20 = U[:, :500].astype(np.float64)  
#     fig, ax = plt.subplots(figsize=(8, 6))
#     proj_err = []
#     X_heat_f64 = X_ks.astype(np.float64)
#     Us_20_f64 = Us_20.astype(np.float64)
#     for i in range(500):
#         proj_err.append(
#             np.linalg.norm(
#                 X_heat_f64 - Us_20_f64[:, :i+1] 
#                 @ (Us_20_f64[:, :i+1].T @ X_heat_f64), 'fro') 
#             / np.linalg.norm(X_heat_f64, 'fro')
#         )
#     ax.semilogy(range(1, 500+1), proj_err)
#     ax.set_xlabel('Number of POD Modes')
#     ax.set_ylabel('projection error (relative)')
#     ax.set_title(f'POD Mode {i+1} vs Projection Errors')
#     ax.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'../figures/kse_pod_mode_vs_recon.png', dpi=300)
#     plt.show()
    
    
#     #%% Plot the λ vs selected modes and λ vs relative error
#     if reg_path == 'dense2sparse':
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
#         # Extract data
#         lambdas    = [freq['lambda']        for freq in freq_tab]
#         num_modes  = [freq['nonzero_count'] for freq in freq_tab]
#         rel_errors = [freq['error']         for freq in freq_tab]
#         # Plot 1: λ vs relative error
#         ax1.loglog(
#             lambdas, rel_errors, 'o-', markersize=8, linewidth=2, color='red')
#         ax1.set_xlabel('Regularization Parameter (λ)', fontsize=16)
#         ax1.set_ylabel('Relative Error', fontsize=16)
#         ax1.set_title('λ vs Relative Error', fontsize=18)
#         ax1.tick_params(axis='both', which='major', labelsize=14)
#         ax1.grid(True, alpha=0.3)
#         # Plot 2: λ vs selected modes
#         ax2.semilogx(
#             lambdas, num_modes, 'o-', markersize=8, linewidth=2, color='blue')
#         ax2.set_xlabel('Regularization Parameter (λ)', fontsize=16)
#         ax2.set_ylabel('Number of POD Modes', fontsize=16)
#         ax2.set_title('λ vs # Modes', fontsize=18)
#         ax2.tick_params(axis='both', which='major', labelsize=14)
#         ax2.grid(True, alpha=0.3)
#         # Plot 3: # Modes vs Relative Error
#         ax3.semilogy(
#             num_modes, rel_errors, 'o-', markersize=8, linewidth=2, color='green')
#         ax3.set_xlabel('Number of POD Modes', fontsize=16)
#         ax3.set_ylabel('Relative Error', fontsize=16)
#         ax3.set_title('# Modes vs Relative Error', fontsize=18)
#         ax3.tick_params(axis='both', which='major', labelsize=14)
#         ax3.grid(True, alpha=0.3)

#         plt.tight_layout()
#         plt.savefig('../figures/kse_path_summary.png', dpi=300)
#         plt.show()
#         plt.close(fig)
    
    
#     #%% Plot L-curve
#     if reg_path == 'dense2sparse':
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.set_xlabel('L1 Regularization Term (||ω||₁)')
#         ax.set_ylabel('Relative Error')
#         ax.set_title('L-curve for Heat Equation')
#         ax.grid(True, alpha=0.3)
#         for freq in freq_tab:
#             ax.loglog(freq['l1_b'], freq['error'], 'o-', markersize=6, linewidth=2)
#         plt.tight_layout()
#         plt.savefig('../figures/kse_lcurve.png', dpi=300)
#         plt.show()
#         plt.close(fig)
    
#     #%% Plot the λ vs selected modes and λ vs relative error
#     if reg_path == 'dense2sparse':
#         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))
#         # Extract data
#         lambdas = [freq['lambda'] for freq in freq_tab]
#         num_modes = [freq['nonzero_count'] for freq in freq_tab]
#         rel_errors = [freq['error'] for freq in freq_tab]
#         # Plot 1: λ vs relative error
#         ax1.loglog(lambdas, rel_errors, 'o-', markersize=8, linewidth=2, color='red')
#         ax1.set_xlabel('Regularization Parameter (λ)', fontsize=16)
#         ax1.set_ylabel('Relative Error', fontsize=16)
#         ax1.set_title('λ vs Relative Error', fontsize=18)
#         ax1.tick_params(axis='both', which='major', labelsize=14)
#         ax1.grid(True, alpha=0.3)
#         # Plot 2: λ vs selected modes
#         ax2.semilogx(lambdas, num_modes, 'o-', markersize=8, linewidth=2, color='blue')
#         ax2.set_xlabel('Regularization Parameter (λ)', fontsize=16)
#         ax2.set_ylabel('Number of Selected Modes', fontsize=16)
#         ax2.set_title('λ vs Number of Selected Modes', fontsize=18)
#         ax2.tick_params(axis='both', which='major', labelsize=14)
#         ax2.grid(True, alpha=0.3)
#         # Plot 3: Superimposed plot with dual y-axes
#         color1 = 'blue'
#         color2 = 'red'
#         ax3.set_xlabel('Regularization Parameter (λ)', fontsize=16)
#         ax3.set_ylabel('Number of Selected Modes', color=color1, fontsize=16)
#         line1 = ax3.semilogx(lambdas, num_modes, 'o-', markersize=8, 
#                              linewidth=2, color=color1, label='Selected Modes')
#         ax3.tick_params(axis='both', which='major', 
#                         labelsize=14, labelcolor='black')
#         ax3.tick_params(axis='y', labelcolor=color1)
#         ax3.grid(True, alpha=0.3)
#         ax3_twin = ax3.twinx()
#         ax3_twin.set_ylabel('Relative Error', color=color2, fontsize=16)
#         line2 = ax3_twin.loglog(lambdas, rel_errors, 's-', markersize=8, 
#                                 linewidth=2, color=color2, label='Relative Error')
#         ax3_twin.tick_params(axis='y', labelcolor=color2, labelsize=14)
#         ax3.set_title('λ vs Selected Modes & Relative Error', fontsize=18)
#         # Add legend
#         lines = line1 + line2
#         labels = [l.get_label() for l in lines]
#         ax3.legend(lines, labels, loc='center left', fontsize=16)
#         plt.tight_layout()
#         plt.savefig('../figures/kse_lambda_analysis.png', dpi=300)
#         plt.show()
#         plt.close(fig)
        

#     #%% Plot the reconstructed flow fields (heatmap)
#     V, _, _ = np.linalg.svd(X_ks, full_matrices=False)
#     V_selected = V[:, selected_ks]
#     fig, ax = plt.subplots(figsize=(12, 6))
#     X_pod_recon = V_selected @ V_selected.T @ X_ks
    
#     # Fix: Convert numpy array to tensor and move to correct device
#     Z_input = torch.from_numpy(
#         (V[:, :s_ks].T @ X_ks).T.astype(np.float32)).to(device)
#     with torch.no_grad():
#         model_ks.eval()
#         _, X_sparse_recon_tensor = model_ks(Z_input)
#         X_sparse_recon = X_sparse_recon_tensor.cpu().numpy().T 
    
#     # Calculate errors
#     pod_error = X_ks - X_pod_recon
#     sparse_error = X_ks - X_sparse_recon
    
#     fig, axes = plt.subplots(2, 2, figsize=(16, 12))
#     # (1,1) POD reconstruction
#     im1 = axes[0,0].imshow(
#         X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
#         extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
#     axes[0,0].set_xlabel('Time')
#     axes[0,0].set_ylabel('Space (x)')
#     axes[0,0].set_title('POD Reconstruction')
#     plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')
    
#     # (1,2) POD error
#     im2 = axes[0,1].imshow(
#         pod_error, aspect='auto', cmap='RdBu', origin='lower',
#         extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
#     axes[0,1].set_xlabel('Time')
#     axes[0,1].set_ylabel('Space (x)')
#     axes[0,1].set_title('POD Error')
#     plt.colorbar(im2, ax=axes[0,1], label='Error')
    
#     # (2,1) Sparse reconstruction
#     im3 = axes[1,0].imshow(
#         X_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
#         extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
#     axes[1,0].set_xlabel('Time')
#     axes[1,0].set_ylabel('Space (x)')
#     axes[1,0].set_title('Sparse Reconstruction')
#     plt.colorbar(im3, ax=axes[1,0], label='u(x,t)')
    
#     # (2,2) Sparse error
#     im4 = axes[1,1].imshow(
#         sparse_error, aspect='auto', cmap='RdBu', origin='lower',
#         extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
#     axes[1,1].set_xlabel('Time')
#     axes[1,1].set_ylabel('Space (x)')
#     axes[1,1].set_title('Sparse Error')
#     plt.colorbar(im4, ax=axes[1,1], label='Error')
    
#     plt.tight_layout()
#     plt.savefig('../figures/kse_comparison.png', dpi=300)
#     plt.show()
#     plt.close(fig)

#     #%% Test the model on a new sample
#     # Generate new test data with different parameters
#     X_test, xspan_test, tspan_test = generate_kse_data(
#         nx=n_grids, nt=5000, L=32*np.pi, t_max=150.0)

#     # Project test data onto the learned POD basis
#     V_test, _, _ = np.linalg.svd(X_test, full_matrices=False)
#     Z_test = torch.from_numpy(
#         (V[:, :s_ks].T @ X_test).T.astype(np.float32)).to(device)

#     # POD reconstruction using selected modes
#     X_test_pod_recon = V_selected @ V_selected.T @ X_test

#     # Test the model
#     with torch.no_grad():
#         model_ks.eval()
#         _, X_test_recon_tensor = model_ks(Z_test)
#         X_test_recon = X_test_recon_tensor.cpu().numpy().T

#     # Calculate reconstruction errors
#     test_error_sparse = X_test - X_test_recon
#     test_error_pod = X_test - X_test_pod_recon
#     relative_error_sparse = np.linalg.norm(test_error_sparse, 'fro') / np.linalg.norm(X_test, 'fro')
#     relative_error_pod = np.linalg.norm(test_error_pod, 'fro') / np.linalg.norm(X_test, 'fro')
#     print(f"Test reconstruction relative error (Sparse): {relative_error_sparse:.4f}")
#     print(f"Test reconstruction relative error (POD): {relative_error_pod:.4f}")

#     # Plot test results
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))

#     # Original test data
#     im1 = axes[0,0].imshow(
#         X_test, aspect='auto', cmap='viridis', origin='lower',
#         extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
#     axes[0,0].set_xlabel('Time')
#     axes[0,0].set_ylabel('Space (x)')
#     axes[0,0].set_title('Test Data (Original)')
#     plt.colorbar(im1, ax=axes[0,0], label='u(x,t)')

#     # POD reconstruction
#     im2 = axes[0,1].imshow(
#         X_test_pod_recon, aspect='auto', cmap='viridis', origin='lower',
#         extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
#     axes[0,1].set_xlabel('Time')
#     axes[0,1].set_ylabel('Space (x)')
#     axes[0,1].set_title('Test Data (POD Reconstruction)')
#     plt.colorbar(im2, ax=axes[0,1], label='u(x,t)')

#     # POD error
#     im3 = axes[0,2].imshow(
#         test_error_pod, aspect='auto', cmap='RdBu', origin='lower',
#         extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
#     axes[0,2].set_xlabel('Time')
#     axes[0,2].set_ylabel('Space (x)')
#     axes[0,2].set_title('POD Test Error')
#     plt.colorbar(im3, ax=axes[0,2], label='Error')

#     # Sparse reconstruction
#     im4 = axes[1,1].imshow(
#         X_test_recon, aspect='auto', cmap='viridis', origin='lower',
#         extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
#     axes[1,1].set_xlabel('Time')
#     axes[1,1].set_ylabel('Space (x)')
#     axes[1,1].set_title('Test Data (Sparse Reconstruction)')
#     plt.colorbar(im4, ax=axes[1,1], label='u(x,t)')

#     # Sparse error
#     im5 = axes[1,2].imshow(
#         test_error_sparse, aspect='auto', cmap='RdBu', origin='lower',
#         extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]])
#     axes[1,2].set_xlabel('Time')
#     axes[1,2].set_ylabel('Space (x)')
#     axes[1,2].set_title('Sparse Test Error')
#     plt.colorbar(im5, ax=axes[1,2], label='Error')

#     # Hide empty subplot
#     axes[1,0].set_visible(False)

#     plt.tight_layout()
#     plt.savefig('../figures/kse_test_results.png', dpi=300)
#     plt.show()
#     plt.close(fig)
# # %%
