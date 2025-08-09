
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

    # ---------- KSE ----------
    X, xspan, tspan = generate_kse_data(
        nx=n_grids, nt=2500, L=32*np.pi, t_max=100.0
    )
    X = X.astype(np.float64) 
    d, n = X.shape
    s = min(d, n)
    s = 100
    r = 15    
    p = r**2

    # Create flow field plot for KSE (sanity check)
    if sanity_check:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            X, aspect='auto', cmap='viridis', origin='lower',
            extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]])
        ax.set_xlabel('Time')
        ax.set_ylabel('Space (x)')
        ax.set_title('Kuramoto-Sivashinsky Equation Solution')
        plt.colorbar(im, ax=ax, label='u(x,t)')
        plt.tight_layout()
        plt.savefig('figures/kse/kse_data.png', dpi=300)
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
    print("Selected modes (I_qm):", I_qm)
    np.save("results/kse/I_qm.npy", I_qm)


#%% %==================== Configuration of SparseModesNet =====================%

    # Configure conveniently using dictionary
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
        'hidden_units': [r, 512, p],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 2,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.01,
        'lasso_epochs': 100,
        'M': 12.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'lasso_bias': True,
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
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
        'label': "Kuramoto-Sivashinsky",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_2, I_nn_2, omegas_2, path_history = smn.fit(X, config)
    torch.save(model_2, "results/kse/sparsemodesnet_model_pi2net.pth")
    np.save("results/kse/I_nn_pi2net.npy", I_nn_2)
    np.save("results/kse/omegas_pi2net.npy", omegas_2)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_2, I_nn_2, config.s, save=True, 
                     filename='figures/kse/omega_evolution_pi2net.png')


#%% %=============== Configuration of SparseModesNet (Pi3Net) =================%

    # Configure conveniently using dictionary
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
        'hidden_units': [r, 512, p],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 3,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.01,
        'lasso_epochs': 100,
        'M': 12.0,
        'lasso_batch_size': 200,
        'lasso_optimizer': 'Adam',
        'lasso_bias': True,
        'device': device,
        'max_no_change': 50,
        'alpha': 1.0,
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
        'label': "Kuramoto-Sivashinsky",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3, I_nn_3, omegas_3, path_history = smn.fit(X, config)
    torch.save(model_3, "results/kse/sparsemodesnet_model_pi3net.pth")
    np.save("results/kse/I_nn.npy", I_nn_3)
    np.save("results/kse/omegas.npy", omegas_3)

    #%%
    np.save("results/kse/I_nn_pi3net.npy", I_nn_3)
    np.save("results/kse/omegas_pi3net.npy", omegas_3)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, 
                     filename='figures/kse/omega_evolution_pi3net.png')

#%% #===================== Plot Reconstruction Errors =========================#
    # Collect reconstruction errors for different numbers of modes
    mode_counts = []
    qm_errors = []
    pod_errors = []
    sparse_2_errors = []
    sparse_3_errors = []

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

        # SparseModesNet (pi2net) with first r_test modes from selected modes
        if len(I_nn_2) >= r_test:
            V_tmp = np.zeros((d, r))
            V_tmp[:, :r_test] = V_all[:, I_nn_2[:r_test]]
            Z_input_test = torch.from_numpy((V_tmp.T @ X_proc_test).T).to(device)
            with torch.no_grad():
                if  config.network.network_type == 'QM' or config.network.network_type == 'CM':
                    # Use analytical decoder
                    X_sparse_recon_test, _, _ = model_2(V_tmp.T @ X_proc_test)
                else:
                    model_2.eval()
                    # Retrain the weight matrix
                    X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
                    _, X_sparse_lin, N_sparse_out = model_2(Z_input_test)
                    resid = X_proc_test_tensor.T - X_sparse_lin
                    model_2.update_nonlinear_weight(resid, N_sparse_out, 
                                                  config.training.reg_param)
                    X_sparse_recon_tensor_test, _, _ = model_2(Z_input_test)
                    X_sparse_recon_test = X_sparse_recon_tensor_test.cpu().numpy().T

                X_sparse_recon_test = config.preprocessing.backward(X_sparse_recon_test)
            
            recon_error_sparse_test = np.linalg.norm(X - X_sparse_recon_test, ord='fro')
            rel_recon_error_sparse_2_test = recon_error_sparse_test / np.linalg.norm(X, ord='fro')
        else:
            rel_recon_error_sparse_2_test = np.nan
        
        # SparseModesNet (pi2net) with first r_test modes from selected modes
        if len(I_nn_3) >= r_test:
            V_tmp = np.zeros((d, r))
            V_tmp[:, :r_test] = V_all[:, I_nn_3[:r_test]]
            Z_input_test = torch.from_numpy((V_tmp.T @ X_proc_test).T).to(device)
            with torch.no_grad():
                if  config.network.network_type == 'QM' or config.network.network_type == 'CM':
                    # Use analytical decoder
                    X_sparse_recon_test, _, _ = model_3(V_tmp.T @ X_proc_test)
                else:
                    model_3.eval()
                    # Retrain the weight matrix
                    X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
                    _, X_sparse_lin, N_sparse_out = model_3(Z_input_test)
                    resid = X_proc_test_tensor.T - X_sparse_lin
                    model_3.update_nonlinear_weight(resid, N_sparse_out, 
                                                  config.training.reg_param)
                    X_sparse_recon_tensor_test, _, _ = model_3(Z_input_test)
                    X_sparse_recon_test = X_sparse_recon_tensor_test.cpu().numpy().T

                X_sparse_recon_test = config.preprocessing.backward(X_sparse_recon_test)
            
            recon_error_sparse_test = np.linalg.norm(X - X_sparse_recon_test, ord='fro')
            rel_recon_error_sparse_3_test = recon_error_sparse_test / np.linalg.norm(X, ord='fro')
        else:
            rel_recon_error_sparse_3_test = np.nan
        
        mode_counts.append(r_test)
        qm_errors.append(rel_recon_error_qm_test)
        pod_errors.append(rel_recon_error_pod_test)
        sparse_2_errors.append(rel_recon_error_sparse_2_test)
        sparse_3_errors.append(rel_recon_error_sparse_3_test)

    # Plot reconstruction errors
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.semilogy(mode_counts, qm_errors, '-^', label='Greedy Quadratic Manifold', 
                markersize=10, linewidth=4)
    ax.semilogy(mode_counts, pod_errors, '-o', label='POD (leading-r)', 
                markersize=10, linewidth=4)
    
    # Only plot valid SparseModesNet errors
    valid_sparse_errors = [err for err in sparse_2_errors if not np.isnan(err)]
    valid_mode_counts = [mode_counts[i] for i, err in enumerate(sparse_2_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts, valid_sparse_errors, '-s', label='SparseModesNet (Pi2Net)', 
                markersize=10, linewidth=4)

    # Only plot valid SparseModesNet errors
    valid_sparse_errors = [err for err in sparse_3_errors if not np.isnan(err)]
    valid_mode_counts = [mode_counts[i] for i, err in enumerate(sparse_3_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts, valid_sparse_errors, '-x', label='SparseModesNet (Pi3Net)', 
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

    # Quadratic Manifold reconstruction
    Z_qm = V_qm.T @ (X - mu_qm)
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Leading-r POD reconstruction (using top r modes)
    r = len(I_nn_2)  # Use Pi2Net length for consistency
    V_leading_r = V[:, :r]
    X_pod_recon = V_leading_r @ V_leading_r.T @ X_proc
    X_pod_recon = config.preprocessing.backward(X_pod_recon)
    
    # SparseModesNet Pi2Net reconstruction
    Z_input_2 = torch.from_numpy((V[:, I_nn_2].T @ X_proc).T).to(device)
    with torch.no_grad():
        if config.network.network_type == 'QM' or config.network.network_type == 'CM':
            X_sparse_recon_2 = model_2(V[:, I_nn_2].T @ X_proc)
        else:
            model_2.eval()
            X_sparse_recon_tensor_2, _, _ = model_2(Z_input_2)
            X_sparse_recon_2 = X_sparse_recon_tensor_2.cpu().numpy().T 
        X_sparse_recon_2 = config.preprocessing.backward(X_sparse_recon_2)

    # SparseModesNet Pi3Net reconstruction
    Z_input_3 = torch.from_numpy((V[:, I_nn_3].T @ X_proc).T).to(device)
    with torch.no_grad():
        if config.network.network_type == 'QM' or config.network.network_type == 'CM':
            X_sparse_recon_3 = model_3(V[:, I_nn_3].T @ X_proc)
        else:
            model_3.eval()
            X_sparse_recon_tensor_3, _, _ = model_3(Z_input_3)
            X_sparse_recon_3 = X_sparse_recon_tensor_3.cpu().numpy().T 
        X_sparse_recon_3 = config.preprocessing.backward(X_sparse_recon_3)
    
    # Calculate errors
    pod_error = X - X_pod_recon
    qm_error = X - X_qm_recon
    sparse_error_2 = X - X_sparse_recon_2
    sparse_error_3 = X - X_sparse_recon_3
    
    # Set consistent color scales for reconstructions
    recon_vmin = min(X.min(), X_pod_recon.min(), X_qm_recon.min(), 
                     X_sparse_recon_2.min(), X_sparse_recon_3.min())
    recon_vmax = max(X.max(), X_pod_recon.max(), X_qm_recon.max(), 
                     X_sparse_recon_2.max(), X_sparse_recon_3.max())
    
    # Set consistent color scales for errors
    error_vmax = max(np.abs(pod_error).max(), np.abs(qm_error).max(), 
                     np.abs(sparse_error_2).max(), np.abs(sparse_error_3).max())
    error_vmin = -error_vmax
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
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
    axes[0,2].set_title('Quadratic Manifold', fontsize=15)
    
    # (1,4) SparseModesNet Pi2Net reconstruction
    im4 = axes[0,3].imshow(
        X_sparse_recon_2, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,3].set_title('SparseModesNet (Pi2Net)', fontsize=15)

    # (1,5) SparseModesNet Pi3Net reconstruction
    im5 = axes[0,4].imshow(
        X_sparse_recon_3, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,4].set_title('SparseModesNet (Pi3Net)', fontsize=15)
    
    # Row 2: Errors
    # (2,1) Empty - no error for original data
    axes[1,0].axis('off')
    
    # (2,2) POD error
    im6 = axes[1,1].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,1].set_xlabel('Time', fontsize=14)
    axes[1,1].set_ylabel('Space (x)', fontsize=14)
    axes[1,1].set_title('POD Error', fontsize=15)
    
    # (2,3) Quadratic Manifold error
    im7 = axes[1,2].imshow(
        qm_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,2].set_xlabel('Time', fontsize=14)
    axes[1,2].set_title('Quadratic Manifold Error', fontsize=15)
    
    # (2,4) SparseModesNet Pi2Net error
    im8 = axes[1,3].imshow(
        sparse_error_2, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,3].set_xlabel('Time', fontsize=14)
    axes[1,3].set_title('SparseModesNet (Pi2Net) Error', fontsize=15)

    # (2,5) SparseModesNet Pi3Net error
    im9 = axes[1,4].imshow(
        sparse_error_3, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,4].set_xlabel('Time', fontsize=14)
    axes[1,4].set_title('SparseModesNet (Pi3Net) Error', fontsize=15)
    
    # Add unified colorbars
    cax1 = fig.add_axes([0.92, 0.57, 0.015, 0.35])
    cbar1 = plt.colorbar(im5, cax=cax1)
    cbar1.set_label('u(x,t)', fontsize=15)
    cbar1.ax.tick_params(labelsize=14)
    
    cax2 = fig.add_axes([0.92, 0.11, 0.015, 0.35])
    cbar2 = plt.colorbar(im9, cax=cax2)
    cbar2.set_label('Error', fontsize=15)
    cbar2.ax.tick_params(labelsize=14)

    plt.subplots_adjust(left=0.04, right=0.9, top=0.92, bottom=0.1, 
                        wspace=0.25, hspace=0.3)
    plt.suptitle('Flow Field Reconstruction Comparison', fontsize=19, y=0.98)
    plt.savefig('figures/kse/flow_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


#%% %================= Plot waves at specific time points =====================%
    # Select 3 equally spaced time points
    n_times = len(tspan)
    time_indices = [n_times//4, n_times//2, 3*n_times//4]
    time_points = [tspan[i] for i in time_indices]

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices, time_points)):
        # Plot original data
        ax.plot(xspan, X[:, t_idx], 'k-', linewidth=3, 
                label='Original', alpha=0.9)
        # Plot leading-r POD reconstruction
        ax.plot(xspan, X_pod_recon[:, t_idx], '--', 
                linewidth=2, label=f'POD (r={r})', alpha=0.8)
        # Plot Quadratic Manifold reconstruction
        ax.plot(xspan, X_qm_recon[:, t_idx], '-.', 
                linewidth=2, label='Quadratic Manifold', alpha=0.8)
        # Plot SparseModesNet Pi2Net reconstruction
        ax.plot(xspan, X_sparse_recon_2[:, t_idx], ':', 
                linewidth=2, label='SparseModesNet (Pi2Net)', alpha=0.8)
        # Plot SparseModesNet Pi3Net reconstruction
        ax.plot(xspan, X_sparse_recon_3[:, t_idx], '-', 
                linewidth=2, label='SparseModesNet (Pi3Net)', alpha=0.8)
        
        ax.set_xlabel('Space (x)', fontsize=14)
        ax.set_ylabel('u(x,t)', fontsize=14)
        ax.set_title(f't = {t_val:.3f}', fontsize=16)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=15)

    plt.tight_layout()
    plt.suptitle('Wave Profiles at Different Time Points', fontsize=19, y=1.02)
    plt.savefig('figures/kse/wave_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


#%% %===================== Additional Plotting and Analysis ===================%
    # Compute SVD for all datasets
    U_orig, S_orig, _ = np.linalg.svd(X, full_matrices=False)
    U_pod, S_pod, _ = np.linalg.svd(X_pod_recon, full_matrices=False)
    U_qm, S_qm, _ = np.linalg.svd(X_qm_recon, full_matrices=False)
    U_sparse_2, S_sparse_2, _ = np.linalg.svd(X_sparse_recon_2, full_matrices=False)
    U_sparse_3, S_sparse_3, _ = np.linalg.svd(X_sparse_recon_3, full_matrices=False)
    
    # Plot singular values comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left plot: Singular values
    n_sv = min(100, len(S_orig))
    modes_range = range(1, n_sv + 1)
    
    axes[0].semilogy(modes_range, S_orig[:n_sv], 'k-o', linewidth=3, markersize=8, 
                     label='Original Data', alpha=0.9)
    axes[0].semilogy(modes_range, S_pod[:n_sv], '--s', linewidth=2, markersize=6,
                     label=f'POD Reconstruction (r={r})', alpha=0.8)
    axes[0].semilogy(modes_range, S_qm[:n_sv], '-.^', linewidth=2, markersize=6,
                     label='Greedy Quadratic Manifold', alpha=0.8)
    axes[0].semilogy(modes_range, S_sparse_2[:n_sv], ':d', linewidth=2, markersize=6,
                     label='SparseModesNet (Pi2Net)', alpha=0.8)
    axes[0].semilogy(modes_range, S_sparse_3[:n_sv], ':', linewidth=2, markersize=6,
                     label='SparseModesNet (Pi3Net)', alpha=0.8)
    
    axes[0].set_xlabel('Mode Number', fontsize=14)
    axes[0].set_ylabel('Singular Value', fontsize=14)
    axes[0].set_title('Singular Values Comparison', fontsize=16)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=18)
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    
    # Right plot: Relative singular value errors
    sv_error_pod = np.abs(S_pod[:n_sv] - S_orig[:n_sv]) / S_orig[:n_sv]
    sv_error_qm = np.abs(S_qm[:n_sv] - S_orig[:n_sv]) / S_orig[:n_sv]
    sv_error_sparse_2 = np.abs(S_sparse_2[:n_sv] - S_orig[:n_sv]) / S_orig[:n_sv]
    sv_error_sparse_3 = np.abs(S_sparse_3[:n_sv] - S_orig[:n_sv]) / S_orig[:n_sv]
    
    axes[1].semilogy(modes_range, sv_error_pod, '--s', linewidth=2, markersize=6,
                     label=f'POD Reconstruction (r={r})', alpha=0.8)
    axes[1].semilogy(modes_range, sv_error_qm, '-.^', linewidth=2, markersize=6,
                     label='Greedy Quadratic Manifold', alpha=0.8)
    axes[1].semilogy(modes_range, sv_error_sparse_2, ':d', linewidth=2, markersize=6,
                     label='SparseModesNet (Pi2Net)', alpha=0.8)
    axes[1].semilogy(modes_range, sv_error_sparse_3, ':', linewidth=2, markersize=6,
                     label='SparseModesNet (Pi3Net)', alpha=0.8)
    
    axes[1].set_xlabel('Mode Number', fontsize=14)
    axes[1].set_ylabel('Relative Error in Singular Values', fontsize=14)
    axes[1].set_title('Singular Value Reconstructions', fontsize=16)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('figures/kse/singular_values_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Plot the first 20 spatial modes comparison
    n_modes = np.hstack([np.arange(15), [16, 24, 37, 61, 95]])
    fig, axes = plt.subplots(5, 4, figsize=(22, 15))
    axes = axes.flatten()

    for (i, md) in enumerate(n_modes):
        ax = axes[i]
        
        # Get the original mode as reference
        mode_orig = U_orig[:, md]
        
        # Get modes from different methods
        mode_pod = U_pod[:, md]
        mode_qm = U_qm[:, md]
        mode_sparse_2 = U_sparse_2[:, md]
        mode_sparse_3 = U_sparse_3[:, md]
        
        # Fix sign flips by checking correlation with original mode
        # If correlation is negative, flip the sign
        if np.corrcoef(mode_orig, mode_pod)[0, 1] < 0:
            mode_pod = -mode_pod
            
        if np.corrcoef(mode_orig, mode_qm)[0, 1] < 0:
            mode_qm = -mode_qm
            
        if np.corrcoef(mode_orig, mode_sparse_2)[0, 1] < 0:
            mode_sparse_2 = -mode_sparse_2
            
        if np.corrcoef(mode_orig, mode_sparse_3)[0, 1] < 0:
            mode_sparse_3 = -mode_sparse_3
        
        # Plot spatial modes with corrected signs
        ax.plot(xspan, mode_orig, 'k-', linewidth=3, label='Original', alpha=0.9)
        ax.plot(xspan, mode_pod, '--', linewidth=2, label='POD', alpha=0.8)
        ax.plot(xspan, mode_qm, '-.', linewidth=2, label='GreedyQM', alpha=0.8)
        ax.plot(xspan, mode_sparse_2, ':', linewidth=2, label='SMN-Pi2', alpha=0.8)
        ax.plot(xspan, mode_sparse_3, '-', linewidth=1.5, label='SMN-Pi3', alpha=0.8)

        # Remove all labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        ax.set_title(f'Mode {md+1} (σ = {S_orig[md]:.2e})', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        if i == 15:
            ax.legend(fontsize=19, ncol=2)

    plt.tight_layout()
    plt.suptitle('Spatial Modes Comparison', fontsize=18, y=1.02)
    plt.savefig('figures/kse/spatial_modes_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    # Plot energy spectrum comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Cumulative energy
    energy_orig = np.cumsum(S_orig**2) / np.sum(S_orig**2)
    energy_pod = np.cumsum(S_pod**2) / np.sum(S_pod**2)
    energy_qm = np.cumsum(S_qm**2) / np.sum(S_qm**2)
    energy_sparse_2 = np.cumsum(S_sparse_2**2) / np.sum(S_sparse_2**2)
    energy_sparse_3 = np.cumsum(S_sparse_3**2) / np.sum(S_sparse_3**2)
    
    n_plot = min(50, len(S_orig))
    mode_range = range(1, n_plot + 1)
    
    ax.plot(mode_range, energy_orig[:n_plot], 'k-o', linewidth=3, markersize=6,
            label='Original Data', alpha=0.9)
    ax.plot(mode_range, energy_pod[:n_plot], '--s', linewidth=2, markersize=4,
            label=f'POD Reconstruction (r={r})', alpha=0.8)
    ax.plot(mode_range, energy_qm[:n_plot], '-.^', linewidth=2, markersize=4,
            label='Greedy Quadratic Manifold', alpha=0.8)
    ax.plot(mode_range, energy_sparse_2[:n_plot], ':d', linewidth=2, markersize=4,
            label='SparseModesNet (Pi2Net)', alpha=0.8)
    ax.plot(mode_range, energy_sparse_3[:n_plot], ':', linewidth=2, markersize=4,
            label='SparseModesNet (Pi3Net)', alpha=0.8)
    
    # Add horizontal lines for energy thresholds
    ax.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5, label='99% Energy')
    ax.axhline(y=0.999, color='gray', linestyle=':', alpha=0.5, label='99.9% Energy')
    
    ax.set_xlabel('Mode Number', fontsize=14)
    ax.set_ylabel('Cumulative Energy Fraction', fontsize=14)
    ax.set_title('Energy Spectrum Comparison: Pi2Net vs Pi3Net', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim([0.8, 1.001])
    
    plt.tight_layout()
    plt.savefig('figures/kse/energy_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
