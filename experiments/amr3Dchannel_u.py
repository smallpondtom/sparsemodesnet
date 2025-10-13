"""
AMR-Wind 3D Channel Flow (u-velocity) simulation experiment using SparseModesNet.
"""

#%% Load modules
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from QM.quadmani import quadmani_greedy, _make_cubic_mapping_jax_fixed
import sparsemodesnet as smn
from utils.channel_data_source import ChannelDataSource

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

def _cubic_mapping_numpy(x):
    """
    Fast vectorized computation of unique cubic terms x ⊗ x ⊗ x (NumPy version).
    Uses meshgrid for efficient index generation.
    
    Args:
        x: np.ndarray of shape (batch_size, n) or (n,)
        
    Returns:
        np.ndarray of shape (batch_size, n*(n+1)*(n+2)//6) or (n*(n+1)*(n+2)//6,)
    """
    if x.ndim == 1:
        n = x.shape[0]
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products
        result = x[i_indices] * x[j_indices] * x[k_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products for all batches
        result = x[:, i_indices] * x[:, j_indices] * x[:, k_indices]
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

    # Load the data source
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[2, 2, 1],
        no_pressure=True,
        which_velocity="u"
    )

    # Data parameters
    n_snapshots = 500

    # Load the data
    X = ds.get_matrix(snapshot_range=slice(0, n_snapshots))

    # Dimensions
    d, n = X.shape
    s = min(d, n)
    s = 100
    r = 20
    p = int(r**2)


#%% #========================= Greedy Cubic Manifold ==========================#
    print("\n" + "="*60)
    print("GREEDY CUBIC MANIFOLD")
    V_cm, W_cm, mu_cm, I_cm = quadmani_greedy(
        X, r, s, 1e-2, np.array([], dtype=int), 
        feature_map=_make_cubic_mapping_jax_fixed(max_r=r))
    
    # Print the selected modes
    print("Selected modes (I_cm):", I_cm)
    np.save("results/amr3Dchannel/u/I_cm.npy", I_cm)


#%% %================= Configuration of SparseModesNet Pi3Net =================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 624,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        'hidden_units': [64, 720, 624],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 3,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.0005,
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
        'label': "Streamwise 3D Channel Flow Pi3Net",
        'enable_logging': True
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3, I_nn_3, omegas_3, path_history = smn.fit(X, config)
    torch.save(model_3, "results/amr3Dchannel/u/sparsemodesnet_model_pi3net.pth")
    np.save("results/amr3Dchannel/u/I_nn_pi2net.npy", I_nn_3)
    np.save("results/amr3Dchannel/u/omegas_pi2net.npy", omegas_3)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, show=False,
                     legend_loc='best',
                     title=r'Streamwise: $\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/u/omega_evolution_pi3net.pdf')

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, show=False,
                     legend_loc='best',
                     title=r'Streamwise: $\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/u/omega_evolution_pi3net.pdf')

#%% %========== Configuration of SparseModesNet Pi3Net (leading-r) ============%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 624,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        'hidden_units': [64, 720, 624],  # PiNet
        'network_type': 'PiNetCCP',
        'poly_order': 3,
        'num_polys': 1,
        'drop_linear': False,
        'drop_constant': False,
        # Mode Selection Phase
        'lam0': 3.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.0005,
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
        'I_nn': range(r),
        'device': device,
        # Experiment Setup
        'label': "Streamwise 3D Channel Flow Pi3Net Leading-r",
        'enable_logging': True
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3p, _, _, _ = smn.fit(X, config)
    torch.save(model_3p, "results/amr3Dchannel/u/sparsemodesnet_model_pi3net_leading.pth")

#%% #===================== Plot Reconstruction Errors =========================#
    # Collect reconstruction errors for different numbers of modes
    mode_counts = []
    qm_errors = []
    cm_errors = []
    pod_errors = []
    sparse_2_errors = []
    sparse_3_errors = []
    sparse_2p_errors = []  # New: Pi2Net with leading modes
    sparse_3p_errors = []  # New: Pi3Net with leading modes

    X_proc = config.preprocessing.forward(X)
    V_all = np.linalg.svd(X_proc, full_matrices=False)[0][:, :s]

    # Test different numbers of modes
    for r_test in range(1, min(r + 1, 21)):  # Test up to 20 modes or r
        # # Quadratic Manifold with r_test modes
        # V_test, W_test, shift_test, I_qm_test = quadmani_greedy(
        #     X, r_test, s, 1e-2, np.array([], dtype=int))
        # shift_test = shift_test.reshape(-1, 1)
        # Z_qm_test = V_test.T @ (X - shift_test)
        # Z_quad_qm_test = quadratic_mapping_numpy(Z_qm_test.T).T
        # recon_error_qm_test = np.linalg.norm(
        #     X - (V_test @ Z_qm_test + W_test @ Z_quad_qm_test + shift_test), ord='fro')
        # rel_recon_error_qm_test = recon_error_qm_test / np.linalg.norm(X, ord='fro')

        # Cubic Manifold with r_test modes
        V_test_3, W_test_3, shift_test, I_qm_test_3 = quadmani_greedy(
            X, r_test, s, 1e-2, np.array([], dtype=int), 
            feature_map=_make_cubic_mapping_jax_fixed(max_r=r_test))
        Z_qm_test_3 = V_test_3.T @ (X - shift_test)
        Z_quad_qm_test_3 = _cubic_mapping_numpy(Z_qm_test_3.T).T
        recon_error_qm_test_3 = np.linalg.norm(
            X - (V_test_3 @ Z_qm_test_3 + W_test_3 @ Z_quad_qm_test_3 + shift_test), ord='fro')
        rel_recon_error_qm_test_3 = recon_error_qm_test_3 / np.linalg.norm(X, ord='fro')
        
        # Leading-r POD reconstruction
        X_proc_test = config.preprocessing.forward(X)
        V_leading_test = np.linalg.svd(X_proc_test, full_matrices=False)[0][:, :r_test]
        X_pod_recon_test = V_leading_test @ V_leading_test.T @ X_proc_test
        X_pod_recon_test = config.preprocessing.backward(X_pod_recon_test)
        recon_error_pod_test = np.linalg.norm(X - X_pod_recon_test, ord='fro')
        rel_recon_error_pod_test = recon_error_pod_test / np.linalg.norm(X, ord='fro')

        # # SparseModesNet (pi2net) with first r_test modes from selected modes
        # if len(I_nn_2) >= r_test:
        #     V_tmp = np.zeros((d, r))
        #     V_tmp[:, :r_test] = V_all[:, I_nn_2[:r_test]]
        #     Z_input_test = torch.from_numpy((V_tmp.T @ X_proc_test).T).to(device)
        #     with torch.no_grad():
        #         if  config.network.network_type == 'QM' or config.network.network_type == 'CM':
        #             # Use analytical decoder
        #             X_sparse_recon_test, _, _ = model_2(V_tmp.T @ X_proc_test)
        #         else:
        #             model_2.eval()
        #             # Retrain the weight matrix
        #             X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
        #             _, X_sparse_lin, N_sparse_out = model_2(Z_input_test)
        #             resid = X_proc_test_tensor.T - X_sparse_lin
        #             model_2.update_nonlinear_weight(resid, N_sparse_out, 
        #                                           config.training.reg_param)
        #             X_sparse_recon_tensor_test, _, _ = model_2(Z_input_test)
        #             X_sparse_recon_test = X_sparse_recon_tensor_test.cpu().numpy().T

        #         X_sparse_recon_test = config.preprocessing.backward(X_sparse_recon_test)
            
        #     recon_error_sparse_test = np.linalg.norm(X - X_sparse_recon_test, ord='fro')
        #     rel_recon_error_sparse_2_test = recon_error_sparse_test / np.linalg.norm(X, ord='fro')
        # else:
        #     rel_recon_error_sparse_2_test = np.nan
        
        # SparseModesNet (pi3net) with first r_test modes from selected modes
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

        # # SparseModesNet Pi2Net with leading r_test modes (model_2p)
        # V_tmp_2p = np.zeros((d, r))
        # V_tmp_2p[:, :r_test] = V_all[:, :r_test]  # Use first r_test leading modes
        # Z_input_test_2p = torch.from_numpy((V_tmp_2p.T @ X_proc_test).T).to(device)
        # with torch.no_grad():
        #     if config.network.network_type == 'QM' or config.network.network_type == 'CM':
        #         # Use analytical decoder
        #         X_sparse_recon_test_2p, _, _ = model_2p(V_tmp_2p.T @ X_proc_test)
        #     else:
        #         model_2p.eval()
        #         # Retrain the weight matrix
        #         X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
        #         _, X_sparse_lin_2p, N_sparse_out_2p = model_2p(Z_input_test_2p)
        #         resid_2p = X_proc_test_tensor.T - X_sparse_lin_2p
        #         model_2p.update_nonlinear_weight(resid_2p, N_sparse_out_2p, 
        #                                        config.training.reg_param)
        #         X_sparse_recon_tensor_test_2p, _, _ = model_2p(Z_input_test_2p)
        #         X_sparse_recon_test_2p = X_sparse_recon_tensor_test_2p.cpu().numpy().T

        #     X_sparse_recon_test_2p = config.preprocessing.backward(X_sparse_recon_test_2p)
        
        # recon_error_sparse_test_2p = np.linalg.norm(X - X_sparse_recon_test_2p, ord='fro')
        # rel_recon_error_sparse_2p_test = recon_error_sparse_test_2p / np.linalg.norm(X, ord='fro')

        # SparseModesNet Pi3Net with leading r_test modes (model_3p)
        V_tmp_3p = np.zeros((d, r))
        V_tmp_3p[:, :r_test] = V_all[:, :r_test]  # Use first r_test leading modes
        Z_input_test_3p = torch.from_numpy((V_tmp_3p.T @ X_proc_test).T).to(device)
        with torch.no_grad():
            if config.network.network_type == 'QM' or config.network.network_type == 'CM':
                # Use analytical decoder
                X_sparse_recon_test_3p, _, _ = model_3p(V_tmp_3p.T @ X_proc_test)
            else:
                model_3p.eval()
                # Retrain the weight matrix
                X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
                _, X_sparse_lin_3p, N_sparse_out_3p = model_3p(Z_input_test_3p)
                resid_3p = X_proc_test_tensor.T - X_sparse_lin_3p
                model_3p.update_nonlinear_weight(resid_3p, N_sparse_out_3p, 
                                               config.training.reg_param)
                X_sparse_recon_tensor_test_3p, _, _ = model_3p(Z_input_test_3p)
                X_sparse_recon_test_3p = X_sparse_recon_tensor_test_3p.cpu().numpy().T

            X_sparse_recon_test_3p = config.preprocessing.backward(X_sparse_recon_test_3p)
        
        recon_error_sparse_test_3p = np.linalg.norm(X - X_sparse_recon_test_3p, ord='fro')
        rel_recon_error_sparse_3p_test = recon_error_sparse_test_3p / np.linalg.norm(X, ord='fro')
        
        mode_counts.append(r_test)
        # qm_errors.append(rel_recon_error_qm_test)
        cm_errors.append(rel_recon_error_qm_test_3)
        pod_errors.append(rel_recon_error_pod_test)
        # sparse_2_errors.append(rel_recon_error_sparse_2_test)
        sparse_3_errors.append(rel_recon_error_sparse_3_test)
        # sparse_2p_errors.append(rel_recon_error_sparse_2p_test)
        sparse_3p_errors.append(rel_recon_error_sparse_3p_test)

    np.savez(
        "results/amr3Dchannel/u/reconstruction_errors.npz",
        mode_counts=mode_counts,
        # qm_errors=qm_errors,
        cm_errors=cm_errors,
        pod_errors=pod_errors,
        # sparse_2_errors=sparse_2_errors,
        sparse_3_errors=sparse_3_errors,
        # sparse_2p_errors=sparse_2p_errors,
        sparse_3p_errors=sparse_3p_errors
    )

    np.savez(
        "results/amr3Dchannel/u/V_all.npz",
        V_all=V_all
    )

#%% #======================= Plot reconstruction errors =======================#
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # ax.semilogy(mode_counts, qm_errors, '-^', label='Greedy Quadratic Manifold', 
    #             markersize=8, linewidth=3)
    ax.semilogy(mode_counts, cm_errors, '--v', label='Greedy Cubic Manifold',
                markersize=8, linewidth=3)
    ax.semilogy(mode_counts, pod_errors, '-o', label='POD (leading-r)', 
                markersize=8, linewidth=3)
    
    # Only plot valid SparseModesNet errors (selected modes)
    # valid_sparse_2_errors = [err for err in sparse_2_errors if not np.isnan(err)]
    # valid_mode_counts_2 = [mode_counts[i] for i, err in enumerate(sparse_2_errors) if not np.isnan(err)]
    # ax.semilogy(valid_mode_counts_2, valid_sparse_2_errors, '-s', label=r'SparseModesNet $\Pi_2$-Net (selected)', 
    #             markersize=8, linewidth=3)

    valid_sparse_3_errors = [err for err in sparse_3_errors if not np.isnan(err)]
    valid_mode_counts_3 = [mode_counts[i] for i, err in enumerate(sparse_3_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts_3, valid_sparse_3_errors, '-x', label=r'SparseModesNet $\Pi_3$-Net (selected)', 
                markersize=10, linewidth=3)
    
    # Plot SparseModesNet with leading modes
    # ax.semilogy(mode_counts, sparse_2p_errors, ':s', label=r'SparseModesNet $\Pi_2$-Net (leading-r)', 
    #             markersize=8, linewidth=3, alpha=0.8)
    ax.semilogy(mode_counts, sparse_3p_errors, ':x', label=r'SparseModesNet $\Pi_3$-Net (leading-r)', 
                markersize=10, linewidth=3, alpha=0.8)
    
    ax.set_xlabel(r'Number of Modes $r$', fontsize=25)
    ax.set_ylabel('Relative Reconstruction Error', fontsize=25)
    ax.set_title('Streamwise: Reconstruction Error vs. Number of Modes', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=22, loc='lower left')
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig('figures/amr3Dchannel/u/reconstruction_errors.pdf', 
                dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)

#%% %============ Plot the reconstructed flow fields (heatmap) ================%
    # Select a snapshot to visualize
    snapshot_idx = 50  # Change this index to visualize different snapshots
    xsnap = X[:, snapshot_idx]
    xsnap_proc = X_proc[:, snapshot_idx]

    # # Quadratic Manifold reconstruction
    # Z_qm = V_qm.T @ (xsnap - mu_qm)
    # Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    # X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Cubic Manifold reconstruction
    Z_cm = V_cm.T @ (xsnap - mu_cm)
    Z_quad_cm = _cubic_mapping_numpy(Z_cm.T).T
    X_cm_recon = V_cm @ Z_cm + W_cm @ Z_quad_cm + mu_cm

    # Leading-r POD reconstruction (using top r modes)
    V_leading_r = V_all[:, :r]
    # X_pod_recon = V_leading_r @ V_leading_r.T @ X_proc
    # X_pod_recon = config.preprocessing.backward(X_pod_recon)
    
    # # SparseModesNet Pi2Net reconstruction
    # Z_input_2 = torch.from_numpy((V_all[:, I_nn_2].T @ X_proc).T).to(device)
    # with torch.no_grad():
    #     model_2.eval()
    #     X_sparse_recon_tensor_2, _, _ = model_2(Z_input_2)
    #     X_sparse_recon_2 = X_sparse_recon_tensor_2.cpu().numpy().T 
    #     X_sparse_recon_2 = config.preprocessing.backward(X_sparse_recon_2)

    # SparseModesNet Pi3Net reconstruction
    Z_input_3 = torch.from_numpy((V_all[:, I_nn_3].T @ X_proc).T).to(device)
    with torch.no_grad():
        model_3.eval()
        X_sparse_recon_tensor_3, _, _ = model_3(Z_input_3)
        X_sparse_recon_3 = X_sparse_recon_tensor_3.cpu().numpy().T 
        X_sparse_recon_3 = config.preprocessing.backward(X_sparse_recon_3)

    # SparseModesNet Pi3Net reconstruction (leading-r)
    Z_input_3p = torch.from_numpy((V_leading_r.T @ xsnap_proc).T).to(device)
    with torch.no_grad():
        model_3p.eval()
        X_sparse_recon_tensor_3p, _, _ = model_3p(Z_input_3p)
        X_sparse_recon_3p = X_sparse_recon_tensor_3p.cpu().numpy().T 
        X_sparse_recon_3p = config.preprocessing.backward(X_sparse_recon_3p)
    
    # Calculate errors
    # pod_error = xsnap - X_pod_recon
    # qm_error = xsnap - X_qm_recon
    cm_error = xsnap - X_cm_recon
    # sparse_error_2 = xsnap - X_sparse_recon_2
    sparse_error_3 = xsnap - X_sparse_recon_3
    sparse_error_3p = xsnap - X_sparse_recon_3p
    
    # Reshape the data here 
    xlen = ds.shape[2]
    ylen = ds.shape[3]
    zlen = ds.shape[4]

    xsnap = xsnap.reshape((xlen, ylen, zlen), order='C')
    # X_qm_recon = X_qm_recon.reshape((xlen, ylen, zlen), order='C')
    X_cm_recon = X_cm_recon.reshape((xlen, ylen, zlen), order='C')
    # X_pod_recon = X_pod_recon.reshape((xlen, ylen, zlen), order='C')
    # X_sparse_recon_2 = X_sparse_recon_2.reshape((xlen, ylen, zlen), order='C')
    X_sparse_recon_3 = X_sparse_recon_3.reshape((xlen, ylen, zlen), order='C')
    X_sparse_recon_3p = X_sparse_recon_3p.reshape((xlen, ylen, zlen), order='C')

    # pod_error = pod_error.reshape((xlen, ylen, zlen), order='C')
    # qm_error = qm_error.reshape((xlen, ylen, zlen), order='C')
    cm_error = cm_error.reshape((xlen, ylen, zlen), order='C')
    # sparse_error_2 = sparse_error_2.reshape((xlen, ylen, zlen), order='C')
    sparse_error_3 = sparse_error_3.reshape((xlen, ylen, zlen), order='C')
    sparse_error_3p = sparse_error_3p.reshape((xlen, ylen, zlen), order='C')

    # Slice 
    zslice_idx = zlen // 2  # Middle slice in z-direction
    xsnap_ = xsnap[:, :, zslice_idx]
    # X_qm_recon_ = X_qm_recon[:, :, zslice_idx]
    X_cm_recon_ = X_cm_recon[:, :, zslice_idx]
    # X_pod_recon_ = X_pod_recon[:, :, zslice_idx
    # X_sparse_recon_2_ = X_sparse_recon_2[:, :, zslice_idx]
    X_sparse_recon_3_ = X_sparse_recon_3[:, :, zslice_idx]
    X_sparse_recon_3p_ = X_sparse_recon_3p[:, :, zslice_idx]

    # pod_error_ = pod_error[:, :, zslice_idx]
    # qm_error_ = qm_error[:, :, zslice_idx]
    cm_error_ = cm_error[:, :, zslice_idx]
    # sparse_error_2_ = sparse_error_2[:, :, zslice_idx]
    sparse_error_3_ = sparse_error_3[:, :, zslice_idx]
    sparse_error_3p_ = sparse_error_3p[:, :, zslice_idx]

    # Set consistent color scales for reconstructions
    recon_vmin = min(xsnap_.min(), X_cm_recon.min(), # X_qm_recon_.min(), X_pod_recon_.min(), 
                     X_sparse_recon_3_.min(), X_sparse_recon_3p_.min())
    recon_vmax = max(xsnap_.max(), X_cm_recon.min(), # X_qm_recon_.max(), X_pod_recon_.max(),
                     X_sparse_recon_3_.max(), X_sparse_recon_3p_.max())
    
    # Set consistent color scales for errors
    error_vmax = max(np.abs(cm_error_).max(), # np.abs(qm_error_).max(), np.abs(pod_error_).max(), 
                     np.abs(sparse_error_3_).max(), np.abs(sparse_error_3p_).max())
    error_vmin = -error_vmax
    
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    
    # Row 1: Reconstructions
    # (1,1) Original data
    im1 = axes[0,0].imshow(
        xsnap_, aspect='auto', cmap='viridis', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,0].set_ylabel(r'$\xi_2$', fontsize=22)
    axes[0,0].set_title('Original Data', fontsize=28)
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])
    
    # # (1,2) Leading-r POD reconstruction
    # im2 = axes[0,1].imshow(
    #     X_pod_recon_, aspect='auto', cmap='viridis', origin='lower',
    #     extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
    #     vmin=recon_vmin, vmax=recon_vmax)
    # axes[0,1].set_title(f'POD Reconstruction (r={r})', fontsize=28)
    # axes[0,1].set_xticks([])
    # axes[0,1].set_yticks([])
    
    # (1,2) Cubic Manifold reconstruction
    im2 = axes[0,1].imshow(
        X_cm_recon_, aspect='auto', cmap='viridis', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,1].set_title('GreedyCM', fontsize=28)
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])
    
    # (1,3) SparseModesNet Pi3Net (leading-r) reconstruction
    im3 = axes[0,2].imshow(
        X_sparse_recon_3p_, aspect='auto', cmap='viridis', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,2].set_title(r'SMN $\Pi_3$-Net (leading-r)', fontsize=28)
    axes[0,2].set_xticks([])
    axes[0,2].set_yticks([])

    # (1,4) SparseModesNet Pi3Net reconstruction
    im4 = axes[0,3].imshow(
        X_sparse_recon_3_, aspect='auto', cmap='viridis', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,3].set_title(r'SMN $\Pi_3$-Net', fontsize=28)
    axes[0,3].set_xticks([])
    axes[0,3].set_yticks([])
    
    # Row 2: Errors
    # (2,1) Empty - no error for original data
    axes[1,0].axis('off')
    
    # # (2,2) POD error
    # im6 = axes[1,1].imshow(
    #     pod_error, aspect='auto', cmap='RdBu', origin='lower',
    #     extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    #     vmin=error_vmin, vmax=error_vmax)
    # axes[1,1].set_xlabel('Time', fontsize=14)
    # axes[1,1].set_ylabel('Space (x)', fontsize=14)
    # axes[1,1].set_title('POD Error', fontsize=15)
    
    # (2,2) Cubic Manifold error
    im7 = axes[1,1].imshow(
        cm_error_, aspect='auto', cmap='RdBu', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,1].set_xlabel(r'$\xi_1$', fontsize=22)
    axes[1,1].set_ylabel(r'$\xi_2$', fontsize=22)
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])
    
    # (2,3) SparseModesNet Pi3Net (leading-r) error
    im8 = axes[1,2].imshow(
        sparse_error_3p_, aspect='auto', cmap='RdBu', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,2].set_xlabel(r'$\xi_1$', fontsize=22)
    axes[1,2].set_xticks([])
    axes[1,2].set_yticks([])

    # (2,4) SparseModesNet Pi3Net error
    im9 = axes[1,3].imshow(
        sparse_error_3_, aspect='auto', cmap='RdBu', origin='lower',
        extent=[ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,3].set_xlabel(r'$\xi_1$', fontsize=22)
    axes[1,3].set_xticks([])
    axes[1,3].set_yticks([])
    
    # Add unified colorbars
    cax1 = fig.add_axes([0.91, 0.515, 0.015, 0.385])
    cbar1 = plt.colorbar(im4, cax=cax1)
    cbar1.set_label(r'$x(\xi_1,\xi_2,t)$', fontsize=23)
    cbar1.ax.tick_params(labelsize=20)
    
    cax2 = fig.add_axes([0.91, 0.1, 0.015, 0.385])
    cbar2 = plt.colorbar(im8, cax=cax2)
    cbar2.set_label('error', fontsize=23)
    cbar2.ax.tick_params(labelsize=20)

    plt.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.1, 
                        wspace=0.05, hspace=0.075)
    plt.suptitle('Sliced Streamwise Velocity Reconstruction Comparison', fontsize=29, y=0.98)
    plt.savefig('figures/amr3Dchannel/u/u_recon_recomparison.pdf', dpi=300, bbox_inches='tight')
    # plt.show()
    plt.close(fig)
