"""
AMR-Wind 3D Channel Flow (w-velocity) simulation experiment using SparseModesNet.
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
        subsample=[1, 1, 1],
        y_slice=96,
        no_pressure=True,
        which_velocity="w" # <- select 'w' velocity
    )

    # Data parameters
    n_snapshots = 1000

    # Load the data
    X = ds.get_matrix(snapshot_range=slice(0, n_snapshots))

    # Dimensions
    d, n = X.shape
    s = min(d, n)
    s = 100
    r = 15
    p = int(r**2)

#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V_qm, W_qm, mu_qm, I_qm = quadmani_greedy(
        X, r, s, 1e-2, np.array([], dtype=int))
    mu_qm = mu_qm.reshape(-1, 1)  

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())
    np.save("results/amr3Dchannel/w/I_qm.npy", I_qm)
    np.savez("results/amr3Dchannel/w/qm_svd.npz", V=V_qm, W=W_qm, mu=mu_qm)

    #%% Or load precomputed GreedyQM results
    I_qm = np.load("results/amr3Dchannel/w/I_qm.npy")
    qm_data = np.load("results/amr3Dchannel/w/qm_svd.npz")
    V_qm = qm_data['V']
    W_qm = qm_data['W']
    mu_qm = qm_data['mu']


#%% #========================= Greedy Cubic Manifold ==========================#
    print("\n" + "="*60)
    print("GREEDY CUBIC MANIFOLD")
    V_cm, W_cm, mu_cm, I_cm = quadmani_greedy(
        X, r, s, 1e-2, np.array([], dtype=int), 
        feature_map=_make_cubic_mapping_jax_fixed(max_r=r))
    
    # Print the selected modes
    print("Selected modes (I_cm):", I_cm)
    np.save("results/amr3Dchannel/w/I_cm.npy", I_cm)
    np.savez("results/amr3Dchannel/w/cm_svd.npz", V=V_cm, W=W_cm, mu=mu_cm)

    #%% Or load precomputed GreedyCM results
    I_cm = np.load("results/amr3Dchannel/w/I_cm.npy")
    cm_data = np.load("results/amr3Dchannel/w/cm_svd.npz")
    V_cm = cm_data['V']
    W_cm = cm_data['W']
    mu_cm = cm_data['mu']


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
        'lam0': 10.0,
        'lasso_lr': 1e-3,
        'lasso_lr_patience': 1000,
        'epsilon': 0.1,
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
        'decoder_lr_patience': 50,
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
        'label': "Wall-Normal 3D Channel Flow Pi3Net",
        'enable_logging': True
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3, I_nn_3, omegas_3, path_history, re = smn.fit(X, config)
    torch.save(model_3, "results/amr3Dchannel/w/sparsemodesnet_model_pi3net.pth")
    np.save("results/amr3Dchannel/w/I_nn_pi2net.npy", I_nn_3)
    np.save("results/amr3Dchannel/w/omegas_pi2net.npy", omegas_3)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, show=False,
                     legend_loc='best',
                     title=r'Wall-Normal: $\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/w/omega_evolution_pi3net.pdf')

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, show=False,
                     legend_loc='best',
                     title=r'Wall-Normal: $\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/w/omega_evolution_pi3net.pdf')
    
#%% Load trained model 
    model_3 = torch.load(
        "results/amr3Dchannel/w/sparsemodesnet_model_pi3net.pth",
        weights_only=False,
    )
    I_nn_3 = np.load("results/amr3Dchannel/w/I_nn_pi2net.npy")
    omegas_3 = np.load("results/amr3Dchannel/w/omegas_pi2net.npy")

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
        'decoder_lr_patience': 50,
        'decoder_epochs': 2000,
        'decoder_batch_size': 200,
        'decoder_optimizer': 'Adam',
        'decoder_momentum': 0.9,
        'decoder_bias': True,
        # General training
        'skip_sparse': True,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'I_nn': range(r),
        'device': device,
        # Experiment Setup
        'label': "Wall-Normal 3D Channel Flow Pi3Net Leading-r",
        'enable_logging': True
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3p, _, _, _, _ = smn.fit(X, config)
    torch.save(model_3p, "results/amr3Dchannel/w/sparsemodesnet_model_pi3net_leading.pth")

#%% Load trained model
    model_3p = torch.load(
        "results/amr3Dchannel/w/sparsemodesnet_model_pi3net_leading.pth",
        weights_only=False,
    )

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

    #%% Test different numbers of modes
    for r_test in range(1, min(r + 1, 21)):  # Test up to 20 modes or r
        # Quadratic Manifold with r_test modes
        V_test, W_test, shift_test, I_qm_test = quadmani_greedy(
            X, r_test, s, 1e-2, np.array([], dtype=int))
        shift_test = shift_test.reshape(-1, 1)
        Z_qm_test = V_test.T @ (X - shift_test)
        Z_quad_qm_test = quadratic_mapping_numpy(Z_qm_test.T).T
        recon_error_qm_test = np.linalg.norm(
            X - (V_test @ Z_qm_test + W_test @ Z_quad_qm_test + shift_test), ord='fro')
        rel_recon_error_qm_test = recon_error_qm_test / np.linalg.norm(X, ord='fro')

        # Cubic Manifold with r_test modes
        V_test_3, W_test_3, shift_test, I_qm_test_3 = quadmani_greedy(
            X, r_test, s, 1e-2, np.array([], dtype=int), 
            feature_map=_make_cubic_mapping_jax_fixed(max_r=r_test))
        shift_test = shift_test.reshape(-1, 1)
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
        qm_errors.append(rel_recon_error_qm_test)
        cm_errors.append(rel_recon_error_qm_test_3)
        pod_errors.append(rel_recon_error_pod_test)
        # sparse_2_errors.append(rel_recon_error_sparse_2_test)
        sparse_3_errors.append(rel_recon_error_sparse_3_test)
        # sparse_2p_errors.append(rel_recon_error_sparse_2p_test)
        sparse_3p_errors.append(rel_recon_error_sparse_3p_test)

    np.savez(
        "results/amr3Dchannel/w/reconstruction_errors.npz",
        mode_counts=mode_counts,
        qm_errors=qm_errors,
        cm_errors=cm_errors,
        pod_errors=pod_errors,
        # sparse_2_errors=sparse_2_errors,
        sparse_3_errors=sparse_3_errors,
        # sparse_2p_errors=sparse_2p_errors,
        sparse_3p_errors=sparse_3p_errors
    )

    np.savez(
        "results/amr3Dchannel/w/V_all.npz",
        V_all=V_all
    )

    #%% Or load 
    errors_data = np.load("results/amr3Dchannel/w/reconstruction_errors.npz")
    mode_counts = errors_data['mode_counts']
    qm_errors = errors_data['qm_errors']
    cm_errors = errors_data['cm_errors']
    pod_errors = errors_data['pod_errors']
    sparse_3_errors = errors_data['sparse_3_errors']
    sparse_3p_errors = errors_data['sparse_3p_errors']


#%% #======================= Plot reconstruction errors =======================#
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.semilogy(mode_counts, pod_errors, '-o', label='POD (leading-r)', 
                markersize=23, linewidth=13, c='black')
    ax.semilogy(mode_counts, qm_errors, '--^', label='Greedy Quadratic Manifold', 
                markersize=23, linewidth=13)
    ax.semilogy(mode_counts, cm_errors, '--v', label='Greedy Cubic Manifold',
                markersize=23, linewidth=13)
    
    # Only plot valid SparseModesNet errors (selected modes)
    # valid_sparse_2_errors = [err for err in sparse_2_errors if not np.isnan(err)]
    # valid_mode_counts_2 = [mode_counts[i] for i, err in enumerate(sparse_2_errors) if not np.isnan(err)]
    # ax.semilogy(valid_mode_counts_2, valid_sparse_2_errors, '-s', label=r'SparseModesNet $\Pi_2$-Net (selected)', 
    #             markersize=8, linewidth=3)

    valid_sparse_3_errors = [err for err in sparse_3_errors if not np.isnan(err)]
    valid_mode_counts_3 = [mode_counts[i] for i, err in enumerate(sparse_3_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts_3, valid_sparse_3_errors, '-x', label=r'SparseModesNet $\Pi_3$-Net (selected)', 
                markersize=23, linewidth=13, mew=8)
    
    # Plot SparseModesNet with leading modes
    # ax.semilogy(mode_counts, sparse_2p_errors, ':s', label=r'SparseModesNet $\Pi_2$-Net (leading-r)', 
    #             markersize=8, linewidth=3, alpha=0.8)
    ax.semilogy(mode_counts, sparse_3p_errors, ':x', label=r'SparseModesNet $\Pi_3$-Net (leading-r)', 
                markersize=23, linewidth=13, alpha=0.8, mew=8)
    
    ax.set_xlabel(r'Number of Modes $r$', fontsize=40)
    # ax.set_ylabel('Relative Reconstruction Error', fontsize=40)
    # ax.set_title('Wall-Normal: Reconstruction Error vs. Number of Modes', fontsize=30)
    ax.set_title('Wall-Normal Velocity', fontsize=40)
    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=32, loc='lower left', frameon=False)
    ax.set_xlim(left=0)
    ax.set_xticks(np.arange(0, r+1, 3))
    plt.tight_layout()
    plt.savefig('figures/amr3Dchannel/w/reconstruction_errors.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


#%% %============ Plot the reconstructed flow fields (heatmap) ================%
    is_sliced = False
    y_or_z_slice = 'y'
    qm_or_cm = 'qm'

    # Select a snapshot to visualize
    snapshot_idx = 50  # Change this index to visualize different snapshots
    xsnap = X[:, snapshot_idx].reshape(-1, 1)
    xsnap_proc = X_proc[:, snapshot_idx].reshape(-1, 1)

    # Quadratic Manifold reconstruction
    mu_qm = mu_qm.reshape(-1, 1)
    Z_qm = V_qm.T @ (xsnap - mu_qm)
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Cubic Manifold reconstruction
    mu_cm = mu_cm.reshape(-1, 1)
    Z_cm = V_cm.T @ (xsnap - mu_cm)
    Z_quad_cm = _cubic_mapping_numpy(Z_cm.T).T
    X_cm_recon = V_cm @ Z_cm + W_cm @ Z_quad_cm + mu_cm

    # Leading-r POD reconstruction (using top r modes)
    V_leading_r = V_all[:, :r]
    
    # SparseModesNet Pi3Net reconstruction
    Z_input_3 = torch.from_numpy((V_all[:, I_nn_3].T @ xsnap_proc).T).to(device)
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
    qm_error = xsnap - X_qm_recon
    cm_error = xsnap - X_cm_recon
    sparse_error_3 = xsnap - X_sparse_recon_3
    sparse_error_3p = xsnap - X_sparse_recon_3p
    
    # Reshape the data here 
    xlen = ds.shape[2]
    if y_or_z_slice == 'y':
        ylen = ds.shape[4] # this is because of the y-slice
        zlen = 1
    elif y_or_z_slice == 'z':
        ylen = ds.shape[3]
        zlen = 1
    else:
        ylen = ds.shape[3]
        zlen = ds.shape[4]

    if zlen != 1:
        xsnap = xsnap.reshape((xlen, ylen, zlen), order='C')
        X_qm_recon = X_qm_recon.reshape((xlen, ylen, zlen), order='C')
        X_cm_recon = X_cm_recon.reshape((xlen, ylen, zlen), order='C')
        X_sparse_recon_3 = X_sparse_recon_3.reshape((xlen, ylen, zlen), order='C')
        X_sparse_recon_3p = X_sparse_recon_3p.reshape((xlen, ylen, zlen), order='C')

        qm_error = qm_error.reshape((xlen, ylen, zlen), order='C')
        cm_error = cm_error.reshape((xlen, ylen, zlen), order='C')
        sparse_error_3 = sparse_error_3.reshape((xlen, ylen, zlen), order='C')
        sparse_error_3p = sparse_error_3p.reshape((xlen, ylen, zlen), order='C')
    else:
        xsnap = xsnap.reshape((xlen, ylen), order='C')
        X_qm_recon = X_qm_recon.reshape((xlen, ylen), order='C')
        X_cm_recon = X_cm_recon.reshape((xlen, ylen), order='C')
        X_sparse_recon_3 = X_sparse_recon_3.reshape((xlen, ylen), order='C')
        X_sparse_recon_3p = X_sparse_recon_3p.reshape((xlen, ylen), order='C')

        qm_error = qm_error.reshape((xlen, ylen), order='C')
        cm_error = cm_error.reshape((xlen, ylen), order='C')
        sparse_error_3 = sparse_error_3.reshape((xlen, ylen), order='C')
        sparse_error_3p = sparse_error_3p.reshape((xlen, ylen), order='C')

    # Slice 
    if is_sliced:
        zslice_idx = zlen // 2  # Middle slice in z-direction
        xsnap_ = xsnap[:, :, zslice_idx]
        X_qm_recon_ = X_qm_recon[:, :, zslice_idx]
        X_cm_recon_ = X_cm_recon[:, :, zslice_idx]
        X_sparse_recon_3_ = X_sparse_recon_3[:, :, zslice_idx]
        X_sparse_recon_3p_ = X_sparse_recon_3p[:, :, zslice_idx]

        qm_error_ = qm_error[:, :, zslice_idx]
        cm_error_ = cm_error[:, :, zslice_idx]
        sparse_error_3_ = sparse_error_3[:, :, zslice_idx]
        sparse_error_3p_ = sparse_error_3p[:, :, zslice_idx]
    else:
        xsnap_ = xsnap
        X_qm_recon_ = X_qm_recon
        X_cm_recon_ = X_cm_recon
        X_sparse_recon_3_ = X_sparse_recon_3
        X_sparse_recon_3p_ = X_sparse_recon_3p

        qm_error_ = qm_error
        cm_error_ = cm_error
        sparse_error_3_ = sparse_error_3
        sparse_error_3p_ = sparse_error_3p

    # Select data for selected greedy approach 
    X_greedy_recon = X_qm_recon_  if qm_or_cm == 'qm' else X_cm_recon_
    greedy_error = qm_error_ if qm_or_cm == 'qm' else cm_error_

    # Set consistent color scales for reconstructions
    recon_vmin = min(xsnap_.min(), X_greedy_recon.min(), 
                     X_sparse_recon_3_.min(), X_sparse_recon_3p_.min())
    recon_vmax = max(xsnap_.max(), X_greedy_recon.max(),
                     X_sparse_recon_3_.max(), X_sparse_recon_3p_.max())
    
    # Set consistent color scales for errors
    error_vmax = max(np.abs(greedy_error).max(), 
                     np.abs(sparse_error_3_).max(), 
                     np.abs(sparse_error_3p_).max())
    error_vmin = -error_vmax

    # Setup the figure 
    fig, axes = plt.subplots(7, 1, figsize=(22, 27))

    # Grid edges for extent
    if y_or_z_slice == 'y':
        ext = [ds.x[0], ds.x[-1], ds.z[0], ds.z[-1]]
    elif y_or_z_slice == 'z':
        ext = [ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]]
    else:
        raise ValueError("y_or_z_slice must be either 'y' or 'z'.")
    
    # Row 1: Reconstructions
    # (1,1) Original data
    im1 = axes[0].imshow(
        xsnap_.T, aspect=1.5, cmap='viridis', origin='lower', extent=ext,
        vmin=recon_vmin, vmax=recon_vmax)
    if y_or_z_slice == 'y':
        axes[0].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[0].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[0].set_title('Original Data', fontsize=38, pad=15)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # (2,1) Greedy Quadratic/Cubic Manifold reconstruction
    im2 = axes[1].imshow(
        X_greedy_recon.T, aspect=1.5, cmap='viridis', origin='lower', 
        extent=ext, vmin=recon_vmin, vmax=recon_vmax)
    if y_or_z_slice == 'y':
        axes[1].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[1].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[1].set_title(
        'Greedy' + ('QM' if qm_or_cm == 'qm' else 'CM'),
        fontsize=38, pad=15
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # (3,1) SparseModesNet Pi3Net (leading-r) reconstruction
    im3 = axes[2].imshow(
        X_sparse_recon_3p_.T, aspect=1.5, cmap='viridis', origin='lower',
        extent=ext, vmin=recon_vmin, vmax=recon_vmax)
    if y_or_z_slice == 'y':
        axes[2].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[2].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[2].set_title(r'SparseModesNet $\Pi_3$-Net (leading-r)', 
                      fontsize=38, pad=15)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    # (4,1) SparseModesNet Pi3Net reconstruction
    im4 = axes[3].imshow(
        X_sparse_recon_3_.T, aspect=1.5, cmap='viridis', origin='lower',
        extent=ext, vmin=recon_vmin, vmax=recon_vmax)
    if y_or_z_slice == 'y':
        axes[3].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[3].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[3].set_title(r'SparseModesNet $\Pi_3$-Net', fontsize=38, pad=15)
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    
    # Errors
    
    # (5,1) Quadratic/Cubic Manifold error
    im7 = axes[4].imshow(
        greedy_error.T, aspect=1.5, cmap='RdBu', origin='lower', extent=ext,
        vmin=error_vmin, vmax=error_vmax)
    if y_or_z_slice == 'y':
        axes[4].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[4].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[4].set_title(
        'Greedy' + ('QM' if qm_or_cm == 'qm' else 'CM') + ' Error',
        fontsize=38, pad=15
    )
    axes[4].set_xticks([])
    axes[4].set_yticks([])
    
    # (6,1) SparseModesNet Pi3Net (leading-r) error
    im8 = axes[5].imshow(
        sparse_error_3p_.T, aspect=1.5, cmap='RdBu', origin='lower',
        extent=ext, vmin=error_vmin, vmax=error_vmax)
    if y_or_z_slice == 'y':
        axes[5].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[5].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[5].set_title(r'SparseModesNet $\Pi_3$-Net (leading-r) Error', 
                      fontsize=38, pad=15)
    axes[5].set_xticks([])
    axes[5].set_yticks([])

    # (7,1) SparseModesNet Pi3Net error
    im9 = axes[6].imshow(
        sparse_error_3_.T, aspect=1.5, cmap='RdBu', origin='lower',
        extent=ext, vmin=error_vmin, vmax=error_vmax)
    if y_or_z_slice == 'y':
        axes[6].set_ylabel(r'$\xi_3$', fontsize=35)
    else:
        axes[6].set_ylabel(r'$\xi_2$', fontsize=35)
    axes[6].set_title(r'SparseModesNet $\Pi_3$-Net Error', fontsize=38, pad=15)
    axes[6].set_xlabel(r'$\xi_1$', fontsize=35)
    axes[6].set_xticks([])
    axes[6].set_yticks([])
    
    # Add unified colorbars
    cax1 = fig.add_axes([0.92, 0.458, 0.02, 0.431])
    cbar1 = plt.colorbar(im4, cax=cax1)
    if y_or_z_slice == 'y':
        cbar1.set_label(r'$u_3(\xi_1,\xi_3)$', fontsize=38)
    else:
        cbar1.set_label(r'$u_3(\xi_1,\xi_2)$', fontsize=38)
    cbar1.ax.tick_params(labelsize=30)
    
    cax2 = fig.add_axes([0.92, 0.11, 0.02, 0.32])
    cbar2 = plt.colorbar(im8, cax=cax2)
    cbar2.set_label('abs. error', fontsize=38)
    cbar2.ax.tick_params(labelsize=30)

    plt.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.1, 
                        wspace=0.05, hspace=0.075)
    if y_or_z_slice == 'y':
        plt.suptitle('Spanwise Sliced Wall-Normal Velocity Reconstruction Comparison', 
                     fontsize=40, y=0.935)
    else:
        plt.suptitle('Wall-Normal Sliced Wall-Normal Velocity Reconstruction Comparison', 
                     fontsize=40, y=0.935)
    plt.savefig(
        'figures/amr3Dchannel/w/w_recon_recomparison_' + qm_or_cm + '.pdf', 
        dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)



#%% %========================== Reconstructed data ============================%
    X_proc = config.preprocessing.forward(X)
    V, _, _ = np.linalg.svd(X_proc, full_matrices=False)

    # # Quadratic Manifold reconstruction
    # Z_qm = V_qm.T @ (X - mu_qm)
    # Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    # X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Cubic Manifold reconstruction
    Z_cm = V_cm.T @ (X - mu_qm)
    Z_cubic_cm = _cubic_mapping_numpy(Z_cm.T).T
    X_cm_recon = V_cm @ Z_cm + W_cm @ Z_cubic_cm + mu_qm
    
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

#%% %===================== Additional Plotting and Analysis ===================%
    from matplotlib import gridspec
    y_or_z_slice = 'y'

    # Compute SVD for all datasets
    U_orig, S_orig, _ = np.linalg.svd(X, full_matrices=False)
    # U_qm, S_qm, _ = np.linalg.svd(X_qm_recon, full_matrices=False)
    U_cm, S_cm, _ = np.linalg.svd(X_cm_recon, full_matrices=False)
    U_sparse_3, S_sparse_3, _ = np.linalg.svd(X_sparse_recon_3, full_matrices=False)
    
    # Order of modes to plot
    data_type = ['orig', 'cm', 'sparse']
    n_types = len(data_type)
    n_modes = np.array([1, r, s-1, 299])

    fig = plt.figure(figsize=(22, 28))
    nrows, ncols = 8, 2
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    axes = np.empty(nrows*ncols, dtype=object)
    # Mode 1
    axes[0] = fig.add_subplot(gs[0, :]) # Original
    axes[1] = fig.add_subplot(gs[1, 0]) # GreedyCM
    axes[2] = fig.add_subplot(gs[1, 1]) # SparseModesNet Pi3Net
    # Mode r+1
    axes[3] = fig.add_subplot(gs[2, :]) # Original
    axes[4] = fig.add_subplot(gs[3, 0]) # GreedyCM
    axes[5] = fig.add_subplot(gs[3, 1]) # SparseModesNet Pi3Net
    # Mode s
    axes[6] = fig.add_subplot(gs[4, :]) # Original
    axes[7] = fig.add_subplot(gs[5, 0]) # GreedyCM
    axes[8] = fig.add_subplot(gs[5, 1]) # SparseModesNet Pi3Net
    # High-freq mode
    axes[9] = fig.add_subplot(gs[6, :]) # Original
    axes[10] = fig.add_subplot(gs[7, 0]) # GreedyCM
    axes[11] = fig.add_subplot(gs[7, 1]) # SparseModesNet Pi3Net

    for (i, md) in enumerate(n_modes):
        ax_orig = axes[n_types*i]
        ax_cm = axes[n_types*i + 1]
        ax_sparse_3 = axes[n_types*i + 2]

        # Get the original mode as reference
        mode_orig = U_orig[:, md]
        # mode_qm = U_qm[:, md]
        mode_cm = U_cm[:, md]
        mode_sparse_3 = U_sparse_3[:, md]
        
        # Fix sign flips by checking correlation with original mode
        # If correlation is negative, flip the sign
        # if np.corrcoef(mode_orig, mode_qm)[0, 1] < 0:
        #     mode_qm = -mode_qm

        if np.corrcoef(mode_orig, mode_cm)[0, 1] < 0:
            mode_cm = -mode_cm
            
        if np.corrcoef(mode_orig, mode_sparse_3)[0, 1] < 0:
            mode_sparse_3 = -mode_sparse_3

        # Reshape modes into 2D for plotting
        xlen = ds.shape[2]
        if y_or_z_slice == 'y':
            ylen = ds.shape[4] # this is because of the y-slice
            zlen = 1
        elif y_or_z_slice == 'z':
            ylen = ds.shape[3]
            zlen = 1
        else:
            ylen = ds.shape[3]
            zlen = ds.shape[4]

        if zlen != 1:
            mode_orig = mode_orig.reshape((xlen, ylen, zlen), order='C')
            # mode_qm = mode_qm.reshape((xlen, ylen, zlen), order='C')
            mode_cm = mode_cm.reshape((xlen, ylen, zlen), order='C')
            mode_sparse_3 = mode_sparse_3.reshape((xlen, ylen, zlen), order='C')
        else:
            mode_orig = mode_orig.reshape((xlen, ylen), order='C')
            # mode_qm = mode_qm.reshape((xlen, ylen), order='C')
            mode_cm = mode_cm.reshape((xlen, ylen), order='C')
            mode_sparse_3 = mode_sparse_3.reshape((xlen, ylen), order='C')

        # Set consistent color scales for modes
        recon_vmin = min(mode_orig.min(), # mode_qm.min(), 
                        mode_cm.min(), mode_sparse_3.min())
        recon_vmax = max(mode_orig.max(), # mode_qm.max(),
                        mode_cm.max(), mode_sparse_3.max())
        
        # Grid edges for extent
        if y_or_z_slice == 'y':
            ext = [ds.x[0], ds.x[-1], ds.z[0], ds.z[-1]]
        elif y_or_z_slice == 'z':
            ext = [ds.x[0], ds.x[-1], ds.y[0], ds.y[-1]]
        else:
            raise ValueError("y_or_z_slice must be either 'y' or 'z'.")
        
        # Title mod
        if md == r:
            title_add_str = rf'Mode {md+1} = r+1'
        elif md < r+1: 
            title_add_str = rf'Mode {md+1}' 
        elif md == s-1:
            title_add_str = rf'Mode {md+1} = s'
        else:
            title_add_str = rf'Mode {md+1}'
        
        # Original data
        im1 = ax_orig.imshow(
            mode_orig.T, cmap='RdBu_r', origin='lower', extent=ext, aspect=1.5,
            vmin=recon_vmin, vmax=recon_vmax)
        if y_or_z_slice == 'y':
            ax_orig.set_ylabel(r'$\xi_3$', fontsize=35)
        else:
            ax_orig.set_ylabel(r'$\xi_2$', fontsize=35)
        ax_orig.set_title(
            'Original Data: ' +  title_add_str 
            + rf' ($\sigma = {S_orig[md]:.2e}$)',
            fontsize=35, pad=15
        )
        ax_orig.set_xticks([])
        ax_orig.set_yticks([])

        # GreedyCM
        im2 = ax_cm.imshow(
            mode_cm.T, cmap='RdBu_r', origin='lower', aspect=2.5,
            extent=ext, vmin=recon_vmin, vmax=recon_vmax)
        if y_or_z_slice == 'y':
            ax_cm.set_ylabel(r'$\xi_3$', fontsize=35)
        else:
            ax_cm.set_ylabel(r'$\xi_2$', fontsize=35)
        ax_cm.set_title('GreedyCM: '+title_add_str, fontsize=35, pad=15)
        ax_cm.set_xticks([])
        ax_cm.set_yticks([])
        if i == len(n_modes) - 1:
            ax_cm.set_xlabel(r'$\xi_1$', fontsize=35)
        
        # SparseModesNet Pi3Net
        im3 = ax_sparse_3.imshow(
            mode_sparse_3.T, cmap='RdBu_r', origin='lower', aspect=2.5,
            extent=ext, vmin=recon_vmin, vmax=recon_vmax)
        ax_sparse_3.set_title(
            r'SparseModesNet $\Pi_3$-Net: '+title_add_str, fontsize=35, pad=15
        )
        ax_sparse_3.set_xticks([])
        ax_sparse_3.set_yticks([])
        if i == len(n_modes) - 1:
            ax_sparse_3.set_xlabel(r'$\xi_1$', fontsize=35)

    plt.tight_layout()
    plt.subplots_adjust(left=0.04, right=0.9, top=0.9, bottom=0.1, 
                        wspace=0.05, hspace=0.075)
    if y_or_z_slice == 'y':
        plt.suptitle(
            'Spanwise Sliced Wall-Normal Velocity POD Mode Reconstruction Comparison',
            fontsize=40, y=0.935)
    else:
        plt.suptitle(
            'Wall-Normal Sliced Wall-Normal Velocity POD Mode Reconstruction Comparison',
            fontsize=40, y=0.935)
    plt.savefig(
        'figures/amr3Dchannel/w/w_spatial_modes_comparison.pdf', 
        dpi=300, bbox_inches='tight'
    )
    plt.show()
    plt.close(fig)


# %%

