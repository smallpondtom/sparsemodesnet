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
from QM.quadmani import quadmani_greedy, _make_cubic_mapping_jax_fixed
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

    #%% Save the selected modes
    np.save("results/kse/I_qm.npy", I_qm)

    #%% Or load the selected modes
    I_qm = np.load("results/kse/I_qm.npy")


#%% #========================= Greedy Cubic Manifold ==========================#
    print("\n" + "="*60)
    print("GREEDY CUBIC MANIFOLD")
    V_cm, W_cm, _, I_cm = quadmani_greedy(
        X, r, s, 1e-12, np.array([], dtype=int), 
        feature_map=_make_cubic_mapping_jax_fixed(max_r=r))
    # Print the selected modes
    print("Selected modes (I_cm):", I_cm)

    #%% Save the selected modes
    np.save("results/kse/I_cm.npy", I_cm)

    #%% Or load the selected modes
    I_cm = np.load("results/kse/I_cm.npy")


#%% %==================== Configuration of SparseModesNet =====================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 300,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 800, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [50, 600, 300],  # PiNet
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
    model_2, I_nn_2, omegas_2, path_history, re = smn.fit(X, config)
    torch.save(model_2, "results/kse/sparsemodesnet_model_pi2net.pth")
    # np.save("results/kse/I_nn_pi2net.npy", I_nn_2)
    # np.save("results/kse/omegas_pi2net.npy", omegas_2)


#%% Or load models
    model_2 = torch.load("results/kse/sparsemodesnet_model_pi2net.pth", weights_only=False)
    I_nn_2 = np.load("results/kse/I_nn_pi2net.npy")
    omegas_2 = np.load("results/kse/omegas_pi2net.npy")


#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_2, I_nn_2, config.s, save=True, 
                     filename='figures/kse/omega_evolution_pi2net.pdf')


#%% %=============== Configuration of SparseModesNet (Pi3Net) =================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 680,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [50, 720, 680],  # PiNet
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
    model_3, I_nn_3, omegas_3, path_history, re = smn.fit(X, config)
    torch.save(model_3, "results/kse/sparsemodesnet_model_pi3net.pth")
    np.save("results/kse/I_nn_pi3net.npy", I_nn_3)
    np.save("results/kse/omegas_pi3net.npy", omegas_3)

#%% Or load models
    model_3 = torch.load("results/kse/sparsemodesnet_model_pi3net.pth", weights_only=False)
    I_nn_3 = np.load("results/kse/I_nn_pi3net.npy")
    omegas_3 = np.load("results/kse/omegas_pi3net.npy")

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, 
                     y_limits=(1e-6, 2e-1, 5), legend_loc='lower right',
                     title=r'$\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/kse/omega_evolution_pi3net.pdf')


#%% %==================== Configuration of SparseModesNet =====================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 300,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [50, 600, 300],  # PiNet
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
        'skip_sparse': True,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'I_nn': range(r),
        'device': device,
        # Experiment Setup
        'label': "Advecting Pulse",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_2p, _, _, _, _ = smn.fit(X, config)
    torch.save(model_2p, "results/pulse/sparsemodesnet_model_pi2net_leading.pth")

#%% Or load model
    model_2p = torch.load(
        "results/kse/sparsemodesnet_model_pi2net_leading.pth", 
        weights_only=False
    )

#%% %==================== Configuration of SparseModesNet =====================%

    # Configure conveniently using dictionary
    config_dict = {
        # Number of modes
        's': s,
        'r': r,
        'p': 680,
        # Preprocessing
        'normalize_data': True,
        'center': True,
        'whiten': False,
        'normalize_type': 'minmax',
        # Architecture
        # 'hidden_units': [400, 400, 400],  # MLP
        # 'hidden_units': [32, 5, 64, 128],  # CNN
        # 'hidden_units': [64, 256],  # UNET
        'hidden_units': [50, 720, 680],  # PiNet
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
        'skip_sparse': True,
        'weight_scale': 1.0,
        'gamma': 1e-8,
        'reg_param': 1e-15,
        'normalize_layer': 'last',
        'I_nn': range(r),
        'device': device,
        # Experiment Setup
        'label': "Advecting Pulse",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3p, _, _, _, _ = smn.fit(X, config)
    torch.save(model_3p, "results/pulse/sparsemodesnet_model_pi3net_leading.pth")

#%% Or load model
    model_3p = torch.load(
        "results/kse/sparsemodesnet_model_pi3net_leading.pth", 
        weights_only=False
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

        # Cubic Manifold with r_test modes
        V_test_3, W_test_3, _, I_qm_test_3 = quadmani_greedy(
            X, r_test, s, 1e-12, np.array([], dtype=int), 
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

        # SparseModesNet Pi2Net with leading r_test modes (model_2p)
        V_tmp_2p = np.zeros((d, r))
        V_tmp_2p[:, :r_test] = V_all[:, :r_test]  # Use first r_test leading modes
        Z_input_test_2p = torch.from_numpy((V_tmp_2p.T @ X_proc_test).T).to(device)
        with torch.no_grad():
            if config.network.network_type == 'QM' or config.network.network_type == 'CM':
                # Use analytical decoder
                X_sparse_recon_test_2p, _, _ = model_2p(V_tmp_2p.T @ X_proc_test)
            else:
                model_2p.eval()
                # Retrain the weight matrix
                X_proc_test_tensor = torch.from_numpy(X_proc_test).to(device)
                _, X_sparse_lin_2p, N_sparse_out_2p = model_2p(Z_input_test_2p)
                resid_2p = X_proc_test_tensor.T - X_sparse_lin_2p
                model_2p.update_nonlinear_weight(resid_2p, N_sparse_out_2p, 
                                               config.training.reg_param)
                X_sparse_recon_tensor_test_2p, _, _ = model_2p(Z_input_test_2p)
                X_sparse_recon_test_2p = X_sparse_recon_tensor_test_2p.cpu().numpy().T

            X_sparse_recon_test_2p = config.preprocessing.backward(X_sparse_recon_test_2p)
        
        recon_error_sparse_test_2p = np.linalg.norm(X - X_sparse_recon_test_2p, ord='fro')
        rel_recon_error_sparse_2p_test = recon_error_sparse_test_2p / np.linalg.norm(X, ord='fro')

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
        sparse_2_errors.append(rel_recon_error_sparse_2_test)
        sparse_3_errors.append(rel_recon_error_sparse_3_test)
        sparse_2p_errors.append(rel_recon_error_sparse_2p_test)
        sparse_3p_errors.append(rel_recon_error_sparse_3p_test)

    #%% Plot reconstruction errors
    fig, ax = plt.subplots(1, 1, figsize=(27, 9))
    
    ax.semilogy(mode_counts, pod_errors, '-o', label='POD (leading-r)', 
                markersize=20, linewidth=10, c='black')
    ax.semilogy(mode_counts, qm_errors, '--^', label='Greedy Quadratic Manifold', 
                markersize=20, linewidth=10)
    ax.semilogy(mode_counts, cm_errors, '--v', label='Greedy Cubic Manifold',
                markersize=20, linewidth=10)

    # Only plot valid SparseModesNet errors (selected modes)
    valid_sparse_2_errors = [err for err in sparse_2_errors if not np.isnan(err)]
    valid_mode_counts_2 = [mode_counts[i] for i, err in enumerate(sparse_2_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts_2, valid_sparse_2_errors, '-s', label=r'SparseModesNet $\Pi_2$-Net (selected)', 
                markersize=20, linewidth=10)

    valid_sparse_3_errors = [err for err in sparse_3_errors if not np.isnan(err)]
    valid_mode_counts_3 = [mode_counts[i] for i, err in enumerate(sparse_3_errors) if not np.isnan(err)]
    ax.semilogy(valid_mode_counts_3, valid_sparse_3_errors, '-x', label=r'SparseModesNet $\Pi_3$-Net (selected)', 
                markersize=20, linewidth=10, mew=5)
    
    # Plot SparseModesNet with leading modes
    ax.semilogy(mode_counts, sparse_2p_errors, ':s', label=r'SparseModesNet $\Pi_2$-Net (leading-r)', 
                markersize=20, linewidth=10, alpha=0.8)
    ax.semilogy(mode_counts, sparse_3p_errors, ':x', label=r'SparseModesNet $\Pi_3$-Net (leading-r)', 
                markersize=20, linewidth=10, alpha=0.8, mew=5)
    
    # Hide the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.set_xlabel(r'Number of Modes $r$', fontsize=40)
    ax.set_ylabel('Relative Reconstruction Error', fontsize=40)
    # ax.set_title('Reconstruction Error vs. Number of Modes', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.grid(True, alpha=0.3)
    # ax.legend(fontsize=22, loc='lower left')
    ax.legend(fontsize=36, loc='center left', bbox_to_anchor=(1.005, 0.5), 
              frameon=False, handlelength=3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    plt.savefig('figures/kse/reconstruction_errors.pdf', 
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% %========================== Reconstructed data ============================%
    X_proc = config.preprocessing.forward(X)
    V, _, _ = np.linalg.svd(X_proc, full_matrices=False)

    # Quadratic Manifold reconstruction
    Z_qm = V_qm.T @ (X - mu_qm)
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    X_qm_recon = V_qm @ Z_qm + W_qm @ Z_quad_qm + mu_qm

    # Cubic Manifold reconstruction
    Z_cm = V_cm.T @ (X - mu_qm)
    Z_cubic_cm = _cubic_mapping_numpy(Z_cm.T).T
    X_cm_recon = V_cm @ Z_cm + W_cm @ Z_cubic_cm + mu_qm

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
                linewidth=2, label='GreedyQM', alpha=0.8)
        # # Plot SparseModesNet Pi2Net reconstruction
        # ax.plot(xspan, X_sparse_recon_2[:, t_idx], ':', 
        #         linewidth=2, label='SparseModesNet (Pi2Net)', alpha=0.8)
        # Plot SparseModesNet Pi3Net reconstruction
        ax.plot(xspan, X_sparse_recon_3[:, t_idx], '-', 
                linewidth=2, label=r'SMN ($\Pi_3$-Net)', alpha=0.8)
        
        ax.set_xlabel(r'Space ($\xi$)', fontsize=25)
        ax.set_ylabel(r'$x(\xi,t)$', fontsize=25)
        ax.set_title(rf'$t = {t_val:.3f}$', fontsize=28)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=20)

    plt.tight_layout()
    plt.suptitle('Wave Profiles at Different Time Points', fontsize=30, y=1.04)
    plt.savefig('figures/kse/wave_profiles.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% %===================== Additional Plotting and Analysis ===================%
    from matplotlib import gridspec

    # Compute SVD for all datasets
    U_orig, S_orig, _ = np.linalg.svd(X, full_matrices=False)
    U_pod, S_pod, _ = np.linalg.svd(X_pod_recon, full_matrices=False)
    U_qm, S_qm, _ = np.linalg.svd(X_qm_recon, full_matrices=False)
    U_cm, S_cm, _ = np.linalg.svd(X_cm_recon, full_matrices=False)
    U_sparse_2, S_sparse_2, _ = np.linalg.svd(X_sparse_recon_2, full_matrices=False)
    U_sparse_3, S_sparse_3, _ = np.linalg.svd(X_sparse_recon_3, full_matrices=False)
    
    # Plot the first 20 spatial modes comparison
    n_modes = np.hstack([np.arange(4), [r, 37, 61, s-1, 239]])
    fig = plt.figure(figsize=(24, 15))
    nrows, ncols = 5, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    axes = np.empty(nrows*ncols, dtype=object)
    # Modes within r
    axes[0] = fig.add_subplot(gs[0, 0])
    axes[1] = fig.add_subplot(gs[0, 1])
    axes[2] = fig.add_subplot(gs[0, 2])
    axes[3] = fig.add_subplot(gs[0, 3])
    # Lower modes beyond r but within s
    axes[4] = fig.add_subplot(gs[1, :2])
    axes[5] = fig.add_subplot(gs[1, 2:])
    # Higher modes beyond r but within s
    axes[6] = fig.add_subplot(gs[2, :2])
    axes[7] = fig.add_subplot(gs[2, 2:])
    # Higher modes beyond s
    axes[8] = fig.add_subplot(gs[3, :])

    # Get default colors 
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for (i, md) in enumerate(n_modes):
        ax = axes[i]
        
        # Get the original mode as reference
        mode_orig = U_orig[:, md]
        max_val = np.max(np.abs(mode_orig))
        
        # Get modes from different methods
        mode_pod = U_pod[:, md]
        mode_qm = U_qm[:, md]
        mode_cm = U_cm[:, md]
        mode_sparse_2 = U_sparse_2[:, md]
        mode_sparse_3 = U_sparse_3[:, md]
        
        # Fix sign flips by checking correlation with original mode
        # If correlation is negative, flip the sign
        if np.corrcoef(mode_orig, mode_pod)[0, 1] < 0:
            mode_pod = -mode_pod
            
        if np.corrcoef(mode_orig, mode_qm)[0, 1] < 0:
            mode_qm = -mode_qm

        if np.corrcoef(mode_orig, mode_cm)[0, 1] < 0:
            mode_cm = -mode_cm
            
        if np.corrcoef(mode_orig, mode_sparse_2)[0, 1] < 0:
            mode_sparse_2 = -mode_sparse_2
            
        if np.corrcoef(mode_orig, mode_sparse_3)[0, 1] < 0:
            mode_sparse_3 = -mode_sparse_3
        
        # Plot spatial modes with corrected signs
        ax.plot(xspan, mode_orig, 'k-', linewidth=7, 
                label='Original', alpha=0.9)
        if md < r+1:
            ax.plot(xspan, mode_pod, '--', linewidth=6, 
                    label='POD', alpha=0.8, color=default_colors[0])
        ax.plot(xspan, mode_qm, '-.', linewidth=5, 
                label='GreedyQM', alpha=0.8, color=default_colors[1])
        ax.plot(xspan, mode_sparse_2, ':', linewidth=5, 
                label=r'SMN $\Pi_2$-Net', alpha=0.8, color=default_colors[2])
        ax.plot(xspan, mode_cm, '-.', linewidth=4, 
                label='GreedyCM', alpha=0.8, color=default_colors[3])
        ax.plot(xspan, mode_sparse_3, '-', linewidth=3, 
                label=r'SMN $\Pi_3$-Net', alpha=0.8, color=default_colors[4])

        # Remove all labels and ticks
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.margins(x=0.015, y=0) 
        if i == 0:
            ax.set_ylim([-1.1*max_val, -0.5*max_val])
        else:
            ax.set_ylim([-1.2*max_val, 1.2*max_val])
        if md == r:
            ax.set_title(
                rf'Mode {md+1} = r+1 ($\sigma = {S_orig[md]:.2e}$) ', 
                fontsize=35)
        elif md < r+1: 
            ax.set_title(
                rf'Mode {md+1} ($\sigma = {S_orig[md]:.2e}$)', fontsize=35)
        elif md == s-1:
            ax.set_title(
                rf'Mode {md+1} = s ($\sigma = {S_orig[md]:.2e}$) ' + 
                rf'without POD', fontsize=35)
        else:
            ax.set_title(
                rf'Mode {md+1} ($\sigma = {S_orig[md]:.2e}$) ' + 
                rf'without POD', fontsize=35)
        ax.grid(True, alpha=0.3)

    # Replot the last high-frequency mode separately for better visibility
    ax = fig.add_subplot(gs[4, :])
    md = n_modes[-1]
    ax.plot(xspan, mode_orig, 'k-', linewidth=7, 
            label='Original', alpha=0.9)
    ax.plot(xspan, mode_cm, '-.', linewidth=4, 
            label='GreedyCM', alpha=0.8, color=default_colors[3])
    ax.plot(xspan, mode_sparse_3, '-', linewidth=3, 
            label=r'SMN $\Pi_3$-Net', alpha=0.8, color=default_colors[4])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.margins(x=0.015, y=0) 
    ax.set_ylim([-1.2*max_val, 1.2*max_val])
    ax.set_title(
        rf'Mode {md+1} ($\sigma = {S_orig[md]:.2e}$) ' + 
        rf'without POD/GreedyQM/SMN $\Pi_2$-Net', 
        fontsize=35
    )

    plt.tight_layout()
    # plt.suptitle('KSE POD Mode Reconstruction Comparison', fontsize=35, y=1.02)

    # Add legend at the bottom outside the plot
    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=35, 
                     bbox_to_anchor=(0.5, -0.06), frameon=False)
    for line in leg.get_lines():
        line.set_linewidth(10)
    plt.savefig('figures/kse/spatial_modes_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)


# %%
