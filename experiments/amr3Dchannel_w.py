"""
AMR-Wind 3D Channel Flow (u-velocity) simulation experiment using SparseModesNet.
"""

#%% Load modules
import numpy as np
import scipy.linalg.interpolative as sli
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from QM.quadmani import quadmani_greedy
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


#%% %============================= Main Script ================================%
if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)

    # For reproducibility
    torch.manual_seed(42)

    # Load the data source
    ds = ChannelDataSource(
        hfname="../../Data/nrel/channel_5200_data_0_10000.h5",
        subsample=[1, 1, 1],
        no_pressure=True,
        which_velocity="w" # <- select 'w' velocity
    )

    # Data parameters
    n_snapshots = 100

    # Load the data
    X = ds.get_matrix(snapshot_range=slice(0, n_snapshots))

    # Dimensions
    d, n = X.shape
    s = min(d, n)
    s = 120
    r = 20
    p = int(r**2)

#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V_qm, W_qm, mu_qm, I_qm = quadmani_greedy(
        X, r, s, 1e-12, np.array([], dtype=int))
    mu_qm = mu_qm.reshape(-1, 1)  

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())
    np.save("results/amr3Dchannel/u/I_qm.npy", I_qm)

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
        'label': "3D Channel Flow Pi3Net",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3, I_nn_3, omegas_3, path_history = smn.fit(X, config)
    torch.save(model_3, "results/amr3Dchannel/sparsemodesnet_model_pi3net.pth")
    np.save("results/amr3Dchannel/I_nn_pi2net.npy", I_nn_3)
    np.save("results/amr3Dchannel/omegas_pi2net.npy", omegas_3)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, 
                     legend_loc='best',
                     title=r'$\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/omega_evolution_pi3net.png')

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
        'label': "3D Channel Flow Pi3Net Leading-r",
        'enable_logging': False
    }
    config = smn.SparseModesNetConfig.from_dict(config_dict)


#%% %======================== Training SparseModesNet =========================%
    model_3, I_nn_3, omegas_3, path_history = smn.fit(X, config)
    torch.save(model_3, "results/amr3Dchannel/sparsemodesnet_model_pi3net.pth")
    np.save("results/amr3Dchannel/I_nn_pi2net.npy", I_nn_3)
    np.save("results/amr3Dchannel/omegas_pi2net.npy", omegas_3)

#%% %=======================Plot the omega evolutions =========================%
    smn.omega_evolve(omegas_3, I_nn_3, config.s, save=True, 
                     legend_loc='best',
                     title=r'$\omega$ Evolution ($\Pi_3$-Net)',
                     filename='figures/amr3Dchannel/omega_evolution_pi3net.png')

