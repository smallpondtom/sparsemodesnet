#%%
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.heat1d import generate_heat_data
from sparsemodesnet import run_sparsemodesnet
from sparsemodesnet.pod import compute_pod_basis
from sparsemodesnet.config import (
    SparseModesNetConfig, NetworkConfig,
    TrainingConfig, SparsityConfig,
    SelectionConfig, ExperimentConfig
)

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
    
    # Number of modes
    r_max = 3
    
    # number of grids
    n_grids = 2**7
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Heat Equation ----------
    X_heat, xspan_h, tspan_h = generate_heat_data(
        nx=n_grids, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = r_max * 10
    
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
        
    #%% Simple POD basis decoder with leading r modes
    U_r, _, _ = compute_pod_basis(X_heat, s=r_max)
    X_pod = U_r @ U_r.T @ X_heat  # POD reconstruction
    
    #%% FF Decoder with leading r modes
    ff_decoder, I_ff, history = run_sparsemodesnet(
        X_heat,
        SparseModesNetConfig(
            s = s_h, 
            network = NetworkConfig(
                hidden_units = [s_h, int(s_h*(s_h+1)/2)],
                network_type = 'FF'
            ),
            training = TrainingConfig(
                lr = 1e-3,
                batch_size = 32,
                optimizer = 'Adam',
                num_epochs = 2000,
                device = device
            ),
            sparsity = SparsityConfig(),
            selection = SelectionConfig(
                mode_selection = None,
                r_max = r_max
            ),
            experiment = ExperimentConfig(
                label = "Heat Equation FF Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    )
    
    #%% Pi-Net decoder with leading r modes
    pinet_decoder, I_pi, history = run_sparsemodesnet(
        X_heat,
        SparseModesNetConfig(
            s = s_h, 
            network = NetworkConfig(
                hidden_units = [s_h, int(s_h*(s_h+1)/2), n_grids],
                network_type = 'PiNetCCP',
                poly_order = 2,
                num_polys = 1,
                drop_linear = False
            ),
            training = TrainingConfig(
                lr = 1e-3,
                batch_size = 32,
                optimizer = 'Adam',
                num_epochs = 1000,
                device = device
            ),
            sparsity = SparsityConfig(),
            selection = SelectionConfig(
                mode_selection = None,
                r_max = r_max
            ),
            experiment = ExperimentConfig(
                label = "Heat Equation Pi-Net Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    ) 
    
    #%% FF decoder + SparseModesNet
    ff_sparse_decoder, I_ff_sparse, history = run_sparsemodesnet(
        X_heat,
        SparseModesNetConfig(
            s               = s_h, 
            network         = NetworkConfig(
                hidden_units    = [s_h, int(s_h*(s_h+1)/2)],
                network_type    = 'FF'
            ),
            training        = TrainingConfig(
                lr              = 1e-3,
                batch_size      = 32,
                optimizer       = 'Adam',
                num_epochs      = 100,
                final_epochs    = 2000,
                device          = device
            ),
            sparsity        = SparsityConfig(
                M               = 4.0,
                nonzero_thresh  = 1e-6,
                lam0            = 1e-3,
                epsilon         = 0.2,
                max_iters       = 100
            ),
            selection       = SelectionConfig(
                mode_selection  = 'dense2sparse',
                knee_method     = 'dfdt', 
                r_max           = r_max
            ),
            experiment      = ExperimentConfig(
                label           = "Heat Equation FF Sparse Decoder",
                enable_logging  = True,  
                logs_dir        = "./logs"
            )
        ) 
    )
    print("FF Sparse Decoder I_NN:", I_ff_sparse)
    
    #%% Pi-Net decoder + SparseModesNet
    pinet_sparse_decoder, I_pi_sparse, history = run_sparsemodesnet(
        X_heat,
        SparseModesNetConfig(
            s               = s_h, 
            network         = NetworkConfig(
                hidden_units    = [s_h, int(s_h*(s_h+1)/2), n_grids],
                network_type    = 'PiNetCCP',
                poly_order      = 2,
                num_polys       = 1,
                drop_linear     = False
            ),
            training        = TrainingConfig(
                lr              = 1e-3,
                batch_size      = 32,
                optimizer       = 'Adam',
                num_epochs      = 100,
                final_epochs    = 1000,
                device          = device
            ),
            sparsity        = SparsityConfig(
                M               = 4.0,
                nonzero_thresh  = 1e-14,
                lam0            = 1e-3,
                epsilon         = 0.2,
                max_iters       = 100
            ),
            selection       = SelectionConfig(
                mode_selection  = 'dense2sparse',
                knee_method     = 'dfdt', 
                r_max           = r_max
            ),
            experiment      = ExperimentConfig(
                label           = "Heat Equation Pi-Net Sparse Decoder",
                enable_logging  = True,  
                logs_dir        = "./logs"
            )
        )
    )
    print("Pi-Net Sparse Decoder I_NN:", I_pi_sparse)
    
    #%% === Plot the reconstructed flow fields (heatmap) ===
    # 1. POD reconstruction
    U_r, _, _ = compute_pod_basis(X_heat, s=r_max)
    Z_input = torch.from_numpy(
        (U_r.T @ X_heat).T.astype(np.float32)).to(device)
    X_pod_recon = U_r @ U_r.T @ X_heat
    
    # 2. FF decoder reconstruction
    with torch.no_grad():
        ff_decoder.eval()
        X_ff_recon_tensor = ff_decoder(Z_input)
        X_ff_recon = X_ff_recon_tensor.cpu().numpy().T
    
    # 3. Pi-Net decoder reconstruction
    with torch.no_grad():
        pinet_decoder.eval()
        X_pinet_recon_tensor = pinet_decoder(Z_input)
        X_pinet_recon = X_pinet_recon_tensor.cpu().numpy().T
    
    # 4. FF sparse decoder reconstruction
    with torch.no_grad():
        ff_sparse_decoder.eval()
        X_ff_sparse_recon_tensor = ff_sparse_decoder(Z_input)
        X_ff_sparse_recon = X_ff_sparse_recon_tensor.cpu().numpy().T
    
    # 5. Pi-Net sparse decoder reconstruction
    with torch.no_grad():
        pinet_sparse_decoder.eval()
        X_pinet_sparse_recon_tensor = pinet_sparse_decoder(Z_input)
        X_pinet_sparse_recon = X_pinet_sparse_recon_tensor.cpu().numpy().T
    
    # Calculate errors
    pod_error = X_heat - X_pod_recon
    ff_error = X_heat - X_ff_recon
    pinet_error = X_heat - X_pinet_recon
    ff_sparse_error = X_heat - X_ff_sparse_recon
    pinet_sparse_error = X_heat - X_pinet_sparse_recon
    
    # Calculate unified color ranges
    all_reconstructions = [X_pod_recon, X_ff_recon, X_pinet_recon, X_ff_sparse_recon, X_pinet_sparse_recon]
    all_errors = [pod_error, ff_error, pinet_error, ff_sparse_error, pinet_sparse_error]
    
    recon_vmin = min([arr.min() for arr in all_reconstructions])
    recon_vmax = max([arr.max() for arr in all_reconstructions])
    
    # For errors, use symmetric range around zero based on smallest absolute max
    error_abs_max = min([np.abs(arr).max() for arr in all_errors])
    error_vmin = -error_abs_max
    error_vmax = error_abs_max
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Row 1: Reconstructions
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    
    # (1,2) FF decoder reconstruction
    im2 = axes[0,1].imshow(
        X_ff_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,1].set_title('FF Decoder Reconstruction')
    
    # (1,3) Pi-Net decoder reconstruction
    im3 = axes[0,2].imshow(
        X_pinet_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,2].set_title('Pi-Net Decoder Reconstruction')
    
    # (1,4) FF sparse decoder reconstruction
    im4 = axes[0,3].imshow(
        X_ff_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,3].set_title('FF Sparse Decoder Reconstruction')
    
    # (1,5) Pi-Net sparse decoder reconstruction
    im5 = axes[0,4].imshow(
        X_pinet_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,4].set_title('Pi-Net Sparse Decoder Reconstruction')
    
    # Row 2: Errors
    # (2,1) POD error
    im6 = axes[1,0].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('POD Error')
    
    # (2,2) FF decoder error
    im7 = axes[1,1].imshow(
        ff_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_title('FF Decoder Error')
    
    # (2,3) Pi-Net decoder error
    im8 = axes[1,2].imshow(
        pinet_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_title('Pi-Net Decoder Error')
    
    # (2,4) FF sparse decoder error
    im9 = axes[1,3].imshow(
        ff_sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,3].set_xlabel('Time')
    axes[1,3].set_title('FF Sparse Decoder Error')
    
    # (2,5) Pi-Net sparse decoder error
    im10 = axes[1,4].imshow(
        pinet_sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_h[0], tspan_h[-1], xspan_h[0], xspan_h[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,4].set_xlabel('Time')
    axes[1,4].set_title('Pi-Net Sparse Decoder Error')
    
    # Add colorbars manually positioned to avoid tight_layout conflicts
    # Colorbar for reconstructions
    cax1 = fig.add_axes([0.92, 0.59, 0.02, 0.35])
    cbar1 = plt.colorbar(im5, cax=cax1, label='u(x,t)')
    
    # Colorbar for errors
    cax2 = fig.add_axes([0.92, 0.11, 0.02, 0.35])
    cbar2 = plt.colorbar(im10, cax=cax2, label='Error')
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.savefig('../figures/heat_comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    #%% Test the model on a new sample
    # Generate new test data with different parameters
    X_test, xspan_test, tspan_test = generate_heat_data(
        nx=n_grids, nt=500, alpha=0.02, x_max=1.0, t_max=0.5)
    
    # Create test input
    U_r_test, _, _ = compute_pod_basis(X_test, s=r_max)
    Z_test = torch.from_numpy(
        (U_r_test.T @ X_test).T.astype(np.float32)).to(device)
    
    # Test reconstructions
    # 1. POD reconstruction
    X_pod_test_recon = U_r_test @ U_r_test.T @ X_test
    
    # 2. FF decoder reconstruction
    with torch.no_grad():
        ff_decoder.eval()
        X_ff_test_recon_tensor = ff_decoder(Z_test)
        X_ff_test_recon = X_ff_test_recon_tensor.cpu().numpy().T
    
    # 3. Pi-Net decoder reconstruction
    with torch.no_grad():
        pinet_decoder.eval()
        X_pinet_test_recon_tensor = pinet_decoder(Z_test)
        X_pinet_test_recon = X_pinet_test_recon_tensor.cpu().numpy().T
    
    # 4. FF sparse decoder reconstruction
    with torch.no_grad():
        ff_sparse_decoder.eval()
        X_ff_sparse_test_recon_tensor = ff_sparse_decoder(Z_test)
        X_ff_sparse_test_recon = X_ff_sparse_test_recon_tensor.cpu().numpy().T
    
    # 5. Pi-Net sparse decoder reconstruction
    with torch.no_grad():
        pinet_sparse_decoder.eval()
        X_pinet_sparse_test_recon_tensor = pinet_sparse_decoder(Z_test)
        X_pinet_sparse_test_recon = X_pinet_sparse_test_recon_tensor.cpu().numpy().T
    
    # Calculate test errors
    pod_test_error = X_test - X_pod_test_recon
    ff_test_error = X_test - X_ff_test_recon
    pinet_test_error = X_test - X_pinet_test_recon
    ff_sparse_test_error = X_test - X_ff_sparse_test_recon
    pinet_sparse_test_error = X_test - X_pinet_sparse_test_recon
    
    # Calculate unified color ranges for test data
    all_test_reconstructions = [X_pod_test_recon, X_ff_test_recon, X_pinet_test_recon, X_ff_sparse_test_recon, X_pinet_sparse_test_recon]
    all_test_errors = [pod_test_error, ff_test_error, pinet_test_error, ff_sparse_test_error, pinet_sparse_test_error]
    
    test_recon_vmin = min([arr.min() for arr in all_test_reconstructions])
    test_recon_vmax = max([arr.max() for arr in all_test_reconstructions])
    
    # For errors, use symmetric range around zero
    test_error_abs_max = min([np.abs(arr).max() for arr in all_test_errors])
    test_error_vmin = -test_error_abs_max
    test_error_vmax = test_error_abs_max
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Row 1: Test Reconstructions
    # (1,1) POD reconstruction
    im1 = axes[0,0].imshow(
        X_pod_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_recon_vmin, vmax=test_recon_vmax)
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Test Reconstruction')
    
    # (1,2) FF decoder reconstruction
    im2 = axes[0,1].imshow(
        X_ff_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_recon_vmin, vmax=test_recon_vmax)
    axes[0,1].set_title('FF Decoder Test Reconstruction')
    
    # (1,3) Pi-Net decoder reconstruction
    im3 = axes[0,2].imshow(
        X_pinet_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_recon_vmin, vmax=test_recon_vmax)
    axes[0,2].set_title('Pi-Net Decoder Test Reconstruction')
    
    # (1,4) FF sparse decoder reconstruction
    im4 = axes[0,3].imshow(
        X_ff_sparse_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_recon_vmin, vmax=test_recon_vmax)
    axes[0,3].set_title('FF Sparse Decoder Test Reconstruction')
    
    # (1,5) Pi-Net sparse decoder reconstruction
    im5 = axes[0,4].imshow(
        X_pinet_sparse_test_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_recon_vmin, vmax=test_recon_vmax)
    axes[0,4].set_title('Pi-Net Sparse Decoder Test Reconstruction')
    
    # Row 2: Test Errors
    # (2,1) POD error
    im6 = axes[1,0].imshow(
        pod_test_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_vmin, vmax=test_error_vmax)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('POD Test Error')
    
    # (2,2) FF decoder error
    im7 = axes[1,1].imshow(
        ff_test_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_vmin, vmax=test_error_vmax)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_title('FF Decoder Test Error')
    
    # (2,3) Pi-Net decoder error
    im8 = axes[1,2].imshow(
        pinet_test_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_vmin, vmax=test_error_vmax)
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_title('Pi-Net Decoder Test Error')
    
    # (2,4) FF sparse decoder error
    im9 = axes[1,3].imshow(
        ff_sparse_test_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_vmin, vmax=test_error_vmax)
    axes[1,3].set_xlabel('Time')
    axes[1,3].set_title('FF Sparse Decoder Test Error')
    
    # (2,5) Pi-Net sparse decoder error
    im10 = axes[1,4].imshow(
        pinet_sparse_test_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_test[0], tspan_test[-1], xspan_test[0], xspan_test[-1]],
        vmin=test_error_vmin, vmax=test_error_vmax)
    axes[1,4].set_xlabel('Time')
    axes[1,4].set_title('Pi-Net Sparse Decoder Test Error')
    
    # Add colorbars
    cax1 = fig.add_axes([0.92, 0.59, 0.02, 0.35])
    cbar1 = plt.colorbar(im5, cax=cax1, label='u(x,t)')
    
    cax2 = fig.add_axes([0.92, 0.11, 0.02, 0.35])
    cbar2 = plt.colorbar(im10, cax=cax2, label='Error')
    
    plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.1, wspace=0.3, hspace=0.3)
    plt.savefig('../figures/heat_test_comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)