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
    r_max = 15
    
    # number of grids
    n_grids = 2**10
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Advecting Pulse ----------
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        # pulse_width=2.0e-4,
        # pulse_shift=0.1,
        # speed=10.0,
        # final_time=0.1,
        # n_time_samples=1000,
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=8.0,
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
        # plt.savefig('../figures/pulse_data.png', dpi=300)
        plt.show()
        plt.close(fig)
        
    #%% Greedy Quadratic Manifold approach
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, 1e-6, np.array([], dtype=int))
    reduced_points = linear_reduce(V, X_pulse, shift_value)
    reconstructed = lift_quadratic(V, W, shift_value, reduced_points)
    rel_rec_error = np.linalg.norm(reconstructed - X_pulse) / np.linalg.norm(X_pulse)
    print('Relative reconstruction error: ', rel_rec_error)
    print("Quadratic manifold indices I_qm:", I_qm)
    shift_value = np.array(shift_value)[:, np.newaxis]
    
    #%% FF Decoder with leading r modes
    ff_decoder, I_ff, _ = run_sparsemodesnet(
        X_pulse - shift_value,
        SparseModesNetConfig(
            s = s_p, 
            network = NetworkConfig(
                hidden_units = [r_max, int(r_max*(r_max+1)/2)],
                network_type = 'QM'
            ),
            training = TrainingConfig(
                lr = 1e-2,
                batch_size = 1024,
                optimizer = 'Adam',
                final_epochs = 10000,
                device = device,
                I_NN = I_qm.sort()
            ),
            sparsity = SparsityConfig(),
            selection = SelectionConfig(
                mode_selection = None,
                r_max = r_max
            ),
            experiment = ExperimentConfig(
                label = "Pulse FF Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    )
    
    #%% Pi-Net decoder with leading r modes
    pinet_decoder, I_pi, _ = run_sparsemodesnet(
        X_pulse - shift_value,
        SparseModesNetConfig(
            s = s_p, 
            network = NetworkConfig(
                hidden_units = [r_max, int(r_max*(r_max+1)/2), n_grids],
                network_type = 'PiNetNCP',
                poly_order = 2,
                num_polys = 1,
                drop_linear = True,
                drop_constant = True,
                normalize = 'last'
            ),
            training = TrainingConfig(
                lr = 1e-2,
                batch_size = 128,
                optimizer = 'Adam',
                final_epochs = 50000,
                device = device,
                I_NN = I_qm
            ),
            sparsity = SparsityConfig(),
            selection = SelectionConfig(
                mode_selection = None,
                r_max = r_max
            ),
            experiment = ExperimentConfig(
                label = "Pulse Pi-Net Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    )
    
    #%% FF decoder + SparseModesNet
    ff_sparse_decoder, I_ff_sparse, history = run_sparsemodesnet(
        X_pulse - shift_value,
        SparseModesNetConfig(
            s = s_p, 
            network = NetworkConfig(
                hidden_units = [s_p, s_p*5],
                network_type = 'PiNetNCP'
            ),
            training = TrainingConfig(
                lr = 1e-3,
                batch_size = 64,
                optimizer = 'Adam',
                num_epochs = 1000,
                final_epochs = 2000,
                device = device,
            ),
            sparsity = SparsityConfig(
                M               = 10.0,
                nonzero_thresh  = 1e-6,
                lam0            = 1.0,
                epsilon         = 0.1,
                max_iters       = 100
            ),
            selection = SelectionConfig(
                mode_selection  = 'dense2sparse',
                knee_method     = 'dfdt', 
                r_max           = r_max
            ),
            experiment = ExperimentConfig(
                label = "Pulse FF Sparse Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    )
    print("FF Sparse Decoder I_NN:", I_ff_sparse)
    
    #%% Pi-Net decoder + SparseModesNet
    pinet_sparse_decoder, I_pi_sparse, history = run_sparsemodesnet(
        X_pulse - shift_value,
        SparseModesNetConfig(
            s = s_p, 
            network = NetworkConfig(
                hidden_units  = [s_p, s_p*5, n_grids],
                network_type  = 'PiNetNCP',
                poly_order    = 2,
                num_polys     = 1,
                drop_linear   = True,
                drop_constant = True
            ),
            training = TrainingConfig(
                lr = 1e-3,
                batch_size = 256,
                optimizer = 'Adam',
                num_epochs = 1000,
                final_epochs = 2000,
                device = device,
            ),
            sparsity = SparsityConfig(
                M               = 10.0,
                nonzero_thresh  = 1e-6,
                lam0            = 1.0,
                epsilon         = 0.1,
                max_iters       = 100
            ),
            selection = SelectionConfig(
                mode_selection  = 'dense2sparse',
                knee_method     = 'dfdt', 
                r_max           = r_max
            ),
            experiment = ExperimentConfig(
                label = "Pulse Pi-Net Sparse Decoder",
                enable_logging = False,
                logs_dir = "./logs"
            )
        )
    )
    print("Pi-Net Sparse Decoder I_NN:", I_pi_sparse)

    #%% === Plot the reconstructed flow fields (heatmap) ===
    # 1. POD reconstruction
    U_s, _, _ = compute_pod_basis(X_pulse, s=s_p)
    U_r = U_s[:, :r_max]  # Use leading r modes
    Z_input = torch.from_numpy(
        (U_r.T @ X_pulse).T.astype(np.float32)).to(device)
    X_pod_recon = U_r @ U_r.T @ X_pulse
    
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
        U_r = U_s[:, I_ff_sparse]
        Z_input = torch.from_numpy(
            (U_r.T @ X_pulse).T.astype(np.float32)).to(device)
        X_ff_sparse_recon_tensor = ff_sparse_decoder(Z_input)
        X_ff_sparse_recon = X_ff_sparse_recon_tensor.cpu().numpy().T
    
    # 5. Pi-Net sparse decoder reconstruction
    with torch.no_grad():
        pinet_sparse_decoder.eval()
        U_r = U_s[:, I_pi_sparse]
        Z_input = torch.from_numpy(
            (U_r.T @ X_pulse).T.astype(np.float32)).to(device)
        X_pinet_sparse_recon_tensor = pinet_sparse_decoder(Z_input)
        X_pinet_sparse_recon = X_pinet_sparse_recon_tensor.cpu().numpy().T
    
    # Calculate errors
    ff_error = X_pulse - X_ff_recon
    pinet_error = X_pulse - X_pinet_recon
    ff_sparse_error = X_pulse - X_ff_sparse_recon
    pinet_sparse_error = X_pulse - X_pinet_sparse_recon
    
    # Print the errors
    print("POD Reconstruction Error:", np.linalg.norm(pod_error, 'fro') / np.linalg.norm(X_pulse, 'fro'))
    print("FF Decoder Reconstruction Error:", np.linalg.norm(ff_error, 'fro') / np.linalg.norm(X_pulse, 'fro'))
    print("Pi-Net Decoder Reconstruction Error:", np.linalg.norm(pinet_error, 'fro') / np.linalg.norm(X_pulse, 'fro'))
    print("FF Sparse Decoder Reconstruction Error:", np.linalg.norm(ff_sparse_error, 'fro') / np.linalg.norm(X_pulse, 'fro'))
    print("Pi-Net Sparse Decoder Reconstruction Error:", np.linalg.norm(pinet_sparse_error, 'fro') / np.linalg.norm(X_pulse, 'fro'))
    
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
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,0].set_ylabel('Space (x)')
    axes[0,0].set_title('POD Reconstruction')
    
    # (1,2) FF decoder reconstruction
    im2 = axes[0,1].imshow(
        X_ff_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,1].set_title('FF Decoder Reconstruction')
    
    # (1,3) Pi-Net decoder reconstruction
    im3 = axes[0,2].imshow(
        X_pinet_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,2].set_title('Pi-Net Decoder Reconstruction')
    
    # (1,4) FF sparse decoder reconstruction
    im4 = axes[0,3].imshow(
        X_ff_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,3].set_title('FF Sparse Decoder Reconstruction')
    
    # (1,5) Pi-Net sparse decoder reconstruction
    im5 = axes[0,4].imshow(
        X_pinet_sparse_recon, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=recon_vmin, vmax=recon_vmax)
    axes[0,4].set_title('Pi-Net Sparse Decoder Reconstruction')
    
    # Row 2: Errors
    # (2,1) POD error
    im6 = axes[1,0].imshow(
        pod_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Space (x)')
    axes[1,0].set_title('POD Error')
    
    # (2,2) FF decoder error
    im7 = axes[1,1].imshow(
        ff_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_title('FF Decoder Error')
    
    # (2,3) Pi-Net decoder error
    im8 = axes[1,2].imshow(
        pinet_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,2].set_xlabel('Time')
    axes[1,2].set_title('Pi-Net Decoder Error')
    
    # (2,4) FF sparse decoder error
    im9 = axes[1,3].imshow(
        ff_sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
        vmin=error_vmin, vmax=error_vmax)
    axes[1,3].set_xlabel('Time')
    axes[1,3].set_title('FF Sparse Decoder Error')
    
    # (2,5) Pi-Net sparse decoder error
    im10 = axes[1,4].imshow(
        pinet_sparse_error, aspect='auto', cmap='RdBu', origin='lower',
        extent=[tspan_p[0], tspan_p[-1], xspan_p[0], xspan_p[-1]],
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
    # plt.savefig('../figures/pulse/pulse_comparison_all_methods.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)

    #%% Plot waves at specific time points
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
        # Plot FF reconstruction
        ax.plot(xspan_p, X_ff_recon[:, t_idx], '-.', linewidth=2, label='FF Decoder', alpha=0.8)
        # Plot PiNet reconstruction
        ax.plot(xspan_p, X_pinet_recon[:, t_idx], '-.', linewidth=2, label='PiNet Decoder', alpha=0.8)
        # Plot FF Sparse reconstruction
        ax.plot(xspan_p, X_ff_sparse_recon[:, t_idx], ':', linewidth=2, label='FF Sparse', alpha=0.8)
        # Plot PiNet Sparse reconstruction
        ax.plot(xspan_p, X_pinet_sparse_recon[:, t_idx], ':', linewidth=2, label='PiNet Sparse', alpha=0.8)
        
        ax.set_xlabel('Space (x)', fontsize=12)
        ax.set_ylabel('u(x,t)', fontsize=12)
        ax.set_title(f't = {t_val:.3f}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    plt.tight_layout()
    # plt.savefig('../figures/pulse/pulse_waves_timepoints_all_methods.png', dpi=300)
    plt.show()
    plt.close(fig)
# %%
