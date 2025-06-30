#%%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse


#%%
def quadratic_mapping(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    Only computes upper triangular part to avoid redundancy.
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if not isinstance(x, torch.Tensor):
        dim = x.ndim
    else:
        dim = x.dim()
    
    if dim == 1:
        n = x.size(0)
        # Create indices for upper triangular part
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products
        result = x[i_indices] * x[j_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create indices for upper triangular part  
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products for all batches
        result = x[:, i_indices] * x[:, j_indices]
        return result   

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        self.register_buffer('U_r', pod_basis)  # (d, r): store as buffer
        self.d, self.r = pod_basis.shape
        if W is None:
            self.weight_mat = nn.Parameter(
                torch.randn(self.r * (self.r + 1) // 2, self.d) * 0.01)
        else:
            self.weight_mat = nn.Parameter(W, requires_grad=True)  # (r*(r+1)//2, d)
        self.gamma = gamma  # Regularization parameter
        
    def forward(self, z_batch):
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
        # Apply the quadratic mapping
        z_quad = quadratic_mapping(z_batch)  # (batch, r*(r+1)//2)
        x_hat_nn = z_quad @ self.weight_mat  # (batch, d)
        # Reconstruct x_hat
        x_hat = x_hat_lin + x_hat_nn
        return x_hat

def train_qm(model: QuadraticManifold, num_epochs: int, lr: float,
             device: str, z_batch: np.ndarray, x_batch: np.ndarray):
    model.to(device)
    # optimizer = optim.LBFGS(model.parameters(), lr=lr, max_iter=20)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=model.gamma)
    
    mse_loss = nn.MSELoss(reduction='mean')

    loss_history = []
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000)
    
    z_batch = torch.tensor(z_batch, dtype=torch.float32)
    x_batch = torch.tensor(x_batch, dtype=torch.float32)

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        model.train()
        z_batch = z_batch.to(device)  # (batch, s)
        x_batch = x_batch.to(device)  # (batch, d)
        
        def closure():
            optimizer.zero_grad()
            x_hat_batch = model(z_batch)
            
            reconstruction_loss = mse_loss(x_hat_batch, x_batch)
            regularization_loss = 0.5 * model.gamma * torch.sum(model.weight_mat ** 2)
            loss = reconstruction_loss + regularization_loss
            
            loss.backward()
            
            # Fix gradient contiguity issue
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.contiguous()
            
            return loss
        
        if isinstance(optimizer, optim.LBFGS):
            loss = optimizer.step(closure)
        else:
            optimizer.zero_grad()
            x_hat_batch = model(z_batch)
            reconstruction_loss = mse_loss(x_hat_batch, x_batch)
            loss = reconstruction_loss
            loss.backward()
            optimizer.step()

        # CRITICAL FIX 6: Proper relative error calculation
        with torch.no_grad():
            x_hat_batch = model(z_batch)
            rel_error = torch.norm(x_hat_batch - x_batch) / torch.norm(x_batch)
            epoch_loss = rel_error.item()
        
        loss_history.append(epoch_loss)
        lr_scheduler.step(epoch_loss)

        # Print every 10 epochs or first:
        if (epoch % 1000 == 0) or (epoch == 1):
            lr_current = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:<6d} | lr={lr_current:.4e} | "
                  f"Rel Error={epoch_loss:.6e}")

    return loss_history


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

    #%% Train Neural Network Quadratic Manifold
    print("\n" + "="*60)
    print("Training Neural Network Quadratic Manifold")
    print("="*60)
    
    # Use the linear basis V from greedy approach
    pod_basis = torch.tensor(V, dtype=torch.float32)
    
    # Create quadratic manifold model
    gamma = 1e-6  # Regularization parameter
    qm_model = QuadraticManifold(
        pod_basis, 
        gamma, 
        torch.tensor(W, dtype=torch.float32).T + torch.rand(W.shape, dtype=torch.float32).T * 0.01
        # torch.tensor(W, dtype=torch.float32).T
    )
    
    print(f"Model parameters:")
    print(f"  Input dimension (d): {qm_model.d}")
    print(f"  Reduced dimension (r): {qm_model.r}")
    print(f"  Quadratic features: {qm_model.r * (qm_model.r + 1) // 2}")
    print(f"  Weight matrix shape: {qm_model.weight_mat.shape}")
    print(f"  Total parameters: {sum(p.numel() for p in qm_model.parameters())}")
    print(f"  Regularization gamma: {gamma}")
    
    # Prepare training data
    # Use reduced coordinates as input and original data as target
    z_train = reduced_points.T              # (n_samples, r)
    x_train = (X_pulse - shift_value).T     # (n_samples, d)
    
    print(f"\nTraining data shapes:")
    print(f"  z_train: {z_train.shape}")
    print(f"  x_train: {x_train.shape}")
    
    # Training parameters
    num_epochs = 20000
    lr = 1e-4
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    
    # Train the model
    print("\nStarting training...")
    loss_history = train_qm(qm_model, num_epochs, lr, device, z_train, x_train)
    
    #%% Evaluate trained model
    print("\n" + "="*60)
    print("Evaluating Trained Model")
    print("="*60)
    
    qm_model.eval()
    with torch.no_grad():
        z_test = torch.tensor(z_train, dtype=torch.float32).to(device)
        x_reconstructed = qm_model(z_test).cpu().numpy().astype(np.float64)  
        x_reconstructed += shift_value.T  # Add the shift back
    
    # manual reconstruction using numpy    
    V_np = V.astype(np.float64) 
    W_np = W.astype(np.float64)
    z_train = np.array(z_train, dtype=np.float64) 
    x_reconstructed_man = V_np @ z_train.T 
    x_reconstructed_man += W_np @ quadratic_mapping(z_train).T
    x_reconstructed_man += shift_value     
    x_reconstructed_man = x_reconstructed_man.T  # (n_samples, d)
    
    # Compute reconstruction error
    rel_error_nn = np.linalg.norm(x_reconstructed.T - X_pulse) / np.linalg.norm(X_pulse)
    rel_error_man = np.linalg.norm(x_reconstructed_man.T - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Relative reconstruction error (NN QM): {rel_error_nn:.6e}")
    print(f"Relative reconstruction error (manual): {rel_error_man:.6e}")
    print(f"Relative reconstruction error (Greedy QM): {rel_rec_error:.6e}")
    print(f"Improvement ratio: {rel_rec_error / rel_error_nn:.3e}x")
    
    #%% Plotting results
    print("\nGenerating plots...")
    
    # Plot training loss
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.semilogy(loss_history)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Quadratic Manifold Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Plot reconstruction comparison
    # Select a few time snapshots to compare
    time_indices = [0, n_p//4, n_p//2, 3*n_p//4, n_p-1]
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    for i, (t_idx, color) in enumerate(zip(time_indices, colors)):
        ax2.plot(xspan_p, X_pulse[:, t_idx], 
                color='black', linestyle='-', alpha=0.7, linewidth=1.5,
                label=f'Original t={tspan_p[t_idx]:.3f}')
        ax2.plot(xspan_p, x_reconstructed[t_idx, :], 
                color=color, linestyle='--', alpha=0.7,
                label=f'NN QM t={tspan_p[t_idx]:.3f}')
    
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x,t)')
    ax2.set_title('Reconstruction Comparison')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plot 3D comparison
    if sanity_check:
        fig = plt.figure(figsize=(18, 6))
        
        # Original data
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u(x,t)')
        ax1.set_title('Original Data')
        
        # NN Quadratic Manifold reconstruction
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(X_mesh, T_mesh, x_reconstructed, cmap='viridis', alpha=0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        ax2.set_title(f'NN QM Reconstruction\nError: {rel_error_nn:.2e}')
        
        # Error
        ax3 = fig.add_subplot(133, projection='3d')
        error = np.abs(x_reconstructed - X_pulse.T)
        surf3 = ax3.plot_surface(X_mesh, T_mesh, error, cmap='Reds', alpha=0.8)
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('|error|')
        ax3.set_title('Absolute Error')
        
        plt.tight_layout()
        plt.show()
    
    print(f"\n{'='*60}")
    print("Quadratic Manifold Training Completed!")
    print(f"{'='*60}")
# %%
