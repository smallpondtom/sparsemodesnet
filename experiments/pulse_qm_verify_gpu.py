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
def quadratic_mapping_torch(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    CRITICAL: Must use LOWER triangular (tril) to match the paper!
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if x.dim() == 1:
        n = x.size(0)
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        _, n = x.shape
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        result = x[:, i_indices] * x[:, j_indices]
        return result

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

# Keep the old function for backward compatibility
def quadratic_mapping(x):
    """Legacy function - calls the appropriate version"""
    if isinstance(x, torch.Tensor):
        return quadratic_mapping_torch(x)
    else:
        return quadratic_mapping_numpy(x)

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        # Ensure consistent precision across devices
        if pod_basis.dtype != torch.float32:
            pod_basis = pod_basis.float()
            
        self.register_buffer('U_r', pod_basis)  # (d, r): store as buffer
        self.d, self.r = pod_basis.shape
        
        if W is None:
            # Random initialization
            self.weight_mat = nn.Parameter(
                torch.randn(self.r * (self.r + 1) // 2, self.d, dtype=torch.float32) * 0.01)
        else:
            # CRITICAL FIX: Don't transpose! Check shapes first
            if W.dtype != torch.float32:
                W = W.float()
            expected_shape = (self.r * (self.r + 1) // 2, self.d)
            
            if W.shape == expected_shape:
                print(f"✓ W shape {W.shape} matches expected {expected_shape}")
                self.weight_mat = nn.Parameter(W.clone())
            elif W.shape == (expected_shape[1], expected_shape[0]):
                print(f"⚠ W shape {W.shape} needs transpose to match {expected_shape}")
                self.weight_mat = nn.Parameter(W.T.clone())
            else:
                raise ValueError(f"W shape {W.shape} doesn't match expected {expected_shape} or its transpose")
        
        self.gamma = gamma  # Regularization parameter
        
    def forward(self, z_batch):
        # Ensure consistent precision
        if z_batch.dtype != torch.float32:
            z_batch = z_batch.float()
            
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
        # Apply the quadratic mapping
        z_quad = quadratic_mapping_torch(z_batch)  # (batch, r*(r+1)//2)
        x_hat_nn = z_quad @ self.weight_mat  # (batch, d)
        # Reconstruct x_hat
        x_hat = x_hat_lin + x_hat_nn
        return x_hat

#======================== Test Functions ======================================#
def test_quadratic_mapping_consistency():
    """
    Test that torch and numpy quadratic mappings give identical results
    """
    print("\n" + "="*50)
    print("QUADRATIC MAPPING CONSISTENCY TEST")
    print("="*50)
    
    # Test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Single vector test
    z_test = np.random.randn(5).astype(np.float32)
    z_torch = torch.tensor(z_test, dtype=torch.float32)
    
    quad_np = quadratic_mapping_numpy(z_test)
    quad_torch = quadratic_mapping_torch(z_torch).numpy()
    
    diff_single = np.max(np.abs(quad_np - quad_torch))
    print(f"Single vector difference: {diff_single:.2e}")
    
    # Batch test
    z_batch = np.random.randn(10, 5).astype(np.float32)
    z_batch_torch = torch.tensor(z_batch, dtype=torch.float32)
    
    quad_batch_np = quadratic_mapping_numpy(z_batch)
    quad_batch_torch = quadratic_mapping_torch(z_batch_torch).numpy()
    
    diff_batch = np.max(np.abs(quad_batch_np - quad_batch_torch))
    print(f"Batch difference: {diff_batch:.2e}")
    
    if diff_single < 1e-6 and diff_batch < 1e-6:  # Float32 precision
        print("✓ Quadratic mappings are consistent!")
        return True
    else:
        print("✗ Quadratic mappings are inconsistent!")
        return False

def test_simple_quadratic_regression(device='cpu'):
    """
    Test quadratic manifold on a simple synthetic regression problem.
    Generate data from X = V*Z + W*quadratic_mapping(Z) + noise
    Then verify both analytical and NN solutions recover the true W.
    """
    print("\n" + "="*60)
    print("SIMPLE QUADRATIC REGRESSION TEST")
    print("="*60)
    print(f"Using device: {device}")
    
    # Set dimensions (small for easy verification)
    d = 20   # ambient dimension
    r = 6    # reduced dimension (smaller for GPU memory)
    n_samples = 500  # Smaller for faster training
    noise_level = 1e-5
    
    # Set seeds for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Generate true V matrix (linear basis)
    V_true = np.random.randn(d, r).astype(np.float32)
    V_true = np.linalg.qr(V_true)[0]  # Orthogonalize
    print(f"V_true shape: {V_true.shape}")
    
    # Generate true W matrix (quadratic weights)
    quad_dim = r * (r + 1) // 2
    W_true = np.random.randn(d, quad_dim).astype(np.float32) * 0.1
    print(f"W_true shape: {W_true.shape}")
    print(f"Quadratic dimension: {quad_dim}")
    
    # Generate random reduced coordinates
    Z_true = np.random.randn(r, n_samples).astype(np.float32)
    print(f"Z_true shape: {Z_true.shape}")
    
    # Generate quadratic features
    Z_quad_list = []
    for i in range(n_samples):
        z_i = Z_true[:, i]
        z_quad_i = quadratic_mapping_numpy(z_i)
        Z_quad_list.append(z_quad_i)
    Z_quad_true = np.array(Z_quad_list).T  # (quad_dim, n_samples)
    print(f"Z_quad_true shape: {Z_quad_true.shape}")
    
    # Generate true data: X = V*Z + W*Z_quad + noise
    X_linear = V_true @ Z_true
    X_quad = W_true @ Z_quad_true
    noise = np.random.randn(d, n_samples) * noise_level
    X_true = X_linear + X_quad + noise
    X_true = X_true.astype(np.float32)
    print(f"X_true shape: {X_true.shape}")
    
    # Compute SNR
    signal_power = np.linalg.norm(X_linear + X_quad)**2
    noise_power = np.linalg.norm(noise)**2
    snr_db = 10 * np.log10(signal_power / noise_power)
    print(f"SNR: {snr_db:.1f} dB")
    
    print("\n" + "-"*40)
    print("ANALYTICAL LEAST SQUARES SOLUTION")
    print("-"*40)
    
    # Solve for W analytically using least squares
    residual = X_true - V_true @ Z_true  # Remove linear part
    print(f"Residual shape: {residual.shape}")
    
    # Solve: W * Z_quad = residual
    ZqZqT = Z_quad_true @ Z_quad_true.T
    ZqZqT_inv = np.linalg.pinv(ZqZqT)
    W_analytical = residual @ Z_quad_true.T @ ZqZqT_inv
    
    print(f"W_analytical shape: {W_analytical.shape}")
    
    # Verify analytical solution
    X_reconstructed_analytical = V_true @ Z_true + W_analytical @ Z_quad_true
    analytical_error = np.linalg.norm(X_reconstructed_analytical - X_true) / np.linalg.norm(X_true)
    print(f"Analytical reconstruction error: {analytical_error:.2e}")
    
    # Compare recovered W with true W
    W_recovery_error = np.linalg.norm(W_analytical - W_true) / np.linalg.norm(W_true)
    print(f"W recovery error: {W_recovery_error:.2e}")
    
    print("\n" + "-"*40)
    print("NEURAL NETWORK SOLUTION")
    print("-"*40)
    
    # Test neural network with analytical W
    pod_basis_torch = torch.tensor(V_true, dtype=torch.float32).to(device)
    W_analytical_torch = torch.tensor(W_analytical.T, dtype=torch.float32).to(device)  # Transpose for NN
    
    # Create model with analytical weights
    qm_model = QuadraticManifold(pod_basis_torch, gamma=1e-8, W=W_analytical_torch)
    qm_model = qm_model.to(device)
    qm_model.eval()
    
    # Prepare NN input/output
    Z_torch = torch.tensor(Z_true.T, dtype=torch.float32).to(device)  # (n_samples, r)
    X_torch = torch.tensor(X_true.T, dtype=torch.float32).to(device)  # (n_samples, d)
    
    print(f"Z_torch shape: {Z_torch.shape}")
    print(f"X_torch shape: {X_torch.shape}")
    
    # Test NN forward pass
    with torch.no_grad():
        X_reconstructed_nn = qm_model(Z_torch)
    
    nn_error = torch.norm(X_reconstructed_nn - X_torch) / torch.norm(X_torch)
    print(f"NN reconstruction error: {nn_error.item():.2e}")
    
    # Compare NN weights with analytical weights
    nn_weights = qm_model.weight_mat.T  # Transpose back
    nn_weight_diff = torch.norm(nn_weights - W_analytical_torch.T) / torch.norm(W_analytical_torch.T)
    print(f"NN vs analytical weight difference: {nn_weight_diff.item():.2e}")
    
    print("\n" + "-"*40)
    print("TRAINING FROM SCRATCH")
    print("-"*40)
    
    # Train NN from random initialization
    qm_model_scratch = QuadraticManifold(pod_basis_torch, gamma=1e-8)
    qm_model_scratch = qm_model_scratch.to(device)
    qm_model_scratch.train()
    
    # Training setup
    optimizer = optim.Adam(qm_model_scratch.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    print("Training NN from scratch...")
    for epoch in range(2000):  # Fewer epochs for faster testing
        optimizer.zero_grad()
        X_pred = qm_model_scratch(Z_torch)
        loss = mse_loss(X_pred, X_torch)
        loss.backward()
        optimizer.step()
        
        if epoch % 400 == 0:
            with torch.no_grad():
                rel_err = torch.norm(X_pred - X_torch) / torch.norm(X_torch)
                print(f"Epoch {epoch:4d}: Loss = {loss.item():.6e}, Rel Error = {rel_err.item():.6e}")
    
    # Final evaluation
    qm_model_scratch.eval()
    with torch.no_grad():
        X_final = qm_model_scratch(Z_torch)
        final_error = torch.norm(X_final - X_torch) / torch.norm(X_torch)
    
    print(f"Final training error: {final_error.item():.2e}")
    
    # Compare learned weights with true/analytical weights
    learned_weights = qm_model_scratch.weight_mat.T
    learned_vs_true_diff = torch.norm(learned_weights - torch.tensor(W_true, dtype=torch.float32).to(device)) / torch.norm(torch.tensor(W_true, dtype=torch.float32).to(device))
    learned_vs_analytical_diff = torch.norm(learned_weights - torch.tensor(W_analytical, dtype=torch.float32).to(device)) / torch.norm(torch.tensor(W_analytical, dtype=torch.float32).to(device))
    
    print(f"Learned vs true W difference: {learned_vs_true_diff.item():.2e}")
    print(f"Learned vs analytical W difference: {learned_vs_analytical_diff.item():.2e}")
    
    print("\n" + "="*60)
    print("REGRESSION TEST SUMMARY")
    print("="*60)
    print(f"Data SNR:                      {snr_db:.1f} dB")
    print(f"Analytical error:              {analytical_error:.2e}")
    print(f"NN (analytical weights) error: {nn_error.item():.2e}")
    print(f"NN (trained) error:            {final_error.item():.2e}")
    print(f"W recovery error:              {W_recovery_error:.2e}")
    print(f"NN weight consistency:         {nn_weight_diff.item():.2e}")
    
    # Success criteria (more lenient for float32)
    success_analytical = analytical_error < noise_level * 100
    success_nn_init = nn_error.item() < noise_level * 100  
    success_nn_trained = final_error.item() < noise_level * 100
    success_weight_recovery = W_recovery_error < 1e-2
    success_nn_consistency = nn_weight_diff.item() < 1e-5
    
    all_success = all([success_analytical, success_nn_init, success_nn_trained, 
                      success_weight_recovery, success_nn_consistency])
    
    if all_success:
        print("✓ ALL TESTS PASSED: Quadratic manifold implementation is correct!")
    else:
        print("✗ SOME TESTS FAILED:")
        if not success_analytical:
            print("  - Analytical solution failed")
        if not success_nn_init:
            print("  - NN with analytical weights failed")
        if not success_nn_trained:
            print("  - NN training failed")
        if not success_weight_recovery:
            print("  - Weight recovery failed")
        if not success_nn_consistency:
            print("  - NN weight consistency failed")
    
    return all_success, {
        'analytical_error': analytical_error,
        'nn_error': nn_error.item(),
        'trained_error': final_error.item(),
        'weight_recovery_error': W_recovery_error,
        'weight_consistency_error': nn_weight_diff.item(),
        'snr_db': snr_db
    }

def train_qm(model: QuadraticManifold, num_epochs: int, lr: float,
             momentum: float, device: str, z_batch: np.ndarray, 
             x_batch: np.ndarray):
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                          nesterov=True, weight_decay=model.gamma)
    
    model.train()
    mse_loss = nn.MSELoss(reduction='mean')
    loss_history = []
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1000)
    
    z_batch = torch.tensor(z_batch, dtype=torch.float32).to(device)
    x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        x_hat_batch = model(z_batch)
        reconstruction_loss = mse_loss(x_hat_batch, x_batch)
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()

        # Calculate relative error
        with torch.no_grad():
            rel_error = torch.norm(x_hat_batch - x_batch) / torch.norm(x_batch)
            epoch_loss = rel_error.item()
        
        loss_history.append(epoch_loss)
        lr_scheduler.step(epoch_loss)

        # Print progress
        if (epoch % 10 == 0) or (epoch == 1):
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
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

#%% #================== First run the simple regression test ==================#
    print("RUNNING SIMPLE REGRESSION TEST FIRST...")
    success, results = test_simple_quadratic_regression(device)
    
    if not success:
        print("\n" + "="*60)
        print("REGRESSION TEST FAILED - STOPPING EXECUTION")
        print("Fix the basic quadratic regression before testing on real data")
        print("="*60)
        exit(1)
    else:
        print("\n" + "="*60)
        print("REGRESSION TEST PASSED - PROCEEDING WITH PULSE DATA")
        print("="*60)

#%% #===================== Main Experiment with Pulse Data ====================#
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
        plt.show()
        plt.close(fig)
        
    #%% Test quadratic mapping consistency
    mapping_consistent = test_quadratic_mapping_consistency()
    
    if not mapping_consistent:
        print("ERROR: Quadratic mappings are inconsistent! Stopping.")
        exit(1)
        
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
        torch.tensor(W, dtype=torch.float32).T
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
    num_epochs = 10000
    lr = 1.2e-4
    
    print(f"\nTraining parameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Device: {device}")
    
    # Train the model
    print("\nStarting training...")
    loss_history = train_qm(qm_model, num_epochs, lr, 0.9,
                            device, z_train, x_train)
    
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
    z_train_np = np.array(z_train, dtype=np.float64) 
    x_reconstructed_man = V_np @ z_train_np.T 
    x_reconstructed_man += W_np @ quadratic_mapping_numpy(z_train_np).T
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
        fig = plt.figure(figsize=(19, 6))
        
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
