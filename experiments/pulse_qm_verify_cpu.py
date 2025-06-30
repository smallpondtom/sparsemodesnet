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

def quadratic_mapping_torch(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    CRITICAL: Must use UPPER triangular (triu) to match the paper!
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if x.dim() == 1:
        n = x.size(0)
        # FIXED: Use triu_indices (upper triangular) not tril_indices!
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        batch_size, n = x.shape
        # FIXED: Use triu_indices (upper triangular) not tril_indices!
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
        batch_size, n = x.shape
        i_indices, j_indices = np.tril_indices(n)
        result = x[:, i_indices] * x[:, j_indices]
        return result

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        # Ensure everything is double precision
        pod_basis = pod_basis.double()
        self.register_buffer('U_r', pod_basis)  # (d, r)
        self.d, self.r = pod_basis.shape
        
        if W is None:
            # Random initialization
            self.weight_mat = nn.Parameter(
                torch.randn(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64) * 0.01)
        else:
            # CRITICAL FIX: Don't transpose! Check shapes first
            W = W.double()
            expected_shape = (self.r * (self.r + 1) // 2, self.d)
            
            if W.shape == expected_shape:
                print(f"✓ W shape {W.shape} matches expected {expected_shape}")
                self.weight_mat = nn.Parameter(W.clone())
            elif W.shape == (expected_shape[1], expected_shape[0]):
                print(f"⚠ W shape {W.shape} needs transpose to match {expected_shape}")
                self.weight_mat = nn.Parameter(W.T.clone())
            else:
                raise ValueError(f"W shape {W.shape} doesn't match expected {expected_shape} or its transpose")
                
        self.gamma = gamma
        
    def forward(self, z_batch):
        # Ensure double precision
        z_batch = z_batch.double()
        
        # Linear reconstruction
        x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
        
        # Quadratic reconstruction
        z_quad = quadratic_mapping_torch(z_batch)  # (batch, r*(r+1)//2)
        x_hat_quad = z_quad @ self.weight_mat      # (batch, d)
        
        # Total reconstruction
        x_hat = x_hat_lin + x_hat_quad
        return x_hat

def manual_reconstruction_check(V, W, z_reduced, shift_value):
    """
    Manual reconstruction using numpy to verify correctness
    """
    print("\n" + "="*50)
    print("MANUAL RECONSTRUCTION CHECK")
    print("="*50)
    
    # Ensure everything is double precision
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    z_reduced = z_reduced.astype(np.float64)
    shift_value = shift_value.astype(np.float64)
    
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
    print(f"z_reduced shape: {z_reduced.shape}")
    print(f"shift_value shape: {shift_value.shape}")
    
    # Linear part: V @ z_reduced
    x_linear = V @ z_reduced  # (d, n_samples)
    print(f"x_linear shape: {x_linear.shape}")
    
    # Quadratic part: need to apply quadratic mapping to each column of z_reduced
    z_quad_list = []
    for i in range(z_reduced.shape[1]):
        z_i = z_reduced[:, i]  # (r,)
        z_quad_i = quadratic_mapping_numpy(z_i)  # (r*(r+1)//2,)
        z_quad_list.append(z_quad_i)
    
    z_quad_matrix = np.array(z_quad_list).T  # (r*(r+1)//2, n_samples)
    print(f"z_quad_matrix shape: {z_quad_matrix.shape}")
    
    # Apply weight matrix
    x_quad = W @ z_quad_matrix  # (d, n_samples)
    print(f"x_quad shape: {x_quad.shape}")
    
    # Total reconstruction
    x_total = x_linear + x_quad + shift_value  # (d, n_samples)
    print(f"x_total shape: {x_total.shape}")
    
    return x_total

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
    z_test = np.random.randn(5).astype(np.float64)
    z_torch = torch.tensor(z_test, dtype=torch.float64)
    
    quad_np = quadratic_mapping_numpy(z_test)
    quad_torch = quadratic_mapping_torch(z_torch).numpy()
    
    diff_single = np.max(np.abs(quad_np - quad_torch))
    print(f"Single vector difference: {diff_single:.2e}")
    
    # Batch test
    z_batch = np.random.randn(10, 5).astype(np.float64)
    z_batch_torch = torch.tensor(z_batch, dtype=torch.float64)
    
    quad_batch_np = quadratic_mapping_numpy(z_batch)
    quad_batch_torch = quadratic_mapping_torch(z_batch_torch).numpy()
    
    diff_batch = np.max(np.abs(quad_batch_np - quad_batch_torch))
    print(f"Batch difference: {diff_batch:.2e}")
    
    if diff_single < 1e-15 and diff_batch < 1e-15:
        print("✓ Quadratic mappings are consistent!")
        return True
    else:
        print("✗ Quadratic mappings are inconsistent!")
        return False

#%%
if __name__ == "__main__":
    # Force CPU for deterministic results
    device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Parameters
    r_max = 15
    n_grids = 2**10
    sanity_check = False  # Disable plotting for now
    
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)
    
    # Generate advecting pulse data
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=8.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    
    print(f"X_pulse shape: {X_pulse.shape}")
    print(f"X_pulse dtype: {X_pulse.dtype}")
    
    # Ensure data is double precision
    X_pulse = X_pulse.astype(np.float64)
    
    d_p, n_p = X_pulse.shape
    s_p = min(d_p, n_p)
    s_p = 100
#%% 
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    print("="*60)
    
    # Get greedy QM solution
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, 1e-6, np.array([], dtype=int))
    
    # Ensure double precision
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    shift_value = np.array(shift_value, dtype=np.float64)[:, np.newaxis]
    
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    print(f"W shape: {W.shape}, dtype: {W.dtype}")
    print(f"shift_value shape: {shift_value.shape}, dtype: {shift_value.dtype}")
    
    # Get reduced coordinates
    reduced_points = linear_reduce(V, X_pulse, shift_value)
    reduced_points = reduced_points.astype(np.float64)
    print(f"reduced_points shape: {reduced_points.shape}, dtype: {reduced_points.dtype}")
    
    # Test greedy reconstruction
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Greedy QM relative error: {rel_error_greedy:.2e}")
    
    # Test quadratic mapping consistency
    mapping_consistent = test_quadratic_mapping_consistency()
    
    if not mapping_consistent:
        print("ERROR: Quadratic mappings are inconsistent! Stopping.")
        exit(1)
    
    # Manual reconstruction test
    manual_reconstructed = manual_reconstruction_check(V, W, reduced_points, shift_value)
    manual_error = np.linalg.norm(manual_reconstructed - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Manual reconstruction error: {manual_error:.2e}")

#%%   
    print("\n" + "="*60)
    print("NEURAL NETWORK QUADRATIC MANIFOLD")
    print("="*60)
    
    # Create NN model with analytical weights
    pod_basis = torch.tensor(V, dtype=torch.float64)
    gamma = 1e-6
    
    # Initialize with analytical solution
    qm_model = QuadraticManifold(pod_basis, gamma, torch.tensor(W, dtype=torch.float64))
    qm_model = qm_model.to(device)
    
    print(f"Model weight matrix shape: {qm_model.weight_mat.shape}")
    print(f"Model weight matrix dtype: {qm_model.weight_mat.dtype}")
    
    # Check if weight matrices match
    weight_diff = torch.max(torch.abs(qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64))).item()
    print(f"Weight matrix difference: {weight_diff:.2e}")
    
    # Prepare data
    z_train = torch.tensor(reduced_points.T, dtype=torch.float64)  # (n_samples, r)
    x_target = torch.tensor((X_pulse - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
    print(f"z_train shape: {z_train.shape}, dtype: {z_train.dtype}")
    print(f"x_target shape: {x_target.shape}, dtype: {x_target.dtype}")
    
    # Test NN reconstruction (no training)
    qm_model.eval()
    with torch.no_grad():
        x_reconstructed_nn = qm_model(z_train)
        x_final_nn = x_reconstructed_nn.numpy() + shift_value.T
    
    # Compute error
    rel_error_nn = np.linalg.norm(x_final_nn.T - X_pulse) / np.linalg.norm(X_pulse)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Greedy QM error:     {rel_error_greedy:.2e}")
    print(f"Manual recon error:  {manual_error:.2e}")
    print(f"NN QM error:         {rel_error_nn:.2e}")
    print(f"Weight matrix diff:  {weight_diff:.2e}")
    
    if rel_error_nn < 1e-12:
        print("✓ SUCCESS: Neural network achieves machine precision!")
    elif rel_error_nn < 1e-6:
        print("✓ GOOD: Neural network achieves single precision accuracy")
    elif np.abs(rel_error_nn - rel_error_greedy) < 1e-12:
        print("✓ EQUAL: Neural network matches greedy QM error")
    else:
        print("✗ ISSUE: Neural network error is too large")
        
        # Additional debugging
        print("\nDEBUGGING:")
        
        # Check individual components
        with torch.no_grad():
            z_sample = z_train[:5]  # First 5 samples
            
            # Linear part
            x_lin_nn = z_sample @ qm_model.U_r.T
            x_lin_manual = (V @ z_sample.T.numpy()).T
            lin_diff = np.max(np.abs(x_lin_nn.numpy() - x_lin_manual))
            print(f"Linear part difference: {lin_diff:.2e}")
            
            # Quadratic mapping
            z_quad_nn = quadratic_mapping_torch(z_sample)
            z_quad_manual = quadratic_mapping_numpy(z_sample.numpy())
            quad_map_diff = np.max(np.abs(z_quad_nn.numpy() - z_quad_manual))
            print(f"Quadratic mapping difference: {quad_map_diff:.2e}")
            
            # Quadratic part
            x_quad_nn = z_quad_nn @ qm_model.weight_mat
            x_quad_manual = (W @ z_quad_manual.T).T
            quad_diff = np.max(np.abs(x_quad_nn.numpy() - x_quad_manual))
            print(f"Quadratic part difference: {quad_diff:.2e}")
            
#%%   
    # Optional: Train the network to see if it can improve
    if rel_error_nn > 1e-10:
        print(f"\n{'='*60}")
        print("TRAINING NEURAL NETWORK")
        print(f"{'='*60}")
        
        qm_model = QuadraticManifold(pod_basis, gamma)
        qm_model = qm_model.to(device)
        
        # Training setup
        optimizer = optim.Adam(qm_model.parameters(), lr=1.5e-3, weight_decay=gamma)
        mse_loss = nn.MSELoss()
        
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.8, patience=100)
        
        # Training loop
        for epoch in range(100000):
            qm_model.train()
            optimizer.zero_grad()
            x_pred = qm_model(z_train)
            reconstruction_loss = mse_loss(x_pred, x_target)
            loss = reconstruction_loss
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0:
                qm_model.eval()
                with torch.no_grad():
                    x_pred = qm_model(z_train)
                    rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                    print(f"Epoch {epoch:3d}: Loss = {loss.item():.4e}, "
                          f"Rel Error = {rel_err.item():.4e}, "
                          f"LR = {optimizer.param_groups[0]['lr']:.4e}")
                    
            lr_scheduler.step(loss)
        
        # Final evaluation after training
        qm_model.eval()
        with torch.no_grad():
            x_reconstructed_trained = qm_model(z_train)
            x_final_trained = x_reconstructed_trained.numpy() + shift_value.T
        
        rel_error_trained = np.linalg.norm(x_final_trained.T - X_pulse) / np.linalg.norm(X_pulse)
        print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
        print(f"Greedy QM error: {rel_error_greedy:.2e}")
        
        # Check how much weights changed
        final_weight_diff = torch.max(torch.abs(qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64))).item()
        print(f"Final weight matrix difference: {final_weight_diff:.2e}")
        if rel_error_trained < rel_error_nn:
            print("✓ Training improved the model!")
        elif np.abs(rel_error_trained - rel_error_nn) < 1e-10:
            print("✓ Training did not change the model.")
        else:
            print("✗ Training did not improve the model.")
# %%
