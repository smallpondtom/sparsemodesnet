import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse

def quadratic_mapping(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    Only computes upper triangular part to avoid redundancy.
    """
    if x.dim() == 1:
        n = x.size(0)
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        batch_size, n = x.shape
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        result = x[:, i_indices] * x[:, j_indices]
        return result   

def quadratic_mapping_numpy(x):
    """
    Numpy version of quadratic mapping for comparison
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
    def __init__(self, pod_basis: torch.Tensor, use_double_precision=True):
        super(QuadraticManifold, self).__init__()
        
        # CRITICAL: Use double precision if requested
        if use_double_precision:
            pod_basis = pod_basis.double()
        
        self.register_buffer('U_r', pod_basis)
        self.d, self.r = pod_basis.shape
        self.use_double_precision = use_double_precision
        
        # Initialize weight matrix
        if use_double_precision:
            self.weight_mat = nn.Parameter(
                torch.zeros(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64))
        else:
            self.weight_mat = nn.Parameter(
                torch.zeros(self.r * (self.r + 1) // 2, self.d, dtype=torch.float32))
        
    def forward(self, z_batch):
        # Ensure input has correct precision
        if self.use_double_precision and z_batch.dtype != torch.float64:
            z_batch = z_batch.double()
        elif not self.use_double_precision and z_batch.dtype != torch.float32:
            z_batch = z_batch.float()
            
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
        # Apply the quadratic mapping
        z_quad = quadratic_mapping(z_batch)  # (batch, r*(r+1)//2)
        x_hat_nn = z_quad @ self.weight_mat.T  # (batch, d)
        # Reconstruct x_hat
        x_hat = x_hat_lin + x_hat_nn
        return x_hat

def manual_quadratic_reconstruction(V, W, z_reduced, shift_value):
    """
    Manual implementation of quadratic manifold reconstruction for comparison
    """
    # Linear part
    x_linear = V @ z_reduced  # (d, n_samples)
    
    # Quadratic part
    z_quad = quadratic_mapping_numpy(z_reduced.T)  # (n_samples, r*(r+1)//2)
    x_quad = W @ z_quad.T  # (d, n_samples)
    
    # Total reconstruction
    x_reconstructed = x_linear + x_quad + shift_value
    return x_reconstructed

def compare_weight_matrices(W_analytical, model_weight_mat, tolerance=1e-10):
    """Compare analytical and neural network weight matrices"""
    print("\n" + "="*60)
    print("WEIGHT MATRIX COMPARISON")
    print("="*60)
    
    # Convert to numpy for comparison
    if isinstance(model_weight_mat, torch.Tensor):
        W_nn = model_weight_mat.detach().cpu().numpy()
    else:
        W_nn = model_weight_mat
    
    print(f"Analytical W shape: {W_analytical.shape}")
    print(f"Neural Net W shape: {W_nn.shape}")
    
    # Check if shapes match
    if W_analytical.shape != W_nn.shape:
        print("ERROR: Shape mismatch!")
        print(f"Expected: {W_analytical.shape}, Got: {W_nn.shape}")
        
        # Check if transpose is needed
        if W_analytical.shape == W_nn.T.shape:
            print("Trying transpose...")
            W_nn = W_nn.T
            print(f"After transpose: {W_nn.shape}")
    
    # Compute differences
    diff = np.abs(W_analytical - W_nn)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    rel_diff = max_diff / (np.max(np.abs(W_analytical)) + 1e-16)
    
    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print(f"Max relative difference: {rel_diff:.2e}")
    
    if max_diff < tolerance:
        print("✓ Weight matrices are IDENTICAL (within tolerance)")
        return True
    else:
        print("✗ Weight matrices are DIFFERENT")
        
        # Show some sample differences
        print("\nSample of differences (first 5x5):")
        print(diff[:5, :5])
        
        print(f"\nAnalytical W (first 5x5):")
        print(W_analytical[:5, :5])
        
        print(f"\nNeural Net W (first 5x5):")
        print(W_nn[:5, :5])
        
        return False

def diagnostic_reconstruction_test():
    """
    Comprehensive test to ensure exact equivalence between methods
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC TEST - EXACT EQUIVALENCE CHECK")
    print("="*80)
    
    # Use device CPU for deterministic results
    device = 'cpu'
    
    # Generate simple test data
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test parameters
    r_max = 10  # Smaller for easier debugging
    n_grids = 2**8  # Smaller for easier debugging
    
    # Generate advecting pulse data
    from examples.pulse import generate_advecting_pulse
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=8.0,
        final_time=0.15,
        n_time_samples=200,  # Smaller for debugging
        n_space_samples=n_grids
    )
    
    print(f"Original data shape: {X_pulse.shape}")
    
    # Get greedy QM solution
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    s_p = 50  # Smaller for debugging
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, 1e-6, np.array([], dtype=int))
    
    print(f"V shape: {V.shape}")
    print(f"W shape: {W.shape}")
    print(f"shift_value shape: {np.array(shift_value).shape}")
    
    # Get reduced coordinates
    reduced_points = linear_reduce(V, X_pulse, shift_value)
    print(f"Reduced points shape: {reduced_points.shape}")
    
    # Test 1: Verify greedy reconstruction
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nGreedy QM relative error: {rel_error_greedy:.2e}")
    
    # Test 2: Manual reconstruction verification
    shift_array = np.array(shift_value)[:, np.newaxis]
    manual_reconstruction = manual_quadratic_reconstruction(V, W, reduced_points, shift_array)
    manual_error = np.linalg.norm(manual_reconstruction - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Manual reconstruction error: {manual_error:.2e}")
    
    # Test 3: Neural network with double precision
    print(f"\n{'='*40}")
    print("TESTING NEURAL NETWORK (DOUBLE PRECISION)")
    print(f"{'='*40}")
    
    # Create model with double precision
    pod_basis_double = torch.tensor(V, dtype=torch.float64)
    qm_model_double = QuadraticManifold(pod_basis_double, use_double_precision=True)
    
    # Set weight matrix to analytical solution
    W_torch = torch.tensor(W, dtype=torch.float64)
    qm_model_double.weight_mat.data = W_torch
    
    print("Set NN weight matrix to analytical solution")
    
    # Compare weight matrices
    weights_match = compare_weight_matrices(W, qm_model_double.weight_mat)
    
    # Test reconstruction
    z_train_double = torch.tensor(reduced_points.T, dtype=torch.float64)
    x_target_double = torch.tensor((X_pulse - shift_array).T, dtype=torch.float64)
    
    qm_model_double.eval()
    with torch.no_grad():
        x_reconstructed_nn = qm_model_double(z_train_double)
        x_reconstructed_final = x_reconstructed_nn.numpy() + shift_array.T
    
    # Compute error
    rel_error_nn_double = np.linalg.norm(x_reconstructed_final.T - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nNN (double precision) relative error: {rel_error_nn_double:.2e}")
    
    # Test 4: Neural network with single precision
    print(f"\n{'='*40}")
    print("TESTING NEURAL NETWORK (SINGLE PRECISION)")
    print(f"{'='*40}")
    
    pod_basis_single = torch.tensor(V, dtype=torch.float32)
    qm_model_single = QuadraticManifold(pod_basis_single, use_double_precision=False)
    
    # Set weight matrix to analytical solution
    W_torch_single = torch.tensor(W, dtype=torch.float32)
    qm_model_single.weight_mat.data = W_torch_single
    
    # Compare weight matrices
    weights_match_single = compare_weight_matrices(W, qm_model_single.weight_mat)
    
    # Test reconstruction
    z_train_single = torch.tensor(reduced_points.T, dtype=torch.float32)
    
    qm_model_single.eval()
    with torch.no_grad():
        x_reconstructed_nn_single = qm_model_single(z_train_single)
        x_reconstructed_final_single = x_reconstructed_nn_single.numpy() + shift_array.T
    
    rel_error_nn_single = np.linalg.norm(x_reconstructed_final_single.T - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nNN (single precision) relative error: {rel_error_nn_single:.2e}")
    
    # Test 5: Check quadratic mapping equivalence
    print(f"\n{'='*40}")
    print("TESTING QUADRATIC MAPPING")
    print(f"{'='*40}")
    
    # Test on a small sample
    z_sample = reduced_points[:, :5]  # First 5 samples
    z_sample_torch = torch.tensor(z_sample.T, dtype=torch.float64)
    
    # Numpy version
    z_quad_numpy = quadratic_mapping_numpy(z_sample.T)
    
    # Torch version
    z_quad_torch = quadratic_mapping(z_sample_torch).numpy()
    
    quad_mapping_diff = np.max(np.abs(z_quad_numpy - z_quad_torch))
    print(f"Quadratic mapping difference: {quad_mapping_diff:.2e}")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Greedy QM error:           {rel_error_greedy:.2e}")
    print(f"Manual reconstruction:     {manual_error:.2e}")
    print(f"NN (double precision):     {rel_error_nn_double:.2e}")
    print(f"NN (single precision):     {rel_error_nn_single:.2e}")
    print(f"Weight matrices match (double): {weights_match}")
    print(f"Weight matrices match (single): {weights_match_single}")
    print(f"Quadratic mapping difference: {quad_mapping_diff:.2e}")
    
    # Expected outcome: All errors should be essentially identical (< 1e-10 for double, < 1e-6 for single)
    if rel_error_nn_double < 1e-10 and weights_match:
        print("✓ SUCCESS: Neural network achieves exact equivalence!")
    else:
        print("✗ ISSUE: Neural network does not match analytical solution")
        
        # Additional debugging
        print("\nDEBUGGING INFO:")
        print(f"Data types - V: {V.dtype}, W: {W.dtype}")
        print(f"PyTorch model dtype: {qm_model_double.U_r.dtype}")
        print(f"Weight matrix dtype: {qm_model_double.weight_mat.dtype}")

if __name__ == "__main__":
    diagnostic_reconstruction_test()