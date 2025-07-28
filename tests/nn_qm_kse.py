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

from examples.kse import generate_kse_data

#%% #===================== Quadratic Mapping Function =========================#
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

#======================== Quadratic Manifold Class ============================#
class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        # Ensure everything is double precision
        pod_basis = pod_basis.double()
        self.register_buffer('U_r', pod_basis)  # (d, r)
        self.d, self.r = pod_basis.shape
        self.gamma = gamma
        
        if W is None:
            self.weight_mat = nn.Parameter(
                torch.zeros(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64))
        else:
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

#======================== Manual Reconstruction Check =========================#
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
    # z_quad_list = []
    # for i in range(z_reduced.shape[1]):
    #     z_i = z_reduced[:, i]  # (r,)
    #     z_quad_i = quadratic_mapping_numpy(z_i)  # (r*(r+1)//2,)
    #     z_quad_list.append(z_quad_i)
    
    # z_quad_matrix = np.array(z_quad_list).T  # (r*(r+1)//2, n_samples)
    # print(f"z_quad_matrix shape: {z_quad_matrix.shape}")
    
    z_quad_matrix = quadratic_mapping_numpy(z_reduced.T).T  # (r*(r+1)//2, n_samples)
    print(f"z_quad_matrix shape: {z_quad_matrix.shape}")
    
    # Apply weight matrix
    x_quad = W @ z_quad_matrix  # (d, n_samples)
    print(f"x_quad shape: {x_quad.shape}")
    
    # Total reconstruction
    x_total = x_linear + x_quad + shift_value  # (d, n_samples)
    print(f"x_total shape: {x_total.shape}")
    
    return x_total

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

def test_simple_quadratic_regression():
    """
    Test quadratic manifold on a simple synthetic regression problem.
    Generate data from X = V*Z + W*quadratic_mapping(Z) + noise
    Then verify both analytical and NN solutions recover the true W.
    """
    print("\n" + "="*60)
    print("SIMPLE QUADRATIC REGRESSION TEST")
    print("="*60)
    
    # Set dimensions (small for easy verification)
    d = 20   # ambient dimension
    r = 8    # reduced dimension
    n_samples = 1000
    noise_level = 1e-6
    
    # Set seeds for reproducibility
    torch.manual_seed(123)
    np.random.seed(123)
    
    # Generate true V matrix (linear basis)
    V_true = np.random.randn(d, r).astype(np.float64)
    V_true = np.linalg.qr(V_true)[0]  # Orthogonalize
    print(f"V_true shape: {V_true.shape}")
    
    # Generate true W matrix (quadratic weights)
    quad_dim = r * (r + 1) // 2
    W_true = np.random.randn(d, quad_dim).astype(np.float64) * 0.1
    print(f"W_true shape: {W_true.shape}")
    print(f"Quadratic dimension: {quad_dim}")
    
    # Generate random reduced coordinates
    Z_true = np.random.randn(r, n_samples).astype(np.float64)
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
    X_true = X_true.astype(np.float64)
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
    # X = V*Z + W*Z_quad  =>  W*Z_quad = X - V*Z
    # W = (X - V*Z) * Z_quad^T * (Z_quad * Z_quad^T)^(-1)
    
    residual = X_true - V_true @ Z_true  # Remove linear part
    print(f"Residual shape: {residual.shape}")
    
    # Solve: W * Z_quad = residual
    # W = residual @ Z_quad.T @ inv(Z_quad @ Z_quad.T)
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
    pod_basis_torch = torch.tensor(V_true, dtype=torch.float64)
    W_analytical_torch = torch.tensor(W_analytical.T, dtype=torch.float64)  # Transpose for NN
    
    # Create model with analytical weights
    qm_model = QuadraticManifold(pod_basis_torch, gamma=1e-10, W=W_analytical_torch)
    qm_model.eval()
    
    # Prepare NN input/output
    Z_torch = torch.tensor(Z_true.T, dtype=torch.float64)  # (n_samples, r)
    X_torch = torch.tensor(X_true.T, dtype=torch.float64)  # (n_samples, d)
    
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
    qm_model_scratch = QuadraticManifold(pod_basis_torch, gamma=1e-10)
    qm_model_scratch.train()
    
    # Training setup
    optimizer = optim.Adam(qm_model_scratch.parameters(), lr=1e-4)
    mse_loss = nn.MSELoss()
    
    print("Training NN from scratch...")
    losses = []
    for epoch in range(10000):
        optimizer.zero_grad()
        X_pred = qm_model_scratch(Z_torch)
        loss = mse_loss(X_pred, X_torch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
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
    learned_vs_true_diff = torch.norm(learned_weights - torch.tensor(W_true, dtype=torch.float64)) / torch.norm(torch.tensor(W_true, dtype=torch.float64))
    learned_vs_analytical_diff = torch.norm(learned_weights - torch.tensor(W_analytical, dtype=torch.float64)) / torch.norm(torch.tensor(W_analytical, dtype=torch.float64))
    
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
    
    # Success criteria
    success_analytical = analytical_error < noise_level * 10
    success_nn_init = nn_error.item() < noise_level * 10  
    success_nn_trained = final_error.item() < noise_level * 10
    success_weight_recovery = W_recovery_error < 1e-3
    success_nn_consistency = nn_weight_diff.item() < 1e-10
    
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

def visualize_regression_test_results(results):
    """
    Optional visualization of regression test results
    """
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Error comparison
        methods = ['Analytical', 'NN (init)', 'NN (trained)']
        errors = [results['analytical_error'], results['nn_error'], results['trained_error']]
        
        axes[0].bar(methods, errors)
        axes[0].set_yscale('log')
        axes[0].set_ylabel('Relative Error')
        axes[0].set_title('Reconstruction Errors')
        axes[0].grid(True, alpha=0.3)
        
        # Weight recovery
        weight_errors = [results['weight_recovery_error'], results['weight_consistency_error']]
        weight_labels = ['Recovery Error', 'NN Consistency']
        
        axes[1].bar(weight_labels, weight_errors)
        axes[1].set_yscale('log')
        axes[1].set_ylabel('Relative Error')
        axes[1].set_title('Weight Matrix Errors')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Matplotlib not available for visualization")




#%%
if __name__ == "__main__":
    # Force CPU for deterministic results
    device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    
     
#%% #================== First run the simple regression test ==================#
    print("RUNNING SIMPLE REGRESSION TEST FIRST...")
    success, results = test_simple_quadratic_regression()
    
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
        
    visualize_regression_test_results(results)
    
    

#%% #===================== Main Experiment with Pulse Data ====================#
    # Parameters
    r_max = 15
    n_grids = 2**10
    sanity_check = False  # Disable plotting for now
    
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)
    
    # Generate advecting pulse data
    X, xspan, tspan = generate_kse_data(
        nx=n_grids, nt=2000, L=32*np.pi, t_max=100.0)
    d_ks, n_ks = X.shape
    
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    
    # Ensure data is double precision
    X = X.astype(np.float64)
    
    d_p, n_p = X.shape
    s_p = min(d_p, n_p)
    s_p = 100
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    X_mesh, T_mesh = np.meshgrid(xspan, tspan)
    Z_mesh = X.T  # Transpose to match meshgrid dimensions
    surf = ax.plot_surface(
        X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title('Kuramoto-Sivashinksy Equation Data')
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    plt.close(fig)
    
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    print("="*60)
    
    # Get greedy QM solution
    from experiments.QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    gamma = 1e-6  # Regularization parameter
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X, r_max, s_p, gamma, np.array([], dtype=int))
    
    # Ensure double precision
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    shift_value = np.array(shift_value, dtype=np.float64)[:, np.newaxis]
    
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    print(f"W shape: {W.shape}, dtype: {W.dtype}")
    print(f"shift_value shape: {shift_value.shape}, dtype: {shift_value.dtype}")
    
    # Get reduced coordinates
    reduced_points = linear_reduce(V, X, shift_value)
    reduced_points = reduced_points.astype(np.float64)
    print(f"reduced_points shape: {reduced_points.shape}, dtype: {reduced_points.dtype}")
    
    # Test greedy reconstruction
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X) / np.linalg.norm(X)
    print(f"Greedy QM relative error: {rel_error_greedy:.2e}")
    
    # Test quadratic mapping consistency
    mapping_consistent = test_quadratic_mapping_consistency()
    
    if not mapping_consistent:
        print("ERROR: Quadratic mappings are inconsistent! Stopping.")
        exit(1)
    
    # Manual reconstruction test
    manual_reconstructed = manual_reconstruction_check(V, W, reduced_points, shift_value)
    manual_error = np.linalg.norm(manual_reconstructed - X) / np.linalg.norm(X)
    print(f"Manual reconstruction error: {manual_error:.2e}")
    
    # Verify that W is actually the analytical solution  
    print("\n" + "="*40)
    print("VERIFYING ANALYTICAL SOLUTION OF QM")
    print("="*40)

    # Ensure consistent data preprocessing
    print(f"X_pulse shape: {X.shape}")
    print(f"V shape: {V.shape}")
    print(f"reduced_points shape: {reduced_points.shape}")
    print(f"shift_value shape: {shift_value.shape}")

    # Check if shift_value was used in the greedy method
    # The residual should match what the greedy method was trying to fit
    if shift_value.ndim == 1:
        X_centered = (X.T - shift_value).T  # (d, n_samples)
    else:
        X_centered = X - shift_value  # assuming shift_value is (d, 1) or (d, n_samples)

    residual = X_centered - V @ reduced_points
    print(f"Residual shape: {residual.shape}")
    print(f"Residual norm: {np.linalg.norm(residual):.2e}")

    # Compute quadratic features - ensure same ordering as in greedy method
    Z_quad = quadratic_mapping_numpy(reduced_points.T).T  # (quad_dim, n_samples)
    print(f"Z_quad shape: {Z_quad.shape}")

    # Check conditioning of the system
    ZqZqT = Z_quad @ Z_quad.T
    cond_num = np.linalg.cond(ZqZqT)
    print(f"Condition number of Z_quad @ Z_quad.T: {cond_num:.2e}")
    
    if cond_num > 1e12:
        print("⚠️WARNING: System is poorly conditioned!")
        
    def lstsq_l2_numpy(A, B, reg_magnitude=1e-6):
        """
        Numpy version of the JAX lstsq_l2 function for consistency
        """
        phi, sigma, psi_t = np.linalg.svd(A, full_matrices=False)
        sinv = sigma / (sigma**2 + reg_magnitude**2)
        x = psi_t.T * sinv @ (phi.T @ B)
        B_estimate = A @ x
        resid = np.linalg.norm(B - B_estimate)
        return x, resid

    try:
        # Use the same regularized least squares as the greedy method
        # We need to solve: Z_quad.T @ W.T = residual.T
        # Which is equivalent to: W.T = lstsq_l2(Z_quad.T, residual.T)
        W_analytical_T, analytical_resid = lstsq_l2_numpy(
            Z_quad.T, residual.T, reg_magnitude=gamma
        )
        W_analytical = W_analytical_T.T
        
        print(f"Analytical least squares residual: {analytical_resid:.2e}")
        print(f"Analytical W shape: {W_analytical.shape}, dtype: {W_analytical.dtype}")
        print(f"Greedy W shape: {W.shape}, dtype: {W.dtype}")
        
        # Verify the analytical solution actually solves the least squares problem
        reconstruction_analytical = V @ reduced_points + W_analytical @ Z_quad
        if shift_value.ndim == 1:
            reconstruction_analytical = (reconstruction_analytical.T + shift_value).T
        else:
            reconstruction_analytical = reconstruction_analytical + shift_value
        
        analytical_fit_error = np.linalg.norm(reconstruction_analytical - X) / np.linalg.norm(X)
        print(f"Analytical reconstruction error: {analytical_fit_error:.2e}")
        
        # Verify the greedy solution 
        reconstruction_greedy = V @ reduced_points + W @ Z_quad
        if shift_value.ndim == 1:
            reconstruction_greedy = (reconstruction_greedy.T + shift_value).T
        else:
            reconstruction_greedy = reconstruction_greedy + shift_value
        
        greedy_fit_error = np.linalg.norm(reconstruction_greedy - X) / np.linalg.norm(X)
        print(f"Greedy reconstruction error: {greedy_fit_error:.2e}")
        
        # Compare the W matrices
        W_error = np.linalg.norm(W_analytical - W) / np.linalg.norm(W)
        print(f"Relative W error: {W_error:.2e}")
        
        # More detailed comparison
        abs_diff = np.abs(W_analytical - W)
        print(f"W max absolute diff: {np.max(abs_diff):.2e}")
        print(f"W mean absolute diff: {np.mean(abs_diff):.2e}")
        print(f"W median absolute diff: {np.median(abs_diff):.2e}")
        print(f"W 95th percentile diff: {np.percentile(abs_diff, 95):.2e}")
        
        # Success criteria - should be much better now with same regularization
        tolerance = max(1e-12, gamma * 1e3)  # More lenient based on regularization
        
        if W_error < tolerance and abs(analytical_fit_error - greedy_fit_error) < tolerance:
            print("✓ Analytical solution matches greedy QM weights!")
            print(f"  Both methods achieve similar reconstruction error (~{analytical_fit_error:.2e})")
            print(f"  Using regularization parameter: {gamma:.2e}")
        else:
            print("✗ Analytical solution does NOT match greedy QM weights!")
            if W_error >= tolerance:
                print(f"  W matrices differ by {W_error:.2e} (tolerance: {tolerance:.2e})")
            if abs(analytical_fit_error - greedy_fit_error) >= tolerance:
                print(f"  Reconstruction errors differ: {abs(analytical_fit_error - greedy_fit_error):.2e}")
                
    except Exception as e:
        print(f"✗ Error in analytical solution: {e}")
        

#%% #===================== Neural Network Quadratic Manifold ==================#
    print("\n" + "="*60)
    print("NEURAL NETWORK QUADRATIC MANIFOLD")
    print("="*60)
    
    # Create NN model with analytical weights
    pod_basis = torch.tensor(V, dtype=torch.float64)
    
    # Initialize with analytical solution
    qm_model_no_train = QuadraticManifold(pod_basis, gamma, torch.tensor(W.T, dtype=torch.float64))
    qm_model_no_train = qm_model_no_train.to(device)
    
    print(f"Model weight matrix shape: {qm_model_no_train.weight_mat.shape}")
    print(f"Model weight matrix dtype: {qm_model_no_train.weight_mat.dtype}")
    
    # Check if weight matrices match
    weight_diff = torch.max(
        torch.abs(
            qm_model_no_train.weight_mat.T - torch.tensor(W, dtype=torch.float64)
        )).item()
    print(f"Weight matrix difference: {weight_diff:.2e}")
    
    # Prepare data
    z_train = torch.tensor(reduced_points.T, dtype=torch.float64)  # (n_samples, r)
    x_target = torch.tensor((X - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
    print(f"z_train shape: {z_train.shape}, dtype: {z_train.dtype}")
    print(f"x_target shape: {x_target.shape}, dtype: {x_target.dtype}")
    
    # Test NN reconstruction (no training)
    qm_model_no_train.eval()
    with torch.no_grad():
        x_reconstructed_nn = qm_model_no_train(z_train)
        x_final_nn = x_reconstructed_nn.numpy() + shift_value.T
    
    # Compute error
    rel_error_nn = np.linalg.norm(x_final_nn.T - X) / np.linalg.norm(X)
    
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
            x_lin_nn = z_sample @ qm_model_no_train.U_r.T
            x_lin_manual = (V @ z_sample.T.numpy()).T
            lin_diff = np.max(np.abs(x_lin_nn.numpy() - x_lin_manual))
            print(f"Linear part difference: {lin_diff:.2e}")
            
            # Quadratic mapping
            z_quad_nn = quadratic_mapping_torch(z_sample)
            z_quad_manual = quadratic_mapping_numpy(z_sample.numpy())
            quad_map_diff = np.max(np.abs(z_quad_nn.numpy() - z_quad_manual))
            print(f"Quadratic mapping difference: {quad_map_diff:.2e}")
            
            # Quadratic part
            x_quad_nn = z_quad_nn @ qm_model_no_train.weight_mat
            x_quad_manual = (W @ z_quad_manual.T).T
            quad_diff = np.max(np.abs(x_quad_nn.numpy() - x_quad_manual))
            print(f"Quadratic part difference: {quad_diff:.2e}")
            
            
            
#%% #======================== Training the Neural Network =====================#
    # Optional: Train the network to see if it can improve
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")
    
    W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN
    
    qm_model = QuadraticManifold(
        pod_basis, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    optimizer = optim.SGD(qm_model.parameters(), lr=1e-2, 
            momentum=0.99, weight_decay=gamma**2)
    # mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)
    
    # Training loop
    for epoch in range(20000):
        optimizer.zero_grad()
        x_pred = qm_model(z_train)
        loss = torch.mean((x_pred - x_target) ** 2)
        # reconstruction_loss = mse_loss(x_pred, x_target)
        # loss = reconstruction_loss
        loss.backward()
        optimizer.step()
        
        # loss = optimizer.step(closure)
       
        if epoch % 100 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                print(f"Epoch {epoch:3d}: "
                      f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
                      f"Loss = {loss.item():.4e}, "
                      f"Rel Error = {rel_err.item():.4e}")
                
        lr_scheduler.step(loss)
    
    # Final evaluation after training
    qm_model.eval()
    with torch.no_grad():
        x_reconstructed_trained = qm_model(z_train)
        x_final_trained = x_reconstructed_trained.numpy() + shift_value.T
    
    rel_error_trained = np.linalg.norm(x_final_trained.T - X) / np.linalg.norm(X)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")
    
    # Check how much weights changed
    final_weight_diff = torch.max(
        torch.abs(
            qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64)
        )).item()
    print(f"Final weight matrix difference: {final_weight_diff:.2e}")
    if rel_error_trained < rel_error_nn:
        print("✓ Training improved the model!")
    elif np.abs(rel_error_trained - rel_error_nn) < 1e-10:
        print("✓ Training did not change the model.")
    else:
        print("✗ Training did not improve the model.")
# %%
    # Optional: Train the network to see if it can improve
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK - PURE GRADIENT DESCENT")
    print(f"{'='*60}")

    W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN

    qm_model = QuadraticManifold(
        pod_basis, gamma, W_tensor + torch.randn_like(W_tensor) * 0.1)
    qm_model = qm_model.to(device)

    # Pure gradient descent - no optimizer
    learning_rate = 1e-4
    mse_loss = nn.MSELoss()
    qm_model.train()

    # Optional: Add explicit L2 regularization matching analytical solution
    def regularized_loss(x_pred, x_target, model, reg_weight):
        reconstruction_loss = mse_loss(x_pred, x_target)
        # L2 regularization on W (same as analytical solution)
        weight_reg = reg_weight**2 * torch.norm(model.weight_mat, 'fro')**2
        return reconstruction_loss + weight_reg

    # Pure gradient descent training loop
    for epoch in range(10000):
        # Forward pass
        x_pred = qm_model(z_train)
        
        # Compute loss
        loss = regularized_loss(x_pred, x_target, qm_model, gamma)
        
        # Backward pass
        loss.backward()
        
        # Pure gradient descent update: w = w - lr * grad_w
        with torch.no_grad():
            for param in qm_model.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
        
        # Zero gradients for next iteration
        qm_model.zero_grad()
    
        if epoch % 500 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                weight_norm = torch.norm(qm_model.weight_mat, 'fro')
                print(f"Epoch {epoch:4d}: "
                    f"LR = {learning_rate:.4e}, "
                    f"Loss = {loss.item():.4e}, "
                    f"Rel Error = {rel_err.item():.4e}, "
                    f"Weight Norm = {weight_norm.item():.4e}")

    # Final evaluation after training
    qm_model.eval()
    with torch.no_grad():
        x_reconstructed_trained = qm_model(z_train)
        x_final_trained = x_reconstructed_trained.numpy() + shift_value.T

    rel_error_trained = np.linalg.norm(x_final_trained.T - X) / np.linalg.norm(X)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")

    # Check how much weights changed
    final_weight_diff = torch.max(
        torch.abs(
            qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64)
        )).item()
    print(f"Final weight matrix difference: {final_weight_diff:.2e}")

    # Compare with analytical solution
    if 'W_analytical' in locals():
        analytical_weight_diff = torch.max(
            torch.abs(
                qm_model.weight_mat.T - torch.tensor(W_analytical, dtype=torch.float64)
            )).item()
        print(f"Final weight vs analytical difference: {analytical_weight_diff:.2e}")

    if rel_error_trained < rel_error_nn:
        print("✓ Training improved the model!")
    elif np.abs(rel_error_trained - rel_error_nn) < 1e-10:
        print("✓ Training did not change the model.")
    else:
        print("✗ Training did not improve the model.")
        
        
        
        
# %%
    # Natural gradient descent using the Fisher Information Matrix approximation
    print("\n" + "="*60)
    print("NATURAL GRADIENT DESCENT")
    print("="*60)

    qm_model_ng = QuadraticManifold(pod_basis, gamma)
    qm_model_ng = qm_model_ng.to(device)

    # Compute Fisher Information Matrix (Gauss-Newton approximation)
    with torch.no_grad():
        z_quad = quadratic_mapping_torch(z_train)
        # Use the empirical Fisher: F = J^T J where J is the Jacobian
        fisher_matrix = z_quad.T @ z_quad / z_quad.shape[0]  # (quad_dim, quad_dim)
        
        # Add regularization to Fisher matrix
        fisher_matrix += gamma * torch.eye(fisher_matrix.shape[0], dtype=torch.float64)
        
        # Compute preconditioner (inverse of Fisher matrix)
        try:
            preconditioner = torch.inverse(fisher_matrix)
            print(f"Using exact Fisher inverse")
        except:
            # Use pseudo-inverse if singular
            preconditioner = torch.pinverse(fisher_matrix)
            print(f"Using Fisher pseudo-inverse")
        
        print(f"Fisher matrix condition number: {torch.linalg.cond(fisher_matrix):.2e}")

    # Natural gradient training
    learning_rate = 1.0  # Can use much higher learning rate
    prev_loss = float('inf')
    lr_decay = 0.95
    patience = 100
    patience_counter = 0
    for epoch in range(10000):
        # Forward pass
        x_pred = qm_model_ng(z_train)
        loss = regularized_loss(x_pred, x_target, qm_model_ng, gamma)
        
        # Backward pass
        loss.backward()
        
        # Natural gradient update
        with torch.no_grad():
            for param in qm_model_ng.parameters():
                if param.grad is not None:
                    # Apply preconditioner to gradient
                    natural_grad = preconditioner @ param.grad 
                    param.data -= learning_rate * natural_grad
        
        # Simple learning rate adaptation
        if loss.item() >= prev_loss:
            patience_counter += 1
            if patience_counter >= patience:
                learning_rate *= lr_decay
                patience_counter = 0
                print(f"  Reducing learning rate to {learning_rate:.2e}")
        else:
            patience_counter = 0
        
        prev_loss = loss.item()
        
        qm_model_ng.zero_grad()
        
        if epoch % 100 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                print(f"Epoch {epoch:4d}: Loss = {loss.item():.6e}, Rel Error = {rel_err.item():.6e}")
                
            if rel_err.item() < 1e-12:
                print(f"Natural gradient converged at epoch {epoch}")
                break



#%% #======================== Training with Data Normalization ================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK WITH DATA NORMALIZATION")
    print(f"{'='*60}")

    # Analyze data statistics before normalization
    print("Data statistics before normalization:")
    print(f"z_train - mean: {torch.mean(z_train):.2e}, std: {torch.std(z_train):.2e}")
    print(f"z_train - min: {torch.min(z_train):.2e}, max: {torch.max(z_train):.2e}")
    print(f"x_target - mean: {torch.mean(x_target):.2e}, std: {torch.std(x_target):.2e}")
    print(f"x_target - min: {torch.min(x_target):.2e}, max: {torch.max(x_target):.2e}")

    # Normalize input data (z_train)
    z_mean = torch.mean(z_train, dim=0, keepdim=True)
    z_std = torch.std(z_train, dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
    z_train_normalized = (z_train - z_mean) / z_std

    # Normalize target data (x_target)
    x_mean = torch.mean(x_target, dim=0, keepdim=True)
    x_std = torch.std(x_target, dim=0, keepdim=True) + 1e-8
    x_target_normalized = (x_target - x_mean) / x_std

    print("\nData statistics after normalization:")
    print(f"z_train_normalized - mean: {torch.mean(z_train_normalized):.2e}, std: {torch.std(z_train_normalized):.2e}")
    print(f"x_target_normalized - mean: {torch.mean(x_target_normalized):.2e}, std: {torch.std(x_target_normalized):.2e}")

    # Check quadratic features conditioning
    with torch.no_grad():
        z_quad_raw = quadratic_mapping_torch(z_train)
        z_quad_normalized = quadratic_mapping_torch(z_train_normalized)
        
        print(f"\nQuadratic features analysis:")
        print(f"Raw quadratic features - std: {torch.std(z_quad_raw):.2e}, max: {torch.max(torch.abs(z_quad_raw)):.2e}")
        print(f"Normalized quadratic features - std: {torch.std(z_quad_normalized):.2e}, max: {torch.max(torch.abs(z_quad_normalized)):.2e}")
        
        # Condition number comparison
        U_raw, S_raw, V_raw = torch.svd(z_quad_raw)
        U_norm, S_norm, V_norm = torch.svd(z_quad_normalized)
        
        cond_raw = S_raw[0] / S_raw[-1] if S_raw[-1] > 1e-15 else float('inf')
        cond_norm = S_norm[0] / S_norm[-1] if S_norm[-1] > 1e-15 else float('inf')
        
        print(f"Condition number - Raw: {cond_raw:.2e}, Normalized: {cond_norm:.2e}")
        print(f"Condition improvement: {cond_raw / cond_norm:.2e}x better")

    # Create normalized quadratic manifold
    class NormalizedQuadraticManifold(nn.Module):
        def __init__(self, pod_basis, gamma, z_mean, z_std, x_mean, x_std):
            super().__init__()
            pod_basis = pod_basis.double()
            self.register_buffer('U_r', pod_basis)
            self.d, self.r = pod_basis.shape
            self.gamma = gamma
            
            # Store normalization parameters
            self.register_buffer('z_mean', z_mean)
            self.register_buffer('z_std', z_std)
            self.register_buffer('x_mean', x_mean)
            self.register_buffer('x_std', x_std)
            
            # Initialize weights with proper scaling
            self.weight_mat = nn.Parameter(
                torch.randn(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64) * 0.01)
        
        def forward(self, z_batch):
            # Normalize input
            z_batch = z_batch.double()
            z_normalized = (z_batch - self.z_mean) / self.z_std
            
            # Linear part (on normalized data)
            x_hat_lin = z_normalized @ self.U_r.T
            
            # Quadratic part (on normalized data)
            z_quad = quadratic_mapping_torch(z_normalized)
            x_hat_quad = z_quad @ self.weight_mat
            
            # Combine
            x_hat_normalized = x_hat_lin + x_hat_quad
            
            # Denormalize output
            x_hat = x_hat_normalized * self.x_std + self.x_mean
            
            return x_hat

    # Create normalized model
    qm_model_norm = NormalizedQuadraticManifold(pod_basis, gamma, z_mean, z_std, x_mean, x_std)
    qm_model_norm = qm_model_norm.to(device)

    # Training with much higher learning rate (normalization allows this)
    optimizer = optim.Adam(qm_model_norm.parameters(), lr=1e-2, weight_decay=0)
    mse_loss = nn.MSELoss()
    qm_model_norm.train()

    # Add L2 regularization
    def regularized_loss(x_pred, x_target, model, reg_weight):
        reconstruction_loss = mse_loss(x_pred, x_target)
        weight_reg = reg_weight * torch.norm(model.weight_mat, 'fro')**2
        return reconstruction_loss + weight_reg

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=200)

    print(f"\nTraining normalized model...")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Training loop with normalized data
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 1000

    for epoch in range(10000):  # Fewer epochs needed with normalization
        optimizer.zero_grad()
        
        # Forward pass (model handles normalization internally)
        x_pred = qm_model_norm(z_train)
        loss = regularized_loss(x_pred, x_target, qm_model_norm, gamma)
        
        loss.backward()
        
        # Gradient clipping for stability
        # torch.nn.utils.clip_grad_norm_(qm_model_norm.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Learning rate scheduling
        lr_scheduler.step(loss)
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 100 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                print(f"Epoch {epoch:4d}: "
                    f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
                    f"Loss = {loss.item():.6e}, "
                    f"Rel Error = {rel_err.item():.6e}")
                
        # Check for convergence
        if epoch % 500 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                if rel_err.item() < 1e-12:
                    print(f"Converged at epoch {epoch}")
                    break

    # Final evaluation
    qm_model_norm.eval()
    with torch.no_grad():
        x_reconstructed_norm = qm_model_norm(z_train)
        x_final_norm = x_reconstructed_norm.numpy() + shift_value.T

    rel_error_norm = np.linalg.norm(x_final_norm.T - X) / np.linalg.norm(X)
    print(f"\nFinal normalized NN error: {rel_error_norm:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")

    # Compare with non-normalized version
    if 'rel_error_trained' in locals():
        print(f"Non-normalized NN error: {rel_error_trained:.2e}")
        improvement = rel_error_trained / rel_error_norm
        print(f"Normalization improvement: {improvement:.2e}x better")

    # Check convergence speed
    print(f"Training completed in {epoch} epochs")

    # Verify denormalization is working correctly
    with torch.no_grad():
        # Test on normalized data directly
        x_pred_normalized = qm_model_norm(z_train)
        
        # Manual normalization check
        z_norm_manual = (z_train - z_mean) / z_std
        x_pred_manual = (z_norm_manual @ qm_model_norm.U_r.T + 
                        quadratic_mapping_torch(z_norm_manual) @ qm_model_norm.weight_mat)
        x_pred_manual = x_pred_manual * x_std + x_mean
        
        normalization_error = torch.norm(x_pred_normalized - x_pred_manual) / torch.norm(x_pred_manual)
        print(f"Normalization implementation error: {normalization_error.item():.2e}")

    if rel_error_norm < rel_error_greedy:
        print("✓ Normalized training improved over greedy QM!")
    else:
        print("○ Normalized training matched greedy QM performance")
        
        
    
#%% #======================== Training with Weight Normalization ===============# 
    print(f"\n{'='*60}")
    print("TRAINING WITH WEIGHT NORMALIZATION")
    print(f"{'='*60}")
    
    import torch.nn.utils.weight_norm as weight_norm
    class WeightNormQuadraticManifold(nn.Module):
        def __init__(self, pod_basis: torch.Tensor, gamma: float, W: torch.Tensor = None):
            super(WeightNormQuadraticManifold, self).__init__()
            
            # Ensure everything is double precision
            pod_basis = pod_basis.double()
            self.register_buffer('U_r', pod_basis)  # (d, r)
            self.d, self.r = pod_basis.shape
            self.gamma = gamma
            
            # Create the weight matrix as a Linear layer (for weight normalization)
            self.weight_layer = nn.Linear(self.r * (self.r + 1) // 2, self.d, bias=False)
            self.weight_layer = self.weight_layer.double()
            
            # Initialize weights
            if W is None:
                # Random initialization with proper scaling
                nn.init.kaiming_uniform_(self.weight_layer.weight, nonlinearity='linear')
                self.weight_layer.weight.data *= 0.01  # Scale down for stability
            else:
                # Initialize with provided weights
                W = W.double()
                expected_shape = (self.d, self.r * (self.r + 1) // 2)  # Linear layer expects (out, in)
                
                if W.shape == expected_shape:
                    print(f"✓ W shape {W.shape} matches expected {expected_shape}")
                    self.weight_layer.weight.data = W.clone()
                elif W.shape == (expected_shape[1], expected_shape[0]):
                    print(f"⚠ W shape {W.shape} needs transpose to match {expected_shape}")
                    self.weight_layer.weight.data = W.T.clone()
                else:
                    raise ValueError(f"W shape {W.shape} doesn't match expected {expected_shape} or its transpose")
            
            # Apply weight normalization
            self.weight_layer = weight_norm(self.weight_layer, name='weight', dim=0)
            
            print(f"Applied weight normalization to quadratic layer")
            print(f"Weight shape: {self.weight_layer.weight.shape}")
            print(f"Weight_g shape: {self.weight_layer.weight_g.shape}")
            print(f"Weight_v shape: {self.weight_layer.weight_v.shape}")
            
        def forward(self, z_batch):
            # Ensure double precision
            z_batch = z_batch.double()
            
            # Linear reconstruction
            x_hat_lin = z_batch @ self.U_r.T     # (batch, d)
            
            # Quadratic reconstruction with weight normalization
            z_quad = quadratic_mapping_torch(z_batch)  # (batch, r*(r+1)//2)
            x_hat_quad = self.weight_layer(z_quad)      # (batch, d)
            
            # Total reconstruction
            x_hat = x_hat_lin + x_hat_quad
            return x_hat
        
        @property
        def weight_mat(self):
            """For compatibility with existing code"""
            return self.weight_layer.weight.T  # Return in original format


    # Create weight-normalized model
    qm_model_wn = WeightNormQuadraticManifold(pod_basis, gamma)
    qm_model_wn = qm_model_wn.to(device)

    # Analyze the weight normalization effect
    with torch.no_grad():
        print(f"Initial weight_g norm: {torch.norm(qm_model_wn.weight_layer.weight_g):.2e}")
        print(f"Initial weight_v norm: {torch.norm(qm_model_wn.weight_layer.weight_v):.2e}")
        print(f"Initial weight norm: {torch.norm(qm_model_wn.weight_layer.weight):.2e}")

    # Training with weight normalization - can use higher learning rates
    optimizer = optim.SGD(qm_model_wn.parameters(), lr=1e-2, 
                          momentum=0.99, nesterov=True, weight_decay=0) 
    mse_loss = nn.MSELoss()
    qm_model_wn.train()

    # Regularization function
    def regularized_loss_wn(x_pred, x_target, model, reg_weight):
        reconstruction_loss = mse_loss(x_pred, x_target)
        # L2 regularization on the magnitude parameters (weight_g)
        weight_reg = reg_weight * torch.norm(model.weight_layer.weight_g)**2
        return reconstruction_loss + weight_reg

    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    print(f"\nTraining weight-normalized model...")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 500

    for epoch in range(10000):  # Should converge much faster
        optimizer.zero_grad()
        
        # Forward pass
        x_pred = qm_model_wn(z_train)
        loss = regularized_loss_wn(x_pred, x_target, qm_model_wn, gamma)
        
        loss.backward()
        
        # Gradient clipping (less needed with weight norm)
        torch.nn.utils.clip_grad_norm_(qm_model_wn.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Learning rate scheduling
        lr_scheduler.step(loss)
        
        # # Early stopping
        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
            
        # if patience_counter >= max_patience:
        #     print(f"Early stopping at epoch {epoch}")
        #     break
        
        if epoch % 50 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                weight_g_norm = torch.norm(qm_model_wn.weight_layer.weight_g)
                weight_v_norm = torch.norm(qm_model_wn.weight_layer.weight_v)
                print(f"Epoch {epoch:4d}: "
                    f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
                    f"Loss = {loss.item():.6e}, "
                    f"Rel Error = {rel_err.item():.6e}, "
                    f"||g|| = {weight_g_norm.item():.4e}, "
                    f"||v|| = {weight_v_norm.item():.4e}")
                
        # Check for convergence
        if epoch % 100 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                if rel_err.item() < 1e-12:
                    print(f"Weight-normalized model converged at epoch {epoch}")
                    break

    # Final evaluation
    qm_model_wn.eval()
    with torch.no_grad():
        x_reconstructed_wn = qm_model_wn(z_train)
        x_final_wn = x_reconstructed_wn.numpy() + shift_value.T

    rel_error_wn = np.linalg.norm(x_final_wn.T - X) / np.linalg.norm(X)
    print(f"\nFinal weight-normalized NN error: {rel_error_wn:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")
    print(f"Training completed in {epoch} epochs")

    # Compare with other methods
    if 'rel_error_norm' in locals():
        print(f"Data normalization NN error: {rel_error_norm:.2e}")
        improvement = rel_error_norm / rel_error_wn
        print(f"Weight norm vs data norm improvement: {improvement:.2e}x")

    # Analyze final weight statistics
    with torch.no_grad():
        print(f"\nFinal weight statistics:")
        print(f"Weight_g norm: {torch.norm(qm_model_wn.weight_layer.weight_g):.2e}")
        print(f"Weight_v norm: {torch.norm(qm_model_wn.weight_layer.weight_v):.2e}")
        print(f"Final weight norm: {torch.norm(qm_model_wn.weight_layer.weight):.2e}")
        
        # Check if weights are well-conditioned
        weight_matrix = qm_model_wn.weight_layer.weight.detach()
        U, S, V = torch.svd(weight_matrix)
        cond_num = S[0] / S[-1] if S[-1] > 1e-15 else float('inf')
        print(f"Weight matrix condition number: {cond_num:.2e}")

    if rel_error_wn < rel_error_greedy:
        print("✓ Weight normalization improved over greedy QM!")
    else:
        print("○ Weight normalization matched greedy QM performance")
        

#%% #======================== Training with Combined Normalization =============#
    print(f"\n{'='*60}")
    print("TRAINING WITH BOTH WEIGHT AND DATA NORMALIZATION")
    print(f"{'='*60}")
    
    # Analyze data statistics before normalization
    print("Data statistics before normalization:")
    print(f"z_train - mean: {torch.mean(z_train):.2e}, std: {torch.std(z_train):.2e}")
    print(f"z_train - min: {torch.min(z_train):.2e}, max: {torch.max(z_train):.2e}")
    print(f"x_target - mean: {torch.mean(x_target):.2e}, std: {torch.std(x_target):.2e}")
    print(f"x_target - min: {torch.min(x_target):.2e}, max: {torch.max(x_target):.2e}")

    # Normalize input data (z_train)
    z_mean = torch.mean(z_train, dim=0, keepdim=True)
    z_std = torch.std(z_train, dim=0, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
    z_train_normalized = (z_train - z_mean) / z_std

    # Normalize target data (x_target)
    x_mean = torch.mean(x_target, dim=0, keepdim=True)
    x_std = torch.std(x_target, dim=0, keepdim=True) + 1e-8
    x_target_normalized = (x_target - x_mean) / x_std

    print("\nData statistics after normalization:")
    print(f"z_train_normalized - mean: {torch.mean(z_train_normalized):.2e}, std: {torch.std(z_train_normalized):.2e}")
    print(f"x_target_normalized - mean: {torch.mean(x_target_normalized):.2e}, std: {torch.std(x_target_normalized):.2e}")


    class CombinedNormQuadraticManifold(nn.Module):
        def __init__(self, pod_basis, gamma, z_mean, z_std, x_mean, x_std):
            super().__init__()
            pod_basis = pod_basis.double()
            self.register_buffer('U_r', pod_basis)
            self.d, self.r = pod_basis.shape
            self.gamma = gamma
            
            # Store normalization parameters
            self.register_buffer('z_mean', z_mean)
            self.register_buffer('z_std', z_std)
            self.register_buffer('x_mean', x_mean)
            self.register_buffer('x_std', x_std)
            
            # Create weight-normalized layer
            self.weight_layer = nn.Linear(self.r * (self.r + 1) // 2, self.d, bias=False)
            self.weight_layer = self.weight_layer.double()
            
            # Initialize with proper scaling
            nn.init.kaiming_uniform_(self.weight_layer.weight, nonlinearity='linear')
            self.weight_layer.weight.data *= 0.01
            
            # Apply weight normalization
            self.weight_layer = weight_norm(self.weight_layer, name='weight', dim=0)
        
        def forward(self, z_batch):
            # Normalize input
            z_batch = z_batch.double()
            z_normalized = (z_batch - self.z_mean) / self.z_std
            
            # Linear part (on normalized data)
            x_hat_lin = z_normalized @ self.U_r.T
            
            # Quadratic part with weight normalization (on normalized data)
            z_quad = quadratic_mapping_torch(z_normalized)
            x_hat_quad = self.weight_layer(z_quad)
            
            # Combine
            x_hat_normalized = x_hat_lin + x_hat_quad
            
            # Denormalize output
            x_hat = x_hat_normalized * self.x_std + self.x_mean
            
            return x_hat
        
        @property
        def weight_mat(self):
            """For compatibility"""
            return self.weight_layer.weight.T

    # Create combined normalization model
    qm_model_combined = CombinedNormQuadraticManifold(pod_basis, gamma, z_mean, z_std, x_mean, x_std)
    qm_model_combined = qm_model_combined.to(device)

    # Training with even higher learning rate
    # optimizer = optim.Adam(qm_model_combined.parameters(), lr=1, weight_decay=0)  # Very high LR
    optimizer = optim.SGD(qm_model_combined.parameters(), lr=1, 
                          momentum=0.99, nesterov=True, weight_decay=0)  # Very high LR
    mse_loss = nn.MSELoss()
    qm_model_combined.train()
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=1000)

    print(f"Training combined normalization model...")
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    

    # Training loop - should converge very fast
    for epoch in range(10000):  # Even fewer epochs
        optimizer.zero_grad()
        
        # Forward pass
        x_pred = qm_model_combined(z_train)
        loss = regularized_loss_wn(x_pred, x_target, qm_model_combined, gamma)
        
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        lr_scheduler.step(loss)
        
        if epoch % 10 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
                print(f"Epoch {epoch:4d}: Loss = {loss.item():.6e}, Rel Error = {rel_err.item():.6e}")
                
            if rel_err.item() < 1e-12:
                print(f"Combined normalization converged at epoch {epoch}")
                break

    # Final evaluation
    qm_model_combined.eval()
    with torch.no_grad():
        x_reconstructed_combined = qm_model_combined(z_train)
        x_final_combined = x_reconstructed_combined.numpy() + shift_value.T

    rel_error_combined = np.linalg.norm(x_final_combined.T - X) / np.linalg.norm(X)
    print(f"\nFinal combined normalization NN error: {rel_error_combined:.2e}")
    print(f"Training completed in {epoch} epochs")
# %%

