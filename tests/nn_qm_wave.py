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

#========================= Quadratic Mapping Function =========================#
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
            # Random initialization
            # self.weight_mat = nn.Parameter(
            #     torch.randn(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64) * 0.01)
            self.weight_mat = nn.Parameter(
                torch.zeros(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64))
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


def zca_whitening_matrix(X, epsilon=1e-5):
    """
    Function to compute ZCA whitening matrix (aka Mahalanobis whitening).
    INPUT:  X: [M x N] matrix.
        Rows: Variables
        Columns: Observations
    OUTPUT: ZCAMatrix: [M x M] matrix

    Reference:
    https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
    """
    # Covariance matrix [column-wise variables]: Sigma = (X-mu)' * (X-mu) / N
    sigma = np.cov(X, rowvar=True) # [M x M]
    # Singular Value Decomposition. X = U * np.diag(S) * V
    U,S,_ = np.linalg.svd(sigma)
        # U: [M x M] eigenvectors of sigma.
        # S: [M x 1] eigenvalues of sigma.
        # V: [M x M] transpose of U
    # Whitening constant: prevents division by zero
    # ZCA Whitening matrix: U * Lambda * U'
    ZCAMatrix = np.dot(U, np.dot(np.diag(1.0/np.sqrt(S + epsilon)), U.T)) # [M x M]
    return ZCAMatrix



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
    X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
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
    
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    print("="*60)
    
    # Get greedy QM solution
    from experiments.QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    gamma = 1e-6  # Regularization parameter
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, gamma, np.array([], dtype=int))
    
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
    
    # Verify that W is actually the analytical solution  
    print("\n" + "="*40)
    print("VERIFYING ANALYTICAL SOLUTION OF QM")
    print("="*40)

    # Ensure consistent data preprocessing
    print(f"X_pulse shape: {X_pulse.shape}")
    print(f"V shape: {V.shape}")
    print(f"reduced_points shape: {reduced_points.shape}")
    print(f"shift_value shape: {shift_value.shape}")

    # Check if shift_value was used in the greedy method
    # The residual should match what the greedy method was trying to fit
    if shift_value.ndim == 1:
        X_centered = (X_pulse.T - shift_value).T  # (d, n_samples)
    else:
        X_centered = X_pulse - shift_value  # assuming shift_value is (d, 1) or (d, n_samples)

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
        
        analytical_fit_error = np.linalg.norm(reconstruction_analytical - X_pulse) / np.linalg.norm(X_pulse)
        print(f"Analytical reconstruction error: {analytical_fit_error:.2e}")
        
        # Verify the greedy solution 
        reconstruction_greedy = V @ reduced_points + W @ Z_quad
        if shift_value.ndim == 1:
            reconstruction_greedy = (reconstruction_greedy.T + shift_value).T
        else:
            reconstruction_greedy = reconstruction_greedy + shift_value
        
        greedy_fit_error = np.linalg.norm(reconstruction_greedy - X_pulse) / np.linalg.norm(X_pulse)
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
    x_target = torch.tensor((X_pulse - shift_value).T, dtype=torch.float64)  # (n_samples, d)

    print(f"z_train shape: {z_train.shape}, dtype: {z_train.dtype}")
    print(f"x_target shape: {x_target.shape}, dtype: {x_target.dtype}")
    
    # Test NN reconstruction (no training)
    qm_model_no_train.eval()
    with torch.no_grad():
        x_reconstructed_nn = qm_model_no_train(z_train)
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
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN

    qm_model = QuadraticManifold(pod_basis, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    # optimizer = optim.SGD(qm_model.parameters(), lr=1e-4, 
    #         momentum=0.99, weight_decay=gamma**2)
    optimizer = optim.LBFGS(qm_model.parameters(), lr=1e-1)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)
    
    residual = x_target - (z_train @ pod_basis.T)
    
    # Training loop
    for epoch in range(100):
        # optimizer.zero_grad()
        # x_pred = qm_model(z_train)
        # reconstruction_loss = mse_loss(x_pred, x_target)
        # loss = reconstruction_loss
        # loss.backward()
        # optimizer.step()
        
        def closure():
            optimizer.zero_grad()
            x_pred = qm_model(z_train)
            reconstruction_loss = mse_loss(x_pred, x_target)
            loss = reconstruction_loss
            loss.backward()
            return loss

        x_pred = qm_model(z_train)
        loss = optimizer.step(closure)
       
        if epoch % 1 == 0:
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
        tmp = x_reconstructed_trained.numpy()
        x_final_trained = tmp + shift_value.T
    
    rel_error_trained = np.linalg.norm(x_final_trained.T - X_pulse) / np.linalg.norm(X_pulse)
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

#%% #============== Training the Neural Network with whitening ================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN

    X_shift = X_pulse - shift_value
    zcaMat = zca_whitening_matrix(X_shift, epsilon=1e-4)
    X_white = np.dot(zcaMat, X_shift)  # Apply ZCA whitening
    X_white_tensor = torch.tensor(X_white, dtype=torch.float64)  

    # Perform SVD decomposition
    I_white = np.array([0,1,14,16,18,19,20,21,23,30,31,32,34,48,49])
    V_white = torch.svd(X_white_tensor)[0][:, I_white]
    z_target = torch.matmul(x_target, V_white)
    
    qm_model = QuadraticManifold(V_white, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    optimizer = optim.SGD(qm_model.parameters(), lr=1e-4, 
            momentum=0.99, weight_decay=0.0)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Training loop
    for epoch in range(10000):
        optimizer.zero_grad()
        x_pred = qm_model(z_target)
        reconstruction_loss = mse_loss(x_pred, x_target)
        loss = reconstruction_loss
        loss.backward()
        optimizer.step()
        
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
        x_reconstructed_trained = qm_model(z_target)
        tmp = x_reconstructed_trained.numpy()
        x_final_trained = tmp + shift_value.T
    
    rel_error_trained = np.linalg.norm(x_final_trained.T - X_pulse) / np.linalg.norm(X_pulse)
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
    V_white_np = V_white.numpy()
    Z_nn = V_white_np.T @ X_shift
    residual = X_shift - V_white_np @ Z_nn
    Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
    W_nn_T, analytical_resid = lstsq_l2_numpy(
        Z_quad_nn.T, residual.T, reg_magnitude=1e-15
    )
    W_nn = W_nn_T.T
    recon_error = np.linalg.norm(
        X_pulse - V_white_np @ Z_nn - W_nn @ Z_quad_nn - shift_value, ord='fro')
    rel_recon_error_nn = recon_error / np.linalg.norm(X_pulse, ord='fro') 
    print(f"LassoNet: ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_error:.6e}")
    print(f"Relative error: {rel_recon_error_nn:.6e}")

#%%
    residual_tensor = torch.tensor(residual, dtype=torch.float64)
    Z_quad_nn_tensor = torch.tensor(Z_quad_nn, dtype=torch.float64)
    W_nn_tensor = torch.linalg.solve(
        Z_quad_nn_tensor @ Z_quad_nn_tensor.T + 1e-18 * torch.eye(Z_quad_nn_tensor.shape[0], dtype=torch.float64),
        Z_quad_nn_tensor @ residual_tensor.T
    ).T
    W_nn_tensor_np = W_nn_tensor.numpy()
    recon_error = np.linalg.norm(
        X_pulse - V_white @ Z_nn - W_nn_tensor_np @ Z_quad_nn - shift_value, ord='fro')
    rel_recon_error_nn_tensor = recon_error / np.linalg.norm(X_pulse, ord='fro')
    print(f"LassoNet (tensor): ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_error:.6e}")
    print(f"Relative error (tensor): {rel_recon_error_nn_tensor:.6e}")
# %%
