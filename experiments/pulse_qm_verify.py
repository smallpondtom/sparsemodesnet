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
    

#%% #==================== Quadratic Manifold Class ============================#
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
                torch.rand(self.r * (self.r + 1) // 2, self.d, dtype=torch.float64) * 0.01)
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



#%% #======================= Co-Moving Frame Transform ========================#
class CoMovingFrameTransform:
    """
    Complete co-moving frame transformation with forward and inverse operations.
    """
    
    def __init__(self, x_span, t_span, velocity=None):
        """
        Initialize the transformation.
        
        Args:
            x_span: Spatial grid points
            t_span: Temporal grid points
        """
        self.x_span = x_span
        self.t_span = t_span
        self.dx = x_span[1] - x_span[0]
        self.L = x_span[-1] - x_span[0] + self.dx  # Domain length with periodic BC
        self.velocity = velocity
        self.fitted = False
    
    def detect_velocity(self, X_lab):
        """
        Detect advection velocity by tracking center of mass.
        
        Args:
            X_lab: Laboratory frame data (spatial_points, time_points)
            
        Returns:
            velocity: Detected advection velocity
        """
        centers = []
        
        for i in range(X_lab.shape[1]):
            snapshot = X_lab[:, i]
            if np.sum(np.abs(snapshot)) > 1e-10:
                # Center of mass calculation
                center_of_mass = np.sum(
                    self.x_span * np.abs(snapshot)) / np.sum(np.abs(snapshot))
                centers.append(center_of_mass)
            else:
                centers.append(0.0)
        
        centers = np.array(centers)
        
        # Fit linear trend: center(t) = center_0 + velocity * t
        if len(centers) > 1:
            self.velocity = np.polyfit(self.t_span, centers, 1)[0]
            print(f"Detected advection velocity: {self.velocity:.6f}")
        else:
            self.velocity = 0.0
            print("Warning: Could not detect velocity, using 0.0")
        
        self.fitted = True
        return self.velocity
    
    def transform_to_comoving(self, X_lab):
        """
        Transform from laboratory frame to co-moving frame.
        
        Args:
            X_lab: Laboratory frame data (spatial_points, time_points)
            
        Returns:
            X_comoving: Co-moving frame data
        """
        if not self.fitted and self.velocity is None:
            self.detect_velocity(X_lab)
        else:
            self.fitted = True
        
        nx, nt = X_lab.shape
        X_comoving = np.zeros_like(X_lab)
        
        print("Transforming to co-moving frame...")
        for i, time in enumerate(self.t_span):
            # How far has the pulse moved?
            displacement = self.velocity * time
            
            # Convert displacement to grid points
            shift_points = int(np.round(displacement / self.dx))
            
            # Apply circular shift (negative because we move coordinate 
            # system WITH the pulse)
            X_comoving[:, i] = np.roll(X_lab[:, i], -shift_points)
        
        return X_comoving
    
    def transform_to_laboratory(self, X_comoving):
        """
        Transform from co-moving frame back to laboratory frame.
        
        Args:
            X_comoving: Co-moving frame data (spatial_points, time_points)
            
        Returns:
            X_lab: Laboratory frame data
        """
        if not self.fitted:
            raise ValueError("Must fit velocity before inverse transformation")
        
        nx, nt = X_comoving.shape
        X_lab = np.zeros_like(X_comoving)
        
        print("Transforming back to laboratory frame...")
        for i, time in enumerate(self.t_span):
            # How far has the pulse moved?
            displacement = self.velocity * time
            
            # Convert displacement to grid points
            shift_points = int(np.round(displacement / self.dx))
            
            # Apply circular shift (positive because we undo the 
            # co-moving transformation)
            X_lab[:, i] = np.roll(X_comoving[:, i], +shift_points)
        
        return X_lab

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
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
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


#%% #=================== Preprocess the Data for Training =====================#
    print("\n" + "="*60)
    print("PREPROCESSING DATA FOR TRAINING")
    print("="*60)
    
    # Initialize co-moving frame transformer 
    data_transformer =  CoMovingFrameTransform(xspan_p, tspan_p, 5.0)
    
    # Apply co-moving transformation
    X_comoving = data_transformer.transform_to_comoving(X_pulse)
    
    # Recompute QM on co-moving frame 
    V_comoving, W_comoving, shift_comoving, I_qm = quadmani_greedy(
        X_comoving, r_max, s_p, gamma, np.array([], dtype=int))
    
    # Get reduced coordinates for co-moving data
    reduced_comoving = linear_reduce(V_comoving, X_comoving, shift_comoving)
    
    # Compute the reconstruction error for verification 
    foo = lift_quadratic(V_comoving, W_comoving, shift_comoving, reduced_comoving)
    bar = np.linalg.norm(foo - X_comoving) / np.linalg.norm(X_comoving)
    print(f"Greedy QM relative error (co-moving): {bar:.2e}")
    foo = data_transformer.transform_to_laboratory(foo)
    bar = np.linalg.norm(foo - X_pulse) / np.linalg.norm(X_pulse)
    print(f"Greedy QM relative error (laboratory): {bar:.2e}")
    
    # Get training data on co-moving frame
    shift_comoving = shift_comoving[:, np.newaxis]  
    z_train = torch.tensor(reduced_comoving.T, dtype=torch.float64)
    x_target = torch.tensor((X_comoving - shift_comoving).T, dtype=torch.float64)
    
    
    
#%% #======================== Training the Neural Network =====================#
    # Optional: Train the network to see if it can improve
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")
    
    # # Create NN model with analytical weights
    # pod_basis = torch.tensor(V, dtype=torch.float64)
    
    # # Prepare data
    # z_train = torch.tensor(reduced_points.T, dtype=torch.float64)  # (n_samples, r)
    # x_target = torch.tensor((X_pulse - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
    # W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN
    
    pod_basis_comoving = torch.tensor(V_comoving, dtype=torch.float64)
    
    qm_model = QuadraticManifold(
        pod_basis_comoving, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    optimizer = optim.AdamW(qm_model.parameters(), lr=1.0e-2, weight_decay=gamma)
    # optimizer = optim.SGD(qm_model.parameters(), lr=1.0e-2, 
    #         momentum=0.99, weight_decay=0)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)
    
    # Training loop
    for epoch in range(50000):
        optimizer.zero_grad()
        x_pred = qm_model(z_train)
        reconstruction_loss = mse_loss(x_pred, x_target)
        loss = reconstruction_loss # + gamma * torch.norm(qm_model.weight_mat)**2
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
        x_trained_comoving = qm_model(z_train)
        x_trained_comoving = x_trained_comoving.numpy() + shift_comoving.T
        x_final_trained = data_transformer.transform_to_laboratory(x_trained_comoving.T)
        # x_final_trained = x_reconstructed_trained.numpy() + shift_value.T
    
    # rel_error_trained = np.linalg.norm(x_final_trained.T - X_comoving) / np.linalg.norm(X_comoving)
    rel_error_trained = np.linalg.norm(x_final_trained - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")
    
    # Check how much weights changed
    final_weight_diff = torch.max(
        torch.abs(
            qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64)
        )).item()
    print(f"Final weight matrix difference: {final_weight_diff:.2e}")
    if rel_error_trained < rel_error_greedy:
        print("✓ Training improved the model!")
    elif np.abs(rel_error_trained - rel_error_greedy) < 1e-10:
        print("✓ Training did not change the model.")
    else:
        print("✗ Training did worse then the QM model.")
        