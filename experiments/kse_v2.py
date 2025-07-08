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
            # Random initialization
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
    
    

#%%
if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    # if torch.cuda.is_available():
    #     device = 'cuda'
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    # else:
    #     device = 'cpu'
    device = 'cpu'
    print("Using device:", device)
    
    # Number of spatial grids
    n_grids = 2**10  

    # Reduced dimension 
    r_max = 20
    
    # Sanity check flag (plotting)
    sanity_check = True

    # ---------- Kuramoto–Sivashinsky Equation ----------
    # Note: smaller nt for speed, adjust as desired
    X_ks, xspan_ks, tspan_ks = generate_kse_data(
        nx=n_grids, nt=5000, L=32*np.pi, t_max=150.0)
    d_ks, n_ks = X_ks.shape
    s_ks = 200
    
    # Create flow-field for Kuramoto-Sivashinsky Equation
    if sanity_check:
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(
            X_ks, aspect='auto', cmap='viridis', origin='lower',
            extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
        ax.set_xlabel('Time')
        ax.set_ylabel('Space (x)')
        ax.set_title('Kuramoto-Sivashinsky Equation Solution')
        plt.colorbar(im, ax=ax, label='u(x,t)')
        plt.tight_layout()
        plt.savefig('../figures/kse_data.png', dpi=300)
        plt.show()
        plt.close(fig)


#%% #========================== Preprocess the Data ===========================#
    # def apply_minmax_normalization(data, data_min=None, data_max=None):
    #     """Apply min-max normalization to scale data to [0,1]"""
    #     if isinstance(data, torch.Tensor):
    #         # PyTorch version
    #         if data_min is None:
    #             data_min = torch.min(data, dim=0, keepdim=True)[0]
    #         if data_max is None:
    #             data_max = torch.max(data, dim=0, keepdim=True)[0]
            
    #         data_range = data_max - data_min
    #         data_range = torch.where(
    #             data_range < 1e-8, torch.ones_like(data_range), data_range)
            
    #         normalized_data = (data - data_min) / data_range
    #         return normalized_data, data_min, data_max, data_range
    #     else:
    #         # NumPy version
    #         if data_min is None:
    #             data_min = np.min(data, axis=0, keepdims=True)
    #         if data_max is None:
    #             data_max = np.max(data, axis=0, keepdims=True)
            
    #         data_range = data_max - data_min
    #         data_range = np.where(
    #             data_range < 1e-8, np.ones_like(data_range), data_range)
            
    #         normalized_data = (data - data_min) / data_range
    #         return normalized_data, data_min, data_max, data_range

    # def denormalize_data(normalized_data, data_min, data_range):
    #     """Denormalize data back to original scale"""
    #     return normalized_data * data_range + data_min
    
    # X_ks_norm, x_min_alt, x_max_alt, x_range_alt = apply_minmax_normalization(X_ks) 
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    print("="*60)
    
    # Get greedy QM solution
    from QM.quadmani import quadmani_greedy, lift_quadratic, linear_reduce
    
    gamma = 1e-6  # Regularization parameter
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X_ks, r_max, s_ks, gamma, np.array([], dtype=int))
    
    # Ensure double precision
    V = V.astype(np.float64)
    W = W.astype(np.float64)
    shift_value = np.array(shift_value, dtype=np.float64)[:, np.newaxis]
    
    print(f"V shape: {V.shape}, dtype: {V.dtype}")
    print(f"W shape: {W.shape}, dtype: {W.dtype}")
    print(f"shift_value shape: {shift_value.shape}, dtype: {shift_value.dtype}")
    
    # Get reduced coordinates
    reduced_points = linear_reduce(V, X_ks, shift_value)
    reduced_points = reduced_points.astype(np.float64)
    print(f"reduced_points shape: {reduced_points.shape}, dtype: {reduced_points.dtype}")
    
    # Test greedy reconstruction
    reconstructed_greedy = lift_quadratic(V, W, shift_value, reduced_points)
    # reconstructed_greedy = denormalize_data(
    #     reconstructed_greedy, x_min_alt, x_range_alt)
    rel_error_greedy = np.linalg.norm(reconstructed_greedy - X_ks) / np.linalg.norm(X_ks)
    print(f"Greedy QM relative error: {rel_error_greedy:.2e}")
    linear_reconstruction = V @ reduced_points + shift_value
    rel_error_linear = np.linalg.norm(linear_reconstruction - X_ks) / np.linalg.norm(X_ks)
    print(f"Linear reconstruction relative error: {rel_error_linear:.2e}")


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
    x_target = torch.tensor((X_ks - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
    print(f"z_train shape: {z_train.shape}, dtype: {z_train.dtype}")
    print(f"x_target shape: {x_target.shape}, dtype: {x_target.dtype}")
    
    # Test NN reconstruction (no training)
    qm_model_no_train.eval()
    with torch.no_grad():
        x_reconstructed_nn = qm_model_no_train(z_train)
        x_final_nn = x_reconstructed_nn.numpy() + shift_value.T
    
    # Compute error
    rel_error_nn = np.linalg.norm(x_final_nn.T - X_ks) / np.linalg.norm(X_ks)
    
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"Greedy QM error:     {rel_error_greedy:.2e}")
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
    
    def apply_minmax_normalization(data, data_min=None, data_max=None):
        """Apply min-max normalization to scale data to [0,1]"""
        if data_min is None:
            data_min = torch.min(data, dim=0, keepdim=True)[0]
        if data_max is None:
            data_max = torch.max(data, dim=0, keepdim=True)[0]
        
        data_range = data_max - data_min
        data_range = torch.where(data_range < 1e-8, torch.ones_like(data_range), data_range)
        
        normalized_data = (data - data_min) / data_range
        return normalized_data, data_min, data_max, data_range

    def denormalize_data(normalized_data, data_min, data_max, data_range):
        """Denormalize data back to original scale"""
        return normalized_data * data_range + data_min

    # Normalize the data
    X_ks_tensor = torch.tensor(X_ks, dtype=torch.float64)
    x_target, x_min_alt, x_max_alt, x_range_alt = apply_minmax_normalization(X_ks_tensor.T)
    # X_ks_tensor = torch.tensor(X_ks, dtype=torch.float64)
    
    # pod_basis = torch.svd(x_target.T).U[:, :r_max]
    # z_train = x_target @ pod_basis
    
    # Create NN model with analytical weights
    pod_basis = torch.tensor(V, dtype=torch.float64)
    
    # Prepare data
    z_train = torch.tensor(reduced_points.T, dtype=torch.float64)  # (n_samples, r)
    x_target = torch.tensor((X_ks - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
    # W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN
    
    qm_model = QuadraticManifold(
        pod_basis, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    optimizer = optim.SGD(qm_model.parameters(), lr=1.0e-2, 
            momentum=0.99, weight_decay=gamma)
    # optimizer = optim.AdamW(qm_model.parameters(), lr=1.0e-3, weight_decay=gamma)
    # optimizer = optim.Adamax(qm_model.parameters(), lr=1.0e-3, weight_decay=gamma)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)
    
    # Training loop
    for epoch in range(20000):
        optimizer.zero_grad()
        x_pred = qm_model(z_train)
        reconstruction_loss = mse_loss(x_pred, x_target)
        loss = reconstruction_loss # + gamma * torch.norm(qm_model.weight_mat)**2
        # torch.nn.utils.clip_grad_norm_(qm_model.parameters(), max_norm=100.0)
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
        x_reconstructed_trained = qm_model(z_train)
        x_final_trained = x_reconstructed_trained.numpy() + shift_value.T
        # x_final_trained = denormalize_data(
        #     x_reconstructed_trained, x_min_alt, x_max_alt, x_range_alt
        # )
    
    rel_error_trained = np.linalg.norm(x_final_trained.T - X_ks) / np.linalg.norm(X_ks)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")
    print(f"Improvement rate: {rel_error_greedy / rel_error_trained:.2f}")
    
    #%% Check how much weights changed
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
        
        
# # %%
#     from torchmin import Minimizer
    
#     # Optional: Train the network to see if it can improve
#     print(f"\n{'='*60}")
#     print("TRAINING NEURAL NETWORK")
#     print(f"{'='*60}")
    
#     # Create NN model with analytical weights
#     pod_basis = torch.tensor(V, dtype=torch.float64)
    
#     # Prepare data
#     z_train = torch.tensor(reduced_points.T, dtype=torch.float64)  # (n_samples, r)
#     x_target = torch.tensor((X_ks - shift_value).T, dtype=torch.float64)  # (n_samples, d)
    
#     W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN
    
#     qm_model = QuadraticManifold(
#         pod_basis, gamma)
#     qm_model = qm_model.to(device)
    
#     # Training setup
#     mse_loss = nn.MSELoss()
#     qm_model.train()
    
#     # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#     #     optimizer, mode='min', factor=0.8, patience=100)
    
#     optimizer = Minimizer(qm_model.parameters(),
#                           method='newton-cg',
#                           tol=1e-6,
#                           max_iter=200,
#                           disp=2)

#     def closure():
#         optimizer.zero_grad()
#         x_pred = qm_model(z_train)
#         reconstruction_loss = mse_loss(x_pred, x_target)
#         loss = reconstruction_loss + gamma * torch.norm(qm_model.weight_mat)**2
#         return loss
    
#     loss = optimizer.step(closure)
    
    
#     # # Training loop
#     # for epoch in range(1000):
#     #     loss = optimizer.step(closure)
    
#     #     if epoch % 10 == 0:
#     #         with torch.no_grad():
#     #             rel_err = torch.norm(x_pred - x_target) / torch.norm(x_target)
#     #             print(f"Epoch {epoch:3d}: "
#     #                 f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
#     #                 f"Loss = {loss.item():.4e}, "
#     #                 f"Rel Error = {rel_err.item():.4e}")
                
#         # lr_scheduler.step(loss)
    
#     # Final evaluation after training
#     qm_model.eval()
#     with torch.no_grad():
#         x_reconstructed_trained = qm_model(z_train)
#         x_final_trained = x_reconstructed_trained.numpy() + shift_value.T
    
#     rel_error_trained = np.linalg.norm(x_final_trained.T - X_ks) / np.linalg.norm(X_ks)
#     print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
#     print(f"Greedy QM error: {rel_error_greedy:.2e}")
    
#     # Check how much weights changed
#     final_weight_diff = torch.max(
#         torch.abs(
#             qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64)
#         )).item()
#     print(f"Final weight matrix difference: {final_weight_diff:.2e}")
#     if rel_error_trained < rel_error_greedy:
#         print("✓ Training improved the model!")
#     elif np.abs(rel_error_trained - rel_error_greedy) < 1e-10:
#         print("✓ Training did not change the model.")
#     else:
#         print("✗ Training did worse then the QM model.")
# # %%

# %%
