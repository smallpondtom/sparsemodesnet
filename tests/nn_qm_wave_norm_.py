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
    def __init__(self, d, r):
        super(QuadraticManifold, self).__init__()
        self.W = nn.Parameter(
            torch.zeros(d, r * (r + 1) // 2,
            dtype=torch.float64), requires_grad=True)
                
    def forward(self, z_quad):
        return self.W @ z_quad


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


#%%
if __name__ == "__main__":
    # Force CPU for deterministic results
    device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
     
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
    
    gamma = 1e-3  # Regularization parameter
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X_pulse, r_max, s_p, gamma, np.array([], dtype=int), shifting=True)
    
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

#%% #======================== Training the Neural Network =====================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    # Compute the residual and quadratic data 
    Z_quad = quadratic_mapping_numpy(reduced_points.T).T  
    Z_quad_tensor = torch.tensor(Z_quad, dtype=torch.float64)  
    Res = X_pulse - V @ reduced_points - shift_value
    Res_tensor = torch.tensor(Res, dtype=torch.float64)  

    # Create dataset and dataloader for batch training
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(Z_quad_tensor.T, Res_tensor.T)
    batch_size = 1  # Adjust based on your data size and memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    qm_model = QuadraticManifold(V.shape[0], V.shape[1])
    qm_model = qm_model.to(device)
    
    # Training setup
    # optimizer = optim.SGD(qm_model.parameters(), lr=1e-8, 
    #         momentum=0.99)
    optimizer = optim.LBFGS(qm_model.parameters(), lr=1e-1, 
                            line_search_fn='strong_wolfe')
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for z_quad_batch, res_batch in dataloader:
            z_quad_batch = z_quad_batch.to(device)
            res_batch = res_batch.to(device)
            
            # optimizer.zero_grad()
            # res_pred = qm_model(z_quad_batch.T)
            # reconstruction_loss = mse_loss(res_pred, res_batch.T)
            # loss = reconstruction_loss + gamma**2 * torch.norm(qm_model.W)**2
            # loss.backward()
            # optimizer.step()

            def closure():
                optimizer.zero_grad()
                res_pred = qm_model(z_quad_batch.T)
                reconstruction_loss = mse_loss(res_pred, res_batch.T)
                loss = reconstruction_loss + gamma**2 * torch.norm(qm_model.W)**2
                loss.backward()
                return loss
            x_pred = qm_model(z_quad_batch.T)
            loss = optimizer.step(closure)
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        lr_scheduler.step(epoch_loss)
       
        if epoch % 10 == 0:
            with torch.no_grad():
                # Evaluate on full dataset for monitoring
                res_pred_full = qm_model(Z_quad_tensor.to(device))
                x_pred_full = V @ reduced_points + res_pred_full.cpu().numpy() + shift_value
                rel_err = np.linalg.norm(x_pred_full - X_pulse) / np.linalg.norm(X_pulse)
                print(f"Epoch {epoch:3d}: "
                      f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
                      f"Avg Loss = {epoch_loss:.4e}, "
                      f"Rel Error = {rel_err.item():.4e}")

    # Final evaluation after training
    with torch.no_grad():
        x_quad = qm_model(torch.tensor(Z_quad).to(device))
        tmp = x_quad.cpu().numpy().T
        x_final_trained = V @ V.T @ (X_pulse - shift_value) + tmp + shift_value
    
    rel_error_trained = np.linalg.norm(x_final_trained - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}") 
    
            
#%% #======================= Normalize data Data ==============================#
    print("\n" + "="*60)
    print("NORMALIZING DATA")
    print("="*60)

    # Center data
    X_pulse_mean = X_pulse.mean(axis=1, keepdims=True)
    X_pulse_ = X_pulse - X_pulse_mean
    
    # Normalize each column (time sample)
    X_pulse_min, X_pulse_max = X_pulse_.min(axis=1), X_pulse_.max(axis=1)
    X_pulse_shift = X_pulse_min.reshape(-1, 1)
    X_pulse_scale = (X_pulse_max - X_pulse_min).reshape(-1, 1)
    X_pulse_norm = (X_pulse_ - X_pulse_shift) / X_pulse_scale
    
    print(f"Normalized X_pulse shape: {X_pulse_norm.shape}")
    print(f"Normalized X_pulse dtype: {X_pulse_norm.dtype}")

#%% #======================== Training the Neural Network =====================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    X_pulse_norm_tensor = torch.tensor(X_pulse_norm, dtype=torch.float64)
    I_nn = np.array([38, 37, 31, 13, 1, 12, 4, 0, 25, 6, 9, 8, 36, 15, 7])
    V_norm = torch.svd(X_pulse_norm_tensor)[0][:, I_nn]  # First r_max components
    z_train = V_norm.T @ X_pulse_norm_tensor  # Reduced coordinates
    z_train = z_train.T
    x_target = X_pulse_norm_tensor.T  # Target data (n_samples, d)

    # Create dataset and dataloader for batch training
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(z_train, x_target)
    batch_size = 100  # Adjust based on your data size and memory
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    qm_model = QuadraticManifold(V_norm, gamma, torch.tensor(W_nn))
    qm_model = qm_model.to(device)
    
    # Training setup
    optimizer = optim.SGD(qm_model.parameters(), lr=1e-8, 
            momentum=0.99)
    # optimizer = optim.LBFGS(qm_model.parameters(), lr=1e-1, 
    #                         line_search_fn='strong_wolfe')
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Training loop
    num_epochs = 10000
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)
            
            optimizer.zero_grad()
            x_pred = qm_model(z_batch)
            reconstruction_loss = mse_loss(x_pred, x_batch)
            loss = reconstruction_loss + gamma**2 * torch.norm(qm_model.weight_mat)**2
            loss.backward()
            optimizer.step()

            # def closure():
            #     optimizer.zero_grad()
            #     x_pred = qm_model(z_batch)
            #     reconstruction_loss = mse_loss(x_pred, x_batch)
            #     loss = reconstruction_loss + gamma**2 * torch.norm(qm_model.weight_mat)**2
            #     loss.backward()
            #     return loss
            # x_pred = qm_model(z_batch)
            # loss = optimizer.step(closure)
            
            epoch_loss += loss.item()
            n_batches += 1
        
        epoch_loss /= n_batches
        lr_scheduler.step(epoch_loss)
       
        if epoch % 10 == 0:
            with torch.no_grad():
                # Evaluate on full dataset for monitoring
                x_pred_full = qm_model(z_train.to(device))
                rel_err = torch.norm(x_pred_full - x_target.to(device)) / torch.norm(x_target.to(device))
                print(f"Epoch {epoch:3d}: "
                      f"LR = {optimizer.param_groups[0]['lr']:.4e}, "
                      f"Avg Loss = {epoch_loss:.4e}, "
                      f"Rel Error = {rel_err.item():.4e}")

    # Final evaluation after training
    with torch.no_grad():
        x_reconstructed_trained = qm_model(z_train.to(device))
        tmp = x_reconstructed_trained.cpu().numpy().T
        x_final_trained = (tmp * X_pulse_scale) + X_pulse_shift
        x_final_trained = tmp + X_pulse_mean
    
    rel_error_trained = np.linalg.norm(x_final_trained - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}") 
            
#%% #======================== Training the Neural Network =====================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    X_pulse_norm_tensor = torch.tensor(X_pulse_norm, dtype=torch.float64)
    I_nn = np.array([38, 37, 31, 13, 1, 12, 4, 0, 25, 6, 9, 8, 36, 15, 7])
    V_norm = torch.svd(X_pulse_norm_tensor)[0][:, I_nn]  # First r_max components
    z_train = V_norm.T @ X_pulse_norm_tensor  # Reduced coordinates
    z_train = z_train.T
    x_target = X_pulse_norm_tensor.T  # Target data (n_samples, d)

    qm_model = QuadraticManifold(V_norm, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    # optimizer = optim.SGD(qm_model.parameters(), lr=1e-2, 
    #         momentum=0.99, weight_decay=gamma**2)
    optimizer = optim.LBFGS(qm_model.parameters(), lr=1e-1,
                            max_iter=100,
                            tolerance_grad=1e-30,
                            tolerance_change=1e-30,)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Training loop
    for epoch in range(10000):
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
                
        # lr_scheduler.step(loss)

    # Final evaluation after training
    # qm_model.eval()
    with torch.no_grad():
        x_reconstructed_trained = qm_model(z_train)
        tmp = x_reconstructed_trained.numpy().T
        x_final_trained = (tmp * X_pulse_scale) + X_pulse_shift
        x_final_trained = tmp + X_pulse_mean
    
    rel_error_trained = np.linalg.norm(x_final_trained - X_pulse) / np.linalg.norm(X_pulse)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")
    
    # # Check how much weights changed
    # final_weight_diff = torch.max(
    #     torch.abs(
    #         qm_model.weight_mat.T - torch.tensor(W, dtype=torch.float64)
    #     )).item()
    # print(f"Final weight matrix difference: {final_weight_diff:.2e}")


#%% #============== Training the Neural Network with whitening ================#
    print(f"\n{'='*60}")
    print("TRAINING NEURAL NETWORK")
    print(f"{'='*60}")

    W_tensor = torch.tensor(W.T, dtype=torch.float64)  # Transpose for NN

    X_shift = X_pulse - shift_value
    X_shift_tensor = torch.tensor(X_shift, dtype=torch.float64)
    zcaMat = zca_whitening_matrix(X_shift, epsilon=1e-4)
    X_white = np.dot(zcaMat, X_shift)  # Apply ZCA whitening
    X_white_tensor = torch.tensor(X_white, dtype=torch.float64)  

    # Perform SVD decomposition
    I_white = np.array([0,1,14,16,18,19,20,21,23,30,31,32,34,48,49])
    V_white = torch.svd(X_white_tensor)[0][:, I_white]
    z_target = torch.matmul(X_shift_tensor.T, V_white)
    
    qm_model = QuadraticManifold(V_white, gamma)
    qm_model = qm_model.to(device)
    
    # Training setup
    # optimizer = optim.SGD(qm_model.parameters(), lr=1e-4, 
    #         momentum=0.99, weight_decay=0.0)
    optimizer = optim.LBFGS(qm_model.parameters(), lr=1, 
                            line_search_fn='strong_wolfe',
                            max_iter=100,               
                            tolerance_grad=1e-25,       
                            tolerance_change=1e-25)
    mse_loss = nn.MSELoss()
    qm_model.train()
    
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Training loop
    for epoch in range(10000):
        # optimizer.zero_grad()
        # x_pred = qm_model(z_target)
        # reconstruction_loss = mse_loss(x_pred, X_shift_tensor.T)
        # loss = reconstruction_loss
        # loss.backward()
        # optimizer.step()

        def closure():
            optimizer.zero_grad()
            x_pred = qm_model(z_target)
            reconstruction_loss = mse_loss(x_pred, X_shift_tensor.T)
            loss = reconstruction_loss
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        x_pred = qm_model(z_target)
        
        if epoch % 10 == 0:
            with torch.no_grad():
                rel_err = torch.norm(x_pred - X_shift_tensor.T) / torch.norm(X_shift_tensor.T)
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
    
    rel_error_trained = np.linalg.norm(X_pulse - x_final_trained.T) / np.linalg.norm(X_pulse)
    print(f"\nFinal trained NN error: {rel_error_trained:.2e}")
    print(f"Greedy QM error: {rel_error_greedy:.2e}")

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
    V_white_np = V_norm.numpy()
    Z_nn = V_white_np.T @ X_pulse_norm
    residual = X_pulse_norm - V_white_np @ Z_nn
    Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
    W_nn_T, analytical_resid = lstsq_l2_numpy(
        Z_quad_nn.T, residual.T, reg_magnitude=1.0e-13
    )
    W_nn = W_nn_T.T
    recon_error = np.linalg.norm(
        X_pulse - ((V_white_np @ Z_nn + W_nn @ Z_quad_nn)*X_pulse_scale 
                   + X_pulse_shift) 
        - X_pulse_mean, ord='fro')
    rel_recon_error_nn = recon_error / np.linalg.norm(X_pulse, ord='fro') 
    recon_error_proc = np.linalg.norm(
        X_pulse_norm - (V_white_np @ Z_nn + W_nn @ Z_quad_nn), 'fro'
    )
    rel_recon_error_proc = recon_error_proc / np.linalg.norm(X_pulse_norm, 'fro')
    print(f"LassoNet: ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_error:.6e}")
    print(f"Relative error: {rel_recon_error_nn:.6e}")
    print(f"(processed) Relative error: {rel_recon_error_proc:.6e}")

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


