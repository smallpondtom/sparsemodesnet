#%%
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the src and example directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'QM'))

from pulse import generate_advecting_pulse
from sparsemodesnet.dataset import PODReconDataset
from quadmani import quadmani_greedy

#%% 
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


class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, lam: float, M: float,
                 gamma: float, W: torch.Tensor = None):
        super(QuadraticManifold, self).__init__()
        
        # Ensure everything is double precision
        self.register_buffer('U_s', pod_basis)  # (d, r)
        self.d, self.s = pod_basis.shape
        self.M = float(M)
        self.lam = lam
        self.gamma = gamma
        
        # Skip‐weights ω ∈ R^s
        self.omega = nn.Parameter(torch.ones(self.s))
        
        # First layer (used in proximal step)
        self.first_layer = nn.Linear(self.s, self.s, bias=False)
        
        if W is None:
            # Random initialization
            self.weight_mat = nn.Parameter(
                torch.ones(self.s * (self.s + 1) // 2, self.d, dtype=torch.float32) * 1e-8)
        else:
            expected_shape = (self.s * (self.s + 1) // 2, self.d)
            
            if W.shape == expected_shape:
                print(f"✓ W shape {W.shape} matches expected {expected_shape}")
                self.weight_mat = nn.Parameter(W.clone())
            elif W.shape == (expected_shape[1], expected_shape[0]):
                print(f"⚠ W shape {W.shape} needs transpose to match {expected_shape}")
                self.weight_mat = nn.Parameter(W.T.clone())
            else:
                raise ValueError(f"W shape {W.shape} doesn't match expected {expected_shape} or its transpose")
                
        
    def forward(self, z_batch):
        z_hat = z_batch * self.omega.unsqueeze(0)       # (batch, s)
        
        # Reconstruct the linear part via projection
        x_hat_lin = z_hat @ self.U_s.T                  # (batch, d)
        
        h = self.first_layer(z_hat)          # (batch, inter_dim)
        z_quad = quadratic_mapping_torch(h)  # (batch, r*(r+1)//2)
        x_hat_quad = z_quad @ self.weight_mat      # (batch, d)

        x_hat = x_hat_lin + x_hat_quad

        return z_hat, x_hat
    
    def l1_norm_omega(self):
        """Return ℓ₁-norm of ω."""
        return self.omega.abs().sum()

    def proximal_step(self, lam):
        """Batched implementation of Algorithm 4 (Group-Hierarchical Proximal) 
        with λ̄ = 0, corrected so that ω_new = x_star * θ (no extra 
        soft-threshold on ω).
        
        Arguments
        ---------
        lam : float, the regularization parameter for the proximal step. This is
              multiplied by the learning rate.
        
        Note
        ----
        The `v`, `θ`, and `u` notations are presented in the original paper, but
        here we use `omega` for θ. To clarify confusion with the notation, 
        please refer to the original paper.
        """
        M = self.M

        # 1) Gather first‐layer weights W1 ∈ ℝ^{h×s}, then transpose → W1_T ∈ ℝ^{s×h}
        W1   = self.first_layer.weight.data   # (h, s)
        W1_T = W1.t().contiguous()            # (s, h), call h=K

        s, K = W1_T.shape  # s = #features, K = width of first hidden layer

        # 2) Sort each row of |W1_T| in descending order (batched)
        u_abs_sorted, _ = W1_T.abs().sort(dim=1, descending=True)  # (s, K)

        # 3) Build partial sums a_s(m) = lam - M * sum_{i=1}^m u_abs_sorted[j,i-1]
        zeros_m = torch.zeros((s, 1), device=W1_T.device, dtype=W1_T.dtype)  # (s,1)
        cumsum_vals = torch.cumsum(u_abs_sorted, dim=1)  # (s, K)
        a_s = lam - M * torch.cat([zeros_m, cumsum_vals], dim=1)  # (s, K+1)

        # 4) ‖v‖₂ = |θ|, shape (s,)
        theta_abs = self.omega.data.abs()  # (s,)

        # 5) Broadcast |θ| into (s, K+1)
        norm_v_col = theta_abs.unsqueeze(1).expand(-1, K+1)  # (s, K+1)

        # 6) Build m_index = [0,1,...,K] for each of s rows
        m_index = torch.arange(K+1, device=W1_T.device, dtype=W1_T.dtype).view(1, K+1)
        m_index = m_index.expand(s, -1)  # (s, K+1)

        # 7) Compute x_vals(m) = ReLU(1 - a_s / ‖v‖) / (1 + m*M^2)
        x_vals = F.relu(1.0 - a_s / (norm_v_col + 1e-16)) / (1.0 + m_index * (M**2))  # (s, K+1)

        # 8) Compute w_vals(m) = M * x_vals(m) * ‖v‖
        w_vals = M * x_vals * norm_v_col  # (s, K+1)

        # 9) Build “lower(m)” = [u_abs_sorted, 0], shape (s, K+1)
        lower = torch.cat([u_abs_sorted, zeros_m], dim=1)  # (s, K+1)

        # 10) Find index m* per row:  m*_j = sum_{m=0..K} [ lower[j,m] > w_vals[j,m] ]
        cond = lower > w_vals          # (s, K+1), bool
        idx  = torch.sum(cond, dim=1)  # (s,)  ← m* for each feature j

        # 11) Gather x_star[j] = x_vals[j, idx[j]]  and  w_star[j] = w_vals[j, idx[j]]
        row_idx = torch.arange(s, device=W1_T.device)
        x_star  = x_vals[row_idx, idx]  # (s,)
        w_star  = w_vals[row_idx, idx]  # (s,)

        # 12) ***CORRECTED***  Update skip‐weights:  b_new[j] = x_star[j] * θ_j
        # No extra soft‐threshold here, because λ was already used in building a_s→x_vals.
        b_new = x_star * self.omega.data  # (s,)

        # 13) Coordinate‐wise clip each row of W1_T to ±w_star[j]:
        W1_T_abs    = W1_T.abs()                         # (s, K)
        w_star_col  = w_star.unsqueeze(1).expand(-1, K)  # (s, K)
        clipped_abs = torch.min(W1_T_abs, w_star_col)    # (s, K)
        W1_T_new    = W1_T.sign() * clipped_abs          # (s, K)

        # 14) Write back:
        self.omega.data.copy_(b_new)           # (s,)
        W1_updated = W1_T_new.t().contiguous() # shape: (K, s) → transpose to (h, s)
        self.first_layer.weight.data.copy_(W1_updated)
 
    
def train_quadraticmanifold(model: QuadraticManifold,
                            dataloader: DataLoader,
                            num_epochs: int,
                            lr: float,
                            momentum: float,
                            optimizer: str,
                            nonzero_thresh: float,
                            device: str):
    """
    Train for exactly num_epochs at whatever model.lam currently is.
    """
    model.to(device)
    if optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                               lr=lr, weight_decay=model.gamma)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, 
            nesterov=True, weight_decay=model.gamma)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    mse_loss = nn.MSELoss()
    
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100,
    )
    lr_new = optimizer.param_groups[0]['lr']
    history = {'loss': [], 'l1_b': []}
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1  = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            _, x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            
            loss.backward()
            optimizer.step()

            model.proximal_step(model.lam * lr_new)

            batch_size  = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1   += model.l1_norm_omega().item() * batch_size
            n_samples  += batch_size
            
        lr_schedule.step(loss)  # Update learning rate
        lr_new = optimizer.param_groups[0]['lr']

        epoch_loss /= n_samples
        epoch_l1  /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_b'].append(epoch_l1)

        # Print every 10 epochs or first:
        if (epoch % 100 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:<4d} | lr={lr_new:.4e} | "
              f"Recon MSE={epoch_loss:.6e} | ‖ω‖₁={epoch_l1:.6e}")
            
    # Find the non-zero modes after training
    with torch.no_grad():
        selected_modes = np.where(model.omega.detach().cpu().numpy() > nonzero_thresh)[0]

    return selected_modes, history


#%% #================================ __main__ ================================#
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
#%% #===================== Main Experiment with Pulse Data ====================#
    # Parameters
    r_max = 15
    n_grids = 2**10
    sanity_check = False  # Disable plotting for now
    
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)
    
    # Generate advecting pulse data
    X, xspan, tspan = generate_advecting_pulse(
        pulse_width=5.0e-4,
        pulse_shift=0.1,
        speed=5.0,
        final_time=0.15,
        n_time_samples=1000,
        n_space_samples=n_grids
    )
    
    print(f"X_pulse shape: {X.shape}")
    print(f"X_pulse dtype: {X.dtype}")
    
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
    ax.set_title('Advecting Gaussian Pulse')
    plt.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()
    plt.close(fig)
    
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X, r_max, s_p, 1e-6, np.array([], dtype=int))
    shift_value = shift_value.reshape(-1, 1)  

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())


#%% #======================= LassoNet Mode Selection ==========================#
    print("\n" + "="*60)
    print("LASSO MODE SELECTION")
    print("="*60)
    
    # Shifted data
    shift_value_np = np.array(shift_value)
    X_shift = X - shift_value_np
    
    # Define the count of the selected modes
    I_count = np.zeros(s_p, dtype=int)
    
    # Compute the pod basis
    pod_basis = np.linalg.svd(X_shift, full_matrices=False)[0][:, :s_p].astype(np.float64)
    pod_basis_tensor = torch.from_numpy(pod_basis.astype(np.float32)).to(device)
    
    # Compute the reduced data
    Z_np = pod_basis.T @ X_shift  # (s, n)
    
    # Prep the data
    ds_sub = PODReconDataset(Z_np=Z_np, X_np=X_shift)
    dl_sub = DataLoader(ds_sub, batch_size=n_grids//2, shuffle=True)
    
    # Initialize the regularization parameter and increase factor
    lam = 2.0
    eps = 0.2
    
    # Initialize the model 
    model = QuadraticManifold(
        pod_basis=pod_basis_tensor,
        M=5.0, lam=lam, gamma=0.0,
    ) 
    
    cnt = 1  # loop counter 
    while True:
        print(f"\nTraining with λ = {lam:.3e}")
        
        # Train the model
        selected_modes, history = train_quadraticmanifold(
            model=model, dataloader=dl_sub, num_epochs=2000, 
            lr=1e-3, momentum=0.9, optimizer='Adam', 
            nonzero_thresh=1e-13, device=device
        ) 
        
        if selected_modes.size == 0:
            print("No modes selected. End loop.")
            break
        else:
            print(f"Number of selected modes: {len(selected_modes)}")
        
        # Increment the count of the selected modes
        I_count[selected_modes] += 1
        
        # Update lambda
        lam *= (1 + eps)
        model.lam = lam
        
        cnt += 1
        
    # Select the first largest r_max modes
    I_nn = np.argsort(cnt - I_count, kind='mergesort')[:r_max]
    
    

# %% #======================== Plot selected modes ============================#
    print("\n" + "="*60)
    print("PLOTTING SELECTED MODES")
    print("="*60)

    # Create a 10x10 grid plot
    fig, axes = plt.subplots(10, 10, figsize=(15, 15))
    fig.suptitle('Mode Selection Counts (Circle size = count)\n' +
                 'Orange squares = Quadratic Manifold selected modes', 
                fontsize=14, y=0.95)

    # Normalize counts for circle sizes
    max_count = np.max(I_count) if np.max(I_count) > 0 else 1
    normalized_counts = I_count / max_count

    # Plot each mode
    for i in range(10):
        for j in range(10):
            mode_idx = i * 10 + j
            ax = axes[i, j]
            
            if mode_idx < len(I_count):
                # Circle size based on normalized count
                circle_size = normalized_counts[mode_idx] * 4000  # Scale for visibility
                
                # Plot circle
                ax.scatter(0.5, 0.5, s=circle_size, c='darkblue', alpha=0.6)
                
                # Add red square if this mode was selected by quadratic manifold
                if mode_idx in I_qm:
                    rect = plt.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                    linewidth=3, edgecolor='orange', 
                                    facecolor='none')
                    ax.add_patch(rect)
                
                # Add count text
                if mode_idx in I_nn:
                    ax.text(0.5, 0.5, f'{I_count[mode_idx]}', 
                        ha='center', va='center', 
                        fontsize=25, color='orange',)
                else:
                    ax.text(0.5, 0.5, f'{I_count[mode_idx]}', 
                        ha='center', va='center', 
                        fontsize=25, color='white',)
            
            # Set labels and limits
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f'Mode {mode_idx+1}', fontsize=13)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove axis spines
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
    plt.savefig('figures/lassonet/selected_modes_plot.png', dpi=200)
    plt.show()

    # Print summary statistics
    print(f"\nSummary:")
    print(f"Total modes: {len(I_count)}")
    print(f"Max count: {np.max(I_count)}")
    print(f"Modes with non-zero counts: {np.sum(I_count > 0)}")
    print(f"Quadratic manifold selected modes: {len(I_qm)}")
    print(f"LassoNet top {r_max} modes: {I_nn}")
    print(f"Quadratic manifold modes: {np.sort(I_qm)}")

    # Find overlap between methods
    overlap = np.intersect1d(I_nn, I_qm)
    print(f"Overlap between methods: {len(overlap)} modes")
    print(f"Overlapping modes: {np.sort(overlap)}")
    

# %% #==================== Compute Reconstruction Errors ======================#
    # Compute the reconstruction error (Quadratic Manifold)
    Z_qm = V.T @ X_shift
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    recon_error_quad = np.linalg.norm(
        X - V @ Z_qm - W @ Z_quad_qm - shift_value, ord='fro')
    rel_recon_error_qm = recon_error_quad / np.linalg.norm(X, ord='fro')
    
    # Compute the reconstruction error (LassoNet)
    V_nn = pod_basis[:, I_nn]  
    Z_nn = V_nn.T @ X_shift
    residual = X_shift - V_nn @ Z_nn
    Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
        
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

    W_nn_T, analytical_resid = lstsq_l2_numpy(
        Z_quad_nn.T, residual.T, reg_magnitude=1e-12
    )
    W_nn = W_nn_T.T
    
    recon_error = np.linalg.norm(
        X - V_nn @ Z_nn - W_nn @ Z_quad_nn - shift_value, ord='fro')
    rel_recon_error_nn = recon_error / np.linalg.norm(X, ord='fro') 
    
    # Print results
    print(f"\nReconstruction errors:")
    print(f"Quadratic Manifold: ||X - V @ Z - W @ Z_quad||_F = {recon_error_quad:.6e}")
    print(f"Relative error: {rel_recon_error_qm:.6e}")
    print(f"LassoNet: ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_error:.6e}")
    print(f"Relative error: {rel_recon_error_nn:.6e}")
# %%
