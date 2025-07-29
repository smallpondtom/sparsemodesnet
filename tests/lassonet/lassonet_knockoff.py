#%%
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.covariance import GraphicalLassoCV
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Add the src and example directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'examples'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'experiments', 'QM'))

from pulse import generate_advecting_pulse
from sparsemodesnet.dataset import PODReconDataset
from quadmani import quadmani_greedy

import knockpy as kp
from knockpy.knockoff_filter import KnockoffFilter


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
            # CRITICAL FIX: Don't transpose! Check shapes first
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
        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:3d} | lr={lr_new:.4e} | "
              f"Recon MSE={epoch_loss:.6e} | ‖ω‖₁={epoch_l1:.6e}")
            
    # Find the non-zero modes after training
    with torch.no_grad():
        selected_modes = np.where(model.omega.detach().cpu().numpy() > nonzero_thresh)[0]

    return selected_modes, history

#%% Knockoff Feature Statistics using QuadraticManifold
def compute_feature_statistics_quadratic_manifold(
    Z_orig, Z_knockoff, X_target, pod_basis_tensor, device, 
    num_epochs=500, lr=1e-3, verbose=False
):
    """
    Compute feature statistics using QuadraticManifold.
    
    Parameters:
    -----------
    Z_orig : np.ndarray, shape (s, n)
        Original POD coefficients
    Z_knockoff : np.ndarray, shape (s, n) 
        Knockoff POD coefficients
    X_target : np.ndarray, shape (d, n)
        Target reconstruction data
    pod_basis_tensor : torch.Tensor
        POD basis for reconstruction
    device : str
        Device for training
    num_epochs : int
        Training epochs for each model
    lr : float
        Learning rate
    verbose : bool
        Print training progress
        
    Returns:
    --------
    W : np.ndarray, shape (s,)
        Feature importance statistics (positive = original better)
    """
    s, n = Z_orig.shape
    
    # Combine original and knockoff features
    Z_combined = np.vstack([Z_orig, Z_knockoff])  # (2s, n)
    
    # Dataset with combined features
    ds_combined = PODReconDataset(Z_np=Z_combined, X_np=X_target)
    dl_combined = DataLoader(ds_combined, batch_size=n//4, shuffle=True)
    
    # Dataset with original features only  
    ds_orig = PODReconDataset(Z_np=Z_orig, X_np=X_target)
    dl_orig = DataLoader(ds_orig, batch_size=n//4, shuffle=True)
    
    # Extended POD basis for combined features (duplicate for knockoffs)
    pod_basis_combined = torch.cat([pod_basis_tensor, pod_basis_tensor], dim=1)
    
    # Train model on combined features
    model_combined = QuadraticManifold(
        pod_basis=pod_basis_combined,
        M=5.0, lam=0.0, gamma=1e-6  # No regularization for feature statistics
    ).to(device)
    
    # Train model on original features only
    model_orig = QuadraticManifold(
        pod_basis=pod_basis_tensor,
        M=5.0, lam=0.0, gamma=1e-6
    ).to(device)
    
    # Training function
    def train_model(model, dataloader, epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            n_samples = 0
            
            model.train()
            for z_batch, x_batch in dataloader:
                z_batch = z_batch.to(device)
                x_batch = x_batch.to(device)
                
                optimizer.zero_grad()
                _, x_hat_batch = model(z_batch)
                loss = mse_loss(x_hat_batch, x_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * x_batch.shape[0]
                n_samples += x_batch.shape[0]
            
            if verbose and (epoch % 100 == 0):
                print(f"  Epoch {epoch}, Loss: {epoch_loss/n_samples:.6e}")
        
        return epoch_loss / n_samples
    
    # Train both models
    if verbose:
        print("Training combined model...")
    loss_combined = train_model(model_combined, dl_combined, num_epochs)
    
    if verbose:
        print("Training original model...")
    loss_orig = train_model(model_orig, dl_orig, num_epochs)
    
    # Compute feature statistics using model coefficients
    # We'll use the omega (skip-weights) as importance measures
    with torch.no_grad():
        omega_combined = model_combined.omega.cpu().numpy()  # (2s,)
        omega_orig = model_orig.omega.cpu().numpy()  # (s,)
        
        # Split combined weights
        omega_orig_from_combined = omega_combined[:s]
        omega_knockoff_from_combined = omega_combined[s:]
        
        # Feature statistics: difference in importance
        W = np.abs(omega_orig_from_combined) - np.abs(omega_knockoff_from_combined)
    
    if verbose:
        print(f"Combined model loss: {loss_combined:.6e}")
        print(f"Original model loss: {loss_orig:.6e}")
        print(f"Feature statistics range: [{W.min():.4f}, {W.max():.4f}]")
    
    return W

#%% Alternative: LASSO-based feature statistics
def compute_feature_statistics_lasso(Z_orig, Z_knockoff, X_target, alpha=1.0):
    """
    Alternative feature statistics using LASSO coefficients.
    Simpler and faster than neural network approach.
    """
    
    s, n = Z_orig.shape
    d, _ = X_target.shape
    
    # Combine features
    Z_combined = np.vstack([Z_orig, Z_knockoff]).T  # (n, 2s)
    
    # Fit LASSO for each spatial point
    W_all = []
    
    for i in range(d):
        y = X_target[i, :]  # Target for spatial point i
        
        # Fit LASSO
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(Z_combined, y)
        
        # Feature statistics
        coef_orig = lasso.coef_[:s]
        coef_knockoff = lasso.coef_[s:]
        W_i = np.abs(coef_orig) - np.abs(coef_knockoff)
        W_all.append(W_i)
    
    # Average across spatial points
    W = np.mean(W_all, axis=0)
    return W

#%% Main Knockoff Procedure
def knockoff_pod_selection(Z_np, X_np, pod_basis_tensor, device, 
                          fdr=0.1, method='neural', num_epochs=500, 
                          knockoff_method='gaussian', verbose=True):
    """
    Apply model-X knockoffs for POD mode selection.
    
    Parameters:
    -----------
    Z_np : np.ndarray, shape (s, n)
        POD coefficients  
    X_np : np.ndarray, shape (d, n)
        Original high-dimensional data
    pod_basis_tensor : torch.Tensor
        POD basis
    device : str
        Device for training
    fdr : float
        Target false discovery rate
    method : str
        'neural' or 'lasso' for feature statistics
    num_epochs : int
        Training epochs (for neural method)
    knockoff_method : str
        'gaussian' or other knockoff generation methods
    verbose : bool
        Print progress
        
    Returns:
    --------
    selected_modes : np.ndarray
        Indices of selected POD modes
    W : np.ndarray
        Feature importance statistics
    """
    s, n = Z_np.shape
    
    if verbose:
        print(f"\nApplying Model-X Knockoffs for POD Mode Selection")
        print(f"Data shape: {Z_np.shape}")
        print(f"FDR target: {fdr}")
        print(f"Method: {method}")
        print("="*60)
    
    # Step 1: Generate knockoff features
    if verbose:
        print("Step 1: Generating knockoff features...")
    
    if knockoff_method == 'gaussian':
        # Estimate covariance using GraphicalLasso for better conditioning
        cov_estimator = GraphicalLassoCV(cv=3, max_iter=100)
        Z_trans = Z_np.T  # (n, s) for sklearn
        cov_estimator.fit(Z_trans)
        Sigma = cov_estimator.covariance_
        
        # Generate Gaussian knockoffs using correct knockpy API
        gaussian_sampler = kp.knockoffs.GaussianSampler(
            Z_trans, Sigma=Sigma, method='sdp'
        )
        Z_knockoff_trans = gaussian_sampler.sample_knockoffs()
        Z_knockoff = Z_knockoff_trans.T  # Back to (s, n)
        
    else:
        raise ValueError(f"Knockoff method {knockoff_method} not implemented")
    
    if verbose:
        print(f"Knockoff correlation with original: {np.corrcoef(Z_np.flatten(), Z_knockoff.flatten())[0,1]:.4f}")
    
    # Step 2: Compute feature statistics
    if verbose:
        print("Step 2: Computing feature statistics...")
    
    if method == 'neural':
        W = compute_feature_statistics_quadratic_manifold(
            Z_np, Z_knockoff, X_np, pod_basis_tensor, device, 
            num_epochs=num_epochs, verbose=verbose
        )
    elif method == 'lasso':
        W = compute_feature_statistics_lasso(Z_np, Z_knockoff, X_np)
    else:
        raise ValueError(f"Method {method} not supported")
    
    # Step 3: Apply knockoff filter
    if verbose:
        print("Step 3: Applying knockoff selection...")
    
    kfilter = KnockoffFilter()
    selected_modes = kfilter.make_selections(W, fdr=fdr)
    print(W)
    
    if verbose:
        print(f"Selected {len(selected_modes)} modes out of {s}")
        print(f"Selected mode indices: {selected_modes}")
        print(f"Feature statistics summary:")
        print(f"  Mean: {W.mean():.4f}")
        print(f"  Std:  {W.std():.4f}")
        print(f"  Min:  {W.min():.4f}")
        print(f"  Max:  {W.max():.4f}")
    
    return selected_modes, W

#%% Visualization function
def plot_knockoff_results(W, selected_modes, Z_np, save_path=None):
    """Plot knockoff results."""
    s = len(W)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Feature statistics
    axes[0,0].stem(range(s), W, basefmt=' ')
    axes[0,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,0].scatter(selected_modes, W[selected_modes], color='red', s=50, zorder=5)
    axes[0,0].set_xlabel('POD Mode Index')
    axes[0,0].set_ylabel('Feature Statistic W')
    axes[0,0].set_title('Knockoff Feature Statistics')
    axes[0,0].grid(True, alpha=0.3)
    
    # Histogram of feature statistics
    axes[0,1].hist(W, bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_xlabel('Feature Statistic W')
    axes[0,1].set_ylabel('Count')
    axes[0,1].set_title('Distribution of Feature Statistics')
    axes[0,1].grid(True, alpha=0.3)
    
    # POD coefficient magnitude for selected modes
    if len(selected_modes) > 0:
        Z_selected = Z_np[selected_modes, :]
        axes[1,0].imshow(np.abs(Z_selected), aspect='auto', cmap='viridis')
        axes[1,0].set_xlabel('Time Index')
        axes[1,0].set_ylabel('Selected POD Mode')
        axes[1,0].set_title(f'Selected POD Coefficients (n={len(selected_modes)})')
        
        # POD coefficient variance
        pod_vars = np.var(Z_np, axis=1)
        axes[1,1].semilogy(range(s), pod_vars, 'b-', alpha=0.7, label='All modes')
        axes[1,1].semilogy(selected_modes, pod_vars[selected_modes], 'ro', 
                          markersize=6, label='Selected')
        axes[1,1].set_xlabel('POD Mode Index')
        axes[1,1].set_ylabel('Coefficient Variance')
        axes[1,1].set_title('POD Mode Energy')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,0].text(0.5, 0.5, 'No modes selected', ha='center', va='center', 
                       transform=axes[1,0].transAxes)
        axes[1,1].text(0.5, 0.5, 'No modes selected', ha='center', va='center',
                       transform=axes[1,1].transAxes)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

#%% Integration with existing code
def replace_lasso_with_knockoffs(Z_np, X_np, pod_basis_tensor, device, r_max=15, fdr=0.1):
    """
    Drop-in replacement for the regularization path approach.
    
    Returns the same I_NN format as the original code.
    """
    print("\n" + "="*60)
    print("MODEL-X KNOCKOFFS MODE SELECTION")
    print("="*60)
    
    # Apply knockoffs
    selected_modes, W = knockoff_pod_selection(
        Z_np, X_np, pod_basis_tensor, device, 
        fdr=fdr, method='neural', verbose=True
    )
    
    # If more modes selected than r_max, take top ones by feature statistic
    if len(selected_modes) > r_max:
        W_selected = W[selected_modes]
        top_indices = np.argsort(W_selected)[-r_max:]
        selected_modes = selected_modes[top_indices]
        print(f"Truncated to top {r_max} modes based on feature statistics")
    
    # If fewer modes selected, pad with top unselected modes by variance
    elif len(selected_modes) < r_max:
        pod_vars = np.var(Z_np, axis=1)
        remaining_modes = np.setdiff1d(np.arange(Z_np.shape[0]), selected_modes)
        remaining_vars = pod_vars[remaining_modes]
        top_remaining = remaining_modes[np.argsort(remaining_vars)[-(r_max - len(selected_modes)):]]
        selected_modes = np.concatenate([selected_modes, top_remaining])
        print(f"Padded to {r_max} modes with high-variance unselected modes")
    
    # Sort for consistency
    I_NN = np.sort(selected_modes)
    
    # Plot results
    plot_knockoff_results(W, selected_modes, Z_np)
    
    return I_NN, W

# Example usage to replace the existing LASSO approach:
"""
# Replace this section in your main code:
# for lam in lambdas:
#     ... (regularization path training)
# I_NN = np.argsort(I_count)[-r_max:]

# With this:
I_NN, W = replace_lasso_with_knockoffs(
    Z_np, X, pod_basis_tensor, device, r_max=r_max, fdr=0.1
)
"""
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

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())

#%% #======================= LassoNet Mode Selection ==========================#
    print("\n" + "="*60)
    print("LASSO MODE SELECTION")
    print("="*60)
    
    # Define the regularization path 
    lambdas = np.logspace(-1, 4, 20)
    
    # Define the count of the selected modes
    I_count = np.zeros(s_p, dtype=int)
    
    # Compute the pod basis
    pod_basis = np.linalg.svd(X, full_matrices=False)[0][:, :s_p].astype(np.float64)
    pod_basis_tensor = torch.from_numpy(pod_basis.astype(np.float32)).to(device)
    
    # Compute the reduced data
    Z_np = pod_basis.T @ X  # (s, n)
    
    # # Prep the data
    # ds_sub = PODReconDataset(Z_np=Z_np, X_np=X)
    # dl_sub = DataLoader(ds_sub, batch_size=n_grids//2, shuffle=True)
    
    # I_NN, W = replace_lasso_with_knockoffs(
    #     Z_np, X, pod_basis_tensor, device, r_max=r_max, fdr=0.1
    # )
    
    kfilter = KnockoffFilter(
        ksampler='gaussian',
        fstat='lasso',
        knockoff_kwargs={'method': 'mvr'}
    )
    
# %%
