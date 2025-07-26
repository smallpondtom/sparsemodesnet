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

from kse import generate_kse_data
from sparsemodesnet.dataset import PODReconDataset
from quadmani import quadmani_greedy

#%% 
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
    
class MaskedLayer(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,    # e.g., 6
        out_features: int,   # e.g., 2
        mask: torch.Tensor,  # e.g., shape(2,6)
        dtype: torch.dtype,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,  # Ensure double precision
            bias=False,   # no need to use a bias in our case
        )
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
        return x

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, 
                 lam: float, M: float, alpha: float,
                 gamma: float, dtype: torch.dtype = torch.float32):
        super(QuadraticManifold, self).__init__()
        
        # Ensure everything is double precision
        if dtype == torch.float64:
            pod_basis = pod_basis.double()
        self.register_buffer('U_s', pod_basis) 
        self.d, self.s = pod_basis.shape
        self.M = float(M)
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
        self.dtype = dtype
        
        # Skip‐weights ω ∈ R^s
        self.omega = nn.Parameter(torch.ones(self.s, dtype=dtype) * 0.01)

        self.W = nn.Parameter(
            0.001 * torch.ones(self.s * (self.s + 1) // 2, self.d, dtype=dtype)
        )

        # NOTE: 
        # First layer (used in proximal step) as a gate for selected modes.
        # To make this identical to the quadratic manifold we have to mask
        # the first layer weights to be the identity matrix so that purely
        # the selected modes are passed to the quadratic mapping.
        self.first_layer = MaskedLayer(
            self.s, self.s, torch.eye(self.s), dtype=dtype)
        self.first_layer.weight.data.fill_(0.05)  # Initialize to ones

    # Forward map 1
    def forward(self, z_batch, x_batch):
        if model.dtype == torch.float64:
            z_batch = z_batch.double()  
            x_batch = x_batch.double()  

        z_hat = z_batch * self.omega.unsqueeze(0) 
        x_hat_lin = z_hat @ self.U_s.T                  
        h = self.first_layer(z_hat)  
        z_quad = quadratic_mapping_torch(h) 
        x_hat_quad = z_quad @ self.W
        x_hat = x_hat_lin + x_hat_quad

        return z_hat, x_hat

    @staticmethod 
    def lstsq_l2_torch(A, B, reg_magnitude=1e-6):
        U, sigma, Vt = torch.linalg.svd(A, full_matrices=False)
        sinv = sigma / (sigma**2 + reg_magnitude**2)
        
        # Handle both 1D and 2D B cases
        if B.dim() == 1:
            # B is 1D: shape (m,)
            UTB = U.T @ B  # shape (min(m,n),)
            x = Vt.T @ (sinv * UTB)  # shape (n,)
        else:
            # B is 2D: shape (m, k)
            UTB = U.T @ B  # shape (min(m,n), k)
            x = Vt.T @ (sinv.unsqueeze(-1) * UTB)  # shape (n, k)
        
        # Compute residual
        B_estimate = A @ x
        resid = torch.linalg.norm(B - B_estimate)
        
        return x, resid

    
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
        a_s = self.alpha * lam - M * torch.cat([zeros_m, cumsum_vals], dim=1)  # (s, K+1)

        # 4) ‖v‖₂ = |θ|, shape (s,)
        theta_abs = self.omega.data.abs() # (s,)

        # 5) Broadcast |θ| into (s, K+1)
        norm_v_col = theta_abs.unsqueeze(1).expand(-1, K+1)  # (s, K+1)

        # 6) Build m_index = [0,1,...,K] for each of s rows
        m_index = torch.arange(K+1, device=W1_T.device, dtype=W1_T.dtype).view(1, K+1)
        m_index = m_index.expand(s, -1)  # (s, K+1)

        # 7) Compute x_vals(m) = ReLU(1 - a_s / ‖v‖) / (1 + m*M^2)
        x_vals = F.relu(1.0 - a_s / (norm_v_col + 1e-16)
                        ) / (1.0 + m_index * (M**2) + (1 - self.alpha)*lam)  # (s, K+1)

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
                            rmax: int,
                            device: str):
    """
    Train for exactly num_epochs at whatever model.lam currently is.
    """
    model.to(device)
    if optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), 
                               lr=lr, weight_decay=model.gamma)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, 
            nesterov=True, weight_decay=model.gamma)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    mse_loss = nn.MSELoss()
    
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=1000000,
    )
    lr_new = optimizer.param_groups[0]['lr']
    history = {'loss': [], 'l1_b': [], 'omegas': []}

    exit_flag = False
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1  = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            _, x_hat_batch = model(z_batch, x_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss += model.gamma * torch.norm(model.W)**2

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

        with torch.no_grad():
            omega_ = model.omega.detach().cpu().numpy()
            history['omegas'].append(omega_)
            nonzero_count = np.count_nonzero(omega_)

        # Print every 10 epochs or first:
        if (epoch % 10 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:<4d} | lr={lr_new:.4e} | "
                  f"Recon MSE={epoch_loss:.6e} | ‖ω‖₁={epoch_l1:.6e} | "
                  f"Non-zero modes: {nonzero_count}")
            print(omega_)
            
        if nonzero_count <= rmax:
            print(f"Reached maximum non-zero modes ({rmax}). Stopping training.")
            exit_flag = True
            break
            
        if epoch_l1 == 0:
            print("All modes have zero weights. Stopping training.")
            break
        
            
    # Find the non-zero modes after training
    with torch.no_grad():
        omega_ = model.omega.detach().cpu().numpy()

    return omega_, history, exit_flag


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
    
#%% #===================== Main Experiment with KSE Data ======================#
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
    
    print(f"X_pulse shape: {X.shape}")
    print(f"X_pulse dtype: {X.dtype}")
    
    # Ensure data is double precision
    X = X.astype(np.float64)
    
    d_ks, n_ks = X.shape
    s_ks = min(d_ks, n_ks)
    s_ks = 100

    # Plot flow field    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        X, aspect='auto', cmap='viridis', origin='lower',
        extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]])
    ax.set_xlabel('Time')
    ax.set_ylabel('Space (x)')
    ax.set_title('Kuramoto-Sivashinsky Equation Solution')
    plt.colorbar(im, ax=ax, label='u(x,t)')
    plt.tight_layout()
    plt.show()
    plt.close(fig)
    
    
#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V, W, shift_value, I_qm = quadmani_greedy(
        X, r_max, s_ks, 1e-15, np.array([], dtype=int))
    shift_value = shift_value.reshape(-1, 1)  

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())


#%% #======================= LassoNet Mode Selection ==========================#
    print("\n" + "="*60)
    print("LASSO MODE SELECTION")
    print("="*60)

    WHITENING = True
    USE_ONLY_QM_MODES = False

    # Shifted data
    shift_value_np = np.array(shift_value)
    X_shift = X - shift_value_np

    if WHITENING:
        zcaMat = zca_whitening_matrix(X_shift, epsilon=1e-1)
        X_white = np.dot(zcaMat, X_shift)  # Apply ZCA whitening

        # from whitening_algorithms import whiten
        # X_white = whiten(X_shift, method='zca-cor', eps=1e-4)
    else:
        X_white = X_shift

    # Compute the pod basis
    V_white, _, _ = np.linalg.svd(X_white, full_matrices=False)
    V_white = V_white[:, :s_ks]  
    V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)

    if USE_ONLY_QM_MODES:
        # Process the data so that it only includes the modes selected by the 
        # greedy quadratic manifold algorithm.
        I_c = set(np.arange(s_ks)) - set(np.array(I_qm))
        I_c = np.random.choice(list(I_c), 20, replace=False)
        V_tmp = np.hstack((V, V_white[:, I_c]))
        X_proc = (V_tmp @ V_tmp.T @ X_shift) # + W @ quadratic_mapping_numpy(X_shift.T @ V).T
        X_proc = np.array(X_proc, dtype=np.float32)  
    else:
        X_proc = X_white
    
    # Compute the reduced data
    Z_np = V_white.T @ X_proc  # (s, n)
    
    # Prep the data
    ds_sub = PODReconDataset(Z_np=Z_np, X_np=X_proc, type="float32")
    dl_sub = DataLoader(ds_sub, batch_size=500, shuffle=True)
    
    # Initialize the regularization parameter and increase factor
    lam = 5.0
    eps = 0.001
    alpha = 1.0
    gamma = 1e-10
    threshold = 1e-8
    
    # Initialize the model 
    model = QuadraticManifold(
        pod_basis=V_white_tensor, dtype=torch.float32, 
        M=25.0, lam=lam, gamma=gamma, alpha=alpha,
    ) 

    # Define the count of the selected modes and omegas
    I_count = np.zeros(s_ks, dtype=int)
    omegas = model.omega.detach().numpy().reshape(-1, 1)
    
    while True:
        print(f"\nTraining with λ = {lam:.3e}")
        
        # Train the model
        omega_, history, flag = train_quadraticmanifold(
            model=model, dataloader=dl_sub, num_epochs=100, 
            lr=1e-3, momentum=0.95, optimizer='Adam', 
            device=device, rmax=r_max
        ) 
        
        selected_modes = np.where(omega_ > threshold)[0] 
        if selected_modes.size == 0:
            print("No modes selected. End loop.")
            break
        else:
            print(f"Number of selected modes: {len(selected_modes)}")

        # Increment the count of the selected modes and omegas
        I_count[selected_modes] += 1
        omegas = np.concatenate((omegas, omega_.reshape(-1, 1)), axis=1)

        if flag:
            break
        
        # Update lambda
        lam *= (1 + eps)
        model.lam = lam
        
    # Select the first largest r_max modes
    I_nn = np.where(omegas[:, -1] > 0)[0]

#%% #======================= Greedy Quadratic Manifold ========================#
    print("\n" + "="*60)
    print("GREEDY QUADRATIC MANIFOLD")
    
    V, W, shift_tmp, I_qm = quadmani_greedy(
        X_white, r_max, s_ks, 1e-6, np.array([], dtype=int))
    shift_tmp = shift_tmp.reshape(-1, 1)

    # Print the selected modes
    print("Selected modes (I_qm):", I_qm.sort())


# %% #==================== Compute Reconstruction Errors ======================#
    # Compute the reconstruction error (Quadratic Manifold)
    Z_qm = V.T @ X_shift
    Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
    recon_error_quad = np.linalg.norm(
        X - V @ Z_qm - W @ Z_quad_qm - shift_value, ord='fro')
    rel_recon_error_qm = recon_error_quad / np.linalg.norm(X, ord='fro')
    
    # Compute the reconstruction error (LassoNet)
    V_nn = V_white[:, I_nn[:r_max]]  
    Z_nn = V_nn.T @ X_shift
    residual = X_shift - V_nn @ Z_nn
    Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
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

# %% #====================== Plot the singular values =========================#
    # Compute the singular values of the shifted data
    pod_basis, Sig, _ = np.linalg.svd(X_shift, full_matrices=False)
    pod_basis = pod_basis[:, :s_ks]  
    Sig = Sig[:s_ks]  
    
    # Plot singular values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Singular values (linear scale)
    ax1.plot(range(1, len(Sig) + 1), Sig, 'b-o', markersize=4)
    ax1.set_xlabel('Mode Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title('Singular Values (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Highlight quadratic manifold modes
    for mode in I_qm:
        if mode < len(Sig):
            ax1.plot(mode + 1, Sig[mode], 'ro', markersize=8, 
                    markerfacecolor='none', markeredgewidth=2)
    
    # Plot 2: Singular values (log scale)
    ax2.semilogy(range(1, len(Sig) + 1), Sig, 'b-o', markersize=4)
    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('Singular Value (log scale)')
    ax2.set_title('Singular Values (Log Scale)\n(Red circles = Quadratic Manifold modes)')
    ax2.grid(True, alpha=0.3)
    
    # Highlight quadratic manifold modes
    for mode in I_qm:
        if mode < len(Sig):
            ax2.plot(mode + 1, Sig[mode], 'ro', markersize=8, 
                    markerfacecolor='none', markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig('figures/lassonet/kse/singular_values.png', dpi=200)
    plt.show()
    
    # Print statistics
    print(f"\nSingular Value Statistics:")
    print(f"Number of modes: {len(Sig)}")
    print(f"Largest singular value: {Sig[0]:.6e}")
    print(f"Smallest singular value: {Sig[-1]:.6e}")
    print(f"Condition number: {Sig[0]/Sig[-1]:.6e}")
    print(f"Energy captured by first {r_max} modes: {np.sum(Sig[:r_max]**2)/np.sum(Sig**2)*100:.2f}%")
    print(f"Quadratic manifold modes singular values: {Sig[I_qm]}")


# %% #========================= Plot the omegas ===============================#
    print("\n" + "="*60)
    print("PLOTTING OMEGAS")
    print("="*60)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Omega evolution over lambda iterations
    for mode in range(s_ks):
        if mode in I_nn:
            ax1.plot(np.abs(omegas[mode, :]), linewidth=2, 
                     label=f'Mode {mode+1}', color='orange')
        else:
            ax1.plot(np.abs(omegas[mode, :]), linewidth=1, alpha=0.5, 
                     color='darkblue', linestyle='--')

    ax1.set_xlabel('Lambda iteration')
    ax1.set_ylabel('Omega values')
    ax1.set_title('Evolution of Omega Values During Training')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Final omega values
    final_omegas = np.abs(omegas[:, -1])
    non_zero_modes = np.where(final_omegas > 1e-13)[0]
    mode_indices = np.arange(len(final_omegas))
    bars = ax2.bar(mode_indices, final_omegas, alpha=0.7, color='darkblue')

    # Highlight selected modes
    for mode in non_zero_modes:
        bars[mode].set_color('orange')

    # Highlight quadratic manifold modes
    for mode in I_qm:
        if mode < len(bars):
            bars[mode].set_edgecolor('red')
            bars[mode].set_linewidth(0.5)

    ax2.set_xlabel('Mode Index')
    ax2.set_ylabel('Final Omega (Abs) Value')
    ax2.set_title('Final Omega (Abs) Values\n(Orange = Selected by LassoNet, '
                  'Red border = Quadratic Manifold)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    # Add threshold line
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.5, 
                label='Selection threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('figures/lassonet/kse/omega_evolution.png', dpi=200)
    plt.show()

    # Print statistics about omega values
    print(f"\nOmega Statistics:")
    print(f"Number of lambda iterations: {omegas.shape[1]}")
    print(f"Final non-zero omegas: {len(non_zero_modes)}")
    print(f"Max final omega: {np.max(final_omegas):.6e}")
    print(f"Min final omega: {np.min(final_omegas[final_omegas > 1e-13]):.6e}" 
          if len(non_zero_modes) > 0 else "No non-zero omegas")
    print(f"Final selected modes: {non_zero_modes}")
    

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
    plt.savefig('figures/lassonet/kse/selected_modes_plot.png', dpi=200)
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

#%% #===================== Plot Reconstructions and Errors ====================# 
print("\n" + "="*60)
print("PLOTTING RECONSTRUCTIONS AND ERRORS")
print("="*60)

# Compute reconstructions for comparison
# 1. Original data (shifted back)
X_original = X

# 2. POD reconstruction (using top r_max modes)
pod_basis = np.linalg.svd(X_shift, full_matrices=False)[0]
pod_basis_top = pod_basis[:, :r_max]
Z_pod = pod_basis_top.T @ X_shift
X_pod_recon = pod_basis_top @ Z_pod + shift_value_np

# 3. Quadratic Manifold reconstruction
Z_qm = V.T @ X_shift
Z_quad_qm = quadratic_mapping_numpy(Z_qm.T).T
X_qm_recon = V @ Z_qm + W @ Z_quad_qm + shift_value_np

# 4. LassoNet reconstruction  
V_nn = V_white[:, I_nn[:r_max]]  
Z_nn = V_nn.T @ X_shift
residual = X_shift - V_nn @ Z_nn
Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
W_nn_T, _ = lstsq_l2_numpy(Z_quad_nn.T, residual.T, reg_magnitude=1e-15)
W_nn = W_nn_T.T
X_lassonet_recon = V_nn @ Z_nn + W_nn @ Z_quad_nn + shift_value_np

# Compute errors
pod_error = X_original - X_pod_recon
qm_error = X_original - X_qm_recon  
lassonet_error = X_original - X_lassonet_recon

# Set consistent color scales
recon_vmin = min(X_original.min(), X_pod_recon.min(), X_qm_recon.min(), X_lassonet_recon.min())
recon_vmax = max(X_original.max(), X_pod_recon.max(), X_qm_recon.max(), X_lassonet_recon.max())

error_vals = [pod_error, qm_error, lassonet_error]
error_vmax = max([np.abs(err).max() for err in error_vals])
error_vmin = -error_vmax

# Create the comparison plot
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Reconstructions
# (1,1) Original data
im1 = axes[0,0].imshow(
    X_original, aspect='auto', cmap='viridis', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=recon_vmin, vmax=recon_vmax)
axes[0,0].set_ylabel('Space (x)', fontsize=14)
axes[0,0].set_title('Original Data', fontsize=15)

# (1,2) POD reconstruction
im2 = axes[0,1].imshow(
    X_pod_recon, aspect='auto', cmap='viridis', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=recon_vmin, vmax=recon_vmax)
axes[0,1].set_title('POD Reconstruction', fontsize=15)

# (1,3) Quadratic Manifold reconstruction
im3 = axes[0,2].imshow(
    X_qm_recon, aspect='auto', cmap='viridis', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=recon_vmin, vmax=recon_vmax)
axes[0,2].set_title('Quadratic Manifold Reconstruction', fontsize=15)

# (1,4) LassoNet reconstruction
im4 = axes[0,3].imshow(
    X_lassonet_recon, aspect='auto', cmap='viridis', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=recon_vmin, vmax=recon_vmax)
axes[0,3].set_title('LassoNet Reconstruction', fontsize=15)

# Row 2: Errors
# (2,1) Empty - no error for original data
axes[1,0].axis('off')

# (2,2) POD error
im5 = axes[1,1].imshow(
    pod_error, aspect='auto', cmap='RdBu_r', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=error_vmin, vmax=error_vmax)
axes[1,1].set_xlabel('Time', fontsize=14)
axes[1,1].set_ylabel('Space (x)', fontsize=14)
axes[1,1].set_title('POD Error', fontsize=15)

# (2,3) Quadratic Manifold error
im6 = axes[1,2].imshow(
    qm_error, aspect='auto', cmap='RdBu_r', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=error_vmin, vmax=error_vmax)
axes[1,2].set_xlabel('Time', fontsize=14)
axes[1,2].set_title('Quadratic Manifold Error', fontsize=15)

# (2,4) LassoNet error
im7 = axes[1,3].imshow(
    lassonet_error, aspect='auto', cmap='RdBu_r', origin='lower',
    extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]],
    vmin=error_vmin, vmax=error_vmax)
axes[1,3].set_xlabel('Time', fontsize=14)
axes[1,3].set_title('LassoNet Error', fontsize=15)

# Add colorbars
cax1 = fig.add_axes([0.92, 0.57, 0.02, 0.35])
cbar1 = plt.colorbar(im4, cax=cax1, label='u(x,t)')
cbar1.set_label('u(x,t)', fontsize=14)

cax2 = fig.add_axes([0.92, 0.11, 0.02, 0.35])
cbar2 = plt.colorbar(im7, cax=cax2, label='Abs. Error')
cbar2.set_label('Abs. Error', fontsize=14)

plt.subplots_adjust(left=0.05, right=0.9, top=0.92, bottom=0.1, wspace=0.3, hspace=0.3)
plt.suptitle('Reconstruction Comparison: KSE Data', fontsize=19, y=0.98)
plt.savefig('figures/lassonet/kse/reconstruction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

#%% Plot kse at specific time points
print("\nPlotting kse at specific time points...")

# Select 3 equally spaced time points
n_times = len(tspan)
time_indices = [n_times//4, n_times//2, 3*n_times//4]
time_points = [tspan[i] for i in time_indices]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (ax, t_idx, t_val) in enumerate(zip(axes, time_indices, time_points)):
    # Plot original data
    ax.plot(xspan, X_original[:, t_idx], 'k-', linewidth=3, label='Original', alpha=0.9)
    # Plot POD reconstruction
    ax.plot(xspan, X_pod_recon[:, t_idx], 'b--', linewidth=2, label='POD', alpha=0.8)
    # Plot Quadratic Manifold reconstruction
    ax.plot(xspan, X_qm_recon[:, t_idx], 'g-.', linewidth=2, label='Quad. Manifold', alpha=0.8)
    # Plot LassoNet reconstruction
    ax.plot(xspan, X_lassonet_recon[:, t_idx], 'r:', linewidth=2, label='LassoNet', alpha=0.8)

    ax.set_xlabel('Space (x)', fontsize=14)
    ax.set_ylabel('u(x,t)', fontsize=14)
    ax.set_title(f't = {t_val:.3f}', fontsize=16)
    ax.grid(True, alpha=0.3)
    if i == 0:
        ax.legend(fontsize=16)

plt.tight_layout()
plt.suptitle('KSE Flow Profiles at Different Time Points', fontsize=19, y=1.02)
plt.savefig('figures/lassonet/kse/wave_profiles_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

# Print reconstruction error statistics
print(f"\nReconstruction Error Statistics:")
print(f"POD Error (Frobenius norm): {np.linalg.norm(pod_error):.6e}")
print(f"Quadratic Manifold Error: {np.linalg.norm(qm_error):.6e}")  
print(f"LassoNet Error: {np.linalg.norm(lassonet_error):.6e}")

print(f"\nRelative Errors (normalized by original data norm):")
data_norm = np.linalg.norm(X_original)
print(f"POD Relative Error: {np.linalg.norm(pod_error)/data_norm:.6e}")
print(f"Quadratic Manifold Relative Error: {np.linalg.norm(qm_error)/data_norm:.6e}")
print(f"LassoNet Relative Error: {np.linalg.norm(lassonet_error)/data_norm:.6e}")
