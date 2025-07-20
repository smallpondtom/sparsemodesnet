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
            bias=False,      # no need to use a bias in our case
        )
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
        return x

class QuadraticManifold(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, 
                 lam: float, M: float, alpha: float,
                 gamma: float):
        super(QuadraticManifold, self).__init__()
        
        # Ensure everything is double precision
        pod_basis = pod_basis.double()
        self.register_buffer('U_s', pod_basis)  # (d, r)
        self.d, self.s = pod_basis.shape
        self.M = float(M)
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
        
        # Skip‐weights ω ∈ R^s
        self.omega = nn.Parameter(torch.ones(self.s, dtype=torch.float64))
        
        # NOTE: 
        # First layer (used in proximal step) as a gate for selected modes.
        # To make this identical to the quadratic manifold we have to mask
        # the first layer weights to be the identity matrix so that purely
        # the selected modes are passed to the quadratic mapping.
        self.first_layer = MaskedLayer(
            self.s, self.s, torch.eye(self.s), dtype=torch.float64)
        self.first_layer.weight.data.fill_(0.5)  # Initialize to ones
        
    
    def forward(self, z_batch, x_batch):
        z_batch = z_batch.double()  # Ensure double precision
        z_hat = z_batch * self.omega.unsqueeze(0) 
        
        # Reconstruct the linear part via projection
        x_hat_lin = z_hat @ self.U_s.T                        
        
        h = self.first_layer(z_hat)  
        # z_quad = quadratic_mapping_torch(h) 
        
        # with torch.no_grad():
        #     # Compute the residual for the quadratic part
        #     residual = x_batch - x_hat_lin 
        #     W, _ = self.lstsq_l2_torch(
        #         z_quad, residual
        #     ) 
            
        # x_hat_quad = z_quad @ W
        # x_hat = x_hat_lin + x_hat_quad
        x_hat = x_hat_lin

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


#%% #================================ __main__ ================================#
if __name__ == "__main__":
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    device = 'cpu'
    print("Using device:", device)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    
#%% #=================== Main Experiment with Synthetic Data ==================#
    # Parameters
    d = 2**10
    n = 1200
    s = 100
    
    print("\n" + "="*60)
    print("GENERATING DATA")
    print("="*60)

    # Generate synthetic data
    X = np.random.rand(d, n)
    print(f"X shape: {X.shape}")
    print(f"X dtype: {X.dtype}")
    
#%% #============== Generate the Modes while Adding Dummy Modes ===============#
    print("\n" + "="*60)
    print("SELECT ONLY A FEW MODES AND RECONSTRUCTING DATA")
    print("="*60)

    n_total = 100
    n_mode = 5
    V = np.linalg.svd(X, full_matrices=False)[0][:, :n_total]
    selected_modes = np.random.choice(n_total, n_mode, replace=False)
    V_mode = V[:, selected_modes]
    X = V_mode @ V_mode.T @ X  # Reconstruct X using the modes
    print(f"V shape: {V.shape}")
    print(f"Number of used modes: {n_mode}, Number of total modes: {n_total}")
    print(f"Selected modes indices: {selected_modes}")

#%% #======================= LassoNet Mode Selection ==========================#
    print("\n" + "="*60)
    print("LASSO MODE SELECTION")
    print("="*60)
    
    # Define the count of the selected modes and omegas
    I_count = np.zeros(s, dtype=int)
    omegas = np.zeros(s, dtype=np.float64).reshape(-1, 1)
    
    # Compute the pod basis
    V_tensor = torch.from_numpy(V.astype(np.float64)).to(device)
    
    # Compute the reduced data
    Z_np = V.T @ X  # (s, n)
    
    # Prep the data
    ds_sub = PODReconDataset(Z_np=Z_np, X_np=X, type="float64")
    dl_sub = DataLoader(ds_sub, batch_size=300, shuffle=True)
    
    # Initialize the regularization parameter and increase factor
    lam = 10.0
    eps = 0.05
    alpha = 1.0
    # Threshold
    threshold = 1e-6
    
    # Initialize the model 
    model = QuadraticManifold(
        pod_basis=V_tensor, 
        M=10.0, lam=lam, gamma=0.0, alpha=alpha
    ) 

    all_histories = []

    while True:
        print(f"\nTraining with λ = {lam:.3e}")
        
        # Train the model
        omega_, history, flag = train_quadraticmanifold(
            model=model, dataloader=dl_sub, num_epochs=100, 
            lr=1e-3, momentum=0.95, optimizer='Adam', 
            device=device, rmax=n_mode
        ) 
        all_histories.append(history)
        
        selected_modes = np.where(omega_ > threshold)[0] 

        # Increment the count of the selected modes and omegas
        I_count[selected_modes] += 1
        omegas = np.concatenate((omegas, omega_.reshape(-1, 1)), axis=1)

        if flag:
            break
        
        # Update lambda
        lam *= (1 + eps)
        model.lam = lam

# %% #========================= Verify the Selected Modes =====================#
    # Sort the indices of the selected modes by their omega values
    sort_idx = np.argsort(-I_count, kind="mergesort")
    I_guess = sort_idx[0:n_mode]

    cnt = 0
    for idx in I_guess:
        V_guess_i = V[:, idx]
        for j in range(n_mode):
            V_j = V_mode[:, j]
            if np.allclose(V_guess_i, V_j, atol=1e-14):
                print(f"Selected mode {idx} matches mode {j} in V_mode.")
                cnt += 1
    if cnt == n_mode:
        print(f"All {n_mode} modes were selected correctly.")


# %% #==================== Compute Reconstruction Errors ======================#
    # Basis from LassoNet
    V_nn = V[:, I_guess]  # (d, r_max)
    # Compute the reconstruction error (Quadratic Manifold)
    recon_answer = np.linalg.norm(X - V_mode @ V_mode.T @ X, 'fro')
    recon_answer_rel = recon_answer / np.linalg.norm(X, 'fro')
    recon_lassonet = np.linalg.norm(X - V_nn @ V_nn.T @ X, 'fro')
    recon_lassonet_rel = recon_lassonet / np.linalg.norm(X, 'fro')
    
    # Print results
    print(f"\nReconstruction errors:")
    print(f"Answer : ||X - V @ Z - W @ Z_quad||_F = {recon_answer:.6e}")
    print(f"Relative error: {recon_answer_rel:.6e}")
    print(f"LassoNet: ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_lassonet:.6e}")
    print(f"Relative error: {recon_lassonet_rel:.6e}")

# %% #========================= Plot the omegas ===============================#
    print("\n" + "="*60)
    print("PLOTTING OMEGAS")
    print("="*60)

    threshold = 1e-8

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Omega evolution over lambda iterations
    for mode in range(s):
        if mode in I_guess:
            ax1.plot(np.abs(omegas[mode, :]), linewidth=2, 
                     label=f'Mode {mode+1}', color='orange')
        else:
            ax1.plot(np.abs(omegas[mode, :]), linewidth=1, alpha=0.5, 
                     color='darkblue', linestyle='--')

    # ax1.plot(omegas.T, alpha=0.7, linewidth=1)
    ax1.set_xlabel('Lambda iteration')
    ax1.set_ylabel('Omega values')
    ax1.set_title('Evolution of Omega Values During Training')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot 2: Final omega values
    final_omegas = np.abs(omegas[:, -1])
    mode_indices = np.arange(len(final_omegas))
    bars = ax2.bar(mode_indices, final_omegas, alpha=0.7, color='darkblue')

    # Highlight selected modes
    for mode in I_guess:
        bars[mode].set_color('orange')

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
    plt.savefig('figures/lassonet/linear/select_from_modes/omega_evolution.png', dpi=200)
    plt.show()
    
# %% #======================= Enhanced Visualizations ========================#
print("\n" + "="*60)
print("ENHANCED VISUALIZATIONS")
print("="*60)

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)

# 1. Omega Evolution with Clear Mode Tracking
ax1 = fig.add_subplot(gs[0, :2])
threshold = 1e-8

# Plot background modes in gray
for mode in range(s):
    if mode not in I_guess:
        ax1.plot(np.abs(omegas[mode, :]), linewidth=0.8, alpha=0.3, 
                color='gray', linestyle='-')

# Plot true modes in distinct colors
colors = plt.cm.Set1(np.linspace(0, 1, len(I_guess)))
for idx, mode in enumerate(I_guess):
    ax1.plot(np.abs(omegas[mode, :]), linewidth=2.5, 
             label=f'True Mode {mode+1}', color=colors[idx])

ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
           label=f'Threshold ({threshold:.0e})')
ax1.set_xlabel('Lambda Iteration', fontsize=12)
ax1.set_ylabel('|ω| Values', fontsize=12)
ax1.set_title('Omega Evolution: True Modes vs Others', fontsize=14, fontweight='bold')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
# ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. Final Omega Values with Enhanced Highlighting
ax2 = fig.add_subplot(gs[0, 2:])
mode_indices = np.arange(len(final_omegas))

# Create color map for different categories
colors_final = ['lightgray'] * len(final_omegas)
for mode in I_guess:
    colors_final[mode] = 'green'  # True modes
for mode in selected_modes:
    if mode not in I_guess:
        colors_final[mode] = 'orange'  # False positives
for mode in I_guess:
    if mode in selected_modes:
        colors_final[mode] = 'darkgreen'  # True positives

bars = ax2.bar(mode_indices, final_omegas, color=colors_final, alpha=0.8)
ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7)
ax2.set_xlabel('Mode Index', fontsize=12)
ax2.set_ylabel('Final |ω| Value', fontsize=12)
ax2.set_title('Final Mode Selection Results', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)

# Add legend for colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='darkgreen', label='True Positive'),
    Patch(facecolor='orange', label='False Positive'), 
    Patch(facecolor='green', label='True Mode (missed)'),
    Patch(facecolor='lightgray', label='Inactive Mode')
]
ax2.legend(handles=legend_elements, loc='upper right')

# 3. Confusion Matrix Style Visualization
ax3 = fig.add_subplot(gs[1, :2])
confusion_data = np.zeros((2, 2))
true_positives = len(np.intersect1d(I_guess, selected_modes))
false_positives = len([m for m in selected_modes if m not in I_guess])
false_negatives = len([m for m in I_guess if m not in selected_modes])
true_negatives = s - len(I_guess) - false_positives

confusion_data[0, 0] = true_positives    # TP
confusion_data[0, 1] = false_positives   # FP
confusion_data[1, 0] = false_negatives   # FN
confusion_data[1, 1] = true_negatives    # TN

im = ax3.imshow(confusion_data, cmap='Blues', alpha=0.8)
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Selected', 'Not Selected'], fontsize=12)
ax3.set_yticklabels(['True Mode', 'False Mode'], fontsize=12)
ax3.set_xlabel('LassoNet Prediction', fontsize=12)
ax3.set_ylabel('Ground Truth', fontsize=12)
ax3.set_title('Mode Selection Confusion Matrix', fontsize=14, fontweight='bold')

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, int(confusion_data[i, j]), 
                       ha="center", va="center", color="black", fontsize=16, fontweight='bold')

# 4. Selection Count Heatmap
ax4 = fig.add_subplot(gs[1, 2:])
count_matrix = I_count.reshape(10, 10)
im4 = ax4.imshow(count_matrix, cmap='viridis', alpha=0.8)
ax4.set_title('Mode Selection Frequency (10×10 Grid)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Mode Index (j)', fontsize=12)
ax4.set_ylabel('Mode Index (i)', fontsize=12)

# Add text annotations for non-zero counts
for i in range(10):
    for j in range(10):
        mode_idx = i * 10 + j
        if mode_idx < len(I_count) and I_count[mode_idx] > 0:
            color = 'white' if count_matrix[i, j] > np.max(count_matrix)/2 else 'black'
            ax4.text(j, i, f'{I_count[mode_idx]}', ha="center", va="center", 
                    color=color, fontsize=10, fontweight='bold')

plt.colorbar(im4, ax=ax4, label='Selection Count')

# 5. Performance Metrics Over Iterations
ax5 = fig.add_subplot(gs[2, :2])
n_iterations = omegas.shape[1]
precision_history = []
recall_history = []
f1_history = []

for iter_idx in range(n_iterations):
    omega_iter = omegas[:, iter_idx]
    selected_iter = np.where(np.abs(omega_iter) > threshold)[0]
    
    tp = len(np.intersect1d(I_guess, selected_iter))
    fp = len([m for m in selected_iter if m not in I_guess])
    fn = len([m for m in I_guess if m not in selected_iter])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    precision_history.append(precision)
    recall_history.append(recall)
    f1_history.append(f1)

ax5.plot(precision_history, label='Precision', linewidth=2, color='blue')
ax5.plot(recall_history, label='Recall', linewidth=2, color='green')
ax5.plot(f1_history, label='F1-Score', linewidth=2, color='red')
ax5.set_xlabel('Lambda Iteration', fontsize=12)
ax5.set_ylabel('Metric Value', fontsize=12)
ax5.set_title('Performance Metrics Evolution', fontsize=14, fontweight='bold')
ax5.set_ylim(0, 1.05)
ax5.grid(True, alpha=0.3)
ax5.legend()

# 6. Mode Importance Ranking
ax6 = fig.add_subplot(gs[2, 2:])
final_omega_abs = np.abs(final_omegas)
sorted_indices = np.argsort(final_omega_abs)[::-1]
top_modes = sorted_indices[:20]  # Top 20 modes

colors_rank = ['green' if mode in I_guess else 'orange' if mode in selected_modes else 'gray' 
               for mode in top_modes]

bars6 = ax6.bar(range(len(top_modes)), final_omega_abs[top_modes], 
                color=colors_rank, alpha=0.8)
ax6.set_xlabel('Rank', fontsize=12)
ax6.set_ylabel('Final |ω| Value', fontsize=12)
ax6.set_title('Top 20 Modes by Final |ω| Value', fontsize=14, fontweight='bold')
ax6.set_yscale('log')
ax6.grid(True, alpha=0.3)

# Add mode numbers as x-tick labels
ax6.set_xticks(range(len(top_modes)))
ax6.set_xticklabels([f'{mode+1}' for mode in top_modes], rotation=45)

# 7. Algorithm Convergence Analysis
ax7 = fig.add_subplot(gs[3, :])
n_selected_history = []
for iter_idx in range(n_iterations):
    omega_iter = omegas[:, iter_idx]
    n_selected = np.sum(np.abs(omega_iter) > threshold)
    n_selected_history.append(n_selected)

ax7.plot(n_selected_history, linewidth=2, color='purple', marker='o', markersize=4)
ax7.axhline(y=len(I_guess), color='green', linestyle='--', alpha=0.7,
           label=f'True count ({len(I_guess)} modes)')
ax7.set_xlabel('Lambda Iteration', fontsize=12)
ax7.set_ylabel('Number of Selected Modes', fontsize=12)
ax7.set_title('Algorithm Convergence: Selected Mode Count vs Target', fontsize=14, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.legend()

plt.suptitle('LassoNet Mode Selection Analysis', fontsize=18, fontweight='bold', y=0.98)
plt.savefig('figures/lassonet/linear/select_from_modes/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Print comprehensive summary
print(f"\n{'='*60}")
print("COMPREHENSIVE ANALYSIS SUMMARY")
print(f"{'='*60}")

print(f"\nMode Selection Performance:")
print(f"  True Positives:  {true_positives:2d} (correctly identified)")
print(f"  False Positives: {false_positives:2d} (incorrectly selected)")
print(f"  False Negatives: {false_negatives:2d} (missed true modes)")
print(f"  True Negatives:  {true_negatives:2d} (correctly ignored)")

final_precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
final_recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
final_f1 = 2 * final_precision * final_recall / (final_precision + final_recall) if (final_precision + final_recall) > 0 else 0

print(f"\nFinal Metrics:")
print(f"  Precision: {final_precision:.3f}")
print(f"  Recall:    {final_recall:.3f}")
print(f"  F1-Score:  {final_f1:.3f}")
print(f"  Accuracy:  {(true_positives + true_negatives) / s:.3f}")

print(f"\nConvention Details:")
print(f"  True modes:   {sorted(I_guess)}")
print(f"  Selected:     {sorted(selected_modes)}")
print(f"  Missed:       {sorted([m for m in I_guess if m not in selected_modes])}")
print(f"  False picks:  {sorted([m for m in selected_modes if m not in I_guess])}")


#%% #======================= Plot the singular values =========================#
print("\n" + "="*60)
print("PLOTTING SINGULAR VALUES")
print("="*60)

# Compute singular values for original data X
U_X, sigma_X, Vt_X = np.linalg.svd(X, full_matrices=False)

# Compute singular values for the selected modes reconstruction
X_selected = V_nn @ V_nn.T @ X
U_sel, sigma_sel, Vt_sel = np.linalg.svd(X_selected, full_matrices=False)

# Compute singular values for the true modes reconstruction
X_true = V_mode @ V_mode.T @ X
U_true, sigma_true, Vt_true = np.linalg.svd(X_true, full_matrices=False)

# Create plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Singular values comparison
ax1.semilogy(sigma_X[:50], 'b-', linewidth=2, label='Original Data')
ax1.semilogy(sigma_true[:50], 'g--', linewidth=2, label=f'True Modes ({n_mode})')
ax1.semilogy(sigma_sel[:50], 'r:', linewidth=2, label=f'Selected Modes ({len(I_guess)})')
ax1.set_xlabel('Index')
ax1.set_ylabel('Singular Value')
ax1.set_title('Singular Values Comparison (First 50)')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Cumulative energy content
cumulative_X = np.cumsum(sigma_X**2) / np.sum(sigma_X**2)
cumulative_true = np.cumsum(sigma_true**2) / np.sum(sigma_X**2)
cumulative_sel = np.cumsum(sigma_sel**2) / np.sum(sigma_X**2)

ax2.plot(cumulative_X[:50], 'b-', linewidth=2, label='Original Data')
ax2.plot(cumulative_true[:50], 'g--', linewidth=2, label=f'True Modes ({n_mode})')
ax2.plot(cumulative_sel[:50], 'r:', linewidth=2, label=f'Selected Modes ({len(I_guess)})')
ax2.set_xlabel('Index')
ax2.set_ylabel('Cumulative Energy Fraction')
ax2.set_title('Cumulative Energy Content (First 50)')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('figures/lassonet/linear/select_from_modes/singular_values.png', dpi=200, bbox_inches='tight')
plt.show()

# Print energy statistics
print(f"\nEnergy Statistics:")
print(f"  Original data energy: {np.sum(sigma_X**2):.3e}")
print(f"  True modes energy: {np.sum(sigma_true**2):.3e} ({np.sum(sigma_true**2)/np.sum(sigma_X**2)*100:.1f}%)")
print(f"  Selected modes energy: {np.sum(sigma_sel**2):.3e} ({np.sum(sigma_sel**2)/np.sum(sigma_X**2)*100:.1f}%)")


# %%
