#%%
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add the src and example directory to the path
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from examples.pulse import generate_advecting_pulse
from src.sparsemodesnet.dataset import PODReconDataset
from experiments.QM.quadmani import quadmani_greedy


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


class SparseModesNet(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, 
                 hidden_units: list,
                 lam: float, M: float, alpha: float,
                 gamma: float, dtype: torch.dtype = torch.float32):
        super(SparseModesNet, self).__init__()
        
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

        # FF neural network
        assert hidden_units[0] == self.s 
        self.first_layer = MaskedLayer(
            self.s, hidden_units[0], torch.eye(self.s), dtype=dtype)
        self.first_layer.weight.data.fill_(0.1)  # Initialize to ones
        layers = [self.first_layer, nn.SELU(inplace=True)]
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=False))
            layers.append(nn.SELU(inplace=True))
            # layers.append(nn.Dropout(p=0.1)) 
        layers.append(nn.Linear(hidden_units[-1], self.d, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, z_batch):
        z_hat = z_batch * self.omega.unsqueeze(0) 
        x_hat_lin = z_hat @ self.U_s.T                  
        x_hat_nn = self.net(z_hat)
        x_hat = x_hat_lin + x_hat_nn
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
 
    
def train_sparsemodesnet(model: SparseModesNet,
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
        optimizer, mode='min', factor=0.8, patience=10000,
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


class StateDecoder(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, hidden_units: list):
        super(StateDecoder, self).__init__()
        self.register_buffer('U_r', pod_basis)
        self.d, self.r = pod_basis.shape

        # FF neural network
        self.first_layer = nn.Linear(self.r, hidden_units[0], bias=False)
        layers = [self.first_layer, nn.SELU(inplace=True)]
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=False))
            layers.append(nn.SELU(inplace=True))
            # layers.append(nn.Dropout(p=0.1)) 
        layers.append(nn.Linear(hidden_units[-1], self.d, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, z_batch):
        x_hat_lin = z_batch @ self.U_r.T                  
        x_hat_nn = self.net(z_batch)
        x_hat = x_hat_lin + x_hat_nn
        return x_hat


def train_decoder(model: StateDecoder,
                  dataloader: DataLoader,
                  num_epochs: int,
                  lr: float,
                  momentum: float,
                  optimizer: str,
                  device: str):
    model.to(device)
    if optimizer == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, nesterov=True)
    else:
        raise ValueError("Unsupported optimizer. Use 'Adam' or 'SGD'.")
    mse_loss = nn.MSELoss()
    
    lr_schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100,
    )
    lr_new = optimizer.param_groups[0]['lr']
    history = {'loss': []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss.backward()
            optimizer.step()

            batch_size  = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            n_samples  += batch_size
            
        lr_schedule.step(loss)  # Update learning rate
        lr_new = optimizer.param_groups[0]['lr']

        epoch_loss /= n_samples
        history['loss'].append(epoch_loss)

        # Print every 10 epochs or first:
        if (epoch % 10 == 0) or (epoch == 1):
            print(f" Epoch {epoch:<4d} | lr={lr_new:.4e} | "
                  f"Recon MSE={epoch_loss:.6e}")
            
    return model, history

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


#%% #======================= LassoNet Mode Selection ==========================#
    print("\n" + "="*60)
    print("LASSO MODE SELECTION")
    print("="*60)

    NORMALIZE = True
    WHITENING = True

    # Shifted data
    X_mean = X.mean(axis=1, keepdims=True)
    X_center = X - X_mean

    if NORMALIZE and WHITENING:
        X_center_min, X_center_max = X_center.min(axis=1), X_center.max(axis=1)
        X_center_shift = X_center_min.reshape(-1, 1)
        X_center_scale = (X_center_max - X_center_min).reshape(-1, 1)
        X_center_norm = (X_center - X_center_shift) / X_center_scale
        zcaMat = zca_whitening_matrix(X_center_norm, epsilon=1e-4)
        X_proc = np.dot(zcaMat, X_center_norm)  # Apply ZCA whitening
        V_white, _, _ = np.linalg.svd(X_proc, full_matrices=False)
        V_white = V_white[:, :s_p]  
        V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
    elif NORMALIZE:
        # Normalize each row to [0,1]
        X_center_min, X_center_max = X_center.min(axis=1), X_center.max(axis=1)
        X_center_shift = X_center_min.reshape(-1, 1)
        X_center_scale = (X_center_max - X_center_min).reshape(-1, 1)
        X_center_norm = (X_center - X_center_shift) / X_center_scale
        X_proc = X_center_norm
        V_white = np.linalg.svd(X_proc, full_matrices=False)[0][:, :s_p] 
        V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
    else:
        if WHITENING:
            zcaMat = zca_whitening_matrix(X_center, epsilon=1e-4)
            X_white = np.dot(zcaMat, X_center)  # Apply ZCA whitening
        else:
            X_white = copy.deepcopy(X_center)

        # Compute the pod basis
        V_white, _, _ = np.linalg.svd(X_white, full_matrices=False)
        V_white = V_white[:, :s_p]  
        V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
        X_proc = copy.deepcopy(X_white)

    # Compute the reduced data
    Z_np = V_white.T @ X_proc  # (s, n)
    
    # Prep the data
    ds_sub = PODReconDataset(Z_np=Z_np, X_np=X_proc, type="float32")
    dl_sub = DataLoader(ds_sub, batch_size=200, shuffle=True)
    
    # Initialize the regularization parameter and increase factor
    lam = 5.0
    eps = 0.0005
    alpha = 1.0
    gamma = 1e-6
    threshold = 1e-8
    
    # Initialize the model 
    model = SparseModesNet(
        pod_basis=V_white_tensor, 
        hidden_units=[s_p, s_p*10, int(s_p*(s_p+1)/2)],
        dtype=torch.float32, 
        M=12.0, lam=lam, gamma=gamma, alpha=alpha,
    ) 

    # Define the count of the selected modes and omegas
    I_count = np.zeros(s_p, dtype=int)
    omegas = model.omega.detach().numpy().reshape(-1, 1)

    # Add tracking variables before the while loop
    prev_num_selected = None
    no_change_iterations = 0
    max_no_change = 50  # Number of consecutive iterations with no change before breaking

    while True:
        print(f"\nTraining with λ = {lam:.3e}")
        
        # Train the model
        omega_, history, flag = train_sparsemodesnet(
            model=model, dataloader=dl_sub, num_epochs=100, 
            lr=1e-3, momentum=0.95, optimizer='Adam', 
            device=device, rmax=r_max
        ) 
        
        selected_modes = np.where(omega_ > threshold)[0] 
        num_selected = len(selected_modes)
        
        if selected_modes.size == 0:
            print("No modes selected. End loop.")
            break
        else:
            print(f"Number of selected modes: {num_selected}")

        # Check for convergence based on number of selected modes
        if prev_num_selected is None:
            prev_num_selected = num_selected
            no_change_iterations = 0
        elif num_selected == prev_num_selected:
            no_change_iterations += 1
            print(f"Number of selected modes unchanged for {no_change_iterations} consecutive iterations.")
        else:
            prev_num_selected = num_selected
            no_change_iterations = 0

        # Increment the count of the selected modes and omegas
        I_count[selected_modes] += 1
        omegas = np.concatenate((omegas, omega_.reshape(-1, 1)), axis=1)

        if flag:
            break
            
        # Break if number of selected modes hasn't changed for several iterations
        if no_change_iterations >= max_no_change:
            print(f"Number of selected modes unchanged for {max_no_change} consecutive iterations. Assuming convergence.")
            break
        
        # Update lambda
        lam *= (1 + eps)
        model.lam = lam
    
    # Select the first largest r_max modes
    # I_nn = np.where(omegas[:, -1] > 0)[0][:rmax]
    I_nn = np.argsort(omegas[:, -1])[::-1][:r_max]


# %% #========================= Plot the omegas ===============================#
    print("\n" + "="*60)
    print("PLOTTING OMEGAS")
    print("="*60)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Omega evolution over lambda iterations
    for mode in range(s_p):
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
    # plt.savefig('figures/lassonet/wave/omega_evolution.png', dpi=200)
    plt.show()

    # Print statistics about omega values
    print(f"\nOmega Statistics:")
    print(f"Number of lambda iterations: {omegas.shape[1]}")
    print(f"Final non-zero omegas: {len(non_zero_modes)}")
    print(f"Max final omega: {np.max(final_omegas):.6e}")
    print(f"Min final omega: {np.min(final_omegas[final_omegas > 1e-13]):.6e}" 
          if len(non_zero_modes) > 0 else "No non-zero omegas")
    print(f"Final selected modes: {non_zero_modes}")


# %% #=============== Train Final Decoder using chosen modes ==================#
    print("\n" + "="*60)
    print("TRAINING FINAL DECODER")
    print("="*60)

    # Initialize the model 
    r_nn = len(I_nn)
    model = StateDecoder(
        pod_basis=V_white_tensor[:, list(I_nn)],
        hidden_units=[s_p, s_p*10, int(s_p*(s_p+1)/2)],
    ) 
    model.to(device)

    Z_nn = V_white[:, I_nn].T @ X_center_norm  # (r_nn, n)
    ds_sub = PODReconDataset(Z_np=Z_nn, X_np=X_center_norm, type="float32")
    dl_sub = DataLoader(ds_sub, batch_size=200, shuffle=True)

    # Train the model
    model, history = train_decoder(
        model=model, dataloader=dl_sub, num_epochs=10000, 
        lr=1e-3, momentum=0.95, optimizer='SGD', 
        device=device, 
    ) 
    

# %% #==================== Compute Reconstruction Errors ======================#
    # Compute the reconstruction error (LassoNet)
    V_nn = V_white[:, I_nn]  
    if NORMALIZE and WHITENING:
        Z_nn = V_nn.T @ X_center_norm
        residual = X_center_norm - V_nn @ Z_nn
    elif NORMALIZE:
        Z_nn = V_nn.T @ X_proc
        residual = X_proc - V_nn @ Z_nn
    else:
        Z_nn = V_nn.T @ X_center
        residual = X_center - V_nn @ Z_nn
    Z_quad_nn = quadratic_mapping_numpy(Z_nn.T).T 
    W_nn_T, analytical_resid = lstsq_l2_numpy(
        Z_quad_nn.T, residual.T, reg_magnitude=1e-15
    )
    W_nn = W_nn_T.T
    if NORMALIZE:
        recon_error = np.linalg.norm(
            X - ((V_nn @ Z_nn + W_nn @ Z_quad_nn)*X_center_scale + X_center_shift) 
            - X_mean, ord='fro')
    else:
        recon_error = np.linalg.norm(
            X - V_nn @ Z_nn - W_nn @ Z_quad_nn - X_mean, ord='fro')
    rel_recon_error_nn = recon_error / np.linalg.norm(X, ord='fro') 
    
    # Print results
    print(f"\nReconstruction errors:")
    print(f"LassoNet: ||X - V_nn @ Z - W_nn @ Z_quad||_F = {recon_error:.6e}")
    print(f"Relative error: {rel_recon_error_nn:.6e}")