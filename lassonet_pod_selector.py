"""
lassonet_pod_selector.py

LassoNet-POD Mode Selection code. Includes the method code and examples for 
Heat, Burgers, and Kuramoto-Sivashinsky equations.  We train LassoNet in 
POD-space (dimension s) but minimize reconstruction error in the original space:

   1/n ∑_{i=1}^n || x_i - V_s * (b ⊙ z_i + f_NN(z_i)) ||_2^2  +  λ ||b||_1.

Supports CUDA, MPS (Apple Silicon), or CPU.  Also supports three ways to pick λ:
   1) “path”  → warm-start path from λ₀ to large (as before)
   2) “cv”    → k-fold cross-validation to minimize held‐out MSE
   3) “stability” → Meinshausen‐Bühlmann stability‐selection
"""
#%%
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# <<< Add for reproducibility
torch.manual_seed(0)
np.random.seed(0)
#%% 
# ------------------------------------------------------------
# 1) Generate 1D PDE Data: Heat, Burgers, and Kuramoto–Sivashinsky
# ------------------------------------------------------------

def generate_heat_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0):
    """
    Simulate 1D heat equation u_t = alpha * u_xx on [0, x_max], t ∈ [0, t_max].
    Dirichlet BCs (u=0 at boundaries). Returns data ∈ R^{nx x nt}.
    """
    x = np.linspace(0, x_max, nx)
    dx = x[1] - x[0]
    dt = t_max / (nt - 1)
    if dt > dx**2 / (2 * alpha):
        print("Warning: dt may be too large for stability (heat eq).")

    u = np.exp(-((x - x_max/2)**2 * 50.0))
    snapshots = [u.copy()]

    for k in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[0:-2] - 2*u[1:-1] + u[2:])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        snapshots.append(u_new)
        u = u_new

    data = np.column_stack(snapshots)  # shape (nx, nt)
    return data, x, np.linspace(0, t_max, nt)


def generate_burgers_data(nx=100, nt=200, nu=0.01, x_max=1.0, t_max=0.5):
    """
    Simulate 1D viscous Burgers' equation u_t + u u_x = nu u_xx using FFT method.
    Uses spectral differentiation for high accuracy and stability.
    Returns data ∈ R^{nx x nt}.
    """
    
    # Spatial discretization
    dx = x_max / nx
    x = np.linspace(0, x_max, nx, endpoint=False)  # Periodic domain
    
    # Temporal discretization
    t = np.linspace(0, t_max, nt)
    
    # Wave number discretization for FFT
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    
    # Initial condition - smooth Gaussian-like profile
    u0 = np.exp(-((x - x_max/2)**2) / (2 * (x_max/10)**2))
    
    # Define the Burgers system using FFT for spatial derivatives
    def burgers_system(u, t, k, nu):
        # Compute spatial derivatives in Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * k * u_hat
        u_hat_xx = -k**2 * u_hat
        
        # Transform back to spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)
        
        # Burgers equation: u_t = nu*u_xx - u*u_x
        u_t = nu * u_xx - u * u_x
        return u_t.real
    
    # Solve the PDE system using adaptive ODE solver
    U = odeint(burgers_system, u0, t, args=(k, nu), mxstep=5000).T
    
    return U, x, t


def generate_kse_data(nx=256, nt=200, L=32*np.pi, t_max=150.0):
    """
    Simulate 1D Kuramoto-Sivashinsky equation u_t + u*u_x + u_xx + u_xxxx = 0 using ETDRK4.
    Uses Exponential Time Differencing Runge-Kutta 4th order for high accuracy and stability.
    Returns data ∈ R^{nx x nt}.
    """
    # Spatial grid setup (normalized to [0,1] then scaled)
    x = np.arange(1, nx+1) / nx
    x_scaled = x * L  # Scale to actual domain [0, L]
    
    # Time discretization
    h = t_max / (nt - 1)  # Time step
    
    # Initial condition
    u = np.cos(x/16) * (1 + np.sin(x/16))
    v = np.fft.fft(u)
    
    # Wave numbers
    k = np.concatenate((np.arange(0, nx//2), [0], np.arange(-nx//2+1, 0))) / 16
    
    # Linear operator for KS equation: L = k^2 - k^4
    L_op = k**2 - k**4
    
    # ETDRK4 coefficients
    E = np.exp(h * L_op)
    E_2 = np.exp(h * L_op / 2)
    
    # Contour integral parameters for ETDRK4 coefficients
    Mcnt = 16
    r = np.exp(1j * np.pi * (np.arange(1, Mcnt+1) - 0.5) / Mcnt)
    LR = h * np.outer(L_op, np.ones(Mcnt)) + np.outer(np.ones(nx), r)
    
    # ETDRK4 coefficients computed via contour integrals
    Q  = h * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    
    # Handle potential division by zero at k=0
    zero_idx = np.where(np.abs(L_op) < 1e-14)[0]
    if len(zero_idx) > 0:
        Q[zero_idx]  = h
        f1[zero_idx] = h
        f2[zero_idx] = h/2
        f3[zero_idx] = h
    
    # Nonlinear operator
    g = -0.5j * k
    
    # Storage for solution
    uu = np.zeros((nx, nt))
    uu[:, 0] = u
    
    # Time stepping with ETDRK4
    for n in range(1, nt):
        # Stage 1
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v))**2)
        a  = E_2 * v + Q * Nv
        
        # Stage 2
        Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2)
        b  = E_2 * v + Q * Na
        
        # Stage 3
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2)
        c  = E_2 * a + Q * (2*Nb - Nv)
        
        # Stage 4
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c))**2)
        
        # Final update
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        
        # Store solution
        u = np.real(np.fft.ifft(v))
        uu[:, n] = u
    
    # Time array
    t = np.linspace(0, t_max, nt)
    
    return uu, x_scaled, t


#%%
# ------------------------------------------------------------
# 2) Compute POD Basis via SVD
# ------------------------------------------------------------

def compute_pod_basis(X_np: np.ndarray, s: int = None):
    """
    Given X_np ∈ R^{d x n}, compute first s left singular vectors V_s ∈ R^{d x s}.
    If s is None, take s = min(d, n). Returns:
      V_s: (d, s), Sigma_s: (s,), Wt_s: (s, n).
    """
    U, Sigma, Vt = np.linalg.svd(X_np, full_matrices=False)
    d, n = X_np.shape
    r = min(d, n) if s is None else min(s, min(d, n))
    V_s     = U[:, :r].astype(np.float32)
    Sigma_s = Sigma[:r].astype(np.float32)
    Wt_s    = Vt[:r, :].astype(np.float32)
    return V_s, Sigma_s, Wt_s


#%%
# ------------------------------------------------------------
# 3) Dataset that returns (z_i, x_i)
# ------------------------------------------------------------

class PODReconDataset(Dataset):
    """
    Given:
      - Z_np ∈ R^{s x n} (POD coefficients = V_s^T X)
      - X_np ∈ R^{d x n} (original snapshots)
    Creates n samples; each sample i returns (z_i, x_i).
    """

    def __init__(self, Z_np: np.ndarray, X_np: np.ndarray):
        assert Z_np.ndim == 2 and X_np.ndim == 2
        s, n1 = Z_np.shape
        d, n2 = X_np.shape
        assert n1 == n2, "Mismatch in number of snapshots."
        # store row‐major so Dataset returns (z_i, x_i)
        self.Z = Z_np.T.copy().astype(np.float32)  # (n, s)
        self.X = X_np.T.copy().astype(np.float32)  # (n, d)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return self.Z[idx, :], self.X[idx, :]


#%%
# ------------------------------------------------------------
# 4) LassoNet Autoencoder in POD-Space with Reconstruction Loss
# ------------------------------------------------------------

class LassoNetAutoencoderPODRecon(nn.Module):
    """
    LassoNet in POD-space (R^s → R^s), but loss computed in original x-space:
      For each sample i:
        z_i = V_s^T x_i
        z_hat_i = b ⊙ z_i + f(z_i)
        x_hat_i = V_s z_hat_i
      Minimize ∑ ||x_i - x_hat_i||^2 + λ ||b||_1, subject to
        ||W^(1)[j,:]||_∞ ≤ M |b_j|  for all j ∈ {0..s-1}.
    """

    def __init__(self, pod_basis: torch.Tensor, input_dim: int, hidden_units: list,
                 M: float = 5.0, lam: float = 1e-3):
        """
        pod_basis:  V_s ∈ R^{d x s}  (torch.Tensor)
        input_dim:  s (POD dimension)
        hidden_units:  e.g. [64, 32]
        M: hierarchy multiplier
        lam: ℓ₁ penalty on b
        """
        super(LassoNetAutoencoderPODRecon, self).__init__()

        self.V_s = pod_basis           # (d, s)
        self.d, self.s = pod_basis.shape

        self.M = float(M)
        self.lam = float(lam)

        # Skip‐weights b ∈ R^s
        self.b = nn.Parameter(torch.zeros(self.s))

        # Build f_NN in POD-space
        self.first_layer = nn.Linear(self.s, hidden_units[0], bias=True)
        layers = [self.first_layer, nn.ReLU(inplace=True)]
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_units[-1], self.s, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, z_batch):
        """
        z_batch: (batch_size, s)
        Returns:
          z_hat_batch: (batch_size, s)
          x_hat_batch: (batch_size, d)
        """
        skip = z_batch * self.b.unsqueeze(0)   # (batch, s)
        nn_out = self.net(z_batch)             # (batch, s)
        z_hat = skip + nn_out                  # (batch, s)

        # Reconstruct to x-space: x_hat = V_s @ z_hat^T  → (d, batch) → transpose → (batch, d)
        x_hat_T = self.V_s.matmul(z_hat.t())   # (d, batch)
        x_hat = x_hat_T.t()                    # (batch, d)
        return z_hat, x_hat

    def l1_norm_b(self):
        """Return ℓ₁-norm of b."""
        return self.b.abs().sum()

    @staticmethod
    def _row_inf_norm(mat: torch.Tensor) -> torch.Tensor:
        """
        Given mat: (s, h), return length-s vector of rowwise l-infinity norms.
        """
        return mat.abs().max(dim=1)[0]

    def proximal_step(self):
        """
        Batched implementation of Algorithm 4 (Group-Hierarchical Proximal) with λ̄ = 0,
        corrected so that b_new = x_star * θ (no extra soft-threshold on b).
        """
        lam = self.lam
        M   = self.M

        # 1) Gather first‐layer weights W1 ∈ ℝ^{h×s}, then transpose → W1_T ∈ ℝ^{s×h}
        W1   = self.first_layer.weight.data           # (h, s)
        W1_T = W1.t().contiguous()                    # (s, h), call h=K

        s, K = W1_T.shape  # s = #features, K = width of first hidden layer

        # 2) Sort each row of |W1_T| in descending order (batched)
        u_abs_sorted, _ = W1_T.abs().sort(dim=1, descending=True)  # (s, K)

        # 3) Build partial sums a_s(m) = lam - M * sum_{i=1}^m u_abs_sorted[j,i-1]
        zeros_m     = torch.zeros((s, 1), device=W1_T.device, dtype=W1_T.dtype)  # (s,1)
        cumsum_vals = torch.cumsum(u_abs_sorted, dim=1)  # (s, K)
        a_s = lam - M * torch.cat([zeros_m, cumsum_vals], dim=1)  # (s, K+1)

        # 4) ‖v‖₂ = |θ|, shape (s,)
        theta_abs = self.b.data.abs()         # (s,)

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
        b_new = x_star * self.b.data     # (s,)

        # 13) Coordinate‐wise clip each row of W1_T to ±w_star[j]:
        W1_T_abs   = W1_T.abs()                       # (s, K)
        w_star_col = w_star.unsqueeze(1).expand(-1, K)  # (s, K)
        clipped_abs = torch.min(W1_T_abs, w_star_col)   # (s, K)
        W1_T_new   = W1_T.sign() * clipped_abs          # (s, K)

        # 14) Write back:
        self.b.data.copy_(b_new)               # (s,)
        W1_updated = W1_T_new.t().contiguous() # shape: (K, s) → transpose to (h, s)
        self.first_layer.weight.data.copy_(W1_updated)


#%%
# ------------------------------------------------------------
# 5) Train for B epochs at fixed λ (same as before)
# ------------------------------------------------------------

def train_lassonet_pod_recon(model: LassoNetAutoencoderPODRecon,
                             dataloader: DataLoader,
                             num_epochs: int,
                             lr: float,
                             device: str):
    """
    Train for exactly num_epochs at whatever model.lam currently is.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

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
            z_hat_batch, x_hat_batch = model(z_batch)  # (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss.backward()
            optimizer.step()

            model.proximal_step()

            batch_size = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1  += model.l1_norm_b().item() * batch_size
            n_samples += batch_size

        epoch_loss /= n_samples
        epoch_l1  /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_b'].append(epoch_l1)

        # Print every 20 epochs or first:
        if (epoch % 20 == 0) or (epoch == 1):
            print(f"  λ={model.lam:.3e} | Epoch {epoch:3d} | Recon MSE={epoch_loss:.6e} | ‖b‖₁={epoch_l1:.6e}")

    return history


#%%
# ------------------------------------------------------------
# 6a) Cross‐Validation to pick λ
# ------------------------------------------------------------

def select_lambda_cv(X_np: np.ndarray,
                     s: int,
                     hidden_units: list,
                     M: float,
                     lambdas: np.ndarray,
                     lr: float,
                     num_epochs_cv: int,
                     k_folds: int,
                     batch_size: int,
                     device: str):
    """
    Performs k‐fold CV over a grid of λ values. Returns the λ with lowest average val‐MSE.
    """
    print("\n=== Cross‐Validation λ‐Selection ===")
    d, n = X_np.shape
    # 1) Compute POD basis and coefficients Z
    V_s_np, _, _ = compute_pod_basis(X_np, s=s)      # (d, s)
    Z_np = V_s_np.T.dot(X_np)                        # (s, n)

    # 2) Prepare fold splits
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    best_lambda = None
    best_avg_err = np.inf

    for lam in lambdas:
        val_errors = []
        print(f" CV testing λ = {lam:.3e} ...")
        for fold_idx in range(k_folds):
            # train indices = all except this fold
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(k_folds) if i != fold_idx])

            # Build train+val datasets
            Z_all = Z_np.T  # (n, s)
            X_all = X_np.T  # (n, d)
            ds_train = PODReconDataset(Z_np=Z_all[train_idx].T, X_np=X_all[train_idx].T)
            ds_val   = PODReconDataset(Z_np=Z_all[val_idx].T,   X_np=X_all[val_idx].T)
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
            dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

            # Instantiate a fresh model at λ
            model_cv = LassoNetAutoencoderPODRecon(
                pod_basis    = torch.from_numpy(V_s_np.astype(np.float32)).to(device),
                input_dim    = s,
                hidden_units = hidden_units,
                M            = M,
                lam          = float(lam)
            ).to(device)

            # Train for num_epochs_cv
            train_lassonet_pod_recon(model_cv, dl_train, num_epochs_cv, lr, device)

            # Evaluate on val set
            model_cv.eval()
            mse_loss = nn.MSELoss(reduction='sum')
            total_err = 0.0
            total_samples = 0
            with torch.no_grad():
                for z_b, x_b in dl_val:
                    z_b = z_b.to(device)
                    x_b = x_b.to(device)
                    _, x_hat_b = model_cv(z_b)
                    total_err += mse_loss(x_hat_b, x_b).item()
                    total_samples += x_b.shape[0]
            val_mse = total_err / total_samples
            val_errors.append(val_mse)

        avg_err = np.mean(val_errors)
        print(f"avg val‐MSE = {avg_err:.6e}")
        if avg_err < best_avg_err:
            best_avg_err = avg_err
            best_lambda = lam

    print(f"\n→ CV‐selected λ = {best_lambda:.3e}, avg val‐MSE = {best_avg_err:.6e}\n")
    return best_lambda


#%%
# ------------------------------------------------------------
# 6b) Stability Selection to pick λ
# ------------------------------------------------------------

def select_lambda_stability(X_np: np.ndarray,
                            s: int,
                            hidden_units: list,
                            M: float,
                            lambdas: np.ndarray,
                            B: int,
                            pi_thresh: float,
                            lr: float,
                            num_epochs_sub: int,
                            batch_size: int,
                            device: str):
    """
    Performs Meinshausen–Bühlmann stability selection over a grid of λ.
    Returns the first λ for which no features exceed pi_thresh frequency, 
    or the λ that yields <= a target #features. Here we choose λ s.t. most features 
    drop out. The user can inspect the returned freq table for details.
    """
    print("\n=== Stability Selection λ‐Selection ===")
    d, n = X_np.shape
    V_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = V_s_np.T.dot(X_np)                     # (s, n)

    # We store a list of “selection frequencies” for each λ
    freq_table = []  # list of (lam, freq_vector, stable_count)

    for lam in lambdas:
        counts = np.zeros(s, dtype=int)
        print(f" SS testing λ = {lam:.3e} ...")

        for b in range(B):
            # Random half‐sample of indices
            subsamp = np.random.choice(n, size=n//2, replace=False)
            ds_sub = PODReconDataset(Z_np=Z_np[:, subsamp], X_np=X_np[:, subsamp])
            dl_sub = DataLoader(ds_sub, batch_size=batch_size, shuffle=True, drop_last=False)

            # Train on that subsample
            model_ss = LassoNetAutoencoderPODRecon(
                pod_basis    = torch.from_numpy(V_s_np.astype(np.float32)).to(device),
                input_dim    = s,
                hidden_units = hidden_units,
                M            = M,
                lam          = float(lam)
            ).to(device)

            train_lassonet_pod_recon(model_ss, dl_sub, num_epochs_sub, lr, device)

            # Record which features are nonzero in b
            b_opt = model_ss.b.detach().cpu().numpy()
            counts += (np.abs(b_opt) > 1e-8).astype(int)

        freqs = counts / float(B)             # selection freq per feature
        stable_count = np.sum(freqs >= pi_thresh)
        freq_table.append((lam, freqs.copy(), int(stable_count)))
        print(f"stable_count = {stable_count}", end="  ")
        print(f"(#features with freq ≥ {pi_thresh} = {stable_count})")

        # If no feature is “stable,” we can stop early
        if stable_count == 0:
            print("All features dropped out at this λ; stopping SS path.\n")
            break

    # Choose the smallest λ for which stable_count = 0, or else last λ in list
    for (lam, freqs, sc) in freq_table:
        if sc == 0:
            print(f"→ SS‐selected λ = {lam:.3e} (no stable features remain)\n")
            return lam, freq_table

    # If none hit zero, return the last λ
    lam_last, freqs_last, sc_last = freq_table[-1]
    print(f"→ SS‐selected λ = {lam_last:.3e} (end of grid, stable_count={sc_last})\n")
    return lam_last, freq_table


#%%
# ------------------------------------------------------------
# 7) Combined “train‐&‐select‐λ” driver
# ------------------------------------------------------------

def run_lassonet_pod_recon_with_lambda_selection(X_np: np.ndarray,
                                                 s: int,
                                                 hidden_units: list,
                                                 M: float,
                                                 lambda_method: str,
                                                 # for “path”:
                                                 lam0: float = 1e-6,
                                                 epsilon: float = 0.1,
                                                 B_path: int = 20,
                                                 max_iters: int = 100,
                                                 # for “cv”:
                                                 lambdas_cv: np.ndarray = None,
                                                 k_folds: int = 5,
                                                 num_epochs_cv: int = 20,
                                                 # for “stability”:
                                                 lambdas_ss: np.ndarray = None,
                                                 B_ss: int = 50,
                                                 pi_thresh: float = 0.6,
                                                 num_epochs_sub: int = 20,
                                                 # common:
                                                 lr: float = 1e-3,
                                                 batch_size: int = 16,
                                                 device: str = 'cpu',
                                                 label: str = ''):
    """
    Runs LassoNet-POD-Recon but first picks λ via one of three methods:
      • lambda_method='path'      → warm‐start path (the original behavior)
      • lambda_method='cv'        → k‐fold CV over a list of lambdas_cv
      • lambda_method='stability' → stability selection over lambdas_ss

    After λ is chosen, we train a final model on the full data for ‘final_epochs’ 
    (here we reuse B_path for illustration; you can change it).
    """
    print(f"\n=== LassoNet-POD (λ‐selection={lambda_method}) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    # 1) Compute POD basis and Z
    V_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = V_s_np.T.dot(X_np)                     # (s, n)

    # Convert pod_basis to tensor once
    V_s_tensor = torch.from_numpy(V_s_np.astype(np.float32)).to(device)

    # 2) Decide λ according to method
    if lambda_method == 'cv':
        assert lambdas_cv is not None, "Must pass a grid 'lambdas_cv' for CV."
        lam_star = select_lambda_cv(
            X_np           = X_np,
            s              = s,
            hidden_units   = hidden_units,
            M              = M,
            lambdas        = lambdas_cv,
            lr             = lr,
            num_epochs_cv  = num_epochs_cv,
            k_folds        = k_folds,
            batch_size     = batch_size,
            device         = device
        )

    elif lambda_method == 'stability':
        assert lambdas_ss is not None, "Must pass a grid 'lambdas_ss' for stability selection."
        lam_star, freq_table = select_lambda_stability(
            X_np           = X_np,
            s              = s,
            hidden_units   = hidden_units,
            M              = M,
            lambdas        = lambdas_ss,
            B              = B_ss,
            pi_thresh      = pi_thresh,
            lr             = lr,
            num_epochs_sub = num_epochs_sub,
            batch_size     = batch_size,
            device         = device
        )

    else:  # 'path' (warm‐start) 
        # We simply run the path-based routine from before
        lam_star = None

    # 3) If path‐method: use the old “run along λ‐path until b=0”
    if lambda_method == 'path':
        # We call the old `run_lassonet_pod_recon` logic (warm‐start) directly.
        # Note: That function already does its own internal λ-loop from lam0 to big.
        model_final, history_path, selected_indices = run_lassonet_pod_recon_path(
            X_np         = X_np,
            s            = s,
            hidden_units = hidden_units,
            M            = M,
            lam0         = lam0,
            epsilon      = epsilon,
            lr           = lr,
            B            = B_path,
            max_iters    = max_iters,
            batch_size   = batch_size,
            device       = device,
            label        = label
        )
        return model_final, {'path_history': history_path}, selected_indices

    # 4) Otherwise (CV or stability), we now train a final LassoNet on the **entire** data with λ=lam_star
    print(f"\n→ Final training on full data with λ = {lam_star:.3e} ...")
    # Build full dataset
    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    model_final = LassoNetAutoencoderPODRecon(
        pod_basis    = V_s_tensor,
        input_dim    = s,
        hidden_units = hidden_units,
        M            = M,
        lam          = float(lam_star)
    ).to(device)

    # Train for B_path epochs on full data
    history_full = train_lassonet_pod_recon(
        model      = model_final,
        dataloader = dataloader_full,
        num_epochs = B_path,
        lr         = lr,
        device     = device
    )

    # 5) After final training, record selected indices and final error
    b_opt = model_final.b.detach().cpu().numpy()
    selected_indices = np.where(np.abs(b_opt) > 1e-6)[0]
    print(f"\nFinal skip‐weights b: {b_opt.tolist()[:10]} ...")
    print(f"Selected POD‐mode indices (b_j ≠ 0): {selected_indices.tolist()}  "
          f"(count = {len(selected_indices)} / {s})")

    # Compute final reconstruction error
    model_final.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
        _, x_hat_tensor = model_final(Z_tensor)                            # (n, d)
        X_hat_np = x_hat_tensor.cpu().numpy().T                            # (d, n)
    frob_error     = np.linalg.norm(X_np - X_hat_np, 'fro')
    rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
    mse_per_sample = frob_error / X_np.shape[1]
    print(f"Final relative reconstruction ||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
    print(f"Final MSE per sample = {mse_per_sample:.6e}")

    return model_final, {'history_full': history_full, 'lambda_star': lam_star}, selected_indices


#%%
# ------------------------------------------------------------
# 7a) Original “path‐based” routine (unchanged from before)
# ------------------------------------------------------------

def run_lassonet_pod_recon_path(X_np: np.ndarray,
                                s: int,
                                hidden_units: list,
                                M: float,
                                lam0: float,
                                epsilon: float,
                                lr: float,
                                B: int,
                                max_iters: int,
                                batch_size: int,
                                device: str,
                                label: str):
    """
    The original warm-start λ→(1+ε)λ routine that stops when b=0.
    Exactly the same code we provided earlier in step (3).
    """
    print(f"\n=== LassoNet-POD (path) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    V_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = V_s_np.T.dot(X_np)                     # (s, n)

    V_s_tensor = torch.from_numpy(V_s_np.astype(np.float32)).to(device)
    Z_tensor   = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
    X_torch    = torch.from_numpy(X_np.astype(np.float32)).to(device)    # (d, n)

    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    model = LassoNetAutoencoderPODRecon(
        pod_basis    = V_s_tensor,
        input_dim    = s,
        hidden_units = hidden_units,
        M            = M,
        lam          = lam0
    ).to(device)

    lam = lam0
    prev_nonzero = s
    path_history = []
    iter_count = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e}  (‖b‖₀ prev = {prev_nonzero})")
        model.lam = float(lam)

        history = train_lassonet_pod_recon(model, dataloader_full, B, lr, device)
        b_opt = model.b.detach().cpu().numpy()
        nonzero_idxs = np.where(np.abs(b_opt) > 1e-8)[0]
        curr_nonzero = len(nonzero_idxs)

        model.eval()
        with torch.no_grad():
            _, x_hat_tensor = model(Z_tensor)
            X_hat_np = x_hat_tensor.cpu().numpy().T
        frob_error     = np.linalg.norm(X_np - X_hat_np, 'fro')
        rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')

        path_history.append({
            'lambda': lam,
            'nonzero_count': curr_nonzero,
            'selected_idxs': nonzero_idxs.copy(),
            'rel_error': rel_frob_error
        })

        print(f"  → at λ={lam:.3e}:  nonzero={curr_nonzero}, rel_err={rel_frob_error:.6e}")

        if curr_nonzero == 0:
            print("All skip‐weights have zeroed out. Stopping path.\n")
            break

        lam = lam * (1.0 + epsilon)
        prev_nonzero = curr_nonzero

        if iter_count >= max_iters:
            print(f"Reached max_iters={max_iters} on λ‐path; stopping early.\n")
            break

    b_opt_final = model.b.detach().cpu().numpy()
    selected_indices = np.where(np.abs(b_opt_final) > 1e-6)[0]
    return model, path_history, selected_indices


#%%
# ------------------------------------------------------------
# 8) Entry Point: Parse args & run chosen method
# ------------------------------------------------------------

if __name__ == "__main__":
    # Device selection: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)
    
    # Regularization parameter selection method
    lambda_method = 'cv'  # 'path', 'cv', or 'stability'

    # Common hyperparameters
    hidden_units_heat = [256, 128, 64, 32]
    hidden_units_burg = [128, 64, 32]
    hidden_units_ks   = [256, 128, 64, 32]

    # Parameter‐grid for CV or SS (you can customize)
    lambdas_cv = np.logspace(-6, -2, 10)    # 10 values from 1e-6 to 1e-2
    lambdas_ss = np.logspace(-6, 0, 12)     # 12 values from 1e-6 to 1e0
    
    # Sanity check flag (plotting)
    sanity_check = False

    # ---------- Heat Equation ----------
    X_heat, xspan_h, tspan_h = generate_heat_data(nx=2**7, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    
    # Create 3D surface plot for Heat Equation (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan_h, tspan_h)
        Z_mesh = X_heat.T  # Transpose to match meshgrid dimensions
        surf = ax.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u(x,t)')
        ax.set_title('Heat Equation Solution')
        plt.colorbar(surf, shrink=0.5, aspect=5)
        plt.savefig('figures/heat_data.png', dpi=300)
        # plt.show()
        plt.close(fig)

    model_heat, info_heat, selected_h = run_lassonet_pod_recon_with_lambda_selection(
        X_np            = X_heat,
        s               = s_h,
        hidden_units    = hidden_units_heat,
        M               = 0.1,
        lambda_method   = lambda_method,
        lam0            = 1e-6,         # only used if path
        epsilon         = 0.10,         # only used if path
        B_path          = 20,           # epochs per λ for path or final fit
        max_iters       = 100,          # max iterations for path
        lambdas_cv      = lambdas_cv,   # only used if cv
        k_folds         = 5,            # for cv
        num_epochs_cv   = 20,           # for cv
        lambdas_ss      = lambdas_ss,   # only used if stability
        B_ss            = 2,           # subsamples per λ for stability
        pi_thresh       = 0.6,          # threshold for stability
        num_epochs_sub  = 20,           # epochs per subsample for stability
        lr              = 1e-3,
        batch_size      = 16,
        device          = device,
        label           = "Heat Equation"
    )

    # # ---------- Burgers' Equation ----------
    # X_burgers, xspan_b, tspan_b = generate_burgers_data(nx=2**7, nt=1000, nu=0.01, x_max=1.0, t_max=1.0)
    # d_b, n_b = X_burgers.shape
    # s_b = min(d_b, n_b)
    
    # # Create 3D surface plot for Burgers' Equation (sanity check)
    # if sanity_check:
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     X_mesh, T_mesh = np.meshgrid(xspan_b, tspan_b)
    #     Z_mesh = X_burgers.T  # Transpose to match meshgrid dimensions
    #     surf = ax.plot_surface(X_mesh, T_mesh, Z_mesh, cmap='viridis', alpha=0.8)
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('t')
    #     ax.set_zlabel('u(x,t)')
    #     ax.set_title("Burgers' Equation Solution")
    #     plt.colorbar(surf, shrink=0.5, aspect=5)
    #     plt.savefig('figures/burgers_data.png', dpi=300)
    #     # plt.show()
    #     plt.close(fig)

    # model_burg, info_burg, selected_b = run_lassonet_pod_recon_with_lambda_selection(
    #     X_np            = X_burgers,
    #     s               = s_b,
    #     hidden_units    = hidden_units_burg,
    #     M               = 10.0,
    #     lambda_method   = lambda_method,
    #     lam0            = 1e-6,
    #     epsilon         = 0.10,
    #     B_path          = 20,
    #     max_iters       = 100,
    #     lambdas_cv      = lambdas_cv,
    #     k_folds         = 5,
    #     num_epochs_cv   = 20,
    #     lambdas_ss      = lambdas_ss,
    #     B_ss            = 50,
    #     pi_thresh       = 0.6,
    #     num_epochs_sub  = 20,
    #     lr              = 1e-3,
    #     batch_size      = 16,
    #     device          = device,
    #     label           = "Burgers' Equation"
    # )

    # # ---------- Kuramoto–Sivashinsky Equation ----------
    # # Note: smaller nt for speed, adjust as desired
    # X_ks, xspan_ks, tspan_ks = generate_kse_data(nx=2**10, nt=1000, L=100.0, t_max=100.0)
    # d_ks, n_ks = X_ks.shape
    # s_ks = min(d_ks, n_ks)
    
    # # Create flow-field for Kuramoto-Sivashinsky Equation
    # if sanity_check:
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     im = ax.imshow(
    #         X_ks, aspect='auto', cmap='viridis', origin='lower',
    #         extent=[tspan_ks[0], tspan_ks[-1], xspan_ks[0], xspan_ks[-1]])
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Space (x)')
    #     ax.set_title('Kuramoto-Sivashinsky Equation Solution')
    #     plt.colorbar(im, ax=ax, label='u(x,t)')
    #     plt.tight_layout()
    #     plt.savefig('figures/kse_data.png', dpi=300)
    #     # plt.show()
    #     plt.close(fig)

    # model_ks, info_ks, selected_ks = run_lassonet_pod_recon_with_lambda_selection(
    #     X_np            = X_ks,
    #     s               = s_ks,
    #     hidden_units    = hidden_units_ks,
    #     M               = 1.0,
    #     lambda_method   = lambda_method,
    #     lam0            = 1e-6,
    #     epsilon         = 0.10,
    #     B_path          = 20,
    #     max_iters       = 100,
    #     lambdas_cv      = lambdas_cv,
    #     k_folds         = 5,
    #     num_epochs_cv   = 20,
    #     lambdas_ss      = lambdas_ss,
    #     B_ss            = 50,
    #     pi_thresh       = 0.6,
    #     num_epochs_sub  = 20,
    #     lr              = 1e-3,
    #     batch_size      = 16,
    #     device          = device,
    #     label           = "Kuramoto-Sivashinsky Equation"
    # )