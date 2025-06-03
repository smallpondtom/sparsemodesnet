"""
lassonet_pod_selection.py

LassoNet-POD Mode Selection code. Includes the method code and examples for 
Heat, Burgers, and Kuramoto-Sivashinsky equations.  We train LassoNet in 
POD-space (dimension s) but minimize reconstruction error in the original space:

   1/n ∑_{i=1}^n || x_i - V_s * (b ⊙ z_i + f_NN(z_i)) ||_2^2  +  λ ||b||_1.

Supports CUDA, MPS (Apple Silicon), or CPU.

To run:
    python lassonet_pod_selection.py
"""

#%%
import numpy as np
from scipy.integrate import odeint
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

#%%
# try:
#     from .prox import prox  # Relative import when used as module
# except ImportError:
#     from prox import prox   # Absolute import when run directly

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
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = h * np.outer(L_op, np.ones(M)) + np.outer(np.ones(nx), r)
    
    # ETDRK4 coefficients computed via contour integrals
    Q = h * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    
    # Handle potential division by zero at k=0
    zero_idx = np.where(np.abs(L_op) < 1e-14)[0]
    if len(zero_idx) > 0:
        Q[zero_idx] = h
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
        a = E_2 * v + Q * Nv
        
        # Stage 2
        Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2)
        b = E_2 * v + Q * Na
        
        # Stage 3
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2)
        c = E_2 * a + Q * (2*Nb - Nv)
        
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
    V_s = U[:, :r].astype(np.float32)
    Sigma_s = Sigma[:r].astype(np.float32)
    Wt_s = Vt[:r, :].astype(np.float32)
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
        layers = []
        self.first_layer = nn.Linear(self.s, hidden_units[0], bias=True)
        layers.append(self.first_layer)
        layers.append(nn.ReLU(inplace=True))

        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=True))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(hidden_units[-1], self.s, bias=True))
        self.net = nn.Sequential(*layers)
        
        # Is this actually an autoencoder???
        # Try CNNs too for spatial correlation, might turn out to be better

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
        u_abs       = W1_T.abs()                                 # (s, K)
        u_abs_sorted, _ = torch.sort(u_abs, dim=1, descending=True)  # (s, K)

        # 3) Build partial sums a_s(m) = lam - M * sum_{i=1}^m u_abs_sorted[j,i-1]
        zeros_m    = torch.zeros((s, 1), device=W1_T.device, dtype=W1_T.dtype)  # (s,1)
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

    # def proximal_step(self):
    #     """
    #     Algorithm 4 (Group Hierarchical Proximal) in POD-space:
    #     For each j=0..s-1:
    #       Let b_abs_j = |b_j|, W1 = self.first_layer.weight.data  (h, s),
    #       W1_T = W1^T  (s, h), w_j = || W1_T[j, :] ||_∞.

    #       If w_j ≤ M * b_abs_j:
    #         b_new_j = max(0, b_abs_j - lam)
    #         w_new_j = w_j
    #       Else:
    #         α = (M b_abs_j + w_j)/(M^2 + 1)
    #         b_proj = α,     w_proj = M * α
    #         b_new_j = max(0, b_proj - lam)
    #         w_new_j = M * b_new_j

    #       Then:
    #         b_j ← sign(b_j) * b_new_j
    #         Scale W1_T[j, :] so that || W1_T[j,:] ||_∞ = w_new_j.
    #         (If w_j > 0: scale factor = w_new_j / w_j; else zero the row.)

    #     Finally, write W1 = (W1_T)^T back to self.first_layer.weight.data.
    #     """
    #     b_data = self.b.data.clone()               # (s,)
    #     b_abs = b_data.abs()                       # (s,)
    #     sign_b = b_data.sign()                     # (s,)
    #     lam = self.lam
    #     M = self.M

    #     W1 = self.first_layer.weight.data.clone()  # (h, s)
    #     W1_T = W1.t().contiguous()                 # (s, h)
    #     w = LassoNetAutoencoderPODRecon._row_inf_norm(W1_T)  # (s,)

    #     b_new = torch.zeros_like(b_abs)
    #     w_new = torch.zeros_like(w)

    #     mask_ok = (w <= M * b_abs)
    #     # Case 1: w_j ≤ M * b_abs_j → soft-threshold on b_abs_j
    #     b_case1 = torch.clamp(b_abs - lam, min=0.0)
    #     w_case1 = w.clone()
    #     b_new[mask_ok] = b_case1[mask_ok]
    #     w_new[mask_ok] = w_case1[mask_ok]

    #     # Case 2: w_j > M * b_abs_j → project onto w = M b_abs
    #     mask_bad = ~mask_ok
    #     if mask_bad.any():
    #         b_bad = b_abs[mask_bad]
    #         w_bad = w[mask_bad]
    #         alpha = (M * b_bad + w_bad) / (M*M + 1.0)  # (|mask_bad|,)
    #         b_proj = alpha
    #         b_thr = torch.clamp(b_proj - lam, min=0.0)
    #         w_thr = M * b_thr
    #         b_new[mask_bad] = b_thr
    #         w_new[mask_bad] = w_thr

    #     # Update skip‐weights b
    #     b_updated = sign_b * b_new
    #     self.b.data.copy_(b_updated)

    #     # Scale rows of W1_T so ||W1_T[j,:]||_∞ = w_new[j]
    #     W1_T_updated = W1_T.clone()  # (s, h)
    #     scale = torch.zeros_like(w)  # (s,)
    #     nz = (w > 0)
    #     scale[nz] = w_new[nz] / w[nz]
    #     W1_T_updated[nz, :] = W1_T[nz, :] * scale[nz].unsqueeze(1)
    #     W1_T_updated[~nz, :] = 0.0

    #     # Write back to first_layer.weight
    #     W1_updated = W1_T_updated.t().contiguous()  # (h, s)
    #     self.first_layer.weight.data.copy_(W1_updated)
    
    # def proximal_step(self):
    #     """
    #     Replace the closed-form with the authors' prox(...), setting lambda_bar=0.0.
    #     For each j = 0..s-1:
    #       1) v_j = [ b_j ]           (shape (1,))
    #       2) u_j = W1_T[j, :]        (shape (h,))
    #       3) (b_j', W1_T[j,:]') = prox(v_j, u_j, lambda_=lam, lambda_bar=0.0, M=M)
    #       4) write them back
    #     """
    #     lam = self.lam
    #     M = self.M

    #     # 1) Collect the first‐layer weights W1 ∈ ℝ^{h×s}, then transpose → W1_T ∈ ℝ^{s×h}
    #     W1 = self.first_layer.weight.data        # (h, s)
    #     W1_T = W1.t().contiguous()               # (s, h)

    #     # 2) Loop over features j = 0..s-1
    #     for j in range(self.s):
    #         # 2.1) Extract current skip‐weight θ_j as a (1,) tensor
    #         v_j = self.b.data[j].view(1)          # shape (1,)

    #         # 2.2) Extract the entire j-th column of W1 (i.e. j-th row of W1_T)
    #         u_j = W1_T[j, :].clone()              # shape (h,)

    #         # 2.3) Call prox(...) with lambda_bar = 0.0
    #         #     (we use the same lam for the θ‐penalty, and no penalty on |u|)
    #         v_star, u_star = prox(
    #             v_j, u_j,
    #             lambda_    = lam,
    #             lambda_bar = 0.0,   # <-- note: set to 0, as in the paper
    #             M          = M
    #         )

    #         # 2.4) Write back the updated skip‐weight and the updated W1_T[j, :]
    #         self.b.data[j] = v_star.squeeze()        # new θ_j
    #         W1_T[j, :]     = u_star.squeeze().clone()  # new first-layer column

    #     # 3) Transpose W1_T back to (h, s) and copy into the model
    #     W1_updated = W1_T.t().contiguous()  # (h, s)
    #     self.first_layer.weight.data.copy_(W1_updated)

    def l1_norm_b(self):
        """Return ℓ₁-norm of b."""
        return self.b.abs().sum()
    

#%%
# ------------------------------------------------------------
# 5) Training Loop (Reconstruction Loss)
# ------------------------------------------------------------

def train_lassonet_pod_recon(model: LassoNetAutoencoderPODRecon,
                             dataloader: DataLoader,
                             num_epochs: int = 100,
                             lr: float = 1e-3,
                             device: str = 'cpu'):
    """
    Train LassoNet autoencoder in POD-space to minimize:
      sum_i || x_i - V_s (b ⊙ z_i + f(z_i)) ||_2^2  +  λ ||b||_1.
    Each batch: (z_batch, x_batch):
      z_batch: (batch, s),  x_batch: (batch, d).
    Steps:
      1) (z_hat, x_hat) = model(z_batch)
      2) loss = MSE(x_hat, x_batch)
      3) loss.backward(); optimizer.step()
      4) model.proximal_step()
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = {'loss': [], 'l1_b': []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)  # (batch, s)
            x_batch = x_batch.to(device)  # (batch, d)

            optimizer.zero_grad()
            z_hat_batch, x_hat_batch = model(z_batch)  # x_hat_batch: (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss.backward()
            optimizer.step()

            model.proximal_step()

            batch_size = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1 += model.l1_norm_b().item() * batch_size
            n_samples += batch_size

        epoch_loss /= n_samples
        epoch_l1 /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_b'].append(epoch_l1)

        if (epoch % 20 == 0) or (epoch == 1):
            print(f"Epoch {epoch:3d} | Recon MSE = {epoch_loss:.6e} | ℓ₁‖b‖ = {epoch_l1:.6e}")

    return history

#%%
# ------------------------------------------------------------
# 6) Driver: Run LassoNet-POD-Recon on any X_np
# ------------------------------------------------------------

def run_lassonet_pod_recon(X_np: np.ndarray,
                           s: int,
                           hidden_units: list,
                           M: float,
                           lam: float,
                           lr: float,
                           num_epochs: int,
                           batch_size: int,
                           device: str,
                           label: str):
    """
    1) Compute POD basis V_s from X_np ∈ R^{d x n}.
    2) Compute Z = V_s^T X_np ∈ R^{s x n}.
    3) Build dataset of (z_i, x_i), i=1..n.
    4) Instantiate LassoNetAutoencoderPODRecon(V_s, s, hidden_units, M, lam).
    5) Train with train_lassonet_pod_recon.
    6) After training:
         - Identify indices j where b_j ≠ 0  → selected POD modes.
         - Compute final reconstruction X_hat = V_s Z_hat, report ‖X - X_hat‖_F^2 / n.
    """
    print(f"\n=== LassoNet-POD (recon) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    # 1) Compute POD basis
    V_s_np, Sigma_s_np, Wt_s_np = compute_pod_basis(X_np, s=s)  # V_s_np: (d, s)
    Z_np = V_s_np.T.dot(X_np)  # Z = V_s^T X, shape (s, n)

    # Convert to torch Tensors
    V_s = torch.from_numpy(V_s_np.astype(np.float32)).to(device)  # (d, s)
    Z = torch.from_numpy(Z_np.astype(np.float32))                # (s, n)
    X_torch = torch.from_numpy(X_np.astype(np.float32))          # (d, n)

    # 2) Build dataset and dataloader
    dataset = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 3) Instantiate model
    model = LassoNetAutoencoderPODRecon(
        pod_basis=V_s,
        input_dim=s,
        hidden_units=hidden_units,
        M=M,
        lam=lam
    ).to(device)

    # 4) Train
    history = train_lassonet_pod_recon(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device
    )

    # 5) Identify selected POD modes
    b_opt = model.b.detach().cpu().numpy()  # (s,)
    selected_indices = np.where(np.abs(b_opt) > 1e-6)[0]
    print(f"\nFinal skip-weights b: {b_opt.tolist()}")
    print(f"Selected POD-mode indices (b_j ≠ 0): {selected_indices.tolist()}  "
          f"(count = {len(selected_indices)} / {s})")

    # 6) Compute final reconstruction error
    model.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
        z_hat_tensor, x_hat_tensor = model(Z_tensor)                       # (n, s), (n, d)
        X_hat_np = x_hat_tensor.cpu().numpy().T                            # (d, n)

    frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
    rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
    mse_per_sample = frob_error / X_np.shape[1]
    print(f"Final relative reconstruction ||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
    print(f"Final MSE per sample = {mse_per_sample:.6e}")

    return model, history, selected_indices

#%%
# ------------------------------------------------------------
# 7) Entry Point: Run on Heat, Burgers, and KS
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
    
    # Plot for sanity check
    sanity_check = True

    # ---------- Heat Equation ----------
    X_heat, xspan, tspan = generate_heat_data(
        nx=2**7, nt=1000, alpha=0.01, x_max=1.0, t_max=1.0)
    
    # Create 3D surface plot for Heat Equation (sanity check)
    if sanity_check:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        X_mesh, T_mesh = np.meshgrid(xspan, tspan)
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
    
    # Train SparseModesNet on Heat Equation data
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    run_lassonet_pod_recon(
        X_np = X_heat,
        s = s_h,
        hidden_units = [256, 128, 64, 32],  # hidden layer sizes in POD‐space
        M = 0.1,                       # hierarchy multiplier
        lam = 1e-3,                    # ℓ₁ penalty on b
        lr = 1e-3,                     # learning rate
        num_epochs = 100,              # epochs
        batch_size = 16,               # batch size
        device = device,
        label = "Heat Equation"
    )


    # # ---------- Burgers' Equation ----------
    # X_burgers, xspan, tspan = generate_burgers_data(
    #     nx=2**7, nt=1000, nu=0.01, x_max=1.0, t_max=1.0)

    # # Create 3D surface plot for Burgers' Equation (sanity check)
    # if sanity_check:
    #     fig = plt.figure(figsize=(12, 8))
    #     ax = fig.add_subplot(111, projection='3d')
    #     X_mesh, T_mesh = np.meshgrid(xspan, tspan)
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
        
    # # Train SparseModesNet on Burgers' Equation data 
    # d_b, n_b = X_burgers.shape
    # s_b = min(d_b, n_b)
    # run_lassonet_pod_recon(
    #     X_np = X_burgers,
    #     s = s_b,
    #     hidden_units = [128, 64, 32],       # hidden layer sizes in POD‐space
    #     M = 10.0,                           # hierarchy multiplier
    #     lam = 1e-4,                         # ℓ₁ penalty on b
    #     lr = 1e-3,                          # learning rate
    #     num_epochs = 100,                   # epochs
    #     batch_size = 16,                    # batch size
    #     device = device,
    #     label = "Burgers' Equation"
    # )

    # # ---------- Kuramoto–Sivashinsky Equation ----------
    # X_ks, xspan, tspan = generate_kse_data(nx=2**10, nt=4000, L=100.0, t_max=150.0)
        
    # # Create flow-field for Kuramoto-Sivashinsky Equation
    # if sanity_check:
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     im = ax.imshow(X_ks, aspect='auto', cmap='viridis', origin='lower',
    #                     extent=[tspan[0], tspan[-1], xspan[0], xspan[-1]])
    #     ax.set_xlabel('Time')
    #     ax.set_ylabel('Space (x)')
    #     ax.set_title('Kuramoto-Sivashinsky Equation Solution')
    #     plt.colorbar(im, ax=ax, label='u(x,t)')
    #     plt.tight_layout()
    #     plt.savefig('figures/kse_data.png', dpi=300)
    #     # plt.show()
    #     plt.close(fig)
        
    # # Train SparseModesNet on Kuramoto-Sivashinsky Equation data
    # d_ks, n_ks = X_ks.shape
    # s_ks = min(d_ks, n_ks)
    # run_lassonet_pod_recon(
    #     X_np = X_ks,
    #     s = s_ks,
    #     hidden_units = [256, 128, 64, 32],  # hidden layer sizes in POD‐space
    #     M = 1.0,                            # hierarchy multiplier
    #     lam = 1e-4,                         # ℓ₁ penalty on b
    #     lr = 1e-3,                          # learning rate
    #     num_epochs = 100,                   # epochs
    #     batch_size = 16,                    # batch size
    #     device = device,
    #     label = "Kuramoto-Sivashinsky Equation"
    # )