"""
lassonet_pod_selector_recon_with_KS.py

Extends the previous LassoNet‐POD code to include a Kuramoto–Sivashinsky (KS) equation example.
We train LassoNet in POD‐space (dimension s) but minimize reconstruction error in the original x‐space:
    || x_i - V_s * (b ⊙ z_i + f_NN(z_i)) ||_2^2  +  λ ||b||_1.

Supports CUDA, MPS (Apple Silicon), or CPU.

To run:
    python lassonet_pod_selector_recon_with_KS.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1) Generate 1D PDE Data: Heat, Burgers, and Kuramoto–Sivashinsky
# ------------------------------------------------------------

def generate_heat_equation_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0):
    """
    Simulate 1D heat equation u_t = alpha * u_xx on [0, x_max], t ∈ [0, t_max].
    Dirichlet BCs (u=0 at boundaries). Returns data ∈ R^{nx×nt}.
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
    return data

def generate_burgers_data(nx=100, nt=200, nu=0.001, x_max=1.0, t_max=1.0):
    """
    Simulate 1D viscous Burgers' equation u_t + u u_x = nu u_xx on [0, x_max], periodic BC.
    Returns data ∈ R^{nx×nt}.
    """
    x = np.linspace(0, x_max, nx, endpoint=False)
    dx = x[1] - x[0]
    dt = t_max / (nt - 1)

    u = -np.sin(2 * np.pi * x)
    snapshots = [u.copy()]

    for k in range(1, nt):
        dudx = (np.roll(u, -1) - u) / dx
        lap = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        u_new = u - dt * u * dudx + nu * dt * lap
        snapshots.append(u_new.copy())
        u = u_new

    data = np.column_stack(snapshots)
    return data

def generate_ks_equation_data(nx=128, nt=500, L=32.0, t_max=50.0):
    """
    Simulate 1D Kuramoto–Sivashinsky (KS) equation:
        u_t + u u_x + u_xx + u_xxxx = 0
    on [0, L] with periodic BCs, using finite differences.
    Returns data ∈ R^{nx×nt} where each column is u(x, t_k).
    """
    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]
    dt = t_max / (nt - 1)

    # Initial condition: small random perturbation around zero
    np.random.seed(0)
    u = 0.01 * np.random.randn(nx)
    snapshots = [u.copy()]

    for k in range(1, nt):
        # Compute derivatives with periodic finite differences
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        u_xxxx = (np.roll(u, -2) - 4*np.roll(u, -1) + 6*u - 4*np.roll(u, 1) + np.roll(u, 2)) / dx**4

        # KS update: u_t = -u u_x - u_xx - u_xxxx
        u_new = u + dt * (-u * u_x - u_xx - u_xxxx)

        snapshots.append(u_new.copy())
        u = u_new

    data = np.column_stack(snapshots)  # (nx, nt)
    return data

# ------------------------------------------------------------
# 2) Compute POD Basis via SVD
# ------------------------------------------------------------

def compute_pod_basis(X_np: np.ndarray, s: int = None):
    """
    Given X_np ∈ R^{d×n}, compute first s left singular vectors V_s ∈ R^{d×s}.
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

# ------------------------------------------------------------
# 3) Dataset that returns (z_i, x_i)
# ------------------------------------------------------------

class PODReconDataset(Dataset):
    """
    Given:
      - Z_np ∈ R^{s×n} (POD coefficients = V_s^T X)
      - X_np ∈ R^{d×n} (original snapshots)
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
        pod_basis:  V_s ∈ R^{d×s}  (torch.Tensor)
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

    # This doesn't seem to be a correct implmentation from the original paper.
    # Could be wrong
    def proximal_step(self):
        """
        Algorithm 4 (Group Hierarchical Proximal) in POD-space:
        For each j=0..s-1:
          Let b_abs_j = |b_j|, W1 = self.first_layer.weight.data  (h, s),
          W1_T = W1^T  (s, h), w_j = || W1_T[j, :] ||_∞.

          If w_j ≤ M * b_abs_j:
            b_new_j = max(0, b_abs_j - lam)
            w_new_j = w_j
          Else:
            α = (M b_abs_j + w_j)/(M^2 + 1)
            b_proj = α,     w_proj = M * α
            b_new_j = max(0, b_proj - lam)
            w_new_j = M * b_new_j

          Then:
            b_j ← sign(b_j) * b_new_j
            Scale W1_T[j, :] so that || W1_T[j,:] ||_∞ = w_new_j.
            (If w_j > 0: scale factor = w_new_j / w_j; else zero the row.)

        Finally, write W1 = (W1_T)^T back to self.first_layer.weight.data.
        """
        b_data = self.b.data.clone()               # (s,)
        b_abs = b_data.abs()                       # (s,)
        sign_b = b_data.sign()                     # (s,)
        lam = self.lam
        M = self.M

        W1 = self.first_layer.weight.data.clone()  # (h, s)
        W1_T = W1.t().contiguous()                 # (s, h)
        w = LassoNetAutoencoderPODRecon._row_inf_norm(W1_T)  # (s,)

        b_new = torch.zeros_like(b_abs)
        w_new = torch.zeros_like(w)

        mask_ok = (w <= M * b_abs)
        # Case 1: w_j ≤ M * b_abs_j → soft-threshold on b_abs_j
        b_case1 = torch.clamp(b_abs - lam, min=0.0)
        w_case1 = w.clone()
        b_new[mask_ok] = b_case1[mask_ok]
        w_new[mask_ok] = w_case1[mask_ok]

        # Case 2: w_j > M * b_abs_j → project onto w = M b_abs
        mask_bad = ~mask_ok
        if mask_bad.any():
            b_bad = b_abs[mask_bad]
            w_bad = w[mask_bad]
            alpha = (M * b_bad + w_bad) / (M*M + 1.0)  # (|mask_bad|,)
            b_proj = alpha
            b_thr = torch.clamp(b_proj - lam, min=0.0)
            w_thr = M * b_thr
            b_new[mask_bad] = b_thr
            w_new[mask_bad] = w_thr

        # Update skip‐weights b
        b_updated = sign_b * b_new
        self.b.data.copy_(b_updated)

        # Scale rows of W1_T so ||W1_T[j,:]||_∞ = w_new[j]
        W1_T_updated = W1_T.clone()  # (s, h)
        scale = torch.zeros_like(w)  # (s,)
        nz = (w > 0)
        scale[nz] = w_new[nz] / w[nz]
        W1_T_updated[nz, :] = W1_T[nz, :] * scale[nz].unsqueeze(1)
        W1_T_updated[~nz, :] = 0.0

        # Write back to first_layer.weight
        W1_updated = W1_T_updated.t().contiguous()  # (h, s)
        self.first_layer.weight.data.copy_(W1_updated)

    def l1_norm_b(self):
        """Return ℓ₁‐norm of b."""
        return self.b.abs().sum()

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
    1) Compute POD basis V_s from X_np ∈ R^{d×n}.
    2) Compute Z = V_s^T X_np ∈ R^{s×n}.
    3) Build dataset of (z_i, x_i), i=1..n.
    4) Instantiate LassoNetAutoencoderPODRecon(V_s, s, hidden_units, M, lam).
    5) Train with train_lassonet_pod_recon.
    6) After training:
         - Identify indices j where b_j ≠ 0  → selected POD modes.
         - Compute final reconstruction X_hat = V_s Z_hat, report ‖X - X_hat‖_F^2 / n.
    """
    print(f"\n=== LassoNet‐POD (recon) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

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
    print(f"\nSelected POD‐mode indices (b_j ≠ 0): {selected_indices.tolist()}  "
          f"(count = {len(selected_indices)} / {s})")

    # 6) Compute final reconstruction error
    model.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
        z_hat_tensor, x_hat_tensor = model(Z_tensor)                       # (n, s), (n, d)
        X_hat_np = x_hat_tensor.cpu().numpy().T                             # (d, n)

    frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')**2
    mse_per_sample = frob_error / X_np.shape[1]
    print(f"Final reconstruction ||X - X_hat||_F^2 = {frob_error:.6e},  MSE per sample = {mse_per_sample:.6e}")

    return model, history, selected_indices

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

    # Hyperparameters
    M = 5.0                   # hierarchy multiplier
    lam = 1e-3                # ℓ₁ penalty on b
    lr = 1e-3                 # learning rate
    num_epochs = 100          # epochs
    batch_size = 32           # batch size
    hidden_units = [64, 32]   # hidden layer sizes in POD‐space

    # ---------- Heat Equation ----------
    X_heat = generate_heat_equation_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    run_lassonet_pod_recon(
        X_np = X_heat,
        s = s_h,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        label = "Heat Equation"
    )

    # ---------- Burgers' Equation ----------
    X_burgers = generate_burgers_data(nx=128, nt=200, nu=0.001, x_max=1.0, t_max=1.0)
    d_b, n_b = X_burgers.shape
    s_b = min(d_b, n_b)
    run_lassonet_pod_recon(
        X_np = X_burgers,
        s = s_b,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        label = "Burgers’ Equation"
    )

    # ---------- Kuramoto–Sivashinsky Equation ----------
    X_ks = generate_ks_equation_data(nx=128, nt=500, L=32.0, t_max=50.0)
    d_ks, n_ks = X_ks.shape
    s_ks = min(d_ks, n_ks)
    run_lassonet_pod_recon(
        X_np = X_ks,
        s = s_ks,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        label = "Kuramoto–Sivashinsky Equation"
    )