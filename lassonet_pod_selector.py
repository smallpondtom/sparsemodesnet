"""
lassonet_pod_selector_masked.py

Implements four “modes” of masked-input LassoNet in POD-space:

 mode=0:  ω⊙z + f_NN(z)                     # unmasked NN
 mode=1:  ω⊙z + f_NN(ω⊙z)                   # single-mask on NN input
 mode=2:  ω⊙z + ω⊙f_NN(ω⊙z)                 # double-mask on NN output
 mode=3:  ω1⊙z + ω2⊙f_NN(ω1⊙z)              # two-mask variant

All four train to minimize the reconstruction error:
   1/n ∑ || x_i - V_s \hat{z}_i ||_2^2  + λ ||mask||_1
with hierarchical constraints on W^(1) columns.

Runs three examples: 1D Heat, 1D Burgers', 1D Kuramoto-Sivashinsky.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# 1) PDE Data Generators: Heat, Burgers, KS
# ------------------------------------------------------------

def generate_heat_equation_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0):
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

    return np.column_stack(snapshots)  # (nx, nt)

def generate_burgers_data(nx=100, nt=200, nu=0.001, x_max=1.0, t_max=1.0):
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

    return np.column_stack(snapshots)  # (nx, nt)

def generate_ks_equation_data(nx=128, nt=500, L=32.0, t_max=50.0):
    x = np.linspace(0, L, nx, endpoint=False)
    dx = x[1] - x[0]
    dt = t_max / (nt - 1)

    np.random.seed(0)
    u = 0.01 * np.random.randn(nx)
    snapshots = [u.copy()]

    for k in range(1, nt):
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
        u_xx = (np.roll(u, -1) - 2*u + np.roll(u, 1)) / dx**2
        u_xxxx = (np.roll(u, -2) - 4*np.roll(u, -1) + 6*u - 4*np.roll(u, 1) + np.roll(u, 2)) / dx**4
        u_new = u + dt * (-u * u_x - u_xx - u_xxxx)
        snapshots.append(u_new.copy())
        u = u_new

    return np.column_stack(snapshots)  # (nx, nt)

# ------------------------------------------------------------
# 2) Compute POD basis via SVD
# ------------------------------------------------------------

def compute_pod_basis(X_np: np.ndarray, s: int = None):
    U, Sigma, Vt = np.linalg.svd(X_np, full_matrices=False)
    d, n = X_np.shape
    r = min(d, n) if s is None else min(s, min(d, n))
    V_s = U[:, :r].astype(np.float32)
    Sigma_s = Sigma[:r].astype(np.float32)
    Wt_s = Vt[:r, :].astype(np.float32)
    return V_s, Sigma_s, Wt_s

# ------------------------------------------------------------
# 3) Dataset: returns (z_i, x_i) for POD-Reconstruction
# ------------------------------------------------------------

class PODReconDataset(Dataset):
    def __init__(self, Z_np: np.ndarray, X_np: np.ndarray):
        assert Z_np.ndim == 2 and X_np.ndim == 2
        s, n1 = Z_np.shape
        d, n2 = X_np.shape
        assert n1 == n2, "Snapshot counts must match"
        self.Z = Z_np.T.copy().astype(np.float32)  # (n, s)
        self.X = X_np.T.copy().astype(np.float32)  # (n, d)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return self.Z[idx, :], self.X[idx, :]

# ------------------------------------------------------------
# 4) Four Variants of LassoNet in POD-space, all share proximal logic
# ------------------------------------------------------------

class LassoNetAutoencoderPODBase(nn.Module):
    """
    Base class that implements:
      - storage of V_s (d×s)
      - two learnable masks ω or (ω1, ω2)
      - feedforward net f_NN in POD-space
      - proximal_step() that enforces:
          ||W^(1)[j,:]||_∞ ≤ M * mask_j  (mask_j = |ω| or |ω1+ω2|)
      - l1_norm_mask() returns sum of absolute masks.
    Subclasses must override forward() to implement:
      mode=0: ω⊙z + f_NN(z)
      mode=1: ω⊙z + f_NN(ω⊙z)
      mode=2: ω⊙z + ω⊙f_NN(ω⊙z)
      mode=3: ω1⊙z + ω2⊙f_NN(ω1⊙z)
    """

    def __init__(self, pod_basis: torch.Tensor, input_dim: int, hidden_units: list,
                 M: float, lam: float, mode: int):
        super().__init__()
        self.V_s = pod_basis            # (d, s)
        self.d, self.s = pod_basis.shape
        self.M = float(M)
        self.lam = float(lam)
        self.mode = mode

        # Depending on mode, allocate masks:
        if mode in (0,1,2):
            # Single mask ω ∈ R^s
            self.omega = nn.Parameter(torch.zeros(self.s))
        else:
            # Two masks ω1, ω2 ∈ R^s
            self.omega1 = nn.Parameter(torch.zeros(self.s))
            self.omega2 = nn.Parameter(torch.zeros(self.s))

        # Build feedforward net f_NN in POD-space:
        layers = []
        self.first_layer = nn.Linear(self.s, hidden_units[0], bias=True)
        layers.append(self.first_layer)
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, len(hidden_units)):
            layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_units[-1], self.s, bias=True))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _row_inf_norm(mat: torch.Tensor) -> torch.Tensor:
        return mat.abs().max(dim=1)[0]

    def proximal_step(self):
        """
        Hierarchical proximal operator on W^(1) columns and mask(s):
        - Compute effective_mask_j = |ω_j| for mode 0,1,2; or |ω1_j + ω2_j| for mode 3.
        - Let W1 = first_layer.weight.data  (h, s),  W1_T = W1^T  (s, h).
        - Let w_j = max_k |W1_T[j, k]|  (ℓ∞ norm of row j).
        - For each j:
            θ_j = effective_mask_j,  w_j = ℓ∞-norm of row j.
          If w_j ≤ M * θ_j:
            θ_j' = max(0, θ_j - lam),  w_j' = w_j
          Else:
            α = (M θ_j + w_j)/(M^2 + 1)
            θ_proj = α,  w_proj = M α
            θ_j' = max(0, θ_proj - lam),  w_j' = M θ_j'
          Update mask_j ← sign(mask_j) * θ_j'.
          Scale W1_T[j, :] so that its ℓ∞-norm = w_j':
            if original w_j > 0: scale_factor = w_j'/w_j; else zero row.
        - Write back W1 = (W1_T)^T.
        """
        if self.mode in (0,1,2):
            mask_data = self.omega.data.clone()     # (s,)
            theta = mask_data.abs()                 # (s,)
            sign_mask = mask_data.sign()
        else:
            # two masks
            sum_mask = self.omega1.data + self.omega2.data
            theta = sum_mask.abs()                  # (s,)
            sign_mask = sum_mask.sign()             # sign(ω1_j + ω2_j)

        lam = self.lam
        M = self.M

        W1 = self.first_layer.weight.data.clone()  # (h, s)
        W1_T = W1.t().contiguous()                 # (s, h)
        w = LassoNetAutoencoderPODBase._row_inf_norm(W1_T)  # (s,)

        theta_new = torch.zeros_like(theta)
        w_new = torch.zeros_like(w)

        mask_ok = (w <= M * theta)

        # Case 1: w_j ≤ M θ_j  →  θ_j' = max(0, θ_j - lam),  w_j' = w_j
        theta_case1 = torch.clamp(theta - lam, min=0.0)
        w_case1 = w.clone()
        theta_new[mask_ok] = theta_case1[mask_ok]
        w_new[mask_ok] = w_case1[mask_ok]

        # Case 2: w_j > M θ_j  →  project onto w = M θ, then soft‐threshold
        mask_bad = ~mask_ok
        if mask_bad.any():
            theta_bad = theta[mask_bad]
            w_bad = w[mask_bad]
            alpha = (M * theta_bad + w_bad) / (M*M + 1.0)
            theta_proj = alpha
            theta_thr = torch.clamp(theta_proj - lam, min=0.0)
            w_thr = M * theta_thr
            theta_new[mask_bad] = theta_thr
            w_new[mask_bad] = w_thr

        # Update masks
        if self.mode in (0,1,2):
            omega_updated = sign_mask * theta_new   # (s,)
            self.omega.data.copy_(omega_updated)
        else:
            # We split θ_j' back proportionally to ω1_j and ω2_j if both nonzero.
            # If sum_mask_j = 0, then ω1_j=ω2_j=0 → both stay zero.
            # If sum_mask_j ≠ 0, we distribute θ_new_j proportionally:
            sum_mask = self.omega1.data + self.omega2.data
            for j in range(self.s):
                if sum_mask[j] == 0:
                    self.omega1.data[j] = 0.0
                    self.omega2.data[j] = 0.0
                else:
                    frac1 = self.omega1.data[j].abs() / sum_mask[j].abs()
                    frac2 = self.omega2.data[j].abs() / sum_mask[j].abs()
                    new_val = theta_new[j]
                    self.omega1.data[j] = sign_mask[j] * (frac1 * new_val)
                    self.omega2.data[j] = sign_mask[j] * (frac2 * new_val)

        # Scale rows of W1_T so each row j has ℓ∞-norm = w_new[j]
        W1_T_updated = W1_T.clone()          # (s, h)
        scale = torch.zeros_like(w)          # (s,)
        nz = (w > 0)
        scale[nz] = w_new[nz] / w[nz]
        W1_T_updated[nz, :] = W1_T[nz, :] * scale[nz].unsqueeze(1)
        W1_T_updated[~nz, :] = 0.0

        W1_updated = W1_T_updated.t().contiguous()  # (h, s)
        self.first_layer.weight.data.copy_(W1_updated)

    def l1_norm_mask(self):
        if self.mode in (0,1,2):
            return self.omega.abs().sum()
        else:
            return (self.omega1.abs() + self.omega2.abs()).sum()

    def forward(self, z_batch):
        """
        Implement each mode's forward logic.

        Input: z_batch (batch, s)
        Returns: z_hat (batch, s), x_hat (batch, d)
        """
        if self.mode == 0:
            # mode 0:  z_hat = ω⊙z + f_NN(z)
            skip = z_batch * self.omega.unsqueeze(0)       # (batch, s)
            nn_out = self.net(z_batch)                     # (batch, s)
            z_hat = skip + nn_out

        elif self.mode == 1:
            # mode 1:  z_hat = ω⊙z + f_NN(ω⊙z)
            skip = z_batch * self.omega.unsqueeze(0)       # (batch, s)
            masked = skip                                  # (batch, s)
            nn_out = self.net(masked)                      # (batch, s)
            z_hat = skip + nn_out

        elif self.mode == 2:
            # mode 2:  z_hat = ω⊙z + ω⊙f_NN(ω⊙z)
            skip = z_batch * self.omega.unsqueeze(0)       # (batch, s)
            masked = skip                                  # (batch, s)
            nn_out = self.net(masked)                      # (batch, s)
            z_hat = skip + (masked * nn_out)               # (batch, s)

        else:  # mode == 3
            # mode 3:  z_hat = ω1⊙z + ω2⊙f_NN(ω1⊙z)
            skip = z_batch * self.omega1.unsqueeze(0)      # (batch, s)
            masked = skip                                  # (batch, s)
            nn_out = self.net(masked)                      # (batch, s)
            z_hat = skip + (self.omega2.unsqueeze(0) * nn_out)

        # Reconstruct to x-space: x_hat = V_s @ z_hat^T  → (d, batch) → transpose → (batch, d)
        x_hat_T = self.V_s.matmul(z_hat.t())              # (d, batch)
        x_hat = x_hat_T.t()                               # (batch, d)
        return z_hat, x_hat

# ------------------------------------------------------------
# 5) Training Loop (Reconstruction Loss in x-space)
# ------------------------------------------------------------

def train_lassonet_pod_recon_masked(model: LassoNetAutoencoderPODBase,
                                    dataloader: DataLoader,
                                    num_epochs: int = 100,
                                    lr: float = 1e-3,
                                    device: str = 'cpu'):
    """
    Train the masked LassoNet autoencoder to minimize
      ∑ || x_i - V_s z_hat_i ||_2^2  +  λ ||mask||_1,
    subject to hierarchical constraints on W^(1).
    Each batch: (z_batch, x_batch).  
    Steps:
      1) (z_hat, x_hat) = model(z_batch)
      2) loss = MSE(x_hat, x_batch)
      3) loss.backward(), optimizer.step()
      4) model.proximal_step()
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    history = {'loss': [], 'l1_mask': []}

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_l1 = 0.0
        n_samples = 0

        model.train()
        for z_batch, x_batch in dataloader:
            z_batch = z_batch.to(device)
            x_batch = x_batch.to(device)

            optimizer.zero_grad()
            z_hat_batch, x_hat_batch = model(z_batch)    # (batch, s), (batch, d)
            loss = mse_loss(x_hat_batch, x_batch)
            loss.backward()
            optimizer.step()

            # hierarchical proximal step
            model.proximal_step()

            batch_size = x_batch.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_l1 += model.l1_norm_mask().item() * batch_size
            n_samples += batch_size

        epoch_loss /= n_samples
        epoch_l1 /= n_samples
        history['loss'].append(epoch_loss)
        history['l1_mask'].append(epoch_l1)

        if (epoch % 20 == 0) or (epoch == 1):
            print(f"Epoch {epoch:3d} | Recon MSE = {epoch_loss:.6e} | ℓ₁‖mask‖ = {epoch_l1:.6e}")

    return history

# ------------------------------------------------------------
# 6) Driver: Tie Everything Together
# ------------------------------------------------------------

def run_lassonet_pod_recon_masked(X_np: np.ndarray,
                                  s: int,
                                  hidden_units: list,
                                  M: float,
                                  lam: float,
                                  lr: float,
                                  num_epochs: int,
                                  batch_size: int,
                                  device: str,
                                  mode: int,
                                  label: str):
    """
    1) Compute POD basis V_s from X_np ∈ R^{d×n}.
    2) Compute Z = V_s^T X_np ∈ R^{s×n}.
    3) Build dataset (z_i, x_i) for i=1..n.
    4) Instantiate LassoNetAutoencoderPODBase(...) with the given mode.
    5) Train via train_lassonet_pod_recon_masked(...).
    6) After training:
         - Extract mask(s), find selected indices (those j where mask_j ≠ 0).
         - Report final reconstruction error ||X - X_hat||_F^2 / n.
    """
    print(f"\n=== Mode {mode} LassoNet‐POD (masked) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    # 1) POD basis
    V_s_np, Sigma_s_np, Wt_s_np = compute_pod_basis(X_np, s=s)
    Z_np = V_s_np.T.dot(X_np)  # (s, n)

    # Torch Tensors
    V_s = torch.from_numpy(V_s_np.astype(np.float32)).to(device)  # (d, s)
    Z = torch.from_numpy(Z_np.astype(np.float32))                # (s, n)
    # X_torch not needed except for final error

    # 2) Dataset & DataLoader
    dataset = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # 3) Instantiate masked LassoNet model (mode selects variant)
    model = LassoNetAutoencoderPODBase(
        pod_basis=V_s,
        input_dim=s,
        hidden_units=hidden_units,
        M=M,
        lam=lam,
        mode=mode
    ).to(device)

    # 4) Train
    history = train_lassonet_pod_recon_masked(
        model=model,
        dataloader=dataloader,
        num_epochs=num_epochs,
        lr=lr,
        device=device
    )

    # 5) Extract final mask(s) and selected indices
    if mode in (0,1,2):
        omega_opt = model.omega.detach().cpu().numpy()  # (s,)
        selected_indices = np.where(np.abs(omega_opt) > 1e-6)[0]
        print(f"\nSelected POD‐mode indices (ω_j ≠ 0): {selected_indices.tolist()} (count={len(selected_indices)}/{s})")
    else:
        sum_mask = (model.omega1.detach().cpu().numpy() 
                    + model.omega2.detach().cpu().numpy())  # (s,)
        selected_indices = np.where(np.abs(sum_mask) > 1e-6)[0]
        print(f"\nSelected POD‐mode indices (ω1_j+ω2_j ≠ 0): {selected_indices.tolist()} (count={len(selected_indices)}/{s})")

    # 6) Compute final reconstruction error:
    model.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
        z_hat_tensor, x_hat_tensor = model(Z_tensor)                       # (n, s), (n, d)
        X_hat_np = x_hat_tensor.cpu().numpy().T                             # (d, n)

    frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')**2
    mse_per_sample = frob_error / X_np.shape[1]
    print(f"Final reconstruction ||X - X_hat||_F^2 = {frob_error:.6e}, MSE per sample = {mse_per_sample:.6e}")

    return model, history, selected_indices

# ------------------------------------------------------------
# 7) Main: run on Heat, Burgers, and KS for each mode
# ------------------------------------------------------------

if __name__ == "__main__":
    # Device: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print("Using device:", device)

    # Hyperparameters
    M = 5.0
    lam = 1e-3
    lr = 1e-3
    num_epochs = 100
    batch_size = 32
    hidden_units = [64, 32]

    # Choose mode in {0,1,2,3}:
    #   0 = ω⊙z + f_NN(z)
    #   1 = ω⊙z + f_NN(ω⊙z)
    #   2 = ω⊙z + ω⊙f_NN(ω⊙z)
    #   3 = ω1⊙z + ω2⊙f_NN(ω1⊙z)
    mode = 0   # <-- pick whichever variant you want

    # ---------- 1D Heat Equation ----------
    X_heat = generate_heat_equation_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0)
    d_h, n_h = X_heat.shape
    s_h = min(d_h, n_h)
    run_lassonet_pod_recon_masked(
        X_np = X_heat,
        s = s_h,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        mode = mode,
        label = "Heat Equation"
    )

    # ---------- 1D Burgers' Equation ----------
    X_burgers = generate_burgers_data(nx=128, nt=200, nu=0.001, x_max=1.0, t_max=1.0)
    d_b, n_b = X_burgers.shape
    s_b = min(d_b, n_b)
    run_lassonet_pod_recon_masked(
        X_np = X_burgers,
        s = s_b,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        mode = mode,
        label = "Burgers’ Equation"
    )

    # ---------- 1D Kuramoto–Sivashinsky Equation ----------
    X_ks = generate_ks_equation_data(nx=128, nt=500, L=32.0, t_max=50.0)
    d_ks, n_ks = X_ks.shape
    s_ks = min(d_ks, n_ks)
    run_lassonet_pod_recon_masked(
        X_np = X_ks,
        s = s_ks,
        hidden_units = hidden_units,
        M = M,
        lam = lam,
        lr = lr,
        num_epochs = num_epochs,
        batch_size = batch_size,
        device = device,
        mode = mode,
        label = "Kuramoto–Sivashinsky Equation"
    )