import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseModesNet(nn.Module):
    """
    LassoNet in POD-space (R^s → R^s), but loss computed in original x-space:
      For each sample i:
        z_i = V_s^T x_i
        z_hat_i = b ⊙ z_i + f(z_i)
        x_hat_i = V_s z_hat_i
      Minimize ∑ ||x_i - x_hat_i||^2 + λ ||b||_1, subject to
        ||W^(1)[j,:]||_∞ ≤ M |b_j|  for all j ∈ {0..s-1}.
    """

    def __init__(self, pod_basis: torch.Tensor, input_dim: int, 
                 hidden_units: list, M: float = 5.0, lam: float = 1e-3):
        """
        pod_basis:  V_s ∈ R^{d x s}  (torch.Tensor)
        input_dim:  s (POD dimension)
        hidden_units:  e.g. [64, 32]
        M: hierarchy multiplier
        lam: ℓ₁ penalty on b
        """
        super(SparseModesNet, self).__init__()

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
