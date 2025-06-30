import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsemodesnet.pinet import PiNetCCP, PiNetNCP, PiNetNCPSkip, ProdPoly

class SparseModesNet(nn.Module):
    """
    SparseModesNet mapping from reduced (POD) space to the original space 
    (R^s → R^d). The loss is computed as follows:
        For each sample i:
            - z_i = U_s^T x_i where U_s ∈ R^{d x s} is the POD basis
            - ω ∈ R^s are skip-weights
            - ω+ are positive skip-weights, ω+ = softplus(ω)
            - z_hat_i = ω+ ⊙ z_i 
            - x_hat_lin_i = U_s z_hat_i is the projection (skip) connection
            - x_nn_i = f_NN(z_hat_i) is the output of the neural network
            - x_hat_i = x_hat_lin_i + x_nn_i is the final output of the ResNet
            
        Minimize ∑ ||x_i - x_hat_i||^2 + λ ||ω+||_1, subject to the hierarchy 
        constraint:
            ||W^(1)[j,:]||_∞ ≤ M |ω+_j| for all j ∈ {0..s-1}.
    """
    
    def __init__(self, pod_basis: torch.Tensor, input_dim: int, 
                 hidden_units: list, M: float = 5.0, lam: float = 1e-3,
                 network_type: str = 'FF', poly_order: int = 2, 
                 num_polys: int = 1, drop_linear: bool = False,
                 drop_constant: bool = False, 
                 normalize: str | None = None):
        """
        Initialize SparseModesNet.
        
        Arguments
        ---------
        pod_basis :  U_s ∈ R^{d x s}  (torch.Tensor)
        input_dim :  s (POD dimension)
        hidden_units :  e.g. [s, s(s+1)/2, s(s+1)(s+2)/6] (quadratic, cubic, etc.)
        M : hierarchy multiplier
        lam : ℓ₁ penalty on ω
        """
        super(SparseModesNet, self).__init__()

        self.register_buffer('U_s', pod_basis)  # (d, s): store as buffer
        self.d, self.s = pod_basis.shape
        self.M = float(M)
        self.lam = float(lam)

        # Skip‐weights ω ∈ R^s
        self.omega = nn.Parameter(torch.ones(self.s))
        
        # Assertion for network type
        assert network_type in ['FF', 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip'], \
            f"Unsupported network type: {network_type}. " \
            "Use 'FF', 'PiNetCCP', 'PiNetNCP', or 'PiNetNCPSkip'."
        self.network_type = network_type
        
        # Pi-Net dictionary
        PiNet = {
            'PiNetCCP': PiNetCCP,
            'PiNetNCP': PiNetNCP,
            'PiNetNCPSkip': PiNetNCPSkip
        }
        
        if network_type == 'FF': 
            # Build the feedforward network mapping from R^s to R^d
            self.first_layer = nn.Linear(self.s, hidden_units[0], bias=False)
            layers = [self.first_layer, nn.SELU(inplace=True)]
            for i in range(1, len(hidden_units)):
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=False))
                layers.append(nn.SELU(inplace=True))
                layers.append(nn.Dropout(p=0.1)) 
            layers.append(nn.Linear(hidden_units[-1], self.d, bias=False))
            self.net = nn.Sequential(*layers)
        else:
            # PiNetCCP, PiNetNCP, or PiNetNCPSkip (with/without ProdPoly)
            assert len(hidden_units) == 3, \
                "PiNetCCP requires exactly 3 hidden units: \
                [in_dim, inter_dim, out_dim]."
            in_dim, inter_dim, out_dim = hidden_units
            
            assert out_dim == self.d, \
                f"Output dimension {out_dim} must match the original \
                    state dimension {self.d}."
            
            # First layer (used in proximal step)
            self.first_layer = nn.Linear(self.s, in_dim, bias=False)
            
            # Pi-Net blocks
            if num_polys == 1:  # A single Pi-Net block
                if network_type == 'PiNetCCP':
                    self.pinet = PiNet[network_type](
                        in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                        poly_order=poly_order, drop_constant=drop_constant,
                        normalize=normalize
                    ) 
                else:
                    self.pinet = PiNet[network_type](
                        in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                        poly_order=poly_order, drop_linear=drop_linear,
                        drop_constant=drop_constant, normalize=normalize
                    ) 
            else:  # Multiple PiNetCCP blocks
                self.pinet = ProdPoly(
                    pinet_class=PiNet[network_type], num_polys=num_polys,
                    in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                    poly_order=poly_order, drop_linear=drop_linear,
                    drop_constant=drop_constant, normalize=normalize
                )
    
    def forward(self, z_batch):
        """
        Arguments
        ---------
        z_batch : (batch_size, s)
        
        Returns
        -------
        z_hat_batch : (batch_size, s)
        x_hat_batch : (batch_size, d)
        """
        # --- Projection Skip Connection --- 
        # Compute the reduced states with sparsity 
        z_hat = z_batch * self.omega.unsqueeze(0)       # (batch, s)
        # Reconstruct the linear part via projection
        x_hat_lin = z_hat @ self.U_s.T                  # (batch, d)
        
        # --- MLP or MLP + Π-net hybrid ---
        if self.network_type == 'FF': 
            # Apply the NN to the reduced states 
            x_hat_nn = self.net(z_hat)                  # (batch, d)
        else:
            h        = self.first_layer(z_hat)          # (batch, inter_dim)
            x_hat_nn = self.pinet(h)                    # (batch, out_dim)

        # --- Reconstruct x_hat ---
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


class StateDecoder(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, input_dim: int, 
                 hidden_units: list, M: float,
                 network_type: str, poly_order: int, 
                 num_polys: int, drop_linear: bool, 
                 drop_constant: bool, normalize: str | None = None):
        super(StateDecoder, self).__init__()
        
        self.register_buffer('U_r', pod_basis)  # (d, r): store as buffer
        self.d, self.r = pod_basis.shape
        self.M = float(M)
        
        # Assertion for network type
        assert network_type in ['FF', 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip', 'QM'], \
            f"Unsupported network type: {network_type}. " \
            "Use 'FF', 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip', or 'QM'."
        self.network_type = network_type
        
        # Pi-Net dictionary
        PiNet = {
            'PiNetCCP': PiNetCCP,
            'PiNetNCP': PiNetNCP,
            'PiNetNCPSkip': PiNetNCPSkip
        }
        
        if network_type == 'FF': 
            # Build the feedforward network mapping from R^s to R^d
            self.first_layer = nn.Linear(self.r, hidden_units[0], bias=True)
            layers = [self.first_layer, nn.SELU(inplace=True)]
            for i in range(1, len(hidden_units)):
                layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=True))
                layers.append(nn.SELU(inplace=True))
                layers.append(nn.Dropout(p=0.1)) 
            layers.append(nn.Linear(hidden_units[-1], self.d, bias=True))
            self.net = nn.Sequential(*layers)
        elif 'PiNet' in network_type:
            # PiNetCCP, PiNetNCP, or PiNetNCPSkip (with/without ProdPoly)
            assert len(hidden_units) == 3, \
                "PiNetCCP requires exactly 3 hidden units: \
                [in_dim, inter_dim, out_dim]."
            in_dim, inter_dim, out_dim = hidden_units
                    
            assert out_dim == self.d, \
                f"Output dimension {out_dim} must match the original \
                    state dimension {self.d}."
                    
            # First layer (used in proximal step)
            self.first_layer = nn.Linear(self.r, in_dim, bias=False)
            
            # Pi-Net blocks
            if num_polys == 1:  # A single Pi-Net block
                if network_type == 'PiNetCCP':
                    self.pinet = PiNet[network_type](
                        in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                        poly_order=poly_order, drop_constant=drop_constant,
                        normalize=normalize
                    ) 
                else:
                    self.pinet = PiNet[network_type](
                        in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                        poly_order=poly_order, drop_linear=drop_linear,
                        drop_constant=drop_constant, normalize=normalize
                    ) 
            else:  # Multiple PiNetCCP blocks
                self.pinet = ProdPoly(
                    pinet_class=PiNet[network_type], num_polys=num_polys,
                    in_dim=in_dim, out_dim=out_dim, inter_dim=inter_dim,
                    poly_order=poly_order, drop_linear=drop_linear,
                    drop_constant=drop_constant, normalize=normalize
                )
        else:
            self.quadmap = _quadratic_mapping_vectorized
            self.projection = nn.Parameter(
                torch.zeros(self.r * (self.r + 1) // 2, self.d))
        
    def forward(self, z_batch):
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T                # (batch, d)
        
        # --- MLP or MLP + Π-net hybrid ---
        if self.network_type == 'FF': 
            # Apply the NN to the reduced states 
            x_hat_nn = self.net(z_batch)                # (batch, d)
        elif 'PiNet' in self.network_type:
            h        = self.first_layer(z_batch)        # (batch, inter_dim)
            x_hat_nn = self.pinet(h)                    # (batch, out_dim)
        else:
            # Apply the quadratic mapping
            z_quad = self.quadmap(z_batch)              # (batch, r*(r+1)//2)
            x_hat_nn = z_quad @ self.projection         # (batch, d)

        # --- Reconstruct x_hat ---
        x_hat = x_hat_lin + x_hat_nn
       
        return x_hat

def _quadratic_mapping_vectorized(x):
    """
    Vectorized computation of unique Kronecker product x ⊗ x.
    Only computes upper triangular part to avoid redundancy.
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)//2) or (n*(n+1)//2,)
    """
    if x.dim() == 1:
        n = x.size(0)
        # Create indices for upper triangular part
        i_indices, j_indices = torch.triu_indices(n, n, device=x.device)
        # Compute products
        result = x[i_indices] * x[j_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create indices for upper triangular part  
        i_indices, j_indices = torch.triu_indices(n, n, device=x.device)
        # Compute products for all batches
        result = x[:, i_indices] * x[:, j_indices]
        return result   
        