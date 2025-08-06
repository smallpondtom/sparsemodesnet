import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsemodesnet.decoder_models.pinet import (PiNetCCP, 
                                                 PiNetNCP, 
                                                 PiNetNCPSkip, ProdPoly)
from sparsemodesnet.decoder_models.mlp import MLP
from sparsemodesnet.decoder_models.mask import MaskedLayer
from sparsemodesnet.decoder_models.cnn import SpatialCNN
from sparsemodesnet.decoder_models.unet import UNET
from sparsemodesnet.decoder_models.rational import Rational
from sparsemodesnet.linalg.lstsq import lstsq_l2_torch

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
    
    def __init__(self, pod_basis: torch.Tensor, mapping_dim: int, 
                 hidden_units: list, M: float = 5.0, lam: float = 1e-3,
                 gamma: float = 1e-6, alpha: float = 1.0, 
                 network_type: str = 'FF', weight_scale: float = 1e-12, 
                 poly_order: int = 2, num_polys: int = 1, 
                 drop_linear: bool = False, drop_constant: bool = False, 
                 normalize: str | None = None, 
                 dtype: torch.dtype = torch.float32):
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
        self.p = mapping_dim
        self.M = M
        self.lam = lam
        self.gamma = gamma
        self.alpha = alpha
        torch.set_default_dtype(dtype)

        # Skip‐weights ω ∈ R^s
        self.omega = nn.Parameter(torch.ones(self.s) * 0.01)
        
        # Assertion for network type
        assert network_type in ['MLP', 'CNN', 'UNET', 'PiNetCCP', 'PiNetNCP', 
                       'PiNetNCPSkip', 'QM', 'CM'], \
            f"Unsupported network type: {network_type}. " \
            "Use 'MLP', 'CNN', 'UNET', 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip', " \
            "'QM', or 'CM'."
        self.network_type = network_type
        
        # Pi-Net dictionary
        PiNet = {
            'PiNetCCP': PiNetCCP,
            'PiNetNCP': PiNetNCP,
            'PiNetNCPSkip': PiNetNCPSkip
        }
        
        if network_type == 'MLP': 
            # Build the feedforward network mapping from R^s to R^d
            self.mlp = MLP(hidden_units=hidden_units, bias=False,
                           weight_scale=weight_scale)
            self.mlp.initialize(input_dim=self.s, output_dim=self.p)
            self.first_layer = self.mlp.first_layer

            # Output projection
            self.W = nn.Parameter(
                torch.ones(self.p, self.d) * weight_scale 
            )

        elif network_type == 'CNN':
            # Build the convolutional network mapping from R^s to R^d
            # Reshape input to 1D signal for 1D convolutions
            assert len(hidden_units) >= 2, \
               "CNN requires at least 2 values: [num_filters, kernel_size, ...]"
            
            # First layer for proximal operations
            self.first_layer = MaskedLayer(self.s, self.s, torch.eye(self.s))
            self.first_layer.weight.data.fill_(0.1)  # Initialize weights

            # Spatial CNN decoder
            self.cnn = SpatialCNN(hidden_units=hidden_units)
            self.cnn.initialize(input_dim=self.s, output_dim=self.p)

            # Output projection
            self.W = nn.Parameter(
                torch.ones(self.p, self.d) * weight_scale 
            )

        elif network_type == 'UNET':
            # First layer for proximal operations
            self.first_layer = MaskedLayer(self.s, self.s, torch.eye(self.s))
            self.first_layer.weight.data.fill_(0.1)  # Initialize weights

            self.unet = UNET(conv1=hidden_units[0], conv2=hidden_units[1])
            self.unet.initialize(input_dim=self.s, output_dim=self.p)

            # Output projection
            self.W = nn.Parameter(
                torch.ones(self.p, self.d) * weight_scale 
            )

        elif 'PiNet' in network_type:
            # PiNetCCP, PiNetNCP, or PiNetNCPSkip (with/without ProdPoly)
            assert len(hidden_units) == 3, \
                "PiNetCCP requires exactly 3 hidden units: \
                [in_dim, inter_dim, out_dim]."
            in_dim, inter_dim, out_dim = hidden_units

            assert out_dim == self.p, \
                f"Output dimension {out_dim} must match the original \
                    state dimension {self.p}."
            
            # First layer (used in proximal step)
            self.first_layer = nn.Linear(self.s, in_dim, bias=False)
            # self.first_layer = MaskedLayer(self.s, in_dim, torch.eye(self.s))
            # self.first_layer.weight.data.fill_(0.1)  # Initialize weights
            
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

            # Output projection
            self.W = nn.Parameter(
                torch.ones(self.p, self.d) * weight_scale 
            )

        elif network_type == 'QM':
            self.first_layer = MaskedLayer(self.s, self.s, torch.eye(self.s))
            self.first_layer.weight.data.fill_(0.1)  
            self.nonlin_map = _quadratic_mapping_vectorized
            self.W = nn.Parameter(
                torch.ones(self.s * (self.s + 1) // 2, self.d))
        elif network_type == 'CM':
            self.first_layer = MaskedLayer(self.s, self.s, torch.eye(self.s))
            self.first_layer.weight.data.fill_(0.1)  
            self.nonlin_map = _cubic_mapping_vectorized
            self.W = nn.Parameter(
                torch.ones(self.s * (self.s + 1) * (self.s + 2) // 6, self.d))
            
    
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
        if self.network_type == 'MLP': 
            # Apply the NN to the reduced states 
            h        = self.mlp(z_hat)                  # (batch, p)
            x_hat_nn = h @ self.W                       # (batch, d)
        elif self.network_type == 'CNN':
            h        = self.first_layer(z_hat)          # (batch, s)
            h        = self.cnn(h)                      # (batch, p)
            x_hat_nn = h @ self.W                       # (batch, d)
        elif self.network_type == 'UNET':
            h        = self.first_layer(z_hat)          # (batch, s)
            h        = self.unet(h)                     # (batch, p)
            x_hat_nn = h @ self.W                       # (batch, d)
        elif 'PiNet' in self.network_type:
            h        = self.first_layer(z_hat)          # (batch, inter_dim)
            h        = self.pinet(h)                    # (batch, p)
            x_hat_nn = h @ self.W                       # (batch, d)
        else:
            # Apply the quadratic mapping
            h        = self.first_layer(z_hat)          
            z_quad   = self.nonlin_map(h)               # (batch, g(r))
            x_hat_nn = z_quad @ self.W                  # (batch, d)

        # --- Reconstruct x_hat ---
        x_hat = x_hat_lin + x_hat_nn
       
        return z_hat, x_hat


    def l1_norm_omega(self):
        """Return ℓ₁-norm of ω."""
        return self.omega.abs().sum()


    def orthogonalize_W(self):
        """Apply Gram-Schmidt orthogonalization to ensure projection ⊥ Ur"""
        with torch.no_grad():
            # Project out the POD basis components
            # P_orth = P - U_s @ (U_s.T @ P)
            proj_on_Us = (self.W @ self.U_s) @ (self.U_s.T)  
            self.W.data = self.W - proj_on_Us


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


class StateDecoder(nn.Module):
    def __init__(self, pod_basis: torch.Tensor, mapping_dim: int, 
                 hidden_units: list, gamma: float, weight_scale: float,
                 network_type: str, poly_order: int, 
                 num_polys: int, drop_linear: bool, 
                 drop_constant: bool, normalize: str | None = None,
                 dtype: torch.dtype = torch.float32):
        super(StateDecoder, self).__init__()
        
        self.register_buffer('U_r', pod_basis)  # (d, r): store as buffer
        self.d, self.r = pod_basis.shape
        self.p = mapping_dim
        self.gamma = gamma
        torch.set_default_dtype(dtype)
        
        # Assertion for network type
        assert network_type in ['MLP', 'CNN', 'UNET', 'PiNetCCP', 'PiNetNCP', 
                       'PiNetNCPSkip', 'QM', 'CM'], \
            f"Unsupported network type: {network_type}. " \
            "Use 'MLP', 'CNN', 'UNET', 'PiNetCCP', 'PiNetNCP', 'PiNetNCPSkip', " \
            "'QM', or 'CM'."
        self.network_type = network_type
        
        # Pi-Net dictionary
        PiNet = {
            'PiNetCCP': PiNetCCP,
            'PiNetNCP': PiNetNCP,
            'PiNetNCPSkip': PiNetNCPSkip
        }
        
        if network_type == 'MLP': 
            # Build the feedforward network mapping from R^r to R^d
            self.mlp = MLP(hidden_units=hidden_units, bias=False,
                           weight_scale=weight_scale)
            self.mlp.initialize(input_dim=self.r, output_dim=self.p)

            # Output projection
            self.W = torch.nn.Parameter(
                torch.ones(self.p, self.d, dtype=dtype) * weight_scale
            )

        elif network_type == 'CNN':
            # Build the convolutional network mapping from R^s to R^d
            # Reshape input to 1D signal for 1D convolutions
            assert len(hidden_units) >= 2, \
               "CNN requires at least 2 values: [num_filters, kernel_size, ...]"
            
            # Spatial CNN decoder
            self.cnn = SpatialCNN(hidden_units=hidden_units, bias=False)
            self.cnn.initialize(input_dim=self.r, output_dim=self.p)

            # Output projection
            self.W = torch.nn.Parameter(
                torch.ones(self.p, self.d, dtype=dtype) * weight_scale
            )

        elif network_type == 'UNET':
            # Spatial UNET decoder
            self.unet = UNET(conv1=hidden_units[0], conv2=hidden_units[1], bias=False)
            self.unet.initialize(input_dim=self.r, output_dim=self.p)

            # Output projection
            self.W = torch.nn.Parameter(
                torch.ones(self.p, self.d, dtype=dtype) * weight_scale
            )
            
        elif 'PiNet' in network_type:
            # PiNetCCP, PiNetNCP, or PiNetNCPSkip (with/without ProdPoly)
            assert len(hidden_units) == 3, \
                "PiNetCCP requires exactly 3 hidden units: \
                [in_dim, inter_dim, out_dim]."
            in_dim, inter_dim, out_dim = hidden_units

            # in_dim = self.r
            # _, inter_dim, out_dim = hidden_units

            assert out_dim == self.p, \
                f"Output dimension {out_dim} must match the original \
                    state dimension {self.p}."
                    
            # First layer (used in proximal step)
            self.first_layer = nn.Linear(self.r, in_dim, bias=True)
            
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

            # Output projection
            self.W = torch.nn.Parameter(
                torch.ones(self.p, self.d, dtype=dtype) * weight_scale
            )

        elif network_type == 'QM':
            self.nonlin_map = _quadratic_mapping_vectorized
            self.W = torch.nn.Parameter(
                torch.ones(
                self.r * (self.r + 1) // 2, self.d, dtype=dtype) * weight_scale
            )
        elif network_type == 'CM':
            self.nonlin_map = _cubic_mapping_vectorized
            self.W = torch.nn.Parameter(
            torch.ones(self.r * (self.r + 1) * (self.r + 2) // 6, self.d,
                dtype=dtype) * weight_scale
            )

    def forward(self, z_batch):
        # Reconstruct the linear part via projection
        x_hat_lin = z_batch @ self.U_r.T            # (batch, d)
        
        # --- MLP or MLP + Π-net hybrid ---
        if self.network_type == 'MLP': 
            # Apply the NN to the reduced states 
            h = self.mlp(z_batch)                   # (batch, p)
        elif self.network_type == 'CNN':
            h = self.cnn(z_batch)                   # (batch, p)
        elif self.network_type == 'UNET':
            h = self.unet(z_batch)                  # (batch, p)
        elif 'PiNet' in self.network_type:
            h = self.first_layer(z_batch)           # (batch, inter_dim)
            h = self.pinet(h)                       # (batch, p)
        else:
            # Apply the quadratic or cubic mapping
            h = self.nonlin_map(z_batch)            # (batch, g(r))

        # Project output to the original space
        x_hat_nn = h @ self.W                       # (batch, d)

        # --- Reconstruct x_hat ---
        x_hat = x_hat_lin + x_hat_nn

        return x_hat, x_hat_lin, h
    
    def update_nonlinear_weight(self, residual, fnn_out, reg_):
        self.projection.data.copy_(lstsq_l2_torch(
            fnn_out, residual, reg_magnitude=reg_
        )[0])
        return None
    
    def orthogonalize_W(self):
        """Apply Gram-Schmidt orthogonalization to ensure projection ⊥ Ur"""
        with torch.no_grad():
            # Project out the POD basis components
            # P_orth = P - U_r @ (U_r.T @ P)
            proj_on_Ur = (self.W @ self.U_r) @ (self.U_r.T)  
            self.W.data = self.W - proj_on_Ur


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
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products
        result = x[i_indices] * x[j_indices]
        return result
    else:
        _, n = x.shape
        # Create indices for upper triangular part  
        i_indices, j_indices = torch.tril_indices(n, n, device=x.device)
        # Compute products for all batches
        result = x[:, i_indices] * x[:, j_indices]
        return result   


def _cubic_mapping_vectorized(x):
    """
    Fast vectorized computation of unique cubic terms x ⊗ x ⊗ x.
    Uses meshgrid for efficient index generation.
    
    Args:
        x: torch.Tensor of shape (batch_size, n) or (n,)
        
    Returns:
        torch.Tensor of shape (batch_size, n*(n+1)*(n+2)//6) or (n*(n+1)*(n+2)//6,)
    """
    if x.dim() == 1:
        n = x.size(0)
        # Create meshgrid for all combinations
        i_range = torch.arange(n, device=x.device)
        i_grid, j_grid, k_grid = torch.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products
        result = x[i_indices] * x[j_indices] * x[k_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create meshgrid for all combinations
        i_range = torch.arange(n, device=x.device)
        i_grid, j_grid, k_grid = torch.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products for all batches
        result = x[:, i_indices] * x[:, j_indices] * x[:, k_indices]
        return result
