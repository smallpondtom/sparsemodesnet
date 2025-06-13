import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_concrete(logits: torch.Tensor,
                    temperature: float = 0.5,
                    hard: bool = True) -> torch.Tensor:
    """
    Draw a sample from the Concrete (Gumbel-Softmax) distribution for Bernoulli logits.
    Returns a differentiable approximation to {0,1} with optional straight-through hardening.
    """
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
    y = torch.sigmoid((logits + g) / temperature)
    if hard:
        y_hard = (y > 0.5).float()
        return y_hard.detach() - y.detach() + y
    return y


class SparseModesNet(nn.Module):
    """
    SparseModesNet with binary gating via Concrete relaxation and
    feedforward + convolutional proximal updates.
    """
    def __init__(self,
                 pod_basis: torch.Tensor,
                 input_dim: int,
                 hidden_units: list,
                 temperature: float = 0.5,
                 M: float = 5.0,
                 lam: float = 1e-3,
                 network_type: str = 'feedforward',
                 **conv_kwargs):
        super().__init__()
        self.U_s = pod_basis
        self.d, self.s = pod_basis.shape
        self.temperature = temperature
        self.M = float(M)
        self.lam = float(lam)
        self.network_type = network_type

        # logits for Bernoulli gating
        self.omega_logits = nn.Parameter(torch.zeros(self.s))

        # build network
        if network_type == 'feedforward':
            self.net = self._build_feedforward(hidden_units)
            self.first_layer = self.net[0]
        elif network_type == 'convolutional':
            self.net = self._build_convolutional(hidden_units, **conv_kwargs)
            # first conv is net[1]
            self.first_conv = [l for l in self.net if isinstance(l, nn.Conv1d)][0]
        else:
            raise ValueError("Unsupported network_type")

    def _build_feedforward(self, hidden_units):
        layers = [nn.Linear(self.s, hidden_units[0], bias=False), nn.ReLU(inplace=True)]
        for in_dim, out_dim in zip(hidden_units[:-1], hidden_units[1:]):
            layers += [nn.Linear(in_dim, out_dim, bias=False), nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden_units[-1], self.s, bias=False))
        return nn.Sequential(*layers)

    def _build_convolutional(self, hidden_units, kernel_size=3, num_channels=None, padding='same'):
        if num_channels is None:
            num_channels = [16,32,16]
        layers = [nn.Unflatten(1,(1,self.s)),
                  nn.Conv1d(1, num_channels[0], kernel_size, padding=padding, bias=False),
                  nn.ReLU(inplace=True)]
        for in_ch, out_ch in zip(num_channels[:-1], num_channels[1:]):
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, bias=False), nn.ReLU(inplace=True)]
        layers += [nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(num_channels[-1], self.s, bias=False)]
        return nn.Sequential(*layers)

    def forward(self, z_batch):
        self.omega = sample_concrete(self.omega_logits, self.temperature, hard=True)
        z_hat = z_batch * self.omega.unsqueeze(0)
        nn_out = self.net(z_batch)
        x_hat = z_hat @ self.U_s.T + nn_out @ self.U_s.T
        return z_hat, x_hat

    def l1_norm_b(self):
        return torch.sigmoid(self.omega_logits).sum()

    def proximal_step(self):
        if self.network_type == 'feedforward':
            return self._proximal_step_feedforward()
        elif self.network_type == 'convolutional':
            return self._proximal_step_conv()
        else:
            raise ValueError("Unknown network_type")

    def _proximal_step_feedforward(self):
        with torch.no_grad():
            W1 = self.first_layer.weight.data  # (h,s)
            W1_T = W1.t().contiguous()         # (s,h)
            s,K = W1_T.shape

            u_abs_sorted,_ = W1_T.abs().sort(dim=1,descending=True)
            zeros = torch.zeros((s,1),device=W1.device)
            cumsum = torch.cumsum(u_abs_sorted,dim=1)
            a_s = self.lam - self.M*torch.cat([zeros,cumsum],dim=1)

            p = torch.sigmoid(self.omega_logits).unsqueeze(1).expand(-1,K+1)
            m_idx = torch.arange(K+1,device=W1.device).view(1,-1).expand(s,-1)
            x_vals = F.relu(1 - a_s/(p+1e-16)) / (1 + m_idx*(self.M**2))
            w_vals = self.M * x_vals * p

            lower = torch.cat([u_abs_sorted,zeros],dim=1)
            idx = (lower > w_vals).sum(dim=1)
            rows = torch.arange(s,device=W1.device)
            x_star = x_vals[rows,idx]
            b_new = x_star * p[:,0]

            # clip weights
            w_star = w_vals[rows,idx]
            clipped = torch.min(W1_T.abs(), w_star.unsqueeze(1).expand(-1,K))
            W1_T_new = W1_T.sign() * clipped
            self.first_layer.weight.data.copy_(W1_T_new.t().contiguous())

            # update logits
            eps=1e-6
            p_new = b_new.clamp(eps,1-eps)
            self.omega_logits.data.copy_(torch.log(p_new/(1-p_new)))

    def _proximal_step_conv(self):
        with torch.no_grad():
            # gather conv layers
            convs = [l for l in self.net if isinstance(l, nn.Conv1d)]
            if len(convs) < 2:
                return
            first_conv = convs[0]
            second = convs[1]

            omega_p = torch.sigmoid(self.omega_logits)
            out_ch1 = first_conv.out_channels
            num_features = min(out_ch1, self.s)

            # get dependent weights
            if isinstance(second, nn.Conv1d):
                dw = second.weight.data[:, :num_features, :]
                orig = dw.shape
                dw_flat = dw.permute(1,0,2).contiguous().view(num_features,-1)
            else:
                # linear after pooling
                lin = [l for l in self.net if isinstance(l, nn.Linear)][-1]
                dw_flat = lin.weight.data[:,:num_features].t()
                orig = None

            norms = dw_flat.abs().max(dim=1)[0]
            thresh = self.M * omega_p[:num_features]

            needs = norms > thresh
            scale = torch.ones_like(norms)
            valid = thresh > 1e-12
            scale[needs & valid] = thresh[needs & valid]/norms[needs & valid]
            scale[needs & ~valid] = 0.0

            dw_scaled = dw_flat * scale.unsqueeze(1)

            # write back
            if orig is not None:
                oc2, oc1, ks = orig
                dw_r = dw_scaled.view(oc1, oc2, ks).permute(1,0,2)
                second.weight.data[:, :num_features, :] = dw_r
            else:
                lin.weight.data[:,:num_features] = dw_scaled.t()

            # update logits
            new_norms = dw_scaled.abs().max(dim=1)[0]
            soft = F.relu(omega_p[:num_features] - self.lam/self.M)
            new_p = torch.max(new_norms/self.M, soft)
            eps=1e-6
            p_cl = new_p.clamp(eps,1-eps)
            self.omega_logits.data[:num_features] = torch.log(p_cl/(1-p_cl))


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SoftplusParameterization(nn.Module):
#     def forward(self, X):
#         return F.softplus(X)
    
# # Helper class for reshaping in sequential
# class Reshape(nn.Module):
#     def __init__(self, shape):
#         super().__init__()
#         self.shape = shape
    
#     def forward(self, x):
#         return x.view(self.shape)
    
# class SparseModesNet(nn.Module):
#     """
#     LassoNet in POD-space (R^s → R^s), but loss computed in original x-space:
#       For each sample i:
#         z_i = U_s^T x_i
#         z_hat_i = ω ⊙ z_i + f(z_i)
#         x_hat_i = U_s z_hat_i
#       Minimize ∑ ||x_i - x_hat_i||^2 + λ ||ω||_1, subject to the hierarchy constraint:
#         ||W^(1)[j,:]||_∞ ≤ M |ω_j| for all j ∈ {0..s-1}.
#     """
    
#     def __init__(self, pod_basis: torch.Tensor, input_dim: int, 
#                 hidden_units: list, M: float = 5.0, lam: float = 1e-3, 
#                 network_type: str = 'feedforward', **conv_kwargs):
#         """
#         pod_basis:  U_s ∈ R^{d x s}  (torch.Tensor)
#         input_dim:  s (POD dimension)
#         hidden_units:  e.g. [s, s(s+1)/2, s(s+1)(s+2)/6] (quadratic, cubic, etc.)
#         M: hierarchy multiplier
#         lam: ℓ₁ penalty on ω
#         network_type: 'feedforward' or 'convolutional'
#         conv_kwargs: additional arguments for convolutional network
#             - kernel_size: int, default 3
#             - num_channels: list, e.g. [16, 32, 16]
#             - padding: str, default 'same'
#         """
#         super(SparseModesNet, self).__init__()

#         self.U_s = pod_basis  # (d, s)
#         self.d, self.s = pod_basis.shape
#         self.network_type = network_type

#         self.M = float(M)
#         self.lam = float(lam)

#         # Skip‐weights ω ∈ R^s
#         self.omega_raw = nn.Parameter(torch.ones(self.s) * 0.1)
        
#         # Softplus parameterization to ensure omega > 0
#         self.softplus = SoftplusParameterization()

#         # Build f_NN in POD-space based on network type
#         if network_type == 'feedforward':
#             self.net = self._build_feedforward_network(hidden_units)
#         elif network_type == 'convolutional':
#             self.net = self._build_convolutional_network(hidden_units, **conv_kwargs)
#         else:
#             raise ValueError(f"Unsupported network_type: {network_type}. Use 'feedforward' or 'convolutional'.")
        
#     def _build_feedforward_network(self, hidden_units):
#         """Build standard feedforward network"""
#         self.first_layer = nn.Linear(self.s, hidden_units[0], bias=False)
#         layers = [self.first_layer, nn.ReLU(inplace=True)]
        
#         for i in range(1, len(hidden_units)):
#             layers.append(nn.Linear(hidden_units[i-1], hidden_units[i], bias=False))
#             layers.append(nn.ReLU(inplace=True))
        
#         layers.append(nn.Linear(hidden_units[-1], self.d, bias=False))
#         return nn.Sequential(*layers)

#     def _build_convolutional_network(self, hidden_units, kernel_size=3, 
#                                      num_channels=None, padding='same'):
#         """Build 1D convolutional network for POD coefficients with proper LassoNet structure"""
#         if num_channels is None:
#             num_channels = [16, 32, 16]  # Default channel progression
        
#         layers = []
        
#         # Input reshape: (batch, s) -> (batch, 1, s) for 1D conv
#         layers.append(Reshape((-1, 1, self.s)))
        
#         # First conv layer - this is our "first_layer" for hierarchical constraint
#         self.first_conv = nn.Conv1d(1, num_channels[0], kernel_size=kernel_size, padding=padding, bias=False)
#         layers.append(self.first_conv)
#         layers.append(nn.ReLU(inplace=True))
        
#         # Subsequent convolutional layers
#         for i in range(1, len(num_channels)):
#             layers.append(nn.Conv1d(
#                 num_channels[i-1], num_channels[i], 
#                 kernel_size=kernel_size, 
#                 padding=padding, 
#                 bias=False
#             ))
#             layers.append(nn.ReLU(inplace=True))
        
#         # Output projection: (batch, channels[-1], s) -> (batch, s)
#         layers.append(nn.AdaptiveAvgPool1d(1))  # Global average pooling -> (batch, channels[-1], 1)
#         layers.append(nn.Flatten())             # -> (batch, channels[-1])
#         layers.append(nn.Linear(num_channels[-1], self.s, bias=False))
        
#         # Store first layer for proximal step - this is crucial for LassoNet
#         self.first_layer = self.first_conv
        
#         return nn.Sequential(*layers)

#     @property
#     def omega(self):
#         """Get the positive omega values using softplus"""
#         return self.softplus(self.omega_raw)
    
#     def forward(self, z_batch):
#         """
#         z_batch: (batch_size, s)
#         Returns:
#           z_hat_batch: (batch_size, s)
#           x_hat_batch: (batch_size, d)
#         """
#         ## (1)
#         z_hat = z_batch * self.omega.unsqueeze(0)  # (batch, s)
#         nn_out = self.net(z_batch)                # (batch, s)

#         # Reconstruct to x-space: 
#         x_tilde = z_hat @ self.U_s.T   # (d, batch)
#         x_hat = x_tilde + nn_out
       
#         return z_hat, x_hat
    
#     def l1_norm_b(self):
#         """Return ℓ₁-norm of ω."""
#         return self.omega.abs().sum()

#     @staticmethod
#     def _row_inf_norm(mat: torch.Tensor) -> torch.Tensor:
#         """
#         Given mat: (s, h), return length-s vector of rowwise l-infinity norms.
#         """
#         return mat.abs().max(dim=1)[0]
                
#     def _proximal_step_conv(self):
#         """
#         Efficient batched proximal step for convolutional networks
#         """
#         with torch.no_grad():
#             if self.network_type != 'convolutional':
#                 return
                
#             # Find the second layer (conv or linear)
#             second_layer = None
#             conv_layers = [layer for layer in self.net if isinstance(layer, nn.Conv1d)]
            
#             if len(conv_layers) >= 2:
#                 second_layer = conv_layers[1]  # Second conv layer
#                 is_conv_to_conv = True
#             else:
#                 # Find first linear layer after conv
#                 for layer in self.net:
#                     if isinstance(layer, nn.Linear):
#                         second_layer = layer
#                         is_conv_to_conv = False
#                         break
            
#             if second_layer is None:
#                 return
            
#             # Get current omega values and dimensions
#             omega_abs = self.omega.abs()  # (s,)
#             out_channels = self.first_conv.out_channels
#             num_features = min(out_channels, len(omega_abs))
            
#             if num_features == 0:
#                 return
            
#             if is_conv_to_conv:
#                 # Conv-to-conv: shape (out_ch2, out_ch1, kernel_size)
#                 dependent_weights = second_layer.weight[:, :num_features, :]
#                 orig_shape = dependent_weights.shape
#                 # Fix: Use reshape instead of view, and ensure contiguity
#                 dependent_weights = dependent_weights.permute(1, 0, 2).contiguous()  # (num_features, out_ch2, kernel_size)
#                 dependent_weights = dependent_weights.reshape(num_features, -1)  # (num_features, out_ch2*kernel_size)
#             else:
#                 # Conv-to-linear: after global pooling, shape (out_features, num_features)
#                 if second_layer.weight.shape[1] >= num_features:
#                     dependent_weights = second_layer.weight[:, :num_features].t()  # (num_features, out_features)
#                 else:
#                     return
            
#             num_features, K = dependent_weights.shape
            
#             # Apply hierarchical constraint per feature (vectorized)
#             # For each feature j: ||dependent_weights[j,:]||∞ ≤ M * |omega[j]|
            
#             # 1) Get infinity norms per feature
#             weight_norms = torch.max(torch.abs(dependent_weights), dim=1)[0]  # (num_features,)
            
#             # 2) Compute thresholds
#             thresholds = self.M * omega_abs[:num_features]  # (num_features,)
            
#             # 3) Find features that need scaling
#             needs_scaling = weight_norms > thresholds
            
#             # 4) Compute scale factors (vectorized)
#             scale_factors = torch.ones_like(weight_norms)
#             valid_thresholds = thresholds > 1e-12  # Avoid division by zero
#             scale_factors[needs_scaling & valid_thresholds] = (
#                 thresholds[needs_scaling & valid_thresholds] / 
#                 weight_norms[needs_scaling & valid_thresholds]
#             )
#             scale_factors[needs_scaling & ~valid_thresholds] = 0.0
            
#             # 5) Apply scaling (vectorized)
#             dependent_weights *= scale_factors.unsqueeze(1)
            
#             # 6) Write back the results
#             if is_conv_to_conv:
#                 # Reshape back to original conv format
#                 out_ch2, kernel_size = orig_shape[0], orig_shape[2]
#                 # Fix: Use reshape and ensure proper reshaping
#                 dependent_weights_reshaped = dependent_weights.reshape(num_features, out_ch2, kernel_size)
#                 dependent_weights_reshaped = dependent_weights_reshaped.permute(1, 0, 2)  # (out_ch2, num_features, kernel_size)
#                 second_layer.weight.data[:, :num_features, :] = dependent_weights_reshaped
#             else:
#                 # Linear layer
#                 second_layer.weight.data[:, :num_features] = dependent_weights.t()
            
#             # **Key Fix: Update omega using the hierarchical relationship**
#             # Instead of direct soft thresholding, use the constraint relationship
#             # If weights were scaled, omega should be updated to maintain the constraint
            
#             # Compute new effective omega values based on actual weight norms
#             new_weight_norms = torch.max(torch.abs(dependent_weights), dim=1)[0]
            
#             # Omega should satisfy: M * omega >= weight_norm
#             # So: omega >= weight_norm / M
#             # But we also want to minimize omega (L1 penalty), so:
#             # omega = max(weight_norm / M, soft_threshold(omega_old, lambda/M))
            
#             soft_thresh_omega = torch.relu(omega_abs[:num_features] - self.lam / self.M)
#             new_omega_vals = torch.max(new_weight_norms / self.M, soft_thresh_omega)
            
#             # Update omega_raw using inverse softplus (vectorized)
#             new_omega_raw = torch.log(torch.exp(new_omega_vals) - 1 + 1e-8)
#             self.omega_raw.data[:num_features] = new_omega_raw

#     def proximal_step(self):
#         """
#         Main proximal step that handles both feedforward and convolutional networks
#         """
#         if self.network_type == 'feedforward':
#             self._proximal_step_feedforward()
#         elif self.network_type == 'convolutional':
#             self._proximal_step_conv()
#         else:
#             raise ValueError(f"Unknown network_type: {self.network_type}")

#     def _proximal_step_feedforward(self):
#         """
#         Batched implementation of Algorithm 4 (Group-Hierarchical Proximal) with λ̄ = 0,
#         corrected so that ω_new = x_star * θ (no extra soft-threshold on ω).
        
#         Note: The `v`, `θ`, and `u` notations are presented in the origina paper,
#         but here we use `omega` for θ. To clarify confusion with the notation,
#         please refer to the original paper.
#         """
#         lam = self.lam
#         M   = self.M

#         # 1) Gather first‐layer weights W1 ∈ ℝ^{h×s}, then transpose → W1_T ∈ ℝ^{s×h}
#         W1   = self.first_layer.weight.data           # (h, s)
#         W1_T = W1.t().contiguous()                    # (s, h), call h=K

#         s, K = W1_T.shape  # s = #features, K = width of first hidden layer

#         # 2) Sort each row of |W1_T| in descending order (batched)
#         u_abs_sorted, _ = W1_T.abs().sort(dim=1, descending=True)  # (s, K)

#         # 3) Build partial sums a_s(m) = lam - M * sum_{i=1}^m u_abs_sorted[j,i-1]
#         zeros_m     = torch.zeros((s, 1), device=W1_T.device, dtype=W1_T.dtype)  # (s,1)
#         cumsum_vals = torch.cumsum(u_abs_sorted, dim=1)  # (s, K)
#         a_s = lam - M * torch.cat([zeros_m, cumsum_vals], dim=1)  # (s, K+1)

#         # 4) ‖v‖₂ = |θ|, shape (s,)
#         # theta_abs = self.omega.data.abs()  # (s,)
#         theta_abs = self.omega.abs()  # (s,)

#         # 5) Broadcast |θ| into (s, K+1)
#         norm_v_col = theta_abs.unsqueeze(1).expand(-1, K+1)  # (s, K+1)

#         # 6) Build m_index = [0,1,...,K] for each of s rows
#         m_index = torch.arange(K+1, device=W1_T.device, dtype=W1_T.dtype).view(1, K+1)
#         m_index = m_index.expand(s, -1)  # (s, K+1)

#         # 7) Compute x_vals(m) = ReLU(1 - a_s / ‖v‖) / (1 + m*M^2)
#         x_vals = F.relu(1.0 - a_s / (norm_v_col + 1e-16)) / (1.0 + m_index * (M**2))  # (s, K+1)

#         # 8) Compute w_vals(m) = M * x_vals(m) * ‖v‖
#         w_vals = M * x_vals * norm_v_col  # (s, K+1)

#         # 9) Build “lower(m)” = [u_abs_sorted, 0], shape (s, K+1)
#         lower = torch.cat([u_abs_sorted, zeros_m], dim=1)  # (s, K+1)

#         # 10) Find index m* per row:  m*_j = sum_{m=0..K} [ lower[j,m] > w_vals[j,m] ]
#         cond = lower > w_vals          # (s, K+1), bool
#         idx  = torch.sum(cond, dim=1)  # (s,)  ← m* for each feature j

#         # 11) Gather x_star[j] = x_vals[j, idx[j]]  and  w_star[j] = w_vals[j, idx[j]]
#         row_idx = torch.arange(s, device=W1_T.device)
#         x_star  = x_vals[row_idx, idx]  # (s,)
#         w_star  = w_vals[row_idx, idx]  # (s,)

#         # 12) ***CORRECTED***  Update skip‐weights:  b_new[j] = x_star[j] * θ_j
#         # No extra soft‐threshold here, because λ was already used in building a_s→x_vals.
#         b_new = x_star * self.omega.data  # (s,)

#         # 13) Coordinate‐wise clip each row of W1_T to ±w_star[j]:
#         W1_T_abs   = W1_T.abs()                         # (s, K)
#         w_star_col = w_star.unsqueeze(1).expand(-1, K)  # (s, K)
#         clipped_abs = torch.min(W1_T_abs, w_star_col)   # (s, K)
#         W1_T_new   = W1_T.sign() * clipped_abs          # (s, K)

#         # 14) Write back:
#         # self.omega.data.copy_(b_new)               # (s,)
#         W1_updated = W1_T_new.t().contiguous() # shape: (K, s) → transpose to (h, s)
#         self.first_layer.weight.data.copy_(W1_updated)
    
#         with torch.no_grad():
#             # Convert back to unconstrained space using inverse softplus
#             # softplus^(-1)(x) = log(exp(x) - 1) for x > 0
#             # For numerical stability, use: log(exp(x) - 1) ≈ x - log(2) for large x
#             new_omega_raw = torch.log(torch.exp(b_new) - 1 + 1e-8)
#             self.omega_raw.data = new_omega_raw
