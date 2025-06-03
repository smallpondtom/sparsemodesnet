#%%

import torch
import torch.nn.functional as F

# Helper functions from the authors
def soft_threshold(lam: float, x: torch.Tensor) -> torch.Tensor:
    return x.sign() * F.relu(x.abs() - lam)

def sign_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x >= 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.tensor(-1.0, device=x.device, dtype=x.dtype)
    )

def prox(v: torch.Tensor, u: torch.Tensor, *,
         lambda_: float, lambda_bar: float, M: float) -> (torch.Tensor, torch.Tensor):
    onedim = (v.ndim == 1)
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values
    K, batch = u.shape
    s_t = torch.arange(K + 1.0, device=v.device, dtype=v.dtype).view(-1, 1)
    zeros = torch.zeros(1, batch, device=v.device, dtype=v.dtype)

    a_s = lambda_ - M * torch.cat([zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)], dim=0)
    norm_v = torch.norm(v, p=2, dim=0)

    x_vals = F.relu(1 - a_s / norm_v) / (1 + s_t * (M**2))
    w_vals = M * x_vals * norm_v

    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros], dim=0)
    idx = torch.sum(lower > w_vals, dim=0, keepdim=True)

    x_star = torch.gather(x_vals, 0, idx)
    w_star = torch.gather(w_vals, 0, idx)

    beta_star  = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)
    return beta_star, theta_star

# Closed‐form attempt (which we now know is wrong in general)
def closed_form(v: torch.Tensor, u: torch.Tensor, lam: float, M: float):
    w = u.abs().max()  # ||u||_∞
    if w <= M * v.abs():
        b_new_abs = torch.clamp(v.abs() - lam, min=0.0)
        b_new = v.sign() * b_new_abs
        w_new = w
        if w > 0:
            scale = w_new / w
            u_new = u * scale
        else:
            u_new = torch.zeros_like(u)
        return b_new, u_new

    # else w > M*|v|
    alpha = (M * v.abs() + w) / (M*M + 1.0)
    b_new_abs = torch.clamp(alpha - lam, min=0.0)
    b_new = v.sign() * b_new_abs
    w_new = M * b_new_abs
    if w > 0:
        scale = w_new / w
        u_new = u * scale
    else:
        u_new = torch.zeros_like(u)
    return b_new, u_new

#%% Example case where they differ
torch.manual_seed(0)
v = torch.randn(1)            # single scalar
u = torch.tensor([2.0, 5.0, 1.0])  # a small vector

lam = 1.0
M   = 2.0

v_prox, u_prox = prox(v.clone(), u.clone(), lambda_=lam, lambda_bar=0.0, M=M)
v_cf,   u_cf   = closed_form(v.clone(), u.clone(), lam, M)

print("v     =", v.item())
print("u     =", u.tolist())
print()
print("Authors’ prox output:")
print("  v_prox =", v_prox.item())
print("  u_prox =", u_prox.tolist())
print()
print("Closed‐form output:")
print("  v_cf   =", v_cf.item())
print("  u_cf   =", u_cf.tolist())
# %%
import time
import torch

# Create random test data
s, K = 200, 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Fake θ ∈ ℝ^s  and  W1_T ∈ ℝ^{s×K}
theta = torch.randn(s, device=device)
W1_T  = torch.randn(s, K, device=device)

lam = 1e-3
M   = 5.0

# --- 1) Time the old “loop + prox(...)” version --- #
def proximal_old(theta, W1_T):
    # (Assumes prox(v,u,lam,0,M) is already defined and on the same device)
    b = theta.clone()
    W = W1_T.clone()
    s, K = W.shape
    for j in range(s):
        v_j = b[j].view(1)
        u_j = W[j, :].clone()
        v_star, u_star = prox(v_j, u_j, lambda_=lam, lambda_bar=0.0, M=M)
        b[j] = v_star.squeeze()
        W[j, :] = u_star.squeeze()
    return b, W

#%%
# Warm‐up
for _ in range(10):
    _ = proximal_old(theta, W1_T)

# Time it
t0 = time.time()
for _ in range(30):
    u, v = proximal_old(theta, W1_T)
    print(f"b: {u[:5].tolist()}, W: {v[:5, :].tolist()}")  # Print first 5 for brevity
t1 = time.time()
print(f"Old loop+prox: {(t1 - t0)/30:.4f} sec per pass")

#%%
# --- 2) Time the new batched version --- #
def proximal_batched(theta, W1_T):
    b_vals = theta.abs()
    W = W1_T.clone()
    s, K = W.shape

    # 1) sort each row of |W|
    u_abs_sorted, _ = W.abs().sort(dim=1, descending=True)  # (s, K)

    # 2) partial sums + a_s
    zeros_m    = torch.zeros((s, 1), device=W.device, dtype=W.dtype)
    cumsum_vals = torch.cumsum(u_abs_sorted, dim=1)  # (s, K)
    a_s        = lam - M * torch.cat([zeros_m, cumsum_vals], dim=1)  # (s, K+1)

    # 3) broadcast |θ|
    norm_v_col = b_vals.unsqueeze(1).expand(-1, K+1)  # (s, K+1)

    # 4) m index
    m_index = torch.arange(K+1, device=W.device, dtype=W.dtype).view(1, K+1)
    m_index = m_index.expand(s, -1)  # (s, K+1)

    # 5) x(m), w(m)
    x_vals = F.relu(1.0 - a_s / (norm_v_col + 1e-16)) / (1.0 + m_index * (M**2))  # (s, K+1)
    w_vals = M * x_vals * norm_v_col                                           # (s, K+1)

    # 6) lower = [u_abs_sorted, 0]
    lower = torch.cat([u_abs_sorted, zeros_m], dim=1)  # (s, K+1)

    # 7) find idx per row
    cond = lower > w_vals                              # (s, K+1)
    idx  = torch.sum(cond, dim=1)                       # (s,)

    # 8) gather x_star, w_star
    row_idx = torch.arange(s, device=W.device)
    x_star  = x_vals[row_idx, idx]                      # (s,)
    w_star  = w_vals[row_idx, idx]                      # (s,)

    # 9) update b
    b_signed = theta.sign()                             # (s,)
    # raw_b    = x_star * b_vals                          # (s,)
    # b_new    = b_signed * F.relu(raw_b - lam)           # (s,)
    b_new = b_signed * x_star * b_vals

    # 10) clip each coord of W
    W_abs    = W.abs()                                  # (s, K)
    w_star_col = w_star.unsqueeze(1).expand(-1, K)      # (s, K)
    clipped_abs = torch.min(W_abs, w_star_col)          # (s, K)
    W_new    = W.sign() * clipped_abs                   # (s, K)

    return b_new, W_new

#%%
# Warm‐up
for _ in range(10):
    _ = proximal_batched(theta, W1_T)

# Time it
t2 = time.time()
for _ in range(30):
    u, v = proximal_batched(theta, W1_T)
    print(f"b: {u[:5].tolist()}, W: {v[:5, :].tolist()}")  # Print first 5 for brevity
t3 = time.time()
print(f"Batched prox: {(t3 - t2)/30:.4f} sec per pass")
# %%
