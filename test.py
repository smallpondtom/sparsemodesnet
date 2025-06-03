#%%
import torch
import torch.nn.functional as F

#%%
# ————————————————————————————————————————————————————————————————————
# 1) The authors’ prox(...) implementation (with lambda_bar=0)
# ————————————————————————————————————————————————————————————————————

def soft_threshold(lam: float, x: torch.Tensor) -> torch.Tensor:
    return x.sign() * F.relu(x.abs() - lam)

def sign_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0,
                       torch.tensor(1.0, device=x.device, dtype=x.dtype),
                       torch.tensor(-1.0, device=x.device, dtype=x.dtype))

def prox(v: torch.Tensor, u: torch.Tensor, *,
         lambda_: float, lambda_bar: float, M: float) -> tuple[torch.Tensor, torch.Tensor]:
    # v.shape = (1,) or (1, batch)   ← the scalar θ_j
    # u.shape = (K,) or (K, batch)   ← the first-layer weights for feature j
    onedim = (v.ndim == 1)
    if onedim:
        v = v.unsqueeze(-1)  # shape → (1, 1)
        u = u.unsqueeze(-1)  # shape → (K, 1)

    # 1) sort |u| descending
    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values  # (K, batch)

    K, batch = u.shape
    s_t = torch.arange(K + 1.0, device=v.device, dtype=v.dtype).view(-1, 1)  # (K+1, 1)
    zeros = torch.zeros(1, batch, device=v.device, dtype=v.dtype)          # (1, batch)

    # 2) a_s(m) = λ_ - M * [0, cumsum(|u_(1)| - lambda_bar, … )]
    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)], dim=0
    )  # shape (K+1, batch)

    # 3) norm_v = ||v||₂
    norm_v = torch.norm(v, p=2, dim=0)  # shape (batch,)

    # 4) x(m) = max(0, 1 - a_s/‖v‖₂)/(1 + m*M²)
    x_vals = F.relu(1 - a_s / norm_v) / (1 + s_t * (M**2))  # (K+1, batch)

    # 5) w(m) = M * x(m) * ‖v‖₂
    w_vals = M * x_vals * norm_v  # (K+1, batch)

    # 6) lower(m) = S_{lambda_bar}(|u|_sorted)
    intervals = soft_threshold(lambda_bar, u_abs_sorted)   # (K, batch)
    lower = torch.cat([intervals, zeros], dim=0)           # (K+1, batch)

    # 7) find m̃
    idx = torch.sum(lower > w_vals, dim=0, keepdim=True)  # (1, batch)

    # 8) gather x_star, w_star
    x_star = torch.gather(x_vals, 0, idx)  # (1, batch)
    w_star = torch.gather(w_vals, 0, idx)  # (1, batch)

    # 9) updated skip‐weight = x_star * v
    beta_star = x_star * v  # (1, batch)

    # 10) updated W = sign(u) * min( S_{lambda_bar}(|u|), w_star )
    theta_star = sign_binary(u) * torch.min(
        soft_threshold(lambda_bar, u.abs()), w_star
    )

    if onedim:
        beta_star = beta_star.squeeze(-1)
        theta_star = theta_star.squeeze(-1)

    return beta_star, theta_star

#%%
# ————————————————————————————————————————————————————————————————————
# 2) Closed‐form ℓ∞‐projection + soft‐threshold version
# ————————————————————————————————————————————————————————————————————

def closed_form(v: torch.Tensor, u: torch.Tensor, lam: float, M: float):
    """
    Returns exactly the same (b_new, u_new) as prox(v,u,lambda_=lam,lambda_bar=0,M).
    """
    # 1) compute w = ||u||_∞
    w = u.abs().max()

    # 2) Case 1: if w <= M*|v|, do b_new=sign(v)*max(|v|-lam,0), w_new=w
    if w <= M * v.abs():
        b_new_abs = torch.clamp(v.abs() - lam, min=0.0)
        b_new = v.sign() * b_new_abs
        w_new = w
        # scale factor = w_new / w  (unless w=0)
        if w > 0:
            scale = w_new / w
            u_new = u * scale
        else:
            u_new = torch.zeros_like(u)

        return b_new, u_new

    # 3) Case 2: if w > M*|v|, project: α = (M*|v| + w)/(M²+1)
    alpha = (M * v.abs() + w) / (M*M + 1.0)
    # soft‐threshold α by lam
    b_new_abs = torch.clamp(alpha - lam, min=0.0)
    b_new = v.sign() * b_new_abs
    w_new = M * b_new_abs
    # scale u to have ||u_new||_∞ = w_new
    if w > 0:
        scale = w_new / w
        u_new = u * scale
    else:
        u_new = torch.zeros_like(u)

    return b_new, u_new

#%%
# ————————————————————————————————————————————————————————————————————
# 3) Quick check on a few random (v,u) to ensure exact agreement
# ————————————————————————————————————————————————————————————————————

torch.manual_seed(123)
num_tests = 10
K = 16
lam = 0.5
M = 3.0

max_diff_b = 0.0
max_diff_u = 0.0

for _ in range(num_tests):
    v_j = torch.randn(1)       # scalar θ_j
    u_j = torch.randn(K)       # vector of length K

    # 3.a) Run authors’ prox with lambda_bar=0.0
    v_prox, u_prox = prox(v_j.clone(), u_j.clone(),
                          lambda_    = lam,
                          lambda_bar = 0.0,
                          M          = M)

    # 3.b) Run closed‐form version
    v_cf, u_cf = closed_form(v_j.clone(), u_j.clone(), lam, M)

    # 3.c) Compare
    diff_b = (v_prox - v_cf).abs().item()
    diff_u = (u_prox - u_cf).abs().max().item()
    
    print(f"v_prox = {v_prox.item():.4f}, u_prox = {u_prox.tolist()}")
    print(f"v_cf   = {v_cf.item():.4f}, u_cf   = {u_cf.tolist()}")

    if diff_b > max_diff_b:
        max_diff_b = diff_b
    if diff_u > max_diff_u:
        max_diff_u = diff_u

print(f"Maximum |b_prox − b_cf| over {num_tests} tests  = {max_diff_b:.2e}")
print(f"Maximum |u_prox − u_cf|_∞ over {num_tests} tests = {max_diff_u:.2e}")

# If you run this, you should see both differences ~0.0 (up to floating-point eps).  
# %%
