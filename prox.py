import torch
import torch.nn as nn
import torch.nn.functional as F

# ——————————————————————————————————————
# (A) Helper functions from the authors (unchanged)
# ——————————————————————————————————————
def soft_threshold(lam: float, x: torch.Tensor) -> torch.Tensor:
    return x.sign() * F.relu(x.abs() - lam)

def sign_binary(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x >= 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.tensor(-1.0, device=x.device, dtype=x.dtype)
    )

def prox(v: torch.Tensor, u: torch.Tensor, *,
         lambda_: float,  # penalty on |b|
         lambda_bar: float,  # penalty on |W| (we will set this to 0)
         M: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The authors' Group-Hierarchical Proximal operator.
    When lambda_bar = 0, this solves exactly:
      minimize  ½(v - b)^2 + ½||u - W||_2^2 + λ|b|
      s.t. ||W||_∞ ≤ M|b|.
    """
    onedim = (v.ndim == 1)
    if onedim:
        v = v.unsqueeze(-1)  # shape (1, batch=1)
        u = u.unsqueeze(-1)  # shape (K, batch=1)

    # 1) Sort |u| in descending order along the first dimension
    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values  # shape (K, batch)

    K, batch = u.shape
    s_t = torch.arange(K + 1.0, device=v.device, dtype=v.dtype).view(-1, 1)  # shape (K+1,1)
    zeros = torch.zeros(1, batch, device=v.device, dtype=v.dtype)          # shape (1,batch)

    # 2) Build a_s(m) = λ_ - M * [0, cumsum(|u_(1)| - lambda_bar, …, |u_(m)| - lambda_bar)]
    a_s = lambda_ - M * torch.cat([zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)], dim=0)

    # 3) Compute norm_v = ||v||₂
    norm_v = torch.norm(v, p=2, dim=0)  # shape (batch,)

    # 4) Compute x(m) = max(0, 1 - a_s(m)/norm_v)/(1 + m*M^2)
    x_vals = F.relu(1 - a_s / norm_v) / (1 + s_t * (M**2))  # (K+1, batch)

    # 5) Compute w(m) = M * x(m) * norm_v
    w_vals = M * x_vals * norm_v  # shape (K+1, batch)

    # 6) Compute “intervals” = S_{lambda_bar}(|u|_sorted)
    intervals = soft_threshold(lambda_bar, u_abs_sorted)  # (K, batch)
    lower = torch.cat([intervals, zeros], dim=0)          # (K+1, batch)

    # 7) Find m̃ = first m such that lower[m] > w_vals[m]
    idx = torch.sum(lower > w_vals, dim=0, keepdim=True)  # (1, batch)

    # 8) Gather x_star, w_star
    x_star = torch.gather(x_vals, 0, idx)  # (1, batch)
    w_star = torch.gather(w_vals, 0, idx)  # (1, batch)

    # 9) Updated b_star = x_star * v
    beta_star = x_star * v  # (1, batch)

    # 10) Updated W_star = sign(u) * min(S_{lambda_bar}(|u|), w_star)
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star

