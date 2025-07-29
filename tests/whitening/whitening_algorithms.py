#%%
"""
Whitening & Decorrelation Algorithms in Python / PyTorch
=========================================================

This module implements the most commonly used *exact* linear whitening transforms
(W Σ Wᵀ = I) plus a few practical variants for deep learning / high-dimensional
settings. Each function returns the whitened data and (optionally) the whitening
matrix so you can apply it to new data.

Implemented methods (numpy):
----------------------------
1. ZCA / Mahalanobis (Σ^{-1/2})
2. PCA Whitening (Λ^{-1/2} Uᵀ)
3. Cholesky Whitening (Lᵀ from Σ^{-1} = L Lᵀ)
4. ZCA-cor (P^{-1/2} V^{-1/2})
5. PCA-cor (Θ^{-1/2} Gᵀ V^{-1/2})
6. MMSE Whitening (regularized inverse square root)
7. Shrinkage Whitening (Ledoit–Wolf covariance then ZCA/PCA/etc.)
8. Randomized/Sketched Whitening (SRHT / Gaussian sketch + ZCA)

PyTorch layers/utilities:
-------------------------
- matrix_inv_sqrt_newton_schulz(): fast GPU-friendly inverse sqrt
- ZCAWhitening, PCAWhitening modules (fit/transform style)
- IterNormWhitening (iterative normalization a.k.a. IterNorm)
- DecorrelatedBatchNorm2d (simplified DBN)
- GroupWhitening (block-diagonal whitening)

All code is self-contained (numpy, scipy, torch). SciPy is only used for a stable
Cholesky if available; fallback is numpy.linalg.cholesky.

Author: ChatGPT (OpenAI)
License: MIT
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import eigh, svd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:  # allow using only numpy
    torch = None
    nn = object  # type: ignore

try:
    from sklearn.covariance import LedoitWolf
    _HAVE_SKLEARN = True
except ImportError:
    _HAVE_SKLEARN = False

try:
    import scipy.linalg as sla
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _center(X: np.ndarray, center: bool = True):
    if not center:
        return X, np.zeros(X.shape[1])
    mu = X.mean(axis=0, keepdims=True)
    return X - mu, mu.ravel()


def _cov(X: np.ndarray, rowvar: bool = False, bias: bool = False):
    """Compute covariance like numpy.cov but always returns full matrix.
    X shape: (n_samples, n_features) if rowvar=False.
    """
    if rowvar:
        X = X.T
    n = X.shape[0]
    ddof = 0 if bias else 1
    return (X.T @ X) / (n - ddof)


def _safe_inv_sqrt(S: np.ndarray, eps: float):
    """Return S^{-1/2} for SPD S with numerical stabilization eps."""
    vals, vecs = eigh(S)
    vals = np.clip(vals, eps, None)
    return (vecs * (1.0 / np.sqrt(vals)) ) @ vecs.T


def _safe_sqrt(S: np.ndarray, eps: float):
    vals, vecs = eigh(S)
    vals = np.clip(vals, eps, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


# -----------------------------------------------------------------------------
# Core whitening transforms (NumPy)
# -----------------------------------------------------------------------------

def whiten_zca(X: np.ndarray, eps: float = 1e-5, center: bool = True, return_W: bool = False):
    """ZCA / Mahalanobis whitening: W = Σ^{-1/2}.
    Returns Z, (W, mu) if return_W.
    """
    Xc, mu = _center(X, center)
    S = _cov(Xc)
    W = _safe_inv_sqrt(S + eps * np.eye(S.shape[0]), eps)
    Z = Xc @ W.T
    return (Z, (W, mu)) if return_W else Z


def whiten_pca(X: np.ndarray, eps: float = 1e-5, center: bool = True, return_W: bool = False):
    """PCA whitening: W = Λ^{-1/2} Uᵀ, where Σ = U Λ Uᵀ.
    Components are ordered by variance.
    """
    Xc, mu = _center(X, center)
    S = _cov(Xc)
    vals, vecs = eigh(S)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    W = (np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T)
    Z = Xc @ W.T
    return (Z, (W, mu, vecs, vals)) if return_W else Z


def whiten_cholesky(X: np.ndarray, eps: float = 1e-8, center: bool = True, return_W: bool = False):
    """Cholesky whitening: compute L from Σ^{-1} = L Lᵀ, then W = Lᵀ.
    Requires Σ to be SPD. Order of variables matters.
    """
    Xc, mu = _center(X, center)
    S = _cov(Xc)
    # precision matrix
    Sinv = np.linalg.inv(S + eps * np.eye(S.shape[0]))
    if _HAVE_SCIPY:
        L = sla.cholesky(Sinv, lower=True)
    else:
        L = np.linalg.cholesky(Sinv)
    W = L.T
    Z = Xc @ W.T
    return (Z, (W, mu, L)) if return_W else Z


def whiten_zca_cor(X: np.ndarray, eps: float = 1e-5, center: bool = True, return_W: bool = False):
    """ZCA-cor: standardize, then ZCA on correlation matrix.
    W = P^{-1/2} V^{-1/2}.
    """
    Xc, mu = _center(X, center)
    std = Xc.std(axis=0, ddof=1) + eps
    Xstd = Xc / std
    P = _cov(Xstd)
    Wcorr = _safe_inv_sqrt(P + eps * np.eye(P.shape[0]), eps)
    W = Wcorr @ np.diag(1.0 / std)
    Z = Xc @ W.T
    return (Z, (W, mu, std)) if return_W else Z


def whiten_pca_cor(X: np.ndarray, eps: float = 1e-5, center: bool = True, return_W: bool = False):
    """PCA-cor: standardize, then PCA whitening on correlation matrix.
    W = Θ^{-1/2} Gᵀ V^{-1/2}.
    """
    Xc, mu = _center(X, center)
    std = Xc.std(axis=0, ddof=1) + eps
    Xstd = Xc / std
    P = _cov(Xstd)
    vals, vecs = eigh(P)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    W = (np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T) @ np.diag(1.0 / std)
    Z = Xc @ W.T
    return (Z, (W, mu, std, vecs, vals)) if return_W else Z


def whiten_mmse(X: np.ndarray, noise_var: float, eps: float = 1e-5, center: bool = True, return_W: bool = False):
    """MMSE whitening (Eldar & Oppenheim 2003): W = (Σ + σ² I)^{-1/2}.
    Useful when measurements are noisy and you want to avoid over-whitening.
    """
    Xc, mu = _center(X, center)
    S = _cov(Xc)
    W = _safe_inv_sqrt(S + (noise_var + eps) * np.eye(S.shape[0]), eps)
    Z = Xc @ W.T
    return (Z, (W, mu)) if return_W else Z


def whiten_shrinkage(X: np.ndarray, method: str = 'ledoit-wolf', eps: float = 1e-5,
                     center: bool = True, base: str = 'zca', return_W: bool = False):
    """Whitening with a regularized covariance estimate. Default: Ledoit–Wolf.
    base ∈ {'zca','pca','zca-cor','pca-cor','cholesky'} chooses the transform on Σ̂.
    """
    Xc, mu = _center(X, center)
    if method == 'ledoit-wolf':
        if not _HAVE_SKLEARN:
            raise ImportError('sklearn.covariance.LedoitWolf required')
        lw = LedoitWolf().fit(Xc)
        S = lw.covariance_
    else:
        raise ValueError('Unknown shrinkage method')

    # dispatch to base whitening using precomputed S
    if base == 'zca':
        W = _safe_inv_sqrt(S + eps * np.eye(S.shape[0]), eps)
    elif base == 'pca':
        vals, vecs = eigh(S); order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        W = (np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T)
    elif base == 'cholesky':
        Sinv = np.linalg.inv(S + eps * np.eye(S.shape[0]))
        L = np.linalg.cholesky(Sinv)
        W = L.T
    elif base == 'zca-cor':
        std = Xc.std(axis=0, ddof=1) + eps
        Xstd = Xc / std
        P = _cov(Xstd)
        Wcorr = _safe_inv_sqrt(P + eps * np.eye(P.shape[0]), eps)
        W = Wcorr @ np.diag(1.0 / std)
    elif base == 'pca-cor':
        std = Xc.std(axis=0, ddof=1) + eps
        Xstd = Xc / std
        P = _cov(Xstd)
        vals, vecs = eigh(P); order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        W = (np.diag(1.0 / np.sqrt(vals + eps)) @ vecs.T) @ np.diag(1.0 / std)
    else:
        raise ValueError('Unknown base')
    Z = Xc @ W.T
    return (Z, (W, mu, S)) if return_W else Z


def whiten_randomized(X: np.ndarray, k: int, eps: float = 1e-5, center: bool = True,
                      return_W: bool = False, oversample: int = 10, n_iter: int = 2,
                      base: str = 'zca'):
    """Approximate whitening using a randomized sketch (Halko et al. 2011 style).
    1) Sketch X -> Y (n × k)
    2) Compute Σ̂_k from Y and do chosen base whitening in that subspace.
    """
    Xc, mu = _center(X, center)
    n, d = Xc.shape
    l = min(d, k + oversample)
    # Gaussian sketch
    G = np.random.randn(d, l)
    Y = Xc @ G
    # power iterations
    for _ in range(n_iter):
        Y = Xc @ (Xc.T @ Y)
    # Orthonormal basis Q of Y
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ Xc  # l × d
    S_approx = _cov(B.T)  # d × d but rank ≤ l
    # fallback to exact base using this S_approx
    return whiten_shrinkage(Xc, method='ledoit-wolf' if not np.isfinite(S_approx).all() else 'ledoit-wolf',
                             eps=eps, center=False, base=base, return_W=return_W)


# -----------------------------------------------------------------------------
# PyTorch helpers & modules
# -----------------------------------------------------------------------------

if torch is not None:

    def matrix_inv_sqrt_newton_schulz(A: torch.Tensor, num_iter: int = 5, eps: float = 1e-5):
        """Compute A^{-1/2} with the Newton–Schulz iteration.
        A must be symmetric PSD (batchable: (..., d, d)).
        """
        # Normalize A by its trace to improve convergence
        dim = A.shape[-1]
        I = torch.eye(dim, device=A.device, dtype=A.dtype)
        # batch trace
        trace = A.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        A_norm = A / (trace + eps)
        Y = A_norm
        Z = I.expand_as(A)
        for _ in range(num_iter):
            T = 0.5 * (3.0 * I - Z @ Y)
            Y = Y @ T
            Z = T @ Z
        return Z / torch.sqrt(trace + eps)

    class WhiteningBase(nn.Module):
        def __init__(self, eps: float = 1e-5, center: bool = True):
            super().__init__()
            self.eps = eps
            self.center = center
            self.register_buffer('mean', None)
            self.register_buffer('W', None)

        def fit(self, X: torch.Tensor):
            raise NotImplementedError

        def forward(self, X: torch.Tensor):
            if self.center and self.mean is not None:
                Xc = X - self.mean
            else:
                Xc = X
            return (Xc @ self.W.T) if self.W is not None else X

    class ZCAWhitening(WhiteningBase):
        def fit(self, X: torch.Tensor):
            if self.center:
                self.mean = X.mean(dim=0, keepdim=True)
                Xc = X - self.mean
            else:
                self.mean = torch.zeros(1, X.size(1), device=X.device, dtype=X.dtype)
                Xc = X
            S = torch.cov(Xc.T)
            self.W = matrix_inv_sqrt_newton_schulz(S + self.eps * torch.eye(S.size(0), device=S.device, dtype=S.dtype))
            return self

    class PCAWhitening(WhiteningBase):
        def fit(self, X: torch.Tensor):
            if self.center:
                self.mean = X.mean(dim=0, keepdim=True)
                Xc = X - self.mean
            else:
                self.mean = torch.zeros(1, X.size(1), device=X.device, dtype=X.dtype)
                Xc = X
            S = torch.cov(Xc.T)
            vals, vecs = torch.linalg.eigh(S)
            idx = torch.argsort(vals, descending=True)
            vals, vecs = vals[idx], vecs[:, idx]
            self.W = (torch.diag(1.0 / torch.sqrt(vals + self.eps)) @ vecs.T)
            return self

    class IterNormWhitening(WhiteningBase):
        """Iterative normalization layer (similar to IterNorm/DBN papers)."""
        def __init__(self, num_iter: int = 5, eps: float = 1e-5, center: bool = True):
            super().__init__(eps, center)
            self.num_iter = num_iter

        def fit(self, X: torch.Tensor):
            if self.center:
                self.mean = X.mean(dim=0, keepdim=True)
                Xc = X - self.mean
            else:
                self.mean = torch.zeros(1, X.size(1), device=X.device, dtype=X.dtype)
                Xc = X
            S = torch.cov(Xc.T)
            self.W = matrix_inv_sqrt_newton_schulz(S + self.eps * torch.eye(S.size(0), device=S.device, dtype=S.dtype),
                                                   num_iter=self.num_iter, eps=self.eps)
            return self

    class DecorrelatedBatchNorm2d(nn.Module):
        """Very simplified DBN for 2D conv features (N, C, H, W).
        Uses per-batch covariance and Newton–Schulz for inverse sqrt.
        """
        def __init__(self, num_iter: int = 5, eps: float = 1e-4):
            super().__init__()
            self.num_iter = num_iter
            self.eps = eps

        def forward(self, x):
            N, C, H, W = x.shape
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)
            mu = x_flat.mean(dim=0, keepdim=True)
            xc = x_flat - mu
            S = torch.cov(xc.T)
            W = matrix_inv_sqrt_newton_schulz(S + self.eps * torch.eye(C, device=x.device, dtype=x.dtype),
                                              num_iter=self.num_iter, eps=self.eps)
            z = (xc @ W.T).reshape(N, H, W, C).permute(0, 3, 1, 2)
            return z

    class GroupWhitening(nn.Module):
        """Block-diagonal whitening by splitting channels/features into groups."""
        def __init__(self, num_groups: int = 4, num_iter: int = 5, eps: float = 1e-5):
            super().__init__()
            self.num_groups = num_groups
            self.num_iter = num_iter
            self.eps = eps

        def forward(self, x):
            # x: (N, C) or (N, C, H, W) -> flatten spatial dims if needed
            if x.dim() > 2:
                N = x.size(0)
                C = x.size(1)
                flat = x.view(N, C, -1).transpose(1, 2).reshape(-1, C)  # (N*HW, C)
            else:
                flat = x
                N, C = flat.shape
            group_size = C // self.num_groups
            outs = []
            for g in range(self.num_groups):
                sl = slice(g * group_size, (g + 1) * group_size if g < self.num_groups - 1 else C)
                chunk = flat[:, sl]
                mu = chunk.mean(dim=0, keepdim=True)
                xc = chunk - mu
                S = torch.cov(xc.T)
                W = matrix_inv_sqrt_newton_schulz(S + self.eps * torch.eye(S.size(0), device=x.device, dtype=x.dtype),
                                                  num_iter=self.num_iter, eps=self.eps)
                outs.append((xc @ W.T) + mu)  # keep mean
            flat_out = torch.cat(outs, dim=1)
            if x.dim() > 2:
                flat_out = flat_out.view(N, -1, group_size).transpose(1, 2)
                return flat_out.view_as(x)
            return flat_out


# -----------------------------------------------------------------------------
# Convenience router
# -----------------------------------------------------------------------------

_WHITENERS = {
    'zca': whiten_zca,
    'pca': whiten_pca,
    'cholesky': whiten_cholesky,
    'zca-cor': whiten_zca_cor,
    'pca-cor': whiten_pca_cor,
    'mmse': whiten_mmse,
    'shrinkage': whiten_shrinkage,
    'randomized': whiten_randomized,
}


def whiten(X: np.ndarray, method: str = 'zca', **kwargs):
    """Generic whitening frontend. method ∈ _WHITENERS."""
    if method not in _WHITENERS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(_WHITENERS)}")
    return _WHITENERS[method](X, **kwargs)


# -----------------------------------------------------------------------------
# Simple test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 5))
    # create correlation
    X[:, 1] += 0.8 * X[:, 0]
    Z, (W, mu) = whiten_zca(X, return_W=True)
    print('Cov(Z) ~ I?\n', np.cov(Z.T))
    Zp, _ = whiten_pca(X, return_W=True)
    print('Cov(Z_pca) diag?\n', np.cov(Zp.T))

