import numpy as np
import torch

def lstsq_l2(A, B, reg_magnitude=1e-6):
    """
    An efficient l2-penalized linear least-squares solver with a 
    constrant regularization parameter. Given the constant regularization
    parameter we can efficiently apply the regularization to the SVD-based
    least-squares solve. Note that this works for numpy arrays. Use the 
    torch version to work internally in the NN model.

    Parameters
    ---------- 
    :A: The feature matrix of the least-squares
    :B: The target or left-hand side matrix of least-squares
    :reg_magnitude: l2-penalty regularization parameter
    """
    phi, sigma, psi_t = np.linalg.svd(A, full_matrices=False)
    sinv = sigma / (sigma**2 + reg_magnitude**2)
    x = psi_t.T * sinv @ (phi.T @ B)
    B_estimate = A @ x
    resid = np.linalg.norm(B - B_estimate)
    return x, resid

def lstsq_l2_torch(A, B, reg_magnitude=1e-6):
    """
    `lstsq_l2` implementation in torch.
    """
    U, sigma, Vt = torch.linalg.svd(A, full_matrices=False)
    sinv = sigma / (sigma**2 + reg_magnitude**2)
    
    # Handle both 1D and 2D B cases
    if B.dim() == 1:
        # B is 1D: shape (m,)
        UTB = U.T @ B  # shape (min(m,n),)
        x = Vt.T @ (sinv * UTB)  # shape (n,)
    else:
        # B is 2D: shape (m, k)
        UTB = U.T @ B  # shape (min(m,n), k)
        x = Vt.T @ (sinv.unsqueeze(-1) * UTB)  # shape (n, k)
    
    # Compute residual
    B_estimate = A @ x
    resid = torch.linalg.norm(B - B_estimate)
    
    return x, resid