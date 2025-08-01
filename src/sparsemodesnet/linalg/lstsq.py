import numpy as np

def lstsq_l2(A, B, reg_magnitude=1e-6):
    phi, sigma, psi_t = np.linalg.svd(A, full_matrices=False)
    sinv = sigma / (sigma**2 + reg_magnitude**2)
    x = psi_t.T * sinv @ (phi.T @ B)
    B_estimate = A @ x
    resid = np.linalg.norm(B - B_estimate)
    return x, resid
