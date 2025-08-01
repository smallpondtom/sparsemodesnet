import numpy as np

def compute_pod_basis(X_np: np.ndarray, s: int = None):
    """
    Given X_np ∈ R^{d x n}, compute first s left singular vectors U_s ∈ R^{d x s}.
    If s is None, take s = min(d, n). Returns:
      U_s: (d, s), Sigma_s: (s,), Vt_s: (s, n).
    """
    U, Sigma, Vt = np.linalg.svd(X_np, full_matrices=False)
    d, n = X_np.shape
    r = min(d, n) if s is None else min(s, min(d, n))
    U_s     = U[:, :r]
    Sigma_s = Sigma[:r]
    Vt_s    = Vt[:r, :]
    return U_s, Sigma_s, Vt_s
