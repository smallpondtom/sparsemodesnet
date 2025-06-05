import numpy as np

def compute_pod_basis(X_np: np.ndarray, s: int = None):
    """
    Given X_np ∈ R^{d x n}, compute first s left singular vectors V_s ∈ R^{d x s}.
    If s is None, take s = min(d, n). Returns:
      V_s: (d, s), Sigma_s: (s,), Wt_s: (s, n).
    """
    U, Sigma, Vt = np.linalg.svd(X_np, full_matrices=False)
    d, n = X_np.shape
    r = min(d, n) if s is None else min(s, min(d, n))
    V_s     = U[:, :r].astype(np.float32)
    Sigma_s = Sigma[:r].astype(np.float32)
    Wt_s    = Vt[:r, :].astype(np.float32)
    return V_s, Sigma_s, Wt_s