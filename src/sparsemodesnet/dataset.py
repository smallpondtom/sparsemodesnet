import numpy as np
from torch.utils.data import Dataset

class PODReconDataset(Dataset):
    """
    Given:
      - Z_np ∈ R^{s x n} (POD coefficients = V_s^T X)
      - X_np ∈ R^{d x n} (original snapshots)
    Creates n samples; each sample i returns (z_i, x_i).
    """

    def __init__(self, Z_np: np.ndarray, X_np: np.ndarray):
        assert Z_np.ndim == 2 and X_np.ndim == 2
        s, n1 = Z_np.shape
        d, n2 = X_np.shape
        assert n1 == n2, "Mismatch in number of snapshots."
        # store row‐major so Dataset returns (z_i, x_i)
        self.Z = Z_np.T.copy().astype(np.float32)  # (n, s)
        self.X = X_np.T.copy().astype(np.float32)  # (n, d)

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        return self.Z[idx, :], self.X[idx, :]