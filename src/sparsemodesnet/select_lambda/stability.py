import numpy as np
import torch
from torch.utils.data import DataLoader

from sparsemodesnet.pod import compute_pod_basis
from sparsemodesnet.dataset import PODReconDataset
from sparsemodesnet.model import SparseModesNet
from sparsemodesnet.train import train_sparsemodesnet

def select_lambda_stability(X_np: np.ndarray,
                            s: int,
                            hidden_units: list,
                            M: float,
                            lambdas: np.ndarray,
                            B: int,
                            pi_thresh: float,
                            lr: float,
                            num_epochs_sub: int,
                            batch_size: int,
                            optimizer: str,
                            device: str):
    """
    Performs Meinshausen-Bühlmann stability selection over a grid of λ.
    Returns the first λ for which no features exceed pi_thresh frequency, 
    or the λ that yields <= a target #features. Here we choose λ s.t. most features 
    drop out. The user can inspect the returned freq table for details.
    """
    print("\n=== Stability Selection λ-Selection ===")
    d, n = X_np.shape
    V_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = V_s_np.T.dot(X_np)                     # (s, n)

    path_history_ss = []

    for lam in lambdas:
        counts = np.zeros(s, dtype=int)
        print(f" SS testing λ = {lam:.3e} ...")

        for b in range(B):
            # Random half‐sample of indices
            subsamp = np.random.choice(n, size=n//2, replace=False)
            ds_sub = PODReconDataset(Z_np=Z_np[:, subsamp], X_np=X_np[:, subsamp])
            dl_sub = DataLoader(ds_sub, batch_size=batch_size, shuffle=True, drop_last=False)

            # Train on that subsample
            model_ss = SparseModesNet(
                pod_basis    = torch.from_numpy(V_s_np.astype(np.float32)).to(device),
                input_dim    = s,
                hidden_units = hidden_units,
                M            = M,
                lam          = float(lam)
            ).to(device)

            train_sparsemodesnet(model_ss, dl_sub, num_epochs_sub, lr, optimizer, device)

            # Record which features are nonzero in b
            b_opt = model_ss.b.detach().cpu().numpy()
            counts += (np.abs(b_opt) > 1e-8).astype(int)

        freqs = counts / float(B)
        stable_count = int((freqs >= pi_thresh).sum())
        print(f"stable_count = {stable_count} "
              f"(#features with freq ≥ {pi_thresh} = {stable_count})")

        # Now, train on the full data (briefly) to get k(λ) and rel_error
        final_full_epochs = num_epochs_sub
        Z_full = Z_np.T  # (n, s)
        X_full = X_np.T  # (n, d)
        ds_full = PODReconDataset(Z_np=Z_full.T, X_np=X_full.T)
        dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True, drop_last=False)
        model_full = SparseModesNet(
            pod_basis    = torch.from_numpy(V_s_np.astype(np.float32)).to(device),
            input_dim    = s,
            hidden_units = hidden_units,
            M            = M,
            lam          = float(lam)
        ).to(device)
        train_sparsemodesnet(model_full, dl_full, final_full_epochs, lr, optimizer, device)
        with torch.no_grad():
            b_full = model_full.b.detach().cpu().numpy()
            k_full = int((np.abs(b_full) > 1e-6).sum())
            Z_tensor_full = torch.from_numpy(Z_full.astype(np.float32)).to(device)
            _, x_hat_full = model_full(Z_tensor_full)
            X_hat_full_np = x_hat_full.cpu().numpy().T
            frob_err = np.linalg.norm(X_np - X_hat_full_np, 'fro')
            rel_err = frob_err / np.linalg.norm(X_np, 'fro')
        path_history_ss.append({
            'lambda': lam,
            'stable_count': stable_count,
            'k': k_full,
            'rel_error': rel_err
        })

        # If no feature is “stable,” we can stop early
        if stable_count == 0:
            print("All features dropped out at this λ; stopping SS path.\n")
            break

    return path_history_ss