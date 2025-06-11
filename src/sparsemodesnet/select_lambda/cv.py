import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sparsemodesnet.pod import compute_pod_basis
from sparsemodesnet.dataset import PODReconDataset
from sparsemodesnet.model import SparseModesNet
from sparsemodesnet.train import train_sparsemodesnet

def select_lambda_cv(X_np: np.ndarray,
                     s: int,
                     hidden_units: list,
                     M: float,
                     nonzero_thresh: float,
                     lambdas: np.ndarray,
                     lr: float,
                     num_epochs_cv: int,
                     k_folds: int,
                     batch_size: int,
                     optimizer: str,
                     device: str,
                     network_type: str = 'feedforward',  # Add this parameter
                     **conv_kwargs):  # Add this parameter
    """
    Performs k-fold CV over a grid of λ values with support for convolutional networks.
    """
    print("\n=== Cross-Validation λ-Selection ===")
    d, n = X_np.shape
    # 1) Compute POD basis and coefficients Z
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)      # (d, s)
    Z_np = U_s_np.T.dot(X_np)                        # (s, n)

    # 2) Prepare fold splits
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    path_history_cv = []

    for lam in lambdas:
        val_errors = []
        print(f" CV testing λ = {lam:.3e} ...")
        for fold_idx in range(k_folds):
            # train indices = all except this fold
            val_idx = folds[fold_idx]
            train_idx = np.hstack([folds[i] for i in range(k_folds) if i != fold_idx])

            # Build train+val datasets
            Z_all = Z_np.T  # (n, s)
            X_all = X_np.T  # (n, d)
            ds_train = PODReconDataset(Z_np=Z_all[train_idx].T, X_np=X_all[train_idx].T)
            ds_val   = PODReconDataset(Z_np=Z_all[val_idx].T,   X_np=X_all[val_idx].T)
            dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=False)
            dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, drop_last=False)

            # Instantiate a fresh model at λ
            model_cv = SparseModesNet(
                pod_basis    = torch.from_numpy(U_s_np.astype(np.float32)).to(device),
                input_dim    = s,
                hidden_units = hidden_units,
                M            = M,
                lam          = float(lam),
                network_type = network_type,  # Add this
                **conv_kwargs  # Add this
            ).to(device)

            # Train for num_epochs_cv
            train_sparsemodesnet(model_cv, dl_train, num_epochs_cv, lr, optimizer, device)

            # Evaluate on val set
            model_cv.eval()
            mse_loss = nn.MSELoss(reduction='sum')
            total_err = 0.0
            total_samples = 0
            with torch.no_grad():
                for z_b, x_b in dl_val:
                    z_b = z_b.to(device)
                    x_b = x_b.to(device)
                    _, x_hat_b = model_cv(z_b)
                    total_err += mse_loss(x_hat_b, x_b).item()
                    total_samples += x_b.shape[0]
            val_mse = total_err / total_samples
            val_errors.append(val_mse)

        avg_val_error = np.mean(val_errors)
        print(f"avg val-MSE = {avg_val_error:.6e}")

        # Now, train on the full data (briefly) to get k(λ) and rel_error
        final_full_epochs = num_epochs_cv
        Z_full = Z_np.T  # (n, s)
        X_full = X_np.T  # (n, d)
        ds_full = PODReconDataset(Z_np=Z_full.T, X_np=X_full.T)
        dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True, drop_last=False)
        model_full = SparseModesNet(
            pod_basis    = torch.from_numpy(U_s_np.astype(np.float32)).to(device),
            input_dim    = s,
            hidden_units = hidden_units,
            M            = M,
            lam          = float(lam),
            network_type = network_type,  # Add this
            **conv_kwargs  # Add this
        ).to(device)
        train_sparsemodesnet(model_full, dl_full, final_full_epochs, lr, optimizer, device)
        with torch.no_grad():
            omega_opt = model_full.omega.detach().cpu().numpy()
            r_full = int((np.abs(omega_opt) > nonzero_thresh).sum())
            Z_tensor_full = torch.from_numpy(Z_full.astype(np.float32)).to(device)
            _, x_hat_full = model_full(Z_tensor_full)
            X_hat_full_np = x_hat_full.cpu().numpy().T
            frob_err = np.linalg.norm(X_np - X_hat_full_np, 'fro')
            rel_err = frob_err / np.linalg.norm(X_np, 'fro')
        path_history_cv.append({
            'lambda': lam,
            'val_error': avg_val_error,
            'r': r_full,
            'rel_error': rel_err
        })

    return path_history_cv
