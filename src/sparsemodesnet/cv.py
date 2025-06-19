import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sparsemodesnet.pod import compute_pod_basis
from sparsemodesnet.dataset import PODReconDataset
from sparsemodesnet.model import SparseModesNet
from sparsemodesnet.train import train_sparsemodesnet

def run_sparsemodesnet_cv(X_np: np.ndarray,
                          s: int,
                          hidden_units: list,
                          M: float,
                          nonzero_thresh: float,
                          lambdas: np.ndarray,
                          network_type: str,
                          poly_order: int,
                          num_polys: int,
                          drop_linear: bool,
                          lr: float,
                          num_epochs: int,
                          k_folds: int,
                          batch_size: int,
                          optimizer: str,
                          device: str):
    """
    Performs k-fold CV over a grid of λ values.
    """
    print("\n=== Cross-Validation λ-Path ===")
    d, n = X_np.shape
    
    # Compute POD basis and coefficients Z
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)      # (d, s)
    Z_np = U_s_np.T.dot(X_np)                        # (s, n)

    # Prepare fold splits
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)

    path_history_cv = []
    total_lambdas = len(lambdas)
    selected_indices = np.zeros((s, total_lambdas))
    ct = 1

    for index, lam in enumerate(lambdas):
        val_errors = []
        r_folds = []
        print(f" CV testing λ = {lam:.3e}. Currently {ct}/{total_lambdas} ...")
        for fold_idx in range(k_folds):
            # train indices = all except this fold
            val_idx = folds[fold_idx]
            train_idx = np.hstack(
                [folds[i] for i in range(k_folds) if i != fold_idx])

            # Build train+val datasets
            Z_all = Z_np.T  # (n, s)
            X_all = X_np.T  # (n, d)
            ds_train = PODReconDataset(
                Z_np=Z_all[train_idx].T, X_np=X_all[train_idx].T
            )
            ds_val   = PODReconDataset(
                Z_np=Z_all[val_idx].T,   X_np=X_all[val_idx].T
            )
            dl_train = DataLoader(
                ds_train, batch_size=batch_size, shuffle=True, drop_last=False
            )
            dl_val   = DataLoader(
                ds_val,   batch_size=batch_size, shuffle=False, drop_last=False
            )

            # Instantiate a fresh model at λ
            model_cv = SparseModesNet(
                pod_basis=torch.from_numpy(U_s_np.astype(np.float32)).to(device),
                input_dim=s,
                hidden_units=hidden_units,
                M=M,
                lam=float(lam),
                network_type=network_type,
                poly_order=poly_order,
                num_polys=num_polys,
                drop_linear=drop_linear 
            ).to(device)

            # Train for num_epochs
            train_sparsemodesnet(
                model_cv, dl_train, num_epochs, lr, optimizer, device)

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
                    
                # Calculate the number of non-zero modes in the optimal omega
                omega = model_cv.omega.detach().cpu().numpy()
                r_fold = int((np.abs(omega) > nonzero_thresh).sum())
                r_folds.append(r_fold)
                S_stable = np.where(omega >= nonzero_thresh)[0]
                selected_indices[S_stable, index] += 1 / k_folds

            # Compute validation MSE
            val_mse = total_err / total_samples
            val_errors.append(val_mse)
            
        # Average over folds      
        avg_val_error = np.mean(val_errors)
        avg_r = int(np.mean(r_folds))
        print(f" avg val-MSE = {avg_val_error:.6e}, avg r = {avg_r}")
        
        # Store results for this lambda    
        path_history_cv.append({
            'lambda': lam,
            'error': avg_val_error,
            'nonzero_count': avg_r,
        })
        
        if avg_r == 0:
            print("All skip-weights have zeroed out. Stopping path.\n")
            break
        
        ct += 1
        
    # Select the most relevant features/POD modes using weighted selection frequency
    
    if len(path_history_cv) == 0:
        print("No valid models found in CV path.")
        return path_history_cv
    
    # Extract validation errors and compute inverse error weights
    val_errors = np.array([entry['error'] for entry in path_history_cv])
    
    # Use inverse error as weight (add small epsilon to avoid division by zero)
    error_weights = 1.0 / (val_errors + 1e-15)
    error_weights = error_weights / np.sum(error_weights)  # Normalize to sum to 1
    
    # Compute weighted selection frequency for each mode
    # selected_indices shape: (s, total_lambdas)
    # Each column corresponds to a lambda, each row to a POD mode
    weighted_selection_freq = np.zeros(s)
    
    for i, entry in enumerate(path_history_cv):
        weight = error_weights[i]
        # Weight the selection frequencies by model quality
        weighted_selection_freq += selected_indices[:, i] * weight
    
    # Find the 75th percentile threshold (top 25%)
    # Only consider modes that were selected at least once
    nonzero_frequencies = weighted_selection_freq[weighted_selection_freq > 0]
    
    if len(nonzero_frequencies) > 0:
        threshold_75 = np.percentile(nonzero_frequencies, 75)
    else:
        # Fallback if no modes were selected
        threshold_75 = 0.0
        print("Warning: No modes were selected in any CV fold.")
    
    # Select modes in the top 25% of weighted selection frequency
    I_CV = np.where(weighted_selection_freq >= threshold_75)[0]
    
    # Ensure we select at least one mode
    if len(I_CV) == 0:
        # Fallback: select the mode with highest weighted frequency
        I_CV = np.array([np.argmax(weighted_selection_freq)])
        print("Warning: No modes met the 75th percentile threshold. Selecting top mode.")
    
    print(f"\nCV Feature Selection Results:")
    print(f"  Number of lambda values tested: {len(path_history_cv)}")
    print(f"  Validation errors range: [{np.min(val_errors):.2e}, {np.max(val_errors):.2e}]")
    print(f"  75th percentile threshold: {threshold_75:.3f}")
    print(f"  Selected {len(I_CV)} modes out of {s}")
    print(f"  Selected mode indices: {I_CV.tolist()}")
    print(f"  Weighted selection frequencies: {weighted_selection_freq[I_CV]}")
    
    # Store selection results in path_history_cv
    cv_selection_results = {
        'method': 'weighted_selection_frequency_top25',
        'weighted_frequencies': weighted_selection_freq,
        'threshold_75': threshold_75,
        'error_weights': error_weights,
        'raw_selection_matrix': selected_indices,
        'path_history': path_history_cv,
    }
    
    return I_CV, cv_selection_results 



# def run_sparsemodesnet_cv(X_np: np.ndarray,
#                           s: int,
#                           hidden_units: list,
#                           M: float,
#                           nonzero_thresh: float,
#                           lambdas: np.ndarray,
#                           network_type: str,
#                           poly_order: int,
#                           num_polys: int,
#                           drop_linear: bool,
#                           lr: float,
#                           num_epochs_cv: int,
#                           k_folds: int,
#                           batch_size: int,
#                           optimizer: str,
#                           device: str):
#     """
#     Performs k-fold CV over a grid of λ values.
#     """
#     print("\n=== Cross-Validation λ-Path ===")
#     d, n = X_np.shape
    
#     # Compute POD basis and coefficients Z
#     U_s_np, _, _ = compute_pod_basis(X_np, s=s)      # (d, s)
#     Z_np = U_s_np.T.dot(X_np)                        # (s, n)

#     # Prepare fold splits
#     indices = np.arange(n)
#     np.random.shuffle(indices)
#     folds = np.array_split(indices, k_folds)

#     path_history_cv = []
#     total_lambdas = len(lambdas)
#     ct = 1

#     for lam in lambdas:
#         val_errors = []
#         r_folds = []
#         print(f" CV testing λ = {lam:.3e}. Currently {ct}/{total_lambdas} ...")
#         for fold_idx in range(k_folds):
#             # train indices = all except this fold
#             val_idx = folds[fold_idx]
#             train_idx = np.hstack(
#                 [folds[i] for i in range(k_folds) if i != fold_idx])

#             # Build train+val datasets
#             Z_all = Z_np.T  # (n, s)
#             X_all = X_np.T  # (n, d)
#             ds_train = PODReconDataset(
#                 Z_np=Z_all[train_idx].T, X_np=X_all[train_idx].T
#             )
#             ds_val   = PODReconDataset(
#                 Z_np=Z_all[val_idx].T,   X_np=X_all[val_idx].T
#             )
#             dl_train = DataLoader(
#                 ds_train, batch_size=batch_size, shuffle=True, drop_last=False
#             )
#             dl_val   = DataLoader(
#                 ds_val,   batch_size=batch_size, shuffle=False, drop_last=False
#             )

#             # Instantiate a fresh model at λ
#             model_cv = SparseModesNet(
#                 pod_basis=torch.from_numpy(U_s_np.astype(np.float32)).to(device),
#                 input_dim=s,
#                 hidden_units=hidden_units,
#                 M=M,
#                 lam=float(lam),
#                 network_type=network_type,
#                 poly_order=poly_order,
#                 num_polys=num_polys,
#                 drop_linear=drop_linear 
#             ).to(device)

#             # Train for num_epochs_cv
#             train_sparsemodesnet(
#                 model_cv, dl_train, num_epochs_cv, lr, optimizer, device)

#             # Evaluate on val set
#             model_cv.eval()
#             mse_loss = nn.MSELoss(reduction='sum')
#             total_err = 0.0
#             total_samples = 0
#             with torch.no_grad():
#                 for z_b, x_b in dl_val:
#                     z_b = z_b.to(device)
#                     x_b = x_b.to(device)
#                     _, x_hat_b = model_cv(z_b)
#                     total_err += mse_loss(x_hat_b, x_b).item()
#                     total_samples += x_b.shape[0]
                    
#                 # Calculate the number of non-zero modes in the optimal omega
#                 omega_opt = model_cv.omega.detach().cpu().numpy()
#                 r_fold = int((np.abs(omega_opt) > nonzero_thresh).sum())
#                 r_folds.append(r_fold)

#             # Compute validation MSE
#             val_mse = total_err / total_samples
#             val_errors.append(val_mse)
            
#         # Average over folds      
#         avg_val_error = np.mean(val_errors)
#         avg_r = int(np.mean(r_folds))
#         print(f" avg val-MSE = {avg_val_error:.6e}, avg r = {avg_r}")
        
#         # Store results for this lambda    
#         path_history_cv.append({
#             'lambda': lam,
#             'error': avg_val_error,
#             'nonzero_count': avg_r,
#         })
        
#         if avg_r == 0:
#             print("All skip-weights have zeroed out. Stopping path.\n")
#             break
        
#         ct += 1

#     return path_history_cv
