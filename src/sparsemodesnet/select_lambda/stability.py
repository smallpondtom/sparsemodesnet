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
                            nonzero_thresh: float,
                            lambdas: np.ndarray,
                            B: int,
                            pi_thresh: float,
                            lr: float,
                            num_epochs_ss: int,
                            final_epochs: int,
                            batch_size: int,
                            optimizer: str,
                            device: str):
    """
    Canonical stability-selection:
        1) subsample B times for each λ, accumulate counts
        2) compute Π_j(λ), Π_j^max and form S_stable
        3) pick λ* = largest λ whose full-data S(λ) ⊇ S_stable
        4) retrain final model at λ* on full data
    Returns: model_final, lambda_star, S_stable, freqs
    """
    print("\n=== Stability Selection λ-Selection ===")
    d, n = X_np.shape
    V_s_np, _, _ = compute_pod_basis(X_np, s=s)
    Z_np = V_s_np.T.dot(X_np)  # (s, n)

    m = len(lambdas)
    counts = np.zeros((s, m), dtype=int)

    # 1) subsample loop
    for i, lam in enumerate(lambdas):
        print(f" SS testing λ = {lam:.3e} ...")
        for _ in range(B):
            subsamp = np.random.choice(n, size=n//2, replace=False)
            ds_sub = PODReconDataset(Z_np=Z_np[:, subsamp], X_np=X_np[:, subsamp])
            dl_sub = DataLoader(ds_sub, batch_size=batch_size, shuffle=True)

            model_ss = SparseModesNet(
                pod_basis    = torch.from_numpy(V_s_np).to(device),
                input_dim    = s,
                hidden_units = hidden_units,
                M            = M,
                lam          = float(lam)
            ).to(device)
            train_sparsemodesnet(model_ss, dl_sub, num_epochs_ss, lr, optimizer, device)

            omega_opt = model_ss.omega.detach().cpu().numpy()
            counts[:, i] += (np.abs(omega_opt) > nonzero_thresh).astype(int)

        # This is only for logging
        freqs_i = counts[:, i] / float(B)
        stable_count_i = int((freqs_i >= pi_thresh).sum())
        print(f"  → λ = {lam:.3e} | stable features = {stable_count_i} (freq ≥ {pi_thresh})")

    # 2) aggregate into selection probabilities
    freqs   = counts / float(B)           # (s, m)
    pi_max  = freqs.max(axis=1)           # (s,)
    S_stable = set(np.where(pi_max >= pi_thresh)[0])
    print(f"\nComputed Π_j(λ) over all λ. Final stable set size = {len(S_stable)}")

    # 3) pick λ* by checking full-data fits
    ds_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True)

    path_history_ss = []
    lambda_star = None
    S_stable_count = len(S_stable)
    set_diff_min = np.inf
    set_diff_min_lam = None
    print("Finding largest λ that recovers all stable features on full data...")
    for i, lam in enumerate(reversed(lambdas)):  # assuming lambdas are sorted ascending
        print(f"  Testing λ = {lam:.3e} on full data ...")
        model_full = SparseModesNet(
            pod_basis    = torch.from_numpy(V_s_np).to(device),
            input_dim    = s,
            hidden_units = hidden_units,
            M            = M,
            lam          = float(lam)
        ).to(device)
        train_sparsemodesnet(model_full, dl_full,
                             num_epochs_ss if final_epochs is None else final_epochs,
                             lr, optimizer, device)
        b_full = model_full.b.detach().cpu().numpy()
        S_full = set(np.where(np.abs(b_full) > nonzero_thresh)[0])
        
        # Record values for fallback method 
        S_full_count = len(S_full) 
        set_diff_i = abs(S_full_count - S_stable_count)
        if set_diff_i < set_diff_min:
            set_diff_min = set_diff_i
            set_diff_min_lam = lam
            
        path_history_ss.append({
            'lambda': lam,
            'r': S_full_count,
            'rel_error': np.nan
        })
            
        print(f"    → selected {len(S_full)} features; need ≥ {len(S_stable)}")
        if S_stable.issubset(S_full):
            lambda_star = lam
            print(f"  → λ* = {lam:.3e} (covers stable set)\n")
            break
    
    # If no λ covers S_stable, find the first λ satisfies
    # min |r(S_full) - r(S_stable)|, where r(S) = |S|
    if lambda_star is None:
        lambda_star = set_diff_min_lam
        print(f"  No λ covered S_stable; fallback → λ = argmin|r(S)-r(S_stable)| = {lambda_star:.3e}\n")

    # # 4) final retraining at λ*
    # print(f"Retraining final model on full data with λ* = {lambda_star:.3e} ...")
    # model_final = SparseModesNet(
    #     pod_basis    = torch.from_numpy(V_s_np).to(device),
    #     input_dim    = s,
    #     hidden_units = hidden_units,
    #     M            = M,
    #     lam          = float(lambda_star)
    # ).to(device)
    # train_sparsemodesnet(model_final, dl_full,
    #                      num_epochs_ss if final_epochs is None else final_epochs,
    #                      lr, optimizer, device)

    # print("Final model ready.\n")
    return lambda_star, path_history_ss, S_stable, freqs


# def select_lambda_stability(X_np: np.ndarray,
#                             s: int,
#                             hidden_units: list,
#                             M: float,
#                             nonzero_thresh: float,
#                             lambdas: np.ndarray,
#                             B: int,
#                             pi_thresh: float,
#                             lr: float,
#                             num_epochs_sub: int,
#                             batch_size: int,
#                             optimizer: str,
#                             device: str):
#     """
#     Performs Meinshausen-Bühlmann stability selection over a grid of λ.
#     Returns the first λ for which no features exceed pi_thresh frequency, 
#     or the λ that yields <= a target #features. Here we choose λ s.t. most features 
#     drop out. The user can inspect the returned freq table for details.
#     """
#     print("\n=== Stability Selection λ-Selection ===")
#     d, n = X_np.shape
#     U_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
#     Z_np = U_s_np.T.dot(X_np)                     # (s, n)

#     path_history_ss = []

#     for lam in lambdas:
#         counts = np.zeros(s, dtype=int)
#         print(f" SS testing λ = {lam:.3e} ...")

#         for b in range(B):
#             # Random half‐sample of indices
#             subsamp = np.random.choice(n, size=n//2, replace=False)
#             ds_sub = PODReconDataset(Z_np=Z_np[:, subsamp], X_np=X_np[:, subsamp])
#             dl_sub = DataLoader(ds_sub, batch_size=batch_size, shuffle=True, drop_last=False)

#             # Train on that subsample
#             model_ss = SparseModesNet(
#                 pod_basis    = torch.from_numpy(U_s_np.astype(np.float32)).to(device),
#                 input_dim    = s,
#                 hidden_units = hidden_units,
#                 M            = M,
#                 lam          = float(lam)
#             ).to(device)

#             train_sparsemodesnet(model_ss, dl_sub, num_epochs_sub, lr, optimizer, device)

#             # Record which features are nonzero in b
#             omega_opt = model_ss.omega.detach().cpu().numpy()
#             counts += (np.abs(omega_opt) > nonzero_thresh).astype(int)

#         freqs = counts / float(B)
#         stable_count = int((freqs >= pi_thresh).sum())
#         print(f"stable_count = {stable_count} "
#               f"(#features with freq ≥ {pi_thresh} = {stable_count})")

#         # Now, train on the full data (briefly) to get k(λ) and rel_error
#         final_full_epochs = num_epochs_sub
#         Z_full = Z_np.T  # (n, s)
#         X_full = X_np.T  # (n, d)
#         ds_full = PODReconDataset(Z_np=Z_full.T, X_np=X_full.T)
#         dl_full = DataLoader(ds_full, batch_size=batch_size, shuffle=True, drop_last=False)
#         model_full = SparseModesNet(
#             pod_basis    = torch.from_numpy(U_s_np.astype(np.float32)).to(device),
#             input_dim    = s,
#             hidden_units = hidden_units,
#             M            = M,
#             lam          = float(lam)
#         ).to(device)
#         train_sparsemodesnet(model_full, dl_full, final_full_epochs, lr, optimizer, device)
#         with torch.no_grad():
#             omega_full = model_full.omega.detach().cpu().numpy()
#             k_full = int((np.abs(omega_full) > nonzero_thresh).sum())
#             Z_tensor_full = torch.from_numpy(Z_full.astype(np.float32)).to(device)
#             _, x_hat_full = model_full(Z_tensor_full)
#             X_hat_full_np = x_hat_full.cpu().numpy().T
#             frob_err = np.linalg.norm(X_np - X_hat_full_np, 'fro')
#             rel_err = frob_err / np.linalg.norm(X_np, 'fro')
#         path_history_ss.append({
#             'lambda': lam,
#             'stable_count': stable_count,
#             'k': k_full,
#             'rel_error': rel_err
#         })

#         # If no feature is “stable,” we can stop early
#         if stable_count == 0:
#             print("All features dropped out at this λ; stopping SS path.\n")
#             break

#     return path_history_ss