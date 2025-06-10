import numpy as np
import warnings
import torch
from torch.utils.data import DataLoader

from .pod import compute_pod_basis
from .model import SparseModesNet
from .dataset import PODReconDataset
from .train import train_sparsemodesnet
from .select_lambda.cv import select_lambda_cv
from .select_lambda.stability import select_lambda_stability
from .stopping.elbow import pick_elbow
from .stopping.aic import pick_aic

def run_sparsemodesnet(X_np: np.ndarray,
                       s: int,
                       hidden_units: list,
                       M: float,
                       nonzero_thresh: float,
                       lam0: float,
                       epsilon: float,
                       lr: float,
                       B: int,
                       max_iters: int,
                       batch_size: int,
                       optimizer: str,
                       device: str,
                       label: str):
    """
    The original warm-start λ→(1+ε)λ routine that stops when ω=0.
    Exactly the same code we provided earlier in step (3).
    """
    print(f"\n=== LassoNet-POD (path) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    U_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = U_s_np.T.dot(X_np)                     # (s, n)

    U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
    Z_tensor   = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)  # (n, s)
    X_torch    = torch.from_numpy(X_np.astype(np.float32)).to(device)    # (d, n)

    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    model = SparseModesNet(
        pod_basis    = U_s_tensor,
        input_dim    = s,
        hidden_units = hidden_units,
        M            = M,
        lam          = lam0
    ).to(device)

    lam = lam0
    prev_nonzero = s
    path_history = []
    iter_count = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e}  (r(λ) prev = {prev_nonzero})")
        model.lam = float(lam)

        history = train_sparsemodesnet(model, dataloader_full, B, lr, optimizer, device)
        omega_opt = model.omega.detach().cpu().numpy()
        nonzero_idxs = np.where(np.abs(omega_opt) > nonzero_thresh)[0]
        curr_nonzero = len(nonzero_idxs)

        model.eval()
        with torch.no_grad():
            _, x_hat_tensor = model(Z_tensor)
            X_hat_np = x_hat_tensor.cpu().numpy().T
        frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
        rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')

        path_history.append({
            'lambda': lam,
            'nonzero_count': curr_nonzero,
            'selected_idxs': nonzero_idxs.copy(),
            'rel_error': rel_frob_error,
            'l1_b': np.mean(history['l1_b'])
        })

        print(f"  → at λ={lam:.3e}:  nonzero={curr_nonzero}, rel_err={rel_frob_error:.6e}")

        if curr_nonzero == 0:
            print("All skip-weights have zeroed out. Stopping path.\n")
            break

        lam = lam * (1.0 + epsilon)
        prev_nonzero = curr_nonzero

        if iter_count >= max_iters:
            print(f"Reached max_iters={max_iters} on λ-path; stopping early.\n")
            break

    omega_opt_final = model.omega.detach().cpu().numpy()
    selected_indices = np.where(np.abs(omega_opt_final) > nonzero_thresh)[0]
    return model, path_history, selected_indices


def run_sparsemodesnet_with_lambda_selection(
    X_np: np.ndarray,
    s: int,
    hidden_units: list,
    M: float,
    lambda_method: str,
    optimizer: str = 'Adam',
    nonzero_thresh: float = 1e-6,
    r_max: int = None, 
    # for “path”:
    lam0: float = 1e-6,
    epsilon: float = 0.1,
    B_path: int = 20,
    max_iters: int = 100,
    # for “cv”:
    lambdas_cv: np.ndarray = None,
    k_folds: int = 5,
    num_epochs_cv: int = 20,
    # for “stability”:
    lambdas_ss: np.ndarray = None,
    B_ss: int = 50,
    pi_thresh: float = 0.6,
    num_epochs_ss: int = 20,
    final_epochs_ss: int = None,
    # common:
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = 'cpu',
    label: str = ''):
    """
    Runs LassoNet-POD-Recon but first picks λ via one of three methods:
      • lambda_method='path'      → warm-start path (the original behavior)
      • lambda_method='stability' → stability selection over lambdas_ss
      • lambda_method='cv'        → k-fold CV + AIC over a list of lambdas_cv

    After λ is chosen, we train a final model on the full data for 'final_epochs' 
    (here we reuse B_path for illustration; you can change it).
    """
    print(f"\n=== LassoNet-POD (λ-selection={lambda_method}) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    # 1) Compute POD basis and Z
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = U_s_np.T.dot(X_np)                     # (s, n)

    # Convert pod_basis to tensor once
    U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)

    if lambda_method == 'cv':
        assert lambdas_cv is not None, "Must pass a grid 'lambdas_cv' for CV."
        path_history = select_lambda_cv(
            X_np             = X_np,
            s                = s,
            hidden_units     = hidden_units,
            M                = M,
            nonzero_thresh   = nonzero_thresh,
            lambdas          = lambdas_cv,
            lr               = lr,
            num_epochs_cv    = num_epochs_cv,
            k_folds          = k_folds,
            batch_size       = batch_size,
            optimizer        = optimizer,
            device           = device
        )
        n_samples = X_np.shape[1]
        lam_star, r_star, err_star = pick_aic(path_history, n_samples)
        print(f"[CV-AIC] Picked λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
        freq_table = path_history
    elif lambda_method == 'stability':
        assert lambdas_ss is not None, "Must pass a grid 'lambdas_ss' for stability selection."
        lam_star, path_history, S_stable, freqs = select_lambda_stability(
            X_np             = X_np,
            s                = s,
            hidden_units     = hidden_units,
            M                = M,
            nonzero_thresh   = nonzero_thresh,
            lambdas          = lambdas_ss,
            B                = B_ss,
            pi_thresh        = pi_thresh,
            lr               = lr,
            num_epochs_ss    = num_epochs_ss,
            final_epochs     = final_epochs_ss if final_epochs_ss is not None else num_epochs_ss,
            batch_size       = batch_size,
            optimizer        = optimizer,
            device           = device
        )
        print(f"[SS] Picked λ={lam_star:.3e}, S_stable={S_stable}")
        freq_table = {'path_history': path_history, 'stable_modes': S_stable, 'freqs': freqs}
    else:  # 'path' (warm‐start)
        if lambda_method != 'path':
            warnings.warn(
                f"Method {lambda_method!r} does not exist. Using 'path' method instead." 
            )
        _, path_history, _ = run_sparsemodesnet(
            X_np           = X_np,
            s              = s,
            hidden_units   = hidden_units,
            M              = M,
            nonzero_thresh = nonzero_thresh,
            lam0           = lam0,
            epsilon        = epsilon,
            lr             = lr,
            B              = B_path,
            max_iters      = max_iters,
            batch_size     = batch_size,
            optimizer      = optimizer,
            device         = device,
            label          = label
        )
        lam_star, r_star, err_star = pick_elbow(path_history)
        print(f"[Path-Elbow] Picked λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
        freq_table = path_history
        
    # Fallback if r_star is still 0 (no modes selected) 
    if r_star == 0:
        print("No modes selected at λ*. Using last entry in path such that r(λ)≤r_max.")
        if r_max is not None: # Find first lambda where nonzero_count < r_max
            print(f"Searching for first λ with nonzero_count < {r_max} ...")
            nonzero_counts = np.array([entry['nonzero_count'] for entry in path_history])
            valid_indices = np.where(nonzero_counts < r_max)[0]
            if len(valid_indices) > 0:
                idx = valid_indices[0]
                lam_star = path_history[idx]['lambda']
                r_star = path_history[idx]['nonzero_count']
                err_star = path_history[idx]['rel_error']
            else: # If no entry found, use the last one
                lam_star = path_history[-1]['lambda']
                r_star = path_history[-1]['nonzero_count']
                err_star = path_history[-1]['rel_error']
        else: # If r_max not specified, use the last entry from path
            print("r_max not specified. Using last entry from path.")
            lam_star = path_history[-1]['lambda']
            r_star = path_history[-1]['nonzero_count']
            err_star = path_history[-1]['rel_error']
            
        print(f"[Fallback] λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
            
    print(f"\n→ Final training on full data with λ = {lam_star:.3e} ...")
    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    model_final = SparseModesNet(
        pod_basis    = U_s_tensor,
        input_dim    = s,
        hidden_units = hidden_units,
        M            = M,
        lam          = float(lam_star)
    ).to(device)

    history_full = train_sparsemodesnet(
        model      = model_final,
        dataloader = dataloader_full,
        num_epochs = B_path,
        lr         = lr,
        optimizer  = optimizer,
        device     = device
    )

    omega_opt = model_final.omega.detach().cpu().numpy()
    selected_indices = np.where(np.abs(omega_opt) > nonzero_thresh)[0]
    print(f"\nFinal skip-weights ω: {omega_opt.tolist()[:10]} ...")
    print(f"Selected POD-mode indices (ω_j ≠ 0): {selected_indices.tolist()}  "
          f"(count = {len(selected_indices)} / {s})")

    model_final.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)
        _, x_hat_tensor = model_final(Z_tensor)
        X_hat_np = x_hat_tensor.cpu().numpy().T
    frob_error     = np.linalg.norm(X_np - X_hat_np, 'fro')
    rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
    mse_per_sample = frob_error / X_np.shape[1]
    print(f"Final relative reconstruction of SparseModesNet: ||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
    print(f"Final MSE per sample of SparseModesNet: = {mse_per_sample:.6e}")
    
    # Compute the error when using only selected modes
    U_selected = U_s_np[:, selected_indices]  # (d, k)
    frob_error_selected = np.linalg.norm(
        X_np - U_selected @ (U_selected.T @ X_np), 'fro'
    ) 
    rel_frob_error_selected = frob_error_selected / np.linalg.norm(X_np, 'fro')
    mse_per_sample_selected = frob_error_selected / X_np.shape[1]
    print(f"Relative error using only selected modes: {rel_frob_error_selected:.6e}")
    print(f"MSE per sample using only selected modes: {mse_per_sample_selected:.6e}")

    return model_final, {'history_full': history_full, 'lambda_star': lam_star}, selected_indices, freq_table


# def run_sparsemodesnet_with_lambda_selection(
#     X_np: np.ndarray,
#     s: int,
#     hidden_units: list,
#     M: float,
#     lambda_method: str,
#     optimizer: str = 'Adam',
#     nonzero_thresh: float = 1e-6,
#     # for “path”:
#     lam0: float = 1e-6,
#     epsilon: float = 0.1,
#     B_path: int = 20,
#     max_iters: int = 100,
#     # for “cv”:
#     lambdas_cv: np.ndarray = None,
#     k_folds: int = 5,
#     num_epochs_cv: int = 20,
#     # for “stability”:
#     lambdas_ss: np.ndarray = None,
#     B_ss: int = 50,
#     pi_thresh: float = 0.6,
#     num_epochs_sub: int = 20,
#     # new: stopping criterion for path
#     stop_method: str = 'aic',
#     aic_alpha: float = 2.0,
#     K_max: int = None,  # max modes for constraint
#     # common:
#     lr: float = 1e-3,
#     batch_size: int = 16,
#     device: str = 'cpu',
#     label: str = ''):
#     """
#     Runs LassoNet-POD-Recon but first picks λ via one of three methods:
#       • lambda_method='path'      → warm-start path (the original behavior)
#       • lambda_method='cv'        → k-fold CV + AIC over a list of lambdas_cv
#       • lambda_method='stability' → stability selection over lambdas_ss

#     After λ is chosen, we train a final model on the full data for 'final_epochs' 
#     (here we reuse B_path for illustration; you can change it).
#     """
#     print(f"\n=== LassoNet-POD (λ-selection={lambda_method}) on {label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

#     # 1) Compute POD basis and Z
#     U_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
#     Z_np = U_s_np.T.dot(X_np)                     # (s, n)

#     # Convert pod_basis to tensor once
#     U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)

#     if lambda_method == 'cv':
#         assert lambdas_cv is not None, "Must pass a grid 'lambdas_cv' for CV."
#         path_history_cv = select_lambda_cv(
#             X_np             = X_np,
#             s                = s,
#             hidden_units     = hidden_units,
#             M                = M,
#             nonzero_thresh   = nonzero_thresh,
#             lambdas          = lambdas_cv,
#             lr               = lr,
#             num_epochs_cv    = num_epochs_cv,
#             k_folds          = k_folds,
#             batch_size       = batch_size,
#             optimizer        = optimizer,
#             device           = device
#         )
#         n_samples = X_np.shape[1]
#         if stop_method == 'elbow':
#             lam_star, k_star, err_star = pick_elbow(path_history_cv)
#             print(f"[CV-Elbow] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'aic':
#             lam_star, k_star, err_star = pick_aic(path_history_cv, n_samples, aic_alpha)
#             print(f"[CV-AIC, α={aic_alpha:.1e}] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'constraint':
#             if K_max is None:
#                 K_max = s // 2
#             lam_star, k_star, err_star = pick_max_modes(path_history_cv, K_max)
#             print(f"[CV-Constraint] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         else:
#             raise ValueError("stop_method must be 'elbow', 'aic', or 'constraint'")
#         freq_table = None
#     elif lambda_method == 'stability':
#         assert lambdas_ss is not None, "Must pass a grid 'lambdas_ss' for stability selection."
#         path_history_ss = select_lambda_stability(
#             X_np             = X_np,
#             s                = s,
#             hidden_units     = hidden_units,
#             M                = M,
#             nonzero_thresh   = nonzero_thresh,
#             lambdas          = lambdas_ss,
#             B                = B_ss,
#             pi_thresh        = pi_thresh,
#             lr               = lr,
#             num_epochs_sub   = num_epochs_sub,
#             batch_size       = batch_size,
#             optimizer        = optimizer,
#             device           = device
#         )
#         n_samples = X_np.shape[1]
#         if stop_method == 'elbow':
#             lam_star, k_star, err_star = pick_elbow(path_history_ss)
#             print(f"[SS-Elbow] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'aic':
#             lam_star, k_star, err_star = pick_aic(path_history_ss, n_samples, aic_alpha)
#             print(f"[SS-AIC, α={aic_alpha:.2e}] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'constraint':
#             if K_max is None:
#                 K_max = s // 2
#             lam_star, k_star, err_star = pick_max_modes(path_history_ss, K_max)
#             print(f"[SS-Constraint] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         else:
#             raise ValueError("stop_method must be 'elbow', 'aic', or 'constraint'")
#         freq_table = path_history_ss
#     else:  # 'path' (warm‐start)
#         model_path, path_history, _ = run_sparsemodesnet(
#             X_np           = X_np,
#             s              = s,
#             hidden_units   = hidden_units,
#             M              = M,
#             nonzero_thresh = nonzero_thresh,
#             lam0           = lam0,
#             epsilon        = epsilon,
#             lr             = lr,
#             B              = B_path,
#             max_iters      = max_iters,
#             batch_size     = batch_size,
#             optimizer      = optimizer,
#             device         = device,
#             label          = label
#         )
#         n_samples = X_np.shape[1]
#         if stop_method == 'elbow':
#             lam_star, k_star, err_star = pick_elbow(path_history)
#             print(f"[Path-Elbow] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'aic':
#             lam_star, k_star, err_star = pick_aic(path_history, n_samples, aic_alpha)
#             print(f"[Path-AIC, α={aic_alpha:.2e}] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         elif stop_method == 'constraint':
#             if K_max is None:
#                 K_max = s // 2
#             lam_star, k_star, err_star = pick_max_modes(path_history, K_max)
#             print(f"[Path-Constraint] Picked λ={lam_star:.3e}, k={k_star}, err={err_star:.6e}")
#         else:
#             raise ValueError("stop_method must be 'elbow', 'aic', or 'constraint'")
#         freq_table = None

#     print(f"\n→ Final training on full data with λ = {lam_star:.3e} ...")
#     dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
#     dataloader_full = DataLoader(dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

#     model_final = SparseModesNet(
#         pod_basis    = U_s_tensor,
#         input_dim    = s,
#         hidden_units = hidden_units,
#         M            = M,
#         lam          = float(lam_star)
#     ).to(device)

#     history_full = train_sparsemodesnet(
#         model      = model_final,
#         dataloader = dataloader_full,
#         num_epochs = B_path,
#         lr         = lr,
#         optimizer  = optimizer,
#         device     = device
#     )

#     omega_opt = model_final.omega.detach().cpu().numpy()
#     selected_indices = np.where(np.abs(omega_opt) > nonzero_thresh)[0]
#     print(f"\nFinal skip-weights ω: {omega_opt.tolist()[:10]} ...")
#     print(f"Selected POD-mode indices (ω_j ≠ 0): {selected_indices.tolist()}  "
#           f"(count = {len(selected_indices)} / {s})")

#     model_final.eval()
#     with torch.no_grad():
#         Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)
#         _, x_hat_tensor = model_final(Z_tensor)
#         X_hat_np = x_hat_tensor.cpu().numpy().T
#     frob_error     = np.linalg.norm(X_np - X_hat_np, 'fro')
#     rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
#     mse_per_sample = frob_error / X_np.shape[1]
#     print(f"Final relative reconstruction of SparseModesNet: ||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
#     print(f"Final MSE per sample of SparseModesNet: = {mse_per_sample:.6e}")
    
#     # Compute the error when using only selected modes
#     U_selected = U_s_np[:, selected_indices]  # (d, k)
#     frob_error_selected = np.linalg.norm(
#         X_np - U_selected @ (U_selected.T @ X_np), 'fro'
#     ) 
#     rel_frob_error_selected = frob_error_selected / np.linalg.norm(X_np, 'fro')
#     mse_per_sample_selected = frob_error_selected / X_np.shape[1]
#     print(f"Relative error using only selected modes: {rel_frob_error_selected:.6e}")
#     print(f"MSE per sample using only selected modes: {mse_per_sample_selected:.6e}")

#     return model_final, {'history_full': history_full, 'lambda_star': lam_star}, selected_indices, freq_table