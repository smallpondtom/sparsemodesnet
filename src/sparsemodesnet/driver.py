import numpy as np
import kneeliverse.dfdt as dfdt
import kneeliverse.zmethod as zmethod
import warnings
import torch
from torch.utils.data import DataLoader

from .pod import compute_pod_basis
from .model import SparseModesNet
from .dataset import PODReconDataset
from .train import train_sparsemodesnet
from .cv import run_sparsemodesnet_cv

def run_sparsemodesnet_d2s(X_np: np.ndarray,
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
                           device: str):
    """
    The original warm-start λ→(1+ε)λ routine.
    """
    print("\n=== Dense-To-Sparse (default) λ-Path ===")
    
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)   # (d, s)
    Z_np = U_s_np.T.dot(X_np)                     # (s, n)

    U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
    Z_tensor   = torch.from_numpy(Z_np.T.astype(np.float32)).to(device) # (n, s)
    X_torch    = torch.from_numpy(X_np.astype(np.float32)).to(device)   # (d, n)

    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    lam = lam0
    prev_nonzero = s
    path_history = []
    iter_count = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e} ",
              f" (r(λ) prev = {prev_nonzero})")
        
        model = SparseModesNet(
            pod_basis    = U_s_tensor,
            input_dim    = s,
            hidden_units = hidden_units,
            M            = M,
            lam          = lam
        ).to(device)
        
        history = train_sparsemodesnet(
            model, dataloader_full, B, lr, optimizer, device)
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
            'error': rel_frob_error,
            'l1_b': np.mean(history['l1_b'])
        })

        print(f"  → at λ={lam:.3e}:  r(λ)={curr_nonzero},", 
              f" rel_err={rel_frob_error:.6e}")

        if curr_nonzero == 0:
            print("All skip-weights have zeroed out. Stopping path.\n")
            break

        lam = lam * (1.0 + epsilon)
        prev_nonzero = curr_nonzero

        if iter_count >= max_iters:
            print(f"Reached max_iters={max_iters} on λ-path; stopping early.\n")
            break

    return path_history


def run_sparsemodesnet(
    X_np: np.ndarray,
    s: int,
    hidden_units: list,
    M: float,
    reg_path: str = 'dense2sparse',
    lr: float = 1e-3,
    batch_size: int = 16,
    knee_method: str = 'dfdt', 
    optimizer: str = 'Adam',
    nonzero_thresh: float = 1e-6,
    r_max: int = None, 
    # for "path":
    lam0: float = 1e-6,
    epsilon: float = 0.1,
    B_path: int = 20,
    max_iters: int = 100,
    # for "cv":
    lambdas_cv: np.ndarray = None,
    k_folds: int = 5,
    num_epochs_cv: int = 20,
    # other common:
    device: str = 'cpu',
    label: str = ''
):
    """
    Runs SparseModesNet with lambda selection via path or CV.
    
    Returns
    -------
    tuple: (final_model, info_dict, selected_indices, path_history)
    """
    print(f"\n=== SparseModesNet (λ-path={reg_path}) on "
          f"{label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")

    # Compute POD basis and coefficients
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)
    Z_np = U_s_np.T.dot(X_np)
    U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)

    # Select lambda using specified method
    path_history = _regularization_path(
        reg_path, X_np, s, hidden_units, M, nonzero_thresh,
        lambdas_cv, k_folds, num_epochs_cv, lam0, epsilon, B_path, 
        max_iters, lr, batch_size, optimizer, device
    )
    
    # Find optimal lambda using knee detection
    lam_star, r_star, err_star = _find_optimal_lambda(
        path_history, knee_method, r_max, reg_path
    )
    
    # Train final model with selected lambda
    model_final, history_full = _train_final_model(
        U_s_tensor, X_np, Z_np, s, hidden_units, M, lam_star, 
        B_path, lr, optimizer, device, batch_size
    )
    
    # Evaluate and report results
    selected_indices = _evaluate_and_report_results(
        model_final, X_np, U_s_np, Z_np, s, nonzero_thresh, 
        lam_star, device
    )
    
    return (
        model_final, 
        {'history_full': history_full, 'lambda_star': lam_star}, 
        selected_indices, 
        path_history
    )


def _regularization_path(reg_path, X_np, s, hidden_units, M, nonzero_thresh,
                         lambdas_cv, k_folds, num_epochs_cv, lam0, epsilon, 
                         B_path, max_iters, lr, batch_size, optimizer, device):
    """Obtain regularization path using the specified method."""
    if reg_path == 'cv':
        if lambdas_cv is None:
            raise ValueError("Must pass a grid 'lambdas_cv' for CV.")
        return run_sparsemodesnet_cv(
            X_np=X_np, s=s, hidden_units=hidden_units, M=M,
            nonzero_thresh=nonzero_thresh, lambdas=lambdas_cv,
            lr=lr, num_epochs_cv=num_epochs_cv, k_folds=k_folds,
            batch_size=batch_size, optimizer=optimizer, device=device
        )
    else:
        if reg_path != 'dense2sparse':
            warnings.warn(f"Method {reg_path!r} does not exist. "
                          f"Using 'dense2sparse' method instead.")
        return run_sparsemodesnet_d2s(
            X_np=X_np, s=s, hidden_units=hidden_units, M=M,
            nonzero_thresh=nonzero_thresh, lam0=lam0, epsilon=epsilon,
            lr=lr, B=B_path, max_iters=max_iters, batch_size=batch_size,
            optimizer=optimizer, device=device
        )


def _find_optimal_lambda(path_history, knee_method, r_max, reg_path):
    """Find optimal lambda using knee detection or fallback methods."""
    # Prepare data for knee detection
    lam = np.array([h['lambda'] for h in path_history])
    loglam = np.log(lam)
    rs = np.array([h['nonzero_count'] for h in path_history])
    
    # Normalize to [0, 1] range
    loglam_norm = _normalize_array(loglam)
    rs_norm = _normalize_array(rs)
    data = np.column_stack((loglam_norm, rs_norm))
    
    # Apply knee detection
    knee_idx = _detect_knees(data, knee_method)
    
    if len(knee_idx) == 0:
        return _fallback_lambda_selection(path_history, r_max)
    else:
        return _select_from_knees(
            path_history, knee_idx, loglam_norm, loglam, rs, r_max, 
            reg_path
        )


def _normalize_array(arr):
    """Normalize array to [0, 1] range."""
    return (arr - arr.min()) / (arr.max() - arr.min())


def _detect_knees(data, knee_method):
    """Detect knee points using specified method."""
    if knee_method == 'dfdt':
        knee_idx = dfdt.multi_knee(data)
        print(f"Found knees using DFDT method: {knee_idx}")
    elif knee_method == 'zmethod':
        knee_idx = zmethod.knees2(data)
        print(f"Found knees using Z-method: {knee_idx}")
    else:
        raise ValueError(f"Unknown knee_method: {knee_method}. "
                        f"Use 'dfdt' or 'zmethod'.")
    return knee_idx


def _fallback_lambda_selection(path_history, r_max):
    """Fallback lambda selection when knee detection fails."""
    print("No modes selected at λ*. Using fallback selection.")
    
    if r_max is not None:
        print(f"Searching for first λ with nonzero_count <= {r_max} ...")
        nonzero_counts = np.array(
            [entry['nonzero_count'] for entry in path_history])
        valid_indices = np.where(nonzero_counts <= r_max)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[0]
        else:
            idx = -1  # Use last entry
    else:
        print("r_max not specified. Using last entry from path.")
        idx = -1
    
    entry = path_history[idx]
    lam_star, r_star = entry['lambda'], entry['nonzero_count']
    err_star = entry['error']
    print(f"[Fallback] λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
    
    return lam_star, r_star, err_star


def _select_from_knees(path_history, knee_idx, loglam_norm, loglam, 
                       rs, r_max, reg_path):
    """Select lambda from detected knee points."""
    # Convert normalized knee points back to original scale
    loglam_range = loglam.max() - loglam.min()
    knee_normalized = loglam_norm[knee_idx]
    lam_stars = np.exp(knee_normalized * loglam_range + loglam.min())
    
    r_stars = rs[knee_idx]
    err_stars = np.array([path_history[i]['error'] for i in knee_idx])
    
    # Pick first knee point that satisfies r_max constraint
    if r_max is not None:
        valid_mask = r_stars <= r_max
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            i_star = valid_indices[0]
        else:
            i_star = 0  # Fallback to first knee
    else:
        i_star = 0
    
    lam_star, r_star = lam_stars[i_star], r_stars[i_star]
    err_star = err_stars[i_star]
    method_name = {
        'dense2sparse': 'Dense2Sparse', 'cv': 'CV'
    }.get(reg_path, reg_path)
    print(f"[{method_name}] Picked λ={lam_star:.3e}, ",
          f"r={r_star}, err={err_star:.6e}")
    
    return lam_star, r_star, err_star


def _train_final_model(U_s_tensor, X_np, Z_np, s, hidden_units, M, lam_star, 
                      B_path, lr, optimizer, device, batch_size):
    """Train final model with selected lambda."""
    print(f"\n→ Final training on full data with λ = {lam_star:.3e} ...")
    
    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    model_final = SparseModesNet(
        pod_basis=U_s_tensor, input_dim=s, hidden_units=hidden_units,
        M=M, lam=float(lam_star)
    ).to(device)
    
    history_full = train_sparsemodesnet(
        model=model_final, dataloader=dataloader_full, num_epochs=B_path,
        lr=lr, optimizer=optimizer, device=device
    )
    
    return model_final, history_full


def _evaluate_and_report_results(model_final, X_np, U_s_np, Z_np, s, 
                                 nonzero_thresh, lam_star, device):
    """Evaluate final model and report results."""
    # Get selected indices
    omega_opt = model_final.omega.detach().cpu().numpy()
    selected_indices = np.where(np.abs(omega_opt) > nonzero_thresh)[0]
    
    print(f"\nFinal skip-weights ω: {omega_opt.tolist()[:10]} ...")
    print(f"Selected POD-mode indices (ω_j ≠ 0): {selected_indices.tolist()} "
          f"(count = {len(selected_indices)} / {s})")
    
    # Compute reconstruction errors
    model_final.eval()
    with torch.no_grad():
        Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)
        _, x_hat_tensor = model_final(Z_tensor)
        X_hat_np = x_hat_tensor.cpu().numpy().T
    
    # SparseModesNet reconstruction error
    frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
    rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
    mse_per_sample = frob_error / X_np.shape[1]
    
    print(f"Final relative reconstruction of SparseModesNet: "
          f"||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
    print(f"Final MSE per sample of SparseModesNet: = {mse_per_sample:.6e}")
    
    # Selected modes only reconstruction error
    if len(selected_indices) > 0:
        U_selected = U_s_np[:, selected_indices]
        frob_error_selected = np.linalg.norm(
            X_np - U_selected @ (U_selected.T @ X_np), 'fro'
        )
        X_np_norm = np.linalg.norm(X_np, 'fro')
        rel_frob_error_selected = frob_error_selected / X_np_norm
        mse_per_sample_selected = frob_error_selected / X_np.shape[1]
        
        print(f"Relative error using only selected ",
              f"modes: {rel_frob_error_selected:.6e}")
        print(f"MSE per sample using only selected ",
              f"modes: {mse_per_sample_selected:.6e}")
    
    return selected_indices