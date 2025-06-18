import numpy as np
import kneeliverse.dfdt as dfdt
import kneeliverse.zmethod as zmethod
import warnings
import torch
import logging
import os
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from .pod import compute_pod_basis
from .model import SparseModesNet
from .dataset import PODReconDataset
from .train import train_sparsemodesnet
from .cv import run_sparsemodesnet_cv

def _setup_experiment_logging(experiment_name="sparsemodesnet", logs_dir=None):
    """Setup logging with timestamp for experiment tracking."""
    if logs_dir is None:
        # Default to logs directory in current working directory
        logs_dir = Path.cwd() / "logs"
    else:
        logs_dir = Path(logs_dir)
    
    logs_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"{experiment_name}_{timestamp}.log"
    
    # Create a logger specific to this experiment
    logger = logging.getLogger(f"sparsemodesnet.{experiment_name}")
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with simple format (no timestamp/logger name)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Detailed formatter - just the message
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(detailed_formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Console handler keeps the full format for debugging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Initial log entries with timestamp for reference
    logger.info(f"Starting {experiment_name} experiment")
    logger.info(f"Log file: {log_filename}")
    
    return logger, log_filename

def run_sparsemodesnet_d2s(X_np: np.ndarray,
                           s: int,
                           hidden_units: list,
                           M: float,
                           nonzero_thresh: float,
                           lam0: float,
                           epsilon: float,
                           network_type: str,
                           poly_order: int,
                           num_polys: int,
                           drop_linear: bool,
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
            lam          = lam,
            network_type = network_type,
            poly_order   = poly_order,
            num_polys    = num_polys,
            drop_linear  = drop_linear
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
    network_type: str = 'FF',
    poly_order: int = 2,
    num_polys: int = 1,
    drop_linear: bool = False,
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
    label: str = '',
    enable_logging: bool = True,
    logs_dir: str = None
):
    """
    Runs SparseModesNet with lambda selection via path or CV.
    
    Returns
    -------
    tuple: (final_model, info_dict, selected_indices, path_history)
    """
    # Setup logging if enabled
    if enable_logging:
        experiment_name = label.lower().replace(" ", "_") if label else "sparsemodesnet"
        logger, log_file = _setup_experiment_logging(experiment_name, logs_dir)
        
        # Log experiment parameters (these will have timestamps)
        logger.info(f"Experiment: {label}")
        logger.info(f"Data shape: {X_np.shape}")
        logger.info(f"POD dimension: {s}")
        logger.info(f"Hidden units: {hidden_units}")
        logger.info(f"Regularization method: {reg_path}")
        logger.info(f"Device: {device}")
        logger.info(f"Training parameters: M={M}, lr={lr}, batch_size={batch_size}")
        logger.info(f"Optimizer: {optimizer}, knee_method: {knee_method}")
        if network_type == 'FF':
            logger.info("NN type: Feedforward")
        else:
            logger.info(f"NN type: {network_type} with polynomial "
                        f"order of {poly_order}, and {num_polys} number of "
                        f"polynomial blocks")
        logger.info("="*50)
        
        # Create a simple file-only logger for training output
        training_logger = logging.getLogger(f"training.{experiment_name}")
        training_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        for handler in training_logger.handlers[:]:
            training_logger.removeHandler(handler)
        
        # Add only file handler with simple format
        training_file_handler = logging.FileHandler(log_file)
        training_file_handler.setLevel(logging.INFO)
        simple_formatter = logging.Formatter('%(message)s')
        training_file_handler.setFormatter(simple_formatter)
        training_logger.addHandler(training_file_handler)
        training_logger.propagate = False  # Don't propagate to parent loggers
        
        # Modify print function to log to file only (no timestamps)
        original_print = print
        def logged_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            training_logger.info(message)  # Simple format to file
            original_print(*args, **kwargs)  # Normal print to console
        
        # Temporarily replace print with logged version
        import builtins
        builtins.print = logged_print
    
    try:
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
            max_iters, lr, batch_size, optimizer, device, 
            network_type, poly_order, num_polys, drop_linear
        )
        
        # Find optimal lambda using knee detection
        lam_star, r_star, err_star = _find_optimal_lambda(
            path_history, knee_method, r_max, reg_path
        )
        
        # Train final model with selected lambda
        model_final, history_full = _train_final_model(
            U_s_tensor, X_np, Z_np, s, hidden_units, M, lam_star, 
            B_path, lr, optimizer, device, batch_size,
            network_type, poly_order, num_polys, drop_linear
        )
        
        # Evaluate and report results
        selected_indices = _evaluate_and_report_results(
            model_final, X_np, U_s_np, Z_np, s, nonzero_thresh, 
            lam_star, device
        )
        
        if enable_logging:
            # Final summary with timestamps
            logger.info("="*50)
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Selected {len(selected_indices)} modes out of {s}")
            logger.info(f"Final lambda: {lam_star:.3e}")
        
        return (
            model_final, 
            {'history_full': history_full, 'lambda_star': lam_star, 'log_file': log_file if enable_logging else None}, 
            selected_indices, 
            path_history
        )
        
    finally:
        # Restore original print function
        if enable_logging:
            import builtins
            builtins.print = original_print

    
def _regularization_path(reg_path, X_np, s, hidden_units, M, nonzero_thresh,
                         lambdas_cv, k_folds, num_epochs_cv, lam0, epsilon, 
                         B_path, max_iters, lr, batch_size, optimizer, device,
                         network_type, poly_order, num_polys, drop_linear):
    """Obtain regularization path using the specified method."""
    if reg_path == 'cv':
        if lambdas_cv is None:
            raise ValueError("Must pass a grid 'lambdas_cv' for CV.")
        return run_sparsemodesnet_cv(
            X_np=X_np, s=s, hidden_units=hidden_units, M=M,
            nonzero_thresh=nonzero_thresh, lambdas=lambdas_cv,
            lr=lr, num_epochs_cv=num_epochs_cv, k_folds=k_folds,
            batch_size=batch_size, optimizer=optimizer, device=device,
            network_type=network_type, poly_order=poly_order, 
            num_polys=num_polys, drop_linear=drop_linear
        )
    else:
        if reg_path != 'dense2sparse':
            warnings.warn(f"Method {reg_path!r} does not exist. "
                          f"Using 'dense2sparse' method instead.")
        return run_sparsemodesnet_d2s(
            X_np=X_np, s=s, hidden_units=hidden_units, M=M,
            nonzero_thresh=nonzero_thresh, lam0=lam0, epsilon=epsilon,
            lr=lr, B=B_path, max_iters=max_iters, batch_size=batch_size,
            optimizer=optimizer, device=device, 
            network_type=network_type, poly_order=poly_order,
            num_polys=num_polys, drop_linear=drop_linear
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
                      B_path, lr, optimizer, device, batch_size,
                      network_type, poly_order, num_polys, drop_linear):
    """Train final model with selected lambda."""
    print(f"\n→ Final training on full data with λ = {lam_star:.3e} ...")
    
    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=batch_size, shuffle=True, drop_last=False
    )
    
    model_final = SparseModesNet(
        pod_basis=U_s_tensor, input_dim=s, hidden_units=hidden_units,
        M=M, lam=float(lam_star), network_type=network_type,
        poly_order=poly_order, num_polys=num_polys, drop_linear=drop_linear
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