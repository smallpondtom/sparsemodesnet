import numpy as np
import kneeliverse.dfdt as dfdt
import kneeliverse.zmethod as zmethod
import warnings
import torch
import logging
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader

from .pod import compute_pod_basis
from .model import SparseModesNet, StateDecoder
from .dataset import PODReconDataset
from .train import train_sparsemodesnet, train_statedecoder
from .cv import run_sparsemodesnet_cv
from .stability import run_sparsemodesnet_ss


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def _setup_experiment_logging(experiment_name="sparsemodesnet", logs_dir=None):
    """Setup logging with timestamp for experiment tracking."""
    if logs_dir is None:
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
    
    # Create file handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    
    # Detailed formatter
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(detailed_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Initial log entries
    logger.info(f"Starting {experiment_name} experiment")
    logger.info(f"Log file: {log_filename}")
    
    return logger, log_filename


# =============================================================================
# DENSE-TO-SPARSE PATH
# =============================================================================

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
    """Dense-to-sparse regularization path with warm-start λ→(1+ε)λ routine."""
    print("\n=== Dense-To-Sparse (default) λ-Path ===")
    
    U_s_np, _, _ = compute_pod_basis(X_np, s=s)
    Z_np = U_s_np.T.dot(X_np)

    U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
    Z_tensor = torch.from_numpy(Z_np.T.astype(np.float32)).to(device)

    dataset_full = PODReconDataset(Z_np=Z_np, X_np=X_np)
    dataloader_full = DataLoader(
        dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)

    lam = lam0
    prev_nonzero = s
    path_history = []
    iter_count = 0

    while True:
        iter_count += 1
        print(f"\n-- Path iteration {iter_count}, λ = {lam:.3e} "
              f"(r(λ) prev = {prev_nonzero})")
        
        model = SparseModesNet(
            pod_basis=U_s_tensor,
            input_dim=s,
            hidden_units=hidden_units,
            M=M,
            lam=lam,
            network_type=network_type,
            poly_order=poly_order,
            num_polys=num_polys,
            drop_linear=drop_linear
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

        print(f"  → at λ={lam:.3e}: r(λ)={curr_nonzero}, "
              f"rel_err={rel_frob_error:.6e}")
        print(f"  → selected modes: {nonzero_idxs.tolist()}")

        if curr_nonzero == 0:
            print("All skip-weights have zeroed out. Stopping path.\n")
            break

        lam = lam * (1.0 + epsilon)
        prev_nonzero = curr_nonzero

        if iter_count >= max_iters:
            print(f"Reached max_iters={max_iters} on λ-path; stopping early.\n")
            break

    return path_history


# =============================================================================
# KNEE DETECTION AND FEATURE SELECTION
# =============================================================================

def _normalize_array(arr):
    """Normalize array to [0, 1] range."""
    if np.all(arr == arr[0]):  # All elements are the same
        return np.zeros_like(arr)
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
    
    # Ensure knee_idx is a numpy array
    knee_idx = np.array(knee_idx, dtype=int)
    return knee_idx


def _fallback_feature_selection(path_history, r_max):
    """Fallback feature selection when knee detection fails."""
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
    I_NN = entry['selected_idxs']
    print(f"[Fallback] λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
    print(f"Selected modes: {I_NN.tolist()}")
    return I_NN


def _select_from_knees(path_history, knee_idx, r_max):
    """Select lambda from detected knee points."""
    rs = np.array([h['nonzero_count'] for h in path_history])
    r_stars = rs[knee_idx]
    
    # Pick first knee point that satisfies r_max constraint
    if r_max is not None:
        valid_mask = r_stars <= r_max
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            i_star = valid_indices[0]
        else:
            # No knee satisfies r_max constraint, find closest to r_max
            distances = np.abs(r_stars - r_max)
            i_star = np.argmin(distances)
    else:
        i_star = 0
        
    # Make sure knee_idx[i_star] is a valid index
    selected_knee_idx = knee_idx[i_star]
    I_NN = path_history[selected_knee_idx]['selected_idxs']
    if len(I_NN) > r_max:
        I_NN = I_NN[:r_max]  # Truncate if exceeds r_max 
    
    # Print selection info
    selected_entry = path_history[selected_knee_idx]
    print(f"Selected knee at index {selected_knee_idx}: "
          f"λ={selected_entry['lambda']:.3e}, "
          f"r={selected_entry['nonzero_count']}, "
          f"err={selected_entry['error']:.6e}")
    print(f"Selected modes: {I_NN.tolist()}")
    
    return I_NN


def _find_best_features(path_history, knee_method, r_max):
    """Find optimal features using knee detection or fallback methods."""
    # Prepare data for knee detection
    err = np.array([h['error'] for h in path_history])
    logerr = np.log(err + 1e-16)  # Add small epsilon to avoid log(0)
    lasso = np.array([h['l1_b'] for h in path_history])
    loglasso = np.log(lasso + 1e-16)  # Add small epsilon to avoid log(0)
    
    # Normalize to [0, 1] range
    logerr_norm = _normalize_array(logerr)
    loglasso_norm = _normalize_array(loglasso)
    data = np.column_stack((loglasso_norm, logerr_norm))
    
    # Apply knee detection
    knee_idx = _detect_knees(data, knee_method)
    
    if len(knee_idx) == 0:
        return _fallback_feature_selection(path_history, r_max)
    else:
        return _select_from_knees(path_history, knee_idx, r_max)


# =============================================================================
# MAIN DRIVER FUNCTION
# =============================================================================

def run_sparsemodesnet(
    X_np: np.ndarray,
    s: int,
    hidden_units: list,
    M: float = 1.0,
    lr: float = 1e-3,
    batch_size: int = 16,
    mode_selection: str | None = 'dense2sparse',
    knee_method: str = 'dfdt', 
    optimizer: str = 'Adam',
    nonzero_thresh: float = 1e-6,
    num_epochs: int = 100,
    final_epochs: int = 100,
    r_max: int = None, 
    network_type: str = 'FF',
    # for Π-net:
    poly_order: int = 2,
    num_polys: int = 1,
    drop_linear: bool = False,
    # for "dense-to-sparse":
    lam0: float = 1e-6,
    epsilon: float = 0.1,
    max_iters: int = 100,
    # for "stability selection"
    num_subsamples: int = 5,
    pi_thresh: float = 0.9,
    lambdas: np.ndarray = None,
    # for "cv":
    k_folds: int = 5,
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
    tuple: (final_model, selection_history)
    """
    # Setup logging if enabled
    if enable_logging:
        experiment_name = label.lower().replace(" ", "_") if label else "sparsemodesnet"
        logger, log_file = _setup_experiment_logging(experiment_name, logs_dir)
        
        # Log experiment parameters
        logger.info(f"Experiment: {label}")
        logger.info(f"Data shape: {X_np.shape}")
        logger.info(f"POD dimension: {s}")
        logger.info(f"Hidden units: {hidden_units}")
        logger.info(f"Mode Selection Method: {mode_selection}")
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
        
        # Create training logger
        training_logger = logging.getLogger(f"training.{experiment_name}")
        training_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in training_logger.handlers[:]:
            training_logger.removeHandler(handler)
        
        # Add file handler with simple format
        training_file_handler = logging.FileHandler(log_file)
        training_file_handler.setLevel(logging.INFO)
        simple_formatter = logging.Formatter('%(message)s')
        training_file_handler.setFormatter(simple_formatter)
        training_logger.addHandler(training_file_handler)
        training_logger.propagate = False
        
        # Replace print function temporarily
        original_print = print
        def logged_print(*args, **kwargs):
            message = ' '.join(str(arg) for arg in args)
            training_logger.info(message)
            original_print(*args, **kwargs)
        
        import builtins
        builtins.print = logged_print
    
    try:
        # Compute POD basis and coefficients
        U_s_np, _, _ = compute_pod_basis(X_np, s=s)
        Z_np = U_s_np.T.dot(X_np)
        U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
        
        # Mode selection
        if mode_selection is not None:
            print(f"\n=== SparseModesNet (Mode Selection) on "
                  f"{label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")
        else:
            print("Skipping mode selection.")
        
        if mode_selection == 'dense2sparse':
            path_history = run_sparsemodesnet_d2s(
                X_np=X_np, s=s, hidden_units=hidden_units, M=M,
                nonzero_thresh=nonzero_thresh, lam0=lam0, epsilon=epsilon,
                lr=lr, B=num_epochs, max_iters=max_iters, 
                batch_size=batch_size, optimizer=optimizer, device=device, 
                network_type=network_type, poly_order=poly_order,
                num_polys=num_polys, drop_linear=drop_linear
            )
            # Find best features using L-curve
            I_NN = _find_best_features(path_history, knee_method, r_max)
            selection_history = {
                'path_history': path_history,
            }
            
        elif mode_selection == 'ss':
            # CORRECTED: run_sparsemodesnet_ss returns (I_NN, pi_max, freqs)
            I_NN, pi_max, freqs = run_sparsemodesnet_ss(
                X_np=X_np, s=s, hidden_units=hidden_units, M=M,
                nonzero_thresh=nonzero_thresh, lambdas=lambdas,
                network_type=network_type, poly_order=poly_order,
                num_polys=num_polys, drop_linear=drop_linear,
                B=num_subsamples, pi_thresh=pi_thresh,
                lr=lr, num_epochs=num_epochs, batch_size=batch_size,
                optimizer=optimizer, device=device 
            )
            selection_history = {
                'pi_max': pi_max, 'freqs': freqs
            }
            
        elif mode_selection == 'cv':
            # CORRECTED: run_sparsemodesnet_cv returns (I_NN, selection_history)
            I_NN, selection_history = run_sparsemodesnet_cv(
                X_np=X_np, s=s, hidden_units=hidden_units, M=M,
                nonzero_thresh=nonzero_thresh, lambdas=lambdas,
                network_type=network_type, poly_order=poly_order,
                num_polys=num_polys, drop_linear=drop_linear,
                lr=lr, num_epochs=num_epochs, k_folds=k_folds,
                batch_size=batch_size, optimizer=optimizer, device=device 
            )
            
        else:
            warnings.warn("'dense2sparse', 'ss', or 'cv' was not selected for mode "
                          "selection. Hence, training decoder using leading "
                          f"{r_max} modes.")
            I_NN = list(range(r_max)) if r_max is not None else list(range(s//2))
            selection_history = None
             
        # Convert I_NN to list for manipulation if it's a numpy array
        I_NN = list(I_NN) if isinstance(I_NN, np.ndarray) else I_NN
        
        # Adjust and fix the selected modes
        r = len(I_NN)
        if r == 0:
            raise ValueError(
                "No modes selected. Please check your input data and parameters."
            )
            
        if r_max is not None:
            if r > r_max:
                warnings.warn(f"Selected {r} modes, but r_max={r_max} is set. "
                              f"Truncating to r_max.")
                I_NN = I_NN[:r_max]
                r = r_max 
                
            elif r < r_max:
                warnings.warn(f"Selected {r} modes, but r_max={r_max} is set. "
                              f"Will select {r_max-r} additional leading modes.")
                
                # Use numpy operations for efficiency
                all_modes = np.arange(s)
                missing_modes = all_modes[~np.isin(all_modes, I_NN)]
                additional_needed = r_max - len(I_NN)
                additional_modes = missing_modes[:additional_needed]
                I_NN.extend(additional_modes.tolist())
                I_NN.sort()
                r = r_max
                
        # Convert back to numpy array
        I_NN = np.array(I_NN, dtype=int)
        
        # Train decoder with selected modes
        U_r_np = U_s_np[:, I_NN]
        Z_r_np = U_r_np.T.dot(X_np)
        U_r_tensor = torch.from_numpy(U_r_np.astype(np.float32)).to(device)
        
        print(f"\n→ Training decoder model with {r} selected modes ...") 
        decoder = StateDecoder(
            pod_basis=U_r_tensor, 
            input_dim=r, 
            hidden_units=hidden_units,
            M=M, 
            network_type=network_type, 
            poly_order=poly_order,
            num_polys=num_polys, 
            drop_linear=drop_linear
        )
        
        dataset_full = PODReconDataset(Z_np=Z_r_np, X_np=X_np)
        dataloader_full = DataLoader(
            dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)
        
        train_statedecoder(
            decoder, dataloader_full, final_epochs, lr, optimizer, device)
        
        # Evaluate final model
        Z_r_tensor = torch.from_numpy(Z_r_np.T.astype(np.float32)).to(device)
        decoder.eval()
        with torch.no_grad():
            X_hat_tensor = decoder(Z_r_tensor)
            X_hat_np = X_hat_tensor.cpu().numpy().T 
            
        frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
        rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
        mse_per_sample = frob_error / X_np.shape[1]
        
        print(f"Final relative reconstruction error: "
              f"||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
        print(f"Final MSE per sample: {mse_per_sample:.6e}")
        
        # Selected modes only reconstruction error
        if r > 0:
            frob_error_selected = np.linalg.norm(
                X_np - U_r_np @ (U_r_np.T @ X_np), 'fro')
            X_np_norm = np.linalg.norm(X_np, 'fro')
            rel_frob_error_selected = frob_error_selected / X_np_norm
            mse_per_sample_selected = frob_error_selected / X_np.shape[1]
            print(f"Relative error using only selected modes: "
                  f"{rel_frob_error_selected:.6e}")
            print(f"MSE per sample using only selected modes: "
                  f"{mse_per_sample_selected:.6e}")
        
        if enable_logging:
            logger.info("="*50)
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Selected {len(I_NN)} modes out of {s}")
            logger.info(f"Selected modes indices: {I_NN.tolist()}")
            logger.info(f"Final relative error: {rel_frob_error:.6e}")
            
        return decoder, I_NN, selection_history
        
    finally:
        # Restore original print function
        if enable_logging:
            import builtins
            builtins.print = original_print