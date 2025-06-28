import numpy as np
import warnings
import torch
from torch.utils.data import DataLoader

from .config import SparseModesNetConfig
from .pod import compute_pod_basis
from .model import StateDecoder
from .dataset import PODReconDataset
from .train import train_statedecoder
from .dense2sparse import run_sparsemodesnet_d2s
from .cv import run_sparsemodesnet_cv
from .stability import run_sparsemodesnet_ss
from .knee import _find_best_features
from .logging import _setup_experiment_logging, _log_experiment_info

def run_sparsemodesnet(X_np: np.ndarray, 
                       config: SparseModesNetConfig) -> tuple:
    """
    Main driver function using configuration object.
    
    Parameters
    ----------
    X_np : np.ndarray, shape (d, n)
        Data matrix
    config : SparseModesNetConfig
        Complete configuration object
        
    Returns
    -------
    model : SparseModesNet or StateDecoder
        Trained model
    I_NN : np.ndarray or None
        Selected mode indices
    history : dict
        Training/selection history
    """
    # Setup logging
    if config.experiment.enable_logging:
        logger = _setup_experiment_logging(
            experiment_name=config.experiment.label.lower().replace(" ", "_"),
            logs_dir=config.experiment.logs_dir
        )
        _log_experiment_info(logger, X_np, config)
    
    try:
        # Compute POD basis and coefficients
        U_s_np, _, _ = compute_pod_basis(X_np, s=config.s)
        Z_np = U_s_np.T.dot(X_np)
        U_s_tensor = torch.from_numpy(
            U_s_np.astype(np.float32)).to(config.training.device)
        
        # Mode selection
        if config.selection.mode_selection is not None:
            print(f"\n=== SparseModesNet (Mode Selection) on "
                  f"{config.experiment.label}: d={X_np.shape[0]}," 
                  f" n={X_np.shape[1]}, s={config.s} ===")
        else:
            print("Skipping mode selection.")
        
        if config.selection.mode_selection == 'dense2sparse':
            path_history = run_sparsemodesnet_d2s(X_np, config)
            # Find best features using L-curve
            I_NN = _find_best_features(
                path_history, 
                config.selection.knee_method, 
                config.selection.r_max
            )
            selection_history = {
                'path_history': path_history,
            }
            
        elif config.selection.mode_selection == 'ss':
            I_NN, pi_max, freqs = run_sparsemodesnet_ss(X_np, config)
            selection_history = {
                'pi_max': pi_max, 'freqs': freqs
            }
            
        elif config.selection.mode_selection == 'cv':
            I_NN, selection_history = run_sparsemodesnet_cv(X_np, config)
            
        else:
            warnings.warn(
                "'dense2sparse', 'ss', or 'cv' was not selected for mode "
                "selection. Hence, training decoder using leading "
                f"{config.selection.r_max} modes."
            )
            if config.training.I_NN is None:
                I_NN = (list(range(config.selection.r_max)) 
                        if config.selection.r_max is not None 
                        else list(range(config.s//2)))
            else:
                I_NN = config.training.I_NN
            selection_history = None
             
        # Convert I_NN to list for manipulation if it's a numpy array
        I_NN = list(I_NN) if isinstance(I_NN, np.ndarray) else I_NN
        
        # Adjust and fix the selected modes
        r = len(I_NN)
        if r == 0:
            raise ValueError(
                "No modes selected. Please check your input data and parameters."
            )
            
        if config.selection.r_max is not None:
            if r > config.selection.r_max:
                warnings.warn(f"Selected {r} modes, but " 
                              f"r_max={config.selection.r_max} is set. "
                              f"Truncating to r_max.")
                I_NN = I_NN[:config.selection.r_max]
                r = config.selection.r_max 
                
            elif r < config.selection.r_max:
                warnings.warn(f"Selected {r} modes, but " 
                              f"r_max={config.selection.r_max} is set. "
                              f"Will select {config.selection.r_max-r} " 
                              f"additional leading modes.")
                
                # Use numpy operations for efficiency
                all_modes = np.arange(config.s)
                missing_modes = all_modes[~np.isin(all_modes, I_NN)]
                additional_needed = config.selection.r_max - len(I_NN)
                additional_modes = missing_modes[:additional_needed]
                I_NN.extend(additional_modes.tolist())
                I_NN.sort()
                r = config.selection.r_max
                
        # Convert back to numpy array
        I_NN = np.array(I_NN, dtype=int)
        
        # Train decoder with selected modes
        U_r_np = U_s_np[:, I_NN]
        Z_r_np = U_r_np.T.dot(X_np)
        U_r_tensor = torch.from_numpy(
            U_r_np.astype(np.float32)).to(config.training.device)
        
        print(f"\n→ Training decoder model with {r} selected modes ...") 
        decoder = StateDecoder(
            pod_basis      = U_r_tensor, 
            input_dim      = r, 
            hidden_units   = config.network.hidden_units,
            M              = config.sparsity.M, 
            network_type   = config.network.network_type, 
            poly_order     = config.network.poly_order,
            num_polys      = config.network.num_polys, 
            drop_linear    = config.network.drop_linear,
            drop_constant  = config.network.drop_constant
        )
        
        dataset_full = PODReconDataset(Z_np=Z_r_np, X_np=X_np)
        dataloader_full = DataLoader(
            dataset_full, batch_size=config.training.batch_size, 
            shuffle=True, drop_last=False
        )
        
        train_statedecoder(
            decoder, dataloader_full, config.training.num_epochs, 
            config.training.lr, config.training.optimizer, 
            config.training.device
        )
        
        # Evaluate final model
        Z_r_tensor = torch.from_numpy(
            Z_r_np.T.astype(np.float32)).to(config.training.device)
        decoder.eval()
        
        with torch.no_grad():
            X_hat_tensor = decoder(Z_r_tensor)
            X_hat_np = X_hat_tensor.cpu().numpy().T 
            
        frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
        rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
        mse_per_sample = frob_error / X_np.shape[1]
        
        print(f"Selected {len(I_NN)} modes out of {config.s}")
        print(f"Selected modes indices: {I_NN.tolist()}")
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
        
        if config.experiment.enable_logging:
            logger.info("="*50)
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Selected {len(I_NN)} modes out of {config.s}")
            logger.info(f"Selected modes indices: {I_NN.tolist()}")
            logger.info(f"Final relative error: {rel_frob_error:.6e}")
            
        return decoder, I_NN, selection_history
        
    finally:
        # Note: logging cleanup would happen here if needed
        pass

# def run_sparsemodesnet(
#     X_np: np.ndarray,
#     s: int,
#     hidden_units: list,
#     M: float = 1.0,
#     lr: float = 1e-3,
#     batch_size: int = 16,
#     mode_selection: str | None = 'dense2sparse',
#     knee_method: str = 'dfdt', 
#     optimizer: str = 'Adam',
#     nonzero_thresh: float = 1e-6,
#     num_epochs: int = 100,
#     final_epochs: int = 100,
#     r_max: int = None, 
#     network_type: str = 'FF',
#     I_NN: list | np.ndarray = None,
#     # for Π-net:
#     poly_order: int = 2,
#     num_polys: int = 1,
#     drop_linear: bool = False,
#     drop_constant: bool = False,
#     # for "dense-to-sparse":
#     lam0: float = 1e-6,
#     epsilon: float = 0.1,
#     max_iters: int = 100,
#     # for "stability selection"
#     num_subsamples: int = 5,
#     pi_thresh: float = 0.9,
#     lambdas: np.ndarray = None,
#     # for "cv":
#     k_folds: int = 5,
#     # other common:
#     device: str = 'cpu',
#     label: str = '',
#     enable_logging: bool = True,
#     logs_dir: str = None
# ):
#     """
#     Runs SparseModesNet with lambda selection via path or CV.
    
#     Returns
#     -------
#     tuple: (final_model, selection_history)
#     """
#     # Setup logging if enabled
#     if enable_logging:
#         experiment_name = label.lower().replace(" ", "_") if label else "sparsemodesnet"
#         logger, log_file = _setup_experiment_logging(experiment_name, logs_dir)
        
#         # Log experiment parameters
#         logger.info(f"Experiment: {label}")
#         logger.info(f"Data shape: {X_np.shape}")
#         logger.info(f"POD dimension: {s}")
#         logger.info(f"Hidden units: {hidden_units}")
#         logger.info(f"Mode Selection Method: {mode_selection}")
#         logger.info(f"Device: {device}")
#         logger.info(f"Training parameters: M={M}, lr={lr}, batch_size={batch_size}")
#         logger.info(f"Optimizer: {optimizer}, knee_method: {knee_method}")
#         if network_type == 'FF':
#             logger.info("NN type: Feedforward")
#         else:
#             logger.info(f"NN type: {network_type} with polynomial "
#                         f"order of {poly_order}, and {num_polys} number of "
#                         f"polynomial blocks")
#         logger.info("="*50)
        
#         # Create training logger
#         training_logger = logging.getLogger(f"training.{experiment_name}")
#         training_logger.setLevel(logging.INFO)
        
#         # Remove existing handlers
#         for handler in training_logger.handlers[:]:
#             training_logger.removeHandler(handler)
        
#         # Add file handler with simple format
#         training_file_handler = logging.FileHandler(log_file)
#         training_file_handler.setLevel(logging.INFO)
#         simple_formatter = logging.Formatter('%(message)s')
#         training_file_handler.setFormatter(simple_formatter)
#         training_logger.addHandler(training_file_handler)
#         training_logger.propagate = False
        
#         # Replace print function temporarily
#         original_print = print
#         def logged_print(*args, **kwargs):
#             message = ' '.join(str(arg) for arg in args)
#             training_logger.info(message)
#             original_print(*args, **kwargs)
        
#         import builtins
#         builtins.print = logged_print
    
#     try:
#         # Compute POD basis and coefficients
#         U_s_np, _, _ = compute_pod_basis(X_np, s=s)
#         Z_np = U_s_np.T.dot(X_np)
#         U_s_tensor = torch.from_numpy(U_s_np.astype(np.float32)).to(device)
        
#         # Mode selection
#         if mode_selection is not None:
#             print(f"\n=== SparseModesNet (Mode Selection) on "
#                   f"{label}: d={X_np.shape[0]}, n={X_np.shape[1]}, s={s} ===")
#         else:
#             print("Skipping mode selection.")
        
#         if mode_selection == 'dense2sparse':
#             path_history = run_sparsemodesnet_d2s(
#                 X_np=X_np, s=s, hidden_units=hidden_units, M=M,
#                 nonzero_thresh=nonzero_thresh, lam0=lam0, epsilon=epsilon,
#                 lr=lr, B=num_epochs, max_iters=max_iters, 
#                 batch_size=batch_size, optimizer=optimizer, device=device, 
#                 network_type=network_type, poly_order=poly_order,
#                 num_polys=num_polys, drop_linear=drop_linear, 
#                 drop_constant=drop_constant
#             )
#             # Find best features using L-curve
#             I_NN = _find_best_features(path_history, knee_method, r_max)
#             selection_history = {
#                 'path_history': path_history,
#             }
            
#         elif mode_selection == 'ss':
#             # CORRECTED: run_sparsemodesnet_ss returns (I_NN, pi_max, freqs)
#             I_NN, pi_max, freqs = run_sparsemodesnet_ss(
#                 X_np=X_np, s=s, hidden_units=hidden_units, M=M,
#                 nonzero_thresh=nonzero_thresh, lambdas=lambdas,
#                 network_type=network_type, poly_order=poly_order,
#                 num_polys=num_polys, drop_linear=drop_linear,
#                 drop_constant=drop_constant,
#                 B=num_subsamples, pi_thresh=pi_thresh,
#                 lr=lr, num_epochs=num_epochs, batch_size=batch_size,
#                 optimizer=optimizer, device=device, 
#             )
#             selection_history = {
#                 'pi_max': pi_max, 'freqs': freqs
#             }
            
#         elif mode_selection == 'cv':
#             # CORRECTED: run_sparsemodesnet_cv returns (I_NN, selection_history)
#             I_NN, selection_history = run_sparsemodesnet_cv(
#                 X_np=X_np, s=s, hidden_units=hidden_units, M=M,
#                 nonzero_thresh=nonzero_thresh, lambdas=lambdas,
#                 network_type=network_type, poly_order=poly_order,
#                 num_polys=num_polys, drop_linear=drop_linear,
#                 drop_constant=drop_constant,
#                 lr=lr, num_epochs=num_epochs, k_folds=k_folds,
#                 batch_size=batch_size, optimizer=optimizer, device=device,
#             )
            
#         else:
#             warnings.warn("'dense2sparse', 'ss', or 'cv' was not selected for mode "
#                           "selection. Hence, training decoder using leading "
#                           f"{r_max} modes.")
#             if I_NN is None:
#                 I_NN = list(range(r_max)) if r_max is not None else list(range(s//2))
#             selection_history = None
             
#         # Convert I_NN to list for manipulation if it's a numpy array
#         I_NN = list(I_NN) if isinstance(I_NN, np.ndarray) else I_NN
        
#         # Adjust and fix the selected modes
#         r = len(I_NN)
#         if r == 0:
#             raise ValueError(
#                 "No modes selected. Please check your input data and parameters."
#             )
            
#         if r_max is not None:
#             if r > r_max:
#                 warnings.warn(f"Selected {r} modes, but r_max={r_max} is set. "
#                               f"Truncating to r_max.")
#                 I_NN = I_NN[:r_max]
#                 r = r_max 
                
#             elif r < r_max:
#                 warnings.warn(f"Selected {r} modes, but r_max={r_max} is set. "
#                               f"Will select {r_max-r} additional leading modes.")
                
#                 # Use numpy operations for efficiency
#                 all_modes = np.arange(s)
#                 missing_modes = all_modes[~np.isin(all_modes, I_NN)]
#                 additional_needed = r_max - len(I_NN)
#                 additional_modes = missing_modes[:additional_needed]
#                 I_NN.extend(additional_modes.tolist())
#                 I_NN.sort()
#                 r = r_max
                
#         # Convert back to numpy array
#         I_NN = np.array(I_NN, dtype=int)
        
#         # Train decoder with selected modes
#         U_r_np = U_s_np[:, I_NN]
#         Z_r_np = U_r_np.T.dot(X_np)
#         U_r_tensor = torch.from_numpy(U_r_np.astype(np.float32)).to(device)
        
#         print(f"\n→ Training decoder model with {r} selected modes ...") 
#         decoder = StateDecoder(
#             pod_basis=U_r_tensor, 
#             input_dim=r, 
#             hidden_units=hidden_units,
#             M=M, 
#             network_type=network_type, 
#             poly_order=poly_order,
#             num_polys=num_polys, 
#             drop_linear=drop_linear,
#             drop_constant=drop_constant
#         )
        
#         dataset_full = PODReconDataset(Z_np=Z_r_np, X_np=X_np)
#         dataloader_full = DataLoader(
#             dataset_full, batch_size=batch_size, shuffle=True, drop_last=False)
        
#         train_statedecoder(
#             decoder, dataloader_full, final_epochs, lr, optimizer, device)
        
#         # Evaluate final model
#         Z_r_tensor = torch.from_numpy(Z_r_np.T.astype(np.float32)).to(device)
#         decoder.eval()
#         with torch.no_grad():
#             X_hat_tensor = decoder(Z_r_tensor)
#             X_hat_np = X_hat_tensor.cpu().numpy().T 
            
#         frob_error = np.linalg.norm(X_np - X_hat_np, 'fro')
#         rel_frob_error = frob_error / np.linalg.norm(X_np, 'fro')
#         mse_per_sample = frob_error / X_np.shape[1]
        
#         print(f"Final relative reconstruction error: "
#               f"||X - X_hat||_F / ||X||_F = {rel_frob_error:.6e}")
#         print(f"Final MSE per sample: {mse_per_sample:.6e}")
        
#         # Selected modes only reconstruction error
#         if r > 0:
#             frob_error_selected = np.linalg.norm(
#                 X_np - U_r_np @ (U_r_np.T @ X_np), 'fro')
#             X_np_norm = np.linalg.norm(X_np, 'fro')
#             rel_frob_error_selected = frob_error_selected / X_np_norm
#             mse_per_sample_selected = frob_error_selected / X_np.shape[1]
#             print(f"Relative error using only selected modes: "
#                   f"{rel_frob_error_selected:.6e}")
#             print(f"MSE per sample using only selected modes: "
#                   f"{mse_per_sample_selected:.6e}")
        
#         if enable_logging:
#             logger.info("="*50)
#             logger.info(f"Experiment completed successfully!")
#             logger.info(f"Selected {len(I_NN)} modes out of {s}")
#             logger.info(f"Selected modes indices: {I_NN.tolist()}")
#             logger.info(f"Final relative error: {rel_frob_error:.6e}")
            
#         return decoder, I_NN, selection_history
        
#     finally:
#         # Restore original print function
#         if enable_logging:
#             import builtins
#             builtins.print = original_print