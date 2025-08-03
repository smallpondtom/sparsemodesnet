import numpy as np
import torch
from torch.utils.data import DataLoader

from sparsemodesnet.config import SparseModesNetConfig
from sparsemodesnet.linalg.pod import compute_pod_basis
from sparsemodesnet.linalg.lstsq import lstsq_l2
from sparsemodesnet.preprocess import preprocess
from sparsemodesnet.decoder_models.model import StateDecoder
from sparsemodesnet.dataset import PODReconDataset
from sparsemodesnet.training.train import train_statedecoder
from sparsemodesnet.training.dense2sparse import dense2sparse
from .logging import _setup_experiment_logging, _log_experiment_info

def fit(X_np: np.ndarray, config: SparseModesNetConfig) -> tuple:
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
    I_nn : np.ndarray or None
        Selected mode indices
    omegas : np.ndarray
        Mode weights
    path_history : dict
        Training/selection history
    """
    # Setup logging
    if config.experiment.enable_logging:
        logger, _ = _setup_experiment_logging(
            experiment_name=config.experiment.label.lower().replace(" ", "_"),
            logs_dir=config.experiment.logs_dir
        )
        _log_experiment_info(logger, X_np, config)
    
    try:
        # Preprocess data if needed
        X_proc = preprocess(X_np.astype(np.float64), config)

        # Compute POD basis and coefficients
        U_np, _, _ = compute_pod_basis(X_proc, s=config.s)
        
        # Mode selection
        if config.sparsity.skip_sparse:
            print("Skipping mode selection.")

            if config.training.I_nn is None:
                print(
                    "Training decoder using leading "
                    f"{config.r} modes."
                )
                I_nn = list(range(config.r)) 
            else:
                print(
                    "Selected modes are given. Hence, training decoder using "
                    f"the modes {config.training.I_nn}."
                )
                I_nn = config.training.I_nn

            omegas, path_history = None, None
        else:
            print(f"\n=== SparseModesNet (Mode Selection) on "
                  f"{config.experiment.label}: d={X_proc.shape[0]}," 
                  f" n={X_np.shape[1]}, s={config.s} ===")

            # Compute the input (reduced) data 
            Z_np = U_np.T.dot(X_proc)
            U_tensor = torch.from_numpy(U_np).to(
                    config.training.device,
                    dtype=torch.float64 if config.training.device == 'cpu'
                          else torch.float32
                )

            # Run dense-to-sparse mode selection
            I_nn, omegas, path_history = dense2sparse(X_proc, Z_np, 
                                                      U_tensor, config)
             
        # Convert I_nn to np.array 
        I_nn = I_nn if isinstance(I_nn, np.ndarray) else np.array(I_nn)
        
        # Adjust and fix the selected modes
        r = len(I_nn)
        if r == 0:
            raise ValueError(
                "No modes selected. Please check your input data and parameters."
            )

        # Train decoder with selected modes
        U_np = U_np[:, I_nn]
        X_pp = config.preprocessing.forward(X_np)
        Z_pp = U_np.T.dot(X_pp)
        
        # Ensure consistent data types based on device
        target_dtype = np.float64 if config.training.device == 'cpu' else np.float32
        if U_np.dtype != target_dtype:
            U_np = U_np.astype(target_dtype)
        if X_pp.dtype != target_dtype:
            X_pp = X_pp.astype(target_dtype)
        if Z_pp.dtype != target_dtype:
            Z_pp = Z_pp.astype(target_dtype)
        
        U_tensor = torch.from_numpy(U_np).to(
                config.training.device,
                dtype=torch.float64 if config.training.device == 'cpu'
                      else torch.float32
            )

        if config.network.network_type == 'QM' and config.training.analytical:
            residual = X_pp - U_np @ Z_pp
            Z_quad_pp = _quadratic_mapping_numpy(Z_pp.T).T 
            W_nn_T, _ = lstsq_l2(Z_quad_pp.T, residual.T, 
                                 reg_magnitude=config.training.reg_param)
            W_nn = W_nn_T.T
            X_hat_np = U_np @ Z_pp + W_nn @ Z_quad_pp
            decoder = lambda z: U_np @ z + W_nn @ _quadratic_mapping_numpy(z.T).T
        elif config.network.network_type == 'CM' and config.training.analytical:
            residual = X_pp - U_np @ Z_pp
            Z_cubic_pp = _cubic_mapping_numpy(Z_pp.T).T
            W_nn_T, _ = lstsq_l2(Z_cubic_pp.T, residual.T,
                                 reg_magnitude=config.training.reg_param)
            W_nn = W_nn_T.T
            X_hat_np = U_np @ Z_pp + W_nn @ Z_cubic_pp
            decoder = lambda z: U_np @ z + W_nn @ _cubic_mapping_numpy(z.T).T
        else:
            print(f"\n→ Training decoder model with {r} selected modes ...") 
            decoder = StateDecoder(
                pod_basis      = U_tensor, 
                input_dim      = r, 
                hidden_units   = config.network.hidden_units,
                gamma          = config.training.gamma, 
                weight_scale   = config.training.weight_scale,
                network_type   = config.network.network_type, 
                poly_order     = config.network.poly_order,
                num_polys      = config.network.num_polys, 
                drop_linear    = config.network.drop_linear,
                drop_constant  = config.network.drop_constant,
                normalize      = config.network.normalize_layer,
                dtype          = torch.float64 if config.training.device == 'cpu' 
                                 else torch.float32,
            )
            
            dataset_full = PODReconDataset(Z_np=Z_pp, X_np=X_pp, 
                                           type='float64' 
                                                if config.training.device == 'cpu' 
                                                else 'float32')
            dataloader_full = DataLoader(
                dataset_full, batch_size=config.training.decoder_batch_size, 
                shuffle=False, drop_last=False
            )
            
            train_statedecoder(
                model       = decoder, 
                dataloader  = dataloader_full, 
                num_epochs  = config.training.decoder_epochs, 
                lr          = config.training.decoder_lr, 
                lr_patience = config.training.decoder_lr_patience,
                lr_factor   = config.training.decoder_lr_factor, 
                momentum    = config.training.decoder_momentum, 
                optimizer   = config.training.decoder_optimizer, 
                device      = config.training.device,
            )
            
            # Evaluate final model
            Z_pp_tensor = torch.from_numpy(Z_pp.T).to(
                config.training.device,
                dtype=torch.float64 if config.training.device == 'cpu' 
                      else torch.float32
            )
            decoder.eval()
            with torch.no_grad():
                X_hat_tensor = decoder(Z_pp_tensor)
                X_hat_np = X_hat_tensor.cpu().numpy().T 

        X_eval = config.preprocessing.backward(X_hat_np) 
        error = np.linalg.norm(X_np - X_eval, 'fro')
        X_np_norm = np.linalg.norm(X_np, 'fro')
        rel_error = error / X_np_norm
        
        print(f"Selected {len(I_nn)} modes out of {config.s}")
        print(f"Selected modes indices: {I_nn.tolist()}")
        print(f"Final relative reconstruction error: "
              f"||X - X_hat||_F / ||X||_F = {rel_error:.6e}")
        
        # Linear reconstruction error for selected modes
        if r > 0:
            error_linear = np.linalg.norm(
                X_np - U_np @ (U_np.T @ X_np), 'fro')
            rel_error_linear = error_linear / X_np_norm
            print(f"Relative linear reconstruction error using "
                  f"only selected modes: "
                  f"{rel_error_linear:.6e}")
        
        if config.experiment.enable_logging:
            logger.info("="*50)
            logger.info(f"Experiment completed successfully!")
            logger.info(f"Selected {len(I_nn)} modes out of {config.s}")
            logger.info(f"Selected modes indices: {I_nn.tolist()}")
            logger.info(f"Final relative error: {rel_error:.6e}")
            
        return decoder, I_nn, omegas, path_history
        
    finally:
        # Note: logging cleanup would happen here if needed
        pass


def _quadratic_mapping_numpy(x):
    """
    Numpy version - must match the torch version exactly!
    """
    if x.ndim == 1:
        n = x.shape[0]
        i_indices, j_indices = np.tril_indices(n)
        result = x[i_indices] * x[j_indices]
        return result
    else:
        _, n = x.shape
        i_indices, j_indices = np.tril_indices(n)
        result = x[:, i_indices] * x[:, j_indices]
        return result
    

def _cubic_mapping_numpy(x):
    """
    Fast vectorized computation of unique cubic terms x ⊗ x ⊗ x (NumPy version).
    Uses meshgrid for efficient index generation.
    
    Args:
        x: np.ndarray of shape (batch_size, n) or (n,)
        
    Returns:
        np.ndarray of shape (batch_size, n*(n+1)*(n+2)//6) or (n*(n+1)*(n+2)//6,)
    """
    if x.ndim == 1:
        n = x.shape[0]
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products
        result = x[i_indices] * x[j_indices] * x[k_indices]
        return result
    else:
        batch_size, n = x.shape
        # Create meshgrid for all combinations
        i_range = np.arange(n)
        i_grid, j_grid, k_grid = np.meshgrid(i_range, i_range, i_range, indexing='ij')
        
        # Keep only upper triangular combinations (i ≤ j ≤ k)
        mask = (i_grid <= j_grid) & (j_grid <= k_grid)
        i_indices = i_grid[mask]
        j_indices = j_grid[mask]
        k_indices = k_grid[mask]
        
        # Compute cubic products for all batches
        result = x[:, i_indices] * x[:, j_indices] * x[:, k_indices]
        return result