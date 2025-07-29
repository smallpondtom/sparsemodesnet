import numpy as np
import torch
from torch.utils.data import DataLoader

from sparsemodesnet.config import SparseModesNetConfig
from sparsemodesnet.linalg.pod import compute_pod_basis
from sparsemodesnet.preprocess import preprocess
from sparsemodesnet.models.model import StateDecoder
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
        _X_np = preprocess(X_np, config)

        # Compute POD basis and coefficients
        U_np, _, _ = compute_pod_basis(_X_np, s=config.s)
        Z_np = U_np.T.dot(_X_np)
        U_tensor = torch.from_numpy(
            U_np.astype(np.float32)).to(config.training.device)
        
        # Mode selection
        if config.sparsity.skip_sparse:
            print("Skipping mode selection.")

            if config.training.I_nn is None:
                print(
                    "Training decoder using leading "
                    f"{config.sparsity.max_num_modes} modes."
                )
                I_nn = list(range(config.sparsity.max_num_modes)) 
            else:
                print(
                    "Selected modes are given. Hence, training decoder using "
                    f"the modes {config.training.I_nn}."
                )
                I_nn = config.training.I_nn

            omegas, path_history = None
        else:
            print(f"\n=== SparseModesNet (Mode Selection) on "
                  f"{config.experiment.label}: d={_X_np.shape[0]}," 
                  f" n={X_np.shape[1]}, s={config.s} ===")
            
            # Run dense-to-sparse mode selection
            I_nn, omegas, path_history = dense2sparse(_X_np, Z_np, U_np, config)
             
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
        X_lift = config.preprocessing.lift(X_np)
        Z_lift = U_np.T.dot(X_lift)
        U_tensor = torch.from_numpy(
            U_np.astype(np.float32)).to(config.training.device)
        
        print(f"\n→ Training decoder model with {r} selected modes ...") 
        decoder = StateDecoder(
            pod_basis      = U_tensor, 
            input_dim      = r, 
            hidden_units   = config.network.hidden_units,
            M              = config.sparsity.M, 
            network_type   = config.network.network_type, 
            poly_order     = config.network.poly_order,
            num_polys      = config.network.num_polys, 
            drop_linear    = config.network.drop_linear,
            drop_constant  = config.network.drop_constant,
            normalize      = config.network.normalize
        )
        
        dataset_full = PODReconDataset(Z_np=Z_lift, X_np=X_lift)
        dataloader_full = DataLoader(
            dataset_full, batch_size=config.training.batch_size, 
            shuffle=True, drop_last=False
        )
        
        train_statedecoder(
            decoder, dataloader_full, config.training.final_epochs, 
            config.training.lr, config.training.optimizer, 
            config.training.device
        )
        
        # Evaluate final model
        Z_tensor = torch.from_numpy(
            Z_np.T.astype(np.float32)).to(config.training.device)
        decoder.eval()
        with torch.no_grad():
            X_hat_tensor = decoder(Z_tensor)
            X_hat_np = X_hat_tensor.cpu().numpy().T 
        X_unlift = config.preprocessing.unlift(X_hat_np) 
        error = np.linalg.norm(X_np - X_unlift, 'fro')
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
            print(f"Relative error using only selected modes: "
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
