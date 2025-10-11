from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from .config import SparseModesNetConfig

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
    
    # Create training logger
    training_logger = logging.getLogger(f"training.{experiment_name}")
    training_logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in training_logger.handlers[:]:
        training_logger.removeHandler(handler)
    
    # Add file handler with simple format
    training_file_handler = logging.FileHandler(log_filename)
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
    
    return logger, log_filename

def _log_experiment_info(logger: logging.Logger, 
                         X_np: np.ndarray, config: SparseModesNetConfig):
    """
    Log experiment configuration and data information.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    X_np : np.ndarray
        Input data matrix
    config : SparseModesNetConfig
        Configuration object
    """
    logger.info("="*50)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("="*50)
    
    # Data info
    logger.info(f"Data shape: {X_np.shape}")
    logger.info(f"Data type: {X_np.dtype}")
    logger.info(f"Data range: [{X_np.min():.6f}, {X_np.max():.6f}]")
    logger.info(f"Data mean: {X_np.mean():.6f}")
    logger.info(f"Data std: {X_np.std():.6f}")
    
    # Core parameters
    logger.info("-" * 30)
    logger.info("CORE PARAMETERS")
    logger.info("-" * 30)
    logger.info(f"s (total POD modes): {config.s}")
    logger.info(f"r (target active modes): {config.r}")
    logger.info(f"p (latent dimension): {config.p}")
    logger.info(f"I_nn provided: {config.training.I_nn is not None}")
    if config.training.I_nn is not None:
        logger.info(f"I_nn length: {len(config.training.I_nn)}")
        logger.info(f"I_nn indices: {config.training.I_nn}")
    
    # Network configuration
    logger.info("-" * 30)
    logger.info("NETWORK CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Network type: {config.network.network_type}")
    logger.info(f"Hidden units: {config.network.hidden_units}")
    logger.info(f"Polynomial order: {config.network.poly_order}")
    logger.info(f"Number of polynomials: {config.network.num_polys}")
    logger.info(f"Drop linear: {config.network.drop_linear}")
    logger.info(f"Drop constant: {config.network.drop_constant}")
    logger.info(f"Normalize layer: {config.network.normalize_layer}")
    
    # Training configuration - LASSO phase
    logger.info("-" * 30)
    logger.info("TRAINING CONFIGURATION - LASSO PHASE")
    logger.info("-" * 30)
    logger.info(f"LASSO learning rate: {config.training.lasso_lr}")
    logger.info(f"LASSO LR patience: {config.training.lasso_lr_patience}")
    logger.info(f"LASSO LR factor: {config.training.lasso_lr_factor}")
    logger.info(f"LASSO epochs: {config.training.lasso_epochs}")
    logger.info(f"LASSO batch size: {config.training.lasso_batch_size}")
    logger.info(f"LASSO optimizer: {config.training.lasso_optimizer}")
    logger.info(f"LASSO momentum: {config.training.lasso_momentum}")
    logger.info(f"LASSO bias: {config.training.lasso_bias}")
    logger.info(f"Gamma (weight decay): {config.training.gamma}")
    logger.info(f"Max no change: {config.training.max_no_change}")
    logger.info(f"Extra modes: {config.training.extra_modes}")
    logger.info(f"L1 only: {config.training.l1_only}")
    logger.info(f"Full z: {config.training.full_z}")
    
    # Training configuration - Decoder phase
    logger.info("-" * 30)
    logger.info("TRAINING CONFIGURATION - DECODER PHASE")
    logger.info("-" * 30)
    logger.info(f"Decoder learning rate: {config.training.decoder_lr}")
    logger.info(f"Decoder LR patience: {config.training.decoder_lr_patience}")
    logger.info(f"Decoder LR factor: {config.training.decoder_lr_factor}")
    logger.info(f"Decoder epochs: {config.training.decoder_epochs}")
    logger.info(f"Decoder batch size: {config.training.decoder_batch_size}")
    logger.info(f"Decoder optimizer: {config.training.decoder_optimizer}")
    logger.info(f"Decoder momentum: {config.training.decoder_momentum}")
    logger.info(f"Decoder bias: {config.training.decoder_bias}")
    
    # General training parameters
    logger.info("-" * 30)
    logger.info("GENERAL TRAINING PARAMETERS")
    logger.info("-" * 30)
    logger.info(f"Device: {config.training.device}")
    logger.info(f"Regularization parameter: {config.training.reg_param}")
    logger.info(f"Weight scale: {config.training.weight_scale}")
    logger.info(f"Analytical: {config.training.analytical}")

    # Preprocessing configuration
    logger.info("-" * 30)
    logger.info("PREPROCESSING CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Normalize data: {config.preprocessing.normalize_data}")
    logger.info(f"Center: {config.preprocessing.center}")
    logger.info(f"Whiten: {config.preprocessing.whiten}")
    logger.info(f"Whitening epsilon: {config.preprocessing.whitening_epsilon}")
    logger.info(f"Normalize type: {config.preprocessing.normalize_type}")
    logger.info(f"Forward function provided: {config.preprocessing.forward is not None}")
    logger.info(f"Backward function provided: {config.preprocessing.backward is not None}")
    logger.info(f"Mean (mu) provided: {config.preprocessing.mu is not None}")
    logger.info(f"Shift provided: {config.preprocessing.shift is not None}")
    logger.info(f"Scale provided: {config.preprocessing.scale is not None}")
    if config.preprocessing.mu is not None:
        logger.info(f"Mean shape: {config.preprocessing.mu.shape}")
    if config.preprocessing.shift is not None:
        logger.info(f"Shift shape: {config.preprocessing.shift.shape}")
    if config.preprocessing.scale is not None:
        logger.info(f"Scale shape: {config.preprocessing.scale.shape}")
    
    # Sparsity configuration
    logger.info("-" * 30)
    logger.info("SPARSITY CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"M (Lipschitz bound): {config.sparsity.M}")
    logger.info(f"Nonzero threshold: {config.sparsity.nonzero_thresh}")
    logger.info(f"Lambda 0 (initial): {config.sparsity.lam0}")
    logger.info(f"Epsilon (threshold): {config.sparsity.epsilon}")
    logger.info(f"Max iterations: {config.sparsity.max_iters}")
    logger.info(f"Skip sparse: {config.sparsity.skip_sparse}")
    logger.info(f"Selection method: {config.sparsity.selection_method}")
    logger.info(f"Alpha (balance param): {config.sparsity.alpha}")
    
    # Experiment configuration
    logger.info("-" * 30)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Label: {config.experiment.label}")
    logger.info(f"Enable logging: {config.experiment.enable_logging}")
    logger.info(f"Logs directory: {config.experiment.logs_dir}")
    
    logger.info("="*50)
    

def _log_results(logger: logging.Logger, model, 
                 I_nn: np.ndarray, history: dict):
    """
    Log experiment results.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    model : trained model
        The trained model
    I_nn : np.ndarray
        Selected mode indices
    history : dict
        Training/selection history
    """
    logger.info("="*50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*50)
    
    # Selection results
    if I_nn is not None:
        logger.info(f"Number of selected modes: {len(I_nn)}")
        logger.info(f"Selected mode indices: {I_nn.tolist() if hasattr(I_nn, 'tolist') else list(I_nn)}")
        
        # Calculate sparsity if we have path history
        if 'path_history' in history:
            path_hist = history['path_history']
            total_modes = path_hist.get('total_modes', len(I_nn))
            logger.info(f"Selection sparsity: {len(I_nn)} / {total_modes} ({100*len(I_nn)/total_modes:.1f}%)")
    
    # LASSO training results
    if 'lasso_history' in history:
        lasso_hist = history['lasso_history']
        if 'loss' in lasso_hist and lasso_hist['loss']:
            final_loss = lasso_hist['loss'][-1]
            initial_loss = lasso_hist['loss'][0]
            logger.info(f"LASSO - Initial loss: {initial_loss:.6e}")
            logger.info(f"LASSO - Final loss: {final_loss:.6e}")
            logger.info(f"LASSO - Loss reduction: {100*(1-final_loss/initial_loss):.2f}%")
            logger.info(f"LASSO - Epochs completed: {len(lasso_hist['loss'])}")
    
    # Decoder training results
    if 'decoder_history' in history:
        decoder_hist = history['decoder_history']
        if 'loss' in decoder_hist and decoder_hist['loss']:
            final_loss = decoder_hist['loss'][-1]
            initial_loss = decoder_hist['loss'][0]
            logger.info(f"Decoder - Initial loss: {initial_loss:.6e}")
            logger.info(f"Decoder - Final loss: {final_loss:.6e}")
            logger.info(f"Decoder - Loss reduction: {100*(1-final_loss/initial_loss):.2f}%")
            logger.info(f"Decoder - Epochs completed: {len(decoder_hist['loss'])}")
    
    # Path history results (for dense2sparse)
    if 'path_history' in history:
        path_hist = history['path_history']
        if 'path' in path_hist:
            logger.info(f"Path length: {len(path_hist['path'])}")
        if 'knee_point' in path_hist:
            logger.info(f"Knee point: {path_hist['knee_point']}")
        if 'omega_evolution' in path_hist:
            logger.info(f"Omega evolution steps: {len(path_hist['omega_evolution'])}")
    
    # Reconstruction error
    if 'reconstruction_error' in history:
        rec_error = history['reconstruction_error']
        logger.info(f"Final reconstruction error: {rec_error:.6e}")
    
    logger.info("="*50)
