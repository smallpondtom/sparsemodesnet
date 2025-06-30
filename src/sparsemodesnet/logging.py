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
    logger.info(f"s (latent dimension): {config.s}")
    logger.info(f"I_NN provided: {config.training.I_NN is not None}")
    if config.training.I_NN is not None:
        logger.info(f"I_NN shape: {config.training.I_NN.shape}")
    
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
    
    # Training configuration
    logger.info("-" * 30)
    logger.info("TRAINING CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Learning rate: {config.training.lr}")
    logger.info(f"Number of epochs: {config.training.num_epochs}")
    logger.info(f"Batch size: {config.training.batch_size}")
    logger.info(f"Optimizer: {config.training.optimizer}")
    logger.info(f"Device: {config.training.device}")
    
    # Sparsity configuration
    logger.info("-" * 30)
    logger.info("SPARSITY CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"M (penalty parameter): {config.sparsity.M}")
    logger.info(f"Nonzero threshold: {config.sparsity.nonzero_thresh}")
    logger.info(f"Lambda 0: {config.sparsity.lam0}")
    logger.info(f"Epsilon: {config.sparsity.epsilon}")
    logger.info(f"Max iterations: {config.sparsity.max_iters}")
    
    # Selection configuration
    logger.info("-" * 30)
    logger.info("SELECTION CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Mode selection: {config.selection.mode_selection}")
    logger.info(f"Knee method: {config.selection.knee_method}")
    logger.info(f"R max: {config.selection.r_max}")
    
    if config.selection.mode_selection == 'cv':
        logger.info(f"K-folds: {config.selection.k_folds}")
        logger.info(f"Lambdas: {config.selection.lambdas}")
    elif config.selection.mode_selection == 'ss':
        logger.info(f"Number of subsamples: {config.selection.num_subsamples}")
        logger.info(f"Pi threshold: {config.selection.pi_thresh}")
    elif config.selection.mode_selection == 'knockoffs':
        logger.info(f"FDR: {config.selection.fdr}")
        logger.info(f"Knockoff method: {config.selection.knockoff_method}")
        logger.info(f"Feature statistic: {config.selection.feature_stat}")
    
    # Experiment configuration
    logger.info("-" * 30)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("-" * 30)
    logger.info(f"Label: {config.experiment.label}")
    logger.info(f"Logs directory: {config.experiment.logs_dir}")
    
    logger.info("="*50)
    

def _log_results(logger: logging.Logger, model, 
                 I_NN: np.ndarray, history: dict):
    """
    Log experiment results.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    model : trained model
        The trained model
    I_NN : np.ndarray
        Selected mode indices
    history : dict
        Training/selection history
    """
    logger.info("="*50)
    logger.info("EXPERIMENT RESULTS")
    logger.info("="*50)
    
    # Selection results
    if I_NN is not None:
        logger.info(f"Number of selected modes: {len(I_NN)}")
        logger.info(f"Selected mode indices: {I_NN.tolist()}")
        total_modes = len(I_NN) + (history.get('path_history', {}).get(
            'total_modes', 0) - len(I_NN))
        logger.info(f"Selection sparsity: {len(I_NN)} / {total_modes}")
    
    # Training results
    if 'training_history' in history:
        train_hist = history['training_history']
        if 'loss' in train_hist:
            final_loss = train_hist['loss'][-1] if train_hist['loss'] else float('inf')
            logger.info(f"Final training loss: {final_loss:.6e}")
            logger.info(f"Training epochs completed: {len(train_hist['loss'])}")
    
    # Path history results (for dense2sparse)
    if 'path_history' in history:
        path_hist = history['path_history']
        logger.info(f"Path length: {len(path_hist.get('path', []))}")
        if 'knee_point' in path_hist:
            logger.info(f"Knee point: {path_hist['knee_point']}")
    
    logger.info("="*50)