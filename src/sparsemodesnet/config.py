from dataclasses import dataclass
from typing import Optional, List, Callable
import numpy as np

@dataclass
class NetworkConfig:
    """Configuration for network architecture"""
    hidden_units: List[int]
    network_type: str = 'FF'
    poly_order: int = 2
    num_polys: int = 1
    drop_linear: bool = False
    drop_constant: bool = False
    normalize_layer: str | None = None

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    # Lasso part
    lasso_lr: float = 1e-3
    lasso_lr_patience: int = 100
    lasso_lr_factor: float = 0.8
    lasso_epochs: int = 100
    lasso_batch_size: int = 32
    lasso_optimizer: str = 'AdamW'
    lasso_momentum: float = 0.9
    lasso_bias: bool = False
    gamma: float = 1e-6
    max_no_change: int = 50
    extra_modes: int = 0
    # Decoder part
    decoder_lr: float = 1e-3
    decoder_lr_patience: int = 100
    decoder_lr_factor: float = 0.8
    decoder_epochs: int = 100
    decoder_batch_size: int = 32
    decoder_optimizer: str = 'AdamW'
    decoder_momentum: float = 0.9
    decoder_bias: bool = False
    # General
    device: str = 'cpu'
    I_nn: Optional[np.ndarray] = None
    reg_param: float = 1e-15
    weight_scale: float = 1e-12
    analytical: bool = False

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps"""
    normalize_data: bool = True
    center: bool = True
    whiten: bool = False
    whitening_epsilon: float = 1e-5
    forward: Callable | None = None
    backward: Callable | None = None
    mu: np.ndarray | None = None
    shift: np.ndarray | None = None
    scale: np.ndarray | None = None
    normalize_type: str = 'minmax'  # `minmax`, `minmaxsym`

@dataclass
class SparsityConfig:
    """Configuration for sparsity/regularization"""
    M: float = 10.0
    nonzero_thresh: float = 1e-6
    lam0: float = 1e-6
    epsilon: float = 0.1
    max_iters: int = 1000
    skip_sparse: bool = False
    selection_method: str = 'weight'
    alpha: float = 1.0

@dataclass
class ExperimentConfig:
    """Configuration for experiment setup"""
    label: str = "SparseModesNet"
    enable_logging: bool = True
    logs_dir: str = "./logs"

@dataclass
class SparseModesNetConfig:
    """Complete configuration for SparseModesNet experiments"""
    # Core parameters
    s: int # total number of modes used
    r: int # number of modes selected for decoder r <= s
    p: int # nonlinear mapping output dimension
    
    # Configuration groups
    network: NetworkConfig
    training: TrainingConfig
    preprocessing: PreprocessingConfig
    sparsity: SparsityConfig
    experiment: ExperimentConfig
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SparseModesNetConfig':
        """Create config from dictionary for backward compatibility"""
        # Extract nested configs
        network = NetworkConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in NetworkConfig.__dataclass_fields__
            }
        )
        training = TrainingConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in TrainingConfig.__dataclass_fields__
            }
        )
        preprocess = PreprocessingConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in PreprocessingConfig.__dataclass_fields__
            }
        )
        sparsity = SparsityConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in SparsityConfig.__dataclass_fields__
            }
        )
        experiment = ExperimentConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in ExperimentConfig.__dataclass_fields__
            }
        )
        
        # Extract remaining parameters
        remaining = {
            k: v for k, v in config_dict.items() 
            if k not in (
                NetworkConfig.__dataclass_fields__       | 
                TrainingConfig.__dataclass_fields__      |
                PreprocessingConfig.__dataclass_fields__ |
                SparsityConfig.__dataclass_fields__      |
                ExperimentConfig.__dataclass_fields__
            )
        }
        
        return cls(
            network=network,
            training=training, 
            preprocessing=preprocess,
            sparsity=sparsity,
            experiment=experiment,
            **remaining
        )
