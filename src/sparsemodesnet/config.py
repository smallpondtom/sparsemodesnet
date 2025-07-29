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
    normalize: str | None = None

@dataclass
class TrainingConfig:
    """Configuration for training parameters"""
    lr: float = 1e-3
    num_epochs: int = 100
    final_epochs: int = 100
    batch_size: int = 32
    optimizer: str = 'Adam'
    device: str = 'cpu'
    I_nn: Optional[np.ndarray] = None
    momentum: float = 0.9
    max_no_change: int = 50

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing steps"""
    normalize: bool = True
    center: bool = True
    whiten: bool = False
    whitening_epsilon: float = 1e-5
    lift: Callable | None = None
    unlift: Callable | None = None
    mu: np.ndarray | None = None
    shift: np.ndarray | None = None
    scale: np.ndarray | None = None

@dataclass
class SparsityConfig:
    """Configuration for sparsity/regularization"""
    M: float = 10.0
    nonzero_thresh: float = 1e-6
    lam0: float = 1e-6
    epsilon: float = 0.1
    max_iters: int = 1000
    max_num_modes: int = 20
    skip_sparse: bool = False

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
    s: int
    
    # Configuration groups
    network: NetworkConfig
    training: TrainingConfig
    preprocessing: PreprocessingConfig
    sparsity: SparsityConfig
    # selection: SelectionConfig
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
