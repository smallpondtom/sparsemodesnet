from dataclasses import dataclass
from typing import Optional, List
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
    batch_size: int = 32
    optimizer: str = 'Adam'
    device: str = 'cpu'
    I_NN: Optional[np.ndarray] = None

@dataclass
class SparsityConfig:
    """Configuration for sparsity/regularization"""
    M: float = 10.0
    nonzero_thresh: float = 1e-6
    lam0: float = 1e-6
    epsilon: float = 0.1
    max_iters: int = 100

@dataclass
class SelectionConfig:
    """Configuration for mode selection methods"""
    mode_selection: str = 'dense2sparse'
    knee_method: str = 'dfdt'
    r_max: Optional[int] = None
    
    # For CV
    k_folds: int = 5
    lambdas: Optional[np.ndarray] = None
    
    # For Stability Selection
    num_subsamples: int = 100
    pi_thresh: float = 0.6
    
    # For Knockoffs
    fdr: float = 0.1
    knockoff_method: str = 'mvr'
    feature_stat: str = 'lasso'

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
    sparsity: SparsityConfig
    selection: SelectionConfig
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
        sparsity = SparsityConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in SparsityConfig.__dataclass_fields__
            }
        )
        selection = SelectionConfig(
            **{
                k: v for k, v in config_dict.items() 
                if k in SelectionConfig.__dataclass_fields__
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
                NetworkConfig.__dataclass_fields__   | 
                TrainingConfig.__dataclass_fields__  |
                SparsityConfig.__dataclass_fields__  |
                SelectionConfig.__dataclass_fields__ |
                ExperimentConfig.__dataclass_fields__
            )
        }
        
        return cls(
            network=network,
            training=training, 
            sparsity=sparsity,
            selection=selection,
            experiment=experiment,
            **remaining
        )