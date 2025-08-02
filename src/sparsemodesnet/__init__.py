"""
SparseModesNet: POD-based sparse mode selection with neural networks
"""

from .config import (
    SparseModesNetConfig,
    NetworkConfig,
    TrainingConfig,
    PreprocessingConfig,
    SparsityConfig,
    ExperimentConfig
)
from .decoder_models.model import SparseModesNet, StateDecoder
from .linalg.pod import compute_pod_basis
from .linalg.zca import zca_whitening_matrix
from .dataset import PODReconDataset
from .training.train import train_sparsemodesnet
from .fit import fit
from .viz.omega_evolve import omega_evolve

__all__ = [
    # Configurations
    'SparseModesNetConfig',
    'NetworkConfig',
    'TrainingConfig',
    'PreprocessingConfig',
    'SparsityConfig',
    'ExperimentConfig',

    # Core components
    'SparseModesNet',
    'StateDecoder',
    'PODReconDataset',
    'compute_pod_basis',
    'zca_whitening_matrix',
    'train_sparsemodesnet',
    
    # Main driver function
    'fit'

    # Visualizations
    'omega_evolve' 
]

__version__ = "0.1.0"
