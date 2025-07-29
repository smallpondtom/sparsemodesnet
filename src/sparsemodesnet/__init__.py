"""
SparseModesNet: POD-based sparse mode selection with neural networks
"""

from .models.model import SparseModesNet
from .linalg.pod import compute_pod_basis
from .linalg.zca import zca_whitening_matrix
from .dataset import PODReconDataset
from .training.train import train_sparsemodesnet
from .fit import fit

__all__ = [
    # Core components
    'SparseModesNet',
    'PODReconDataset',
    'compute_pod_basis',
    'zca_whitening_matrix',
    'train_sparsemodesnet',
    
    # Main driver function
    'fit'
]

__version__ = "0.1.0"
