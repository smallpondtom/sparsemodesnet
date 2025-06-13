"""
SparseModesNet: POD-based sparse mode selection with neural networks
"""

from .model import SparseModesNet
from .pod import compute_pod_basis
from .dataset import PODReconDataset
from .train import train_sparsemodesnet
from .driver import run_sparsemodesnet, run_sparsemodesnet_d2s
from .cv import run_sparsemodesnet_cv

__all__ = [
    # Core components
    'SparseModesNet',
    'PODReconDataset',
    'compute_pod_basis',
    'train_sparsemodesnet',
    
    # Main driver functions
    'run_sparsemodesnet',
    'run_sparsemodesnet_d2s',
    'run_sparsemodesnet_cv',
]

__version__ = "0.1.0"