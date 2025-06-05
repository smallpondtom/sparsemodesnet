"""
SparseModesNet: POD-based sparse mode selection with neural networks
"""

from .model import SparseModesNet
from .pod import compute_pod_basis
from .dataset import PODReconDataset
from .train import train_sparsemodesnet
from .driver import run_sparsemodesnet, run_sparsemodesnet_with_lambda_selection

# Import lambda selection methods
from .select_lambda.cv import select_lambda_cv
from .select_lambda.stability import select_lambda_stability

# Import stopping criteria
from .stopping.elbow import pick_elbow
from .stopping.aic import pick_aic
from .stopping.bic import pick_bic
from .stopping.maxmodes import pick_max_modes

__all__ = [
    # Core components
    'SparseModesNet',
    'PODReconDataset',
    'compute_pod_basis',
    'train_sparsemodesnet',
    
    # Main driver functions
    'run_sparsemodesnet',
    'run_sparsemodesnet_with_lambda_selection',
    
    # Lambda selection methods
    'select_lambda_cv',
    'select_lambda_stability',
    
    # Stopping criteria
    'pick_elbow',
    'pick_aic', 
    'pick_bic',
    'pick_max_modes'
]

__version__ = "0.1.0"