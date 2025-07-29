import numpy as np
import torch
from sparsemodesnet.config import SparseModesNetConfig
from sparsemodesnet.linalg.zca import zca_whitening_matrix

def preprocess(X: np.ndarray, config: SparseModesNetConfig):
    """
    Preprocess the input data according to the configuration.
    """
    
    if config.preprocessing.center:
        # Compute the row-wise mean
        mu = X.mean(axis=1, keepdims=True)
        config.preprocessing.mu = mu
        X_center = X - mu
    else:
        X_center = np.copy(X)
        mu = 0.0

    if config.preprocessing.normalize and config.preprocessing.whiten:
        # Compute the row-wise max/min
        _min, _max = X_center.min(axis=1), X_center.max(axis=1)

        # Shift = min, Scale = max - min
        _shift = _min.reshape(-1, 1)
        _scale = (_max - _min).reshape(-1, 1)

        # Normalize to [0, 1]
        X_norm = (X_center - _shift) / _scale

        # Compute the ZCA whitening matrix
        zcaMat = zca_whitening_matrix(
            X_norm, 
            epsilon=config.preprocessing.whitening_epsilon
        )

        # Apply ZCA whitening
        X_proc = np.dot(zcaMat, X_norm)

        # Save shift and scale for later use
        config.preprocessing.shift = _shift
        config.preprocessing.scale = _scale
        config.preprocessing.forward = lambda x: ((x - mu) - _shift) / _scale
        config.preprocessing.backward = lambda x: (x * _scale) + _shift + mu

        # # Compute the 

        # V_white, _, _ = np.linalg.svd(X_proc, full_matrices=False)
        # V_white = V_white[:, :s_p]  
        # V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
    elif config.preprocessing.normalize:
        # Compute the row-wise max/min
        _min, _max = X_center.min(axis=1), X_center.max(axis=1)

        # Shift = min, Scale = max - min
        _shift = _min.reshape(-1, 1)
        _scale = (_max - _min).reshape(-1, 1)

        # Normalize to [0, 1]
        X_proc = (X_center - _shift) / _scale

        # Save shift and scale for later use
        config.preprocessing.shift = _shift
        config.preprocessing.scale = _scale
        config.preprocessing.forward = lambda x: ((x - mu) - _shift) / _scale
        config.preprocessing.backward = lambda x: (x * _scale) + _shift + mu

        # V_white = np.linalg.svd(X_proc, full_matrices=False)[0][:, :s_p] 
        # V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
    elif config.preprocessing.whiten:
        zcaMat = zca_whitening_matrix(
            X_center, 
            epsilon=config.preprocessing.whitening_epsilon
        )
        X_proc = np.dot(zcaMat, X_center)  
        
        config.preprocessing.forward = lambda x: x - mu
        config.preprocessing.backward = lambda x: x + mu

        # # Compute the pod basis
        # V_white, _, _ = np.linalg.svd(X_white, full_matrices=False)
        # V_white = V_white[:, :s_p]  
        # V_white_tensor = torch.from_numpy(V_white.astype(np.float32)).to(device)
    else:
        X_proc = X_center
        config.preprocessing.forward = lambda x: x - mu
        config.preprocessing.backward = lambda x: x + mu

    return X_proc