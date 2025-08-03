import numpy as np
from typing import Tuple, Callable, Optional
from sparsemodesnet.config import SparseModesNetConfig
from sparsemodesnet.linalg.zca import zca_whitening_matrix

def _compute_centering_stats(X: np.ndarray, should_center: bool) -> Tuple[np.ndarray, float]:
    """Compute centering statistics."""
    if should_center:
        mu = X.mean(axis=1, keepdims=True)
        X_centered = X - mu
    else:
        mu = 0.0
        X_centered = X.copy()
    
    return X_centered, mu

def _compute_normalization_stats(X: np.ndarray, type: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute min-max normalization parameters."""
    data_min, data_max = X.min(axis=1), X.max(axis=1)
    if type == 'minmax':
        shift = data_min.reshape(-1, 1)
        scale = (data_max - data_min).reshape(-1, 1)
    elif type == 'minmaxsym':
        shift = ((data_max + data_min) / 2).reshape(-1, 1)
        scale = ((data_max - data_min) / 2).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown normalization type: {type}")
    # Avoid division by zero
    scale = np.where(scale == 0, 1.0, scale)  # Prevent division by zero
    return shift, scale


def _normalize_data(X: np.ndarray, shift: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply min-max normalization to [0, 1]."""
    return (X - shift) / scale


def _apply_whitening(X: np.ndarray, epsilon: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply ZCA whitening transformation."""
    zca_matrix = zca_whitening_matrix(X, epsilon=epsilon)
    return np.dot(zca_matrix, X), zca_matrix


def _create_transform_functions(mu: float, shift: Optional[np.ndarray] = None, 
                              scale: Optional[np.ndarray] = None) -> Tuple[Callable, Callable]:
    """Create forward and backward transformation functions."""
    
    def forward(x: np.ndarray) -> np.ndarray:
        result = x - mu
        if shift is not None and scale is not None:
            result = (result - shift) / scale
        return result
    
    def backward(x: np.ndarray) -> np.ndarray:
        result = x
        if shift is not None and scale is not None:
            result = (result * scale) + shift
        return result + mu
    
    return forward, backward


def preprocess(X: np.ndarray, config: SparseModesNetConfig) -> np.ndarray:
    """
    Preprocess input data according to configuration settings.
    
    Args:
        X: Input data array of shape (n_features, n_samples)
        config: Configuration object containing preprocessing parameters
        
    Returns:
        Preprocessed data array
    """
    # Step 1: Handle centering
    X_processed, mu = _compute_centering_stats(X, config.preprocessing.center)
    config.preprocessing.mu = mu
    
    # Step 2: Determine preprocessing pipeline
    normalize = config.preprocessing.normalize_data
    normalize_type = config.preprocessing.normalize_type
    whiten = config.preprocessing.whiten
    
    shift, scale = None, None
    
    # Step 3: Apply normalization if needed
    if normalize:
        shift, scale = _compute_normalization_stats(X_processed, normalize_type)
        X_processed = _normalize_data(X_processed, shift, scale)
        
        # Store normalization parameters
        config.preprocessing.shift = shift
        config.preprocessing.scale = scale
    
    # Step 4: Apply whitening if needed
    if whiten:
        X_processed, zca_matrix = _apply_whitening(
            X_processed, 
            config.preprocessing.whitening_epsilon
        )
        # Store whitening matrix if needed for future use
        config.preprocessing.zca_matrix = zca_matrix
    
    # Step 5: Create and store transformation functions
    forward_fn, backward_fn = _create_transform_functions(mu, shift, scale)
    config.preprocessing.forward = forward_fn
    config.preprocessing.backward = backward_fn
    
    return X_processed

