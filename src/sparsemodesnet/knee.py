import numpy as np
import kneeliverse.dfdt as dfdt
import kneeliverse.zmethod as zmethod

def _normalize_array(arr):
    """Normalize array to [0, 1] range."""
    if np.all(arr == arr[0]):  # All elements are the same
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())


def _detect_knees(data, knee_method):
    """Detect knee points using specified method."""
    if knee_method == 'dfdt':
        knee_idx = dfdt.multi_knee(data)
        print(f"Found knees using DFDT method: {knee_idx}")
    elif knee_method == 'zmethod':
        knee_idx = zmethod.knees2(data)
        print(f"Found knees using Z-method: {knee_idx}")
    else:
        raise ValueError(f"Unknown knee_method: {knee_method}. "
                        f"Use 'dfdt' or 'zmethod'.")
    
    # Ensure knee_idx is a numpy array
    knee_idx = np.array(knee_idx, dtype=int)
    return knee_idx


def _fallback_feature_selection(path_history, r_max):
    """Fallback feature selection when knee detection fails."""
    print("No modes selected at λ*. Using fallback selection.")
    
    if r_max is not None:
        print(f"Searching for first λ with nonzero_count <= {r_max} ...")
        nonzero_counts = np.array(
            [entry['nonzero_count'] for entry in path_history])
        valid_indices = np.where(nonzero_counts <= r_max)[0]
        
        if len(valid_indices) > 0:
            idx = valid_indices[0]
        else:
            idx = -1  # Use last entry
    else:
        print("r_max not specified. Using last entry from path.")
        idx = -1
    
    entry = path_history[idx]
    lam_star, r_star = entry['lambda'], entry['nonzero_count']
    err_star = entry['error']
    I_NN = entry['selected_idxs']
    print(f"[Fallback] λ={lam_star:.3e}, r={r_star}, err={err_star:.6e}")
    print(f"Selected modes: {I_NN.tolist()}")
    return I_NN


def _select_from_knees(path_history, knee_idx, r_max):
    """Select lambda from detected knee points."""
    rs = np.array([h['nonzero_count'] for h in path_history])
    r_stars = rs[knee_idx]
    
    # Pick first knee point that satisfies r_max constraint
    if r_max is not None:
        valid_mask = r_stars <= r_max
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            i_star = valid_indices[0]
        else:
            # No knee satisfies r_max constraint, find closest to r_max
            distances = np.abs(r_stars - r_max)
            i_star = np.argmin(distances)
    else:
        i_star = 0
        
    # Make sure knee_idx[i_star] is a valid index
    selected_knee_idx = knee_idx[i_star]
    I_NN = path_history[selected_knee_idx]['selected_idxs']
    if len(I_NN) > r_max:
        I_NN = I_NN[:r_max]  # Truncate if exceeds r_max 
    
    # Print selection info
    selected_entry = path_history[selected_knee_idx]
    print(f"Selected knee at index {selected_knee_idx}: "
          f"λ={selected_entry['lambda']:.3e}, "
          f"r={selected_entry['nonzero_count']}, "
          f"err={selected_entry['error']:.6e}")
    print(f"Selected modes: {I_NN.tolist()}")
    
    return I_NN


def _find_best_features(path_history, knee_method, r_max):
    """Find optimal features using knee detection or fallback methods."""
    # Prepare data for knee detection
    err = np.array([h['error'] for h in path_history])
    logerr = np.log(err + 1e-16)  # Add small epsilon to avoid log(0)
    lasso = np.array([h['l1_b'] for h in path_history])
    loglasso = np.log(lasso + 1e-16)  # Add small epsilon to avoid log(0)
    
    # Normalize to [0, 1] range
    logerr_norm = _normalize_array(logerr)
    loglasso_norm = _normalize_array(loglasso)
    data = np.column_stack((loglasso_norm, logerr_norm))
    
    # Apply knee detection
    knee_idx = _detect_knees(data, knee_method)
    
    if len(knee_idx) == 0:
        return _fallback_feature_selection(path_history, r_max)
    else:
        return _select_from_knees(path_history, knee_idx, r_max)
