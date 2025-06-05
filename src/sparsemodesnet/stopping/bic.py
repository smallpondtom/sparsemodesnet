import math
import numpy as np
from typing import List, Dict

def pick_bic(path_history: List[Dict], n_samples: int) -> tuple[float, int, float]:
    """
    Given path_history (list of dicts) and n_samples = # snapshots,
    pick the λ that minimizes E(λ) + (k(λ) * log(n_samples) / n_samples).
    Returns (lambda_bic, k_bic, err_bic).
    """
    # For backward compatibility, accept both 'nonzero_count' and 'k'
    best_score = math.inf
    if 'nonzero_count' in path_history[0]:
        get_k = lambda h: h['nonzero_count']
    elif 'k' in path_history[0]:
        get_k = lambda h: h['k']
    else:
        raise ValueError("Path history missing 'nonzero_count' or 'k'.")
    best_tuple = (path_history[0]['lambda'], get_k(path_history[0]), path_history[0]['rel_error'])
    for h in path_history:
        k = get_k(h)
        e = h['rel_error']
        bic = e + (k * math.log(n_samples) / n_samples)
        if bic < best_score:
            best_score = bic
            best_tuple = (h['lambda'], k, e)
    return best_tuple
