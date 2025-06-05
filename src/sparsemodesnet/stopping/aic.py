import math
import numpy as np
from typing import List, Dict

def pick_aic(path_history: List[Dict], n_samples: int, alpha: float) -> tuple[float, int, float]:
    # Accept both 'nonzero_count' and 'k'
    if 'nonzero_count' in path_history[0]:
        get_k = lambda h: h['nonzero_count']
    elif 'k' in path_history[0]:
        get_k = lambda h: h['k']
    else:
        raise ValueError("Path history missing 'nonzero_count' or 'k'.")
    best_score = math.inf
    best_tuple = (path_history[0]['lambda'], get_k(path_history[0]), path_history[0]['rel_error'])
    for h in path_history:
        k = get_k(h)
        e = h['rel_error']
        aic = e + (alpha * k * math.log(n_samples) / n_samples)
        if aic < best_score:
            best_score = aic
            best_tuple = (h['lambda'], k, e)
    return best_tuple
