import math
from typing import List, Dict

def pick_aic(path_history: List[Dict], n_samples: int) -> tuple[float, int, float]:
    get_r = lambda h: h['r']
    best_score = math.inf
    best_tuple = (path_history[0]['lambda'], get_r(path_history[0]), path_history[0]['rel_error'])
    for h in path_history:
        r = get_r(h)
        mse = h['val_error']  # the MSE
        aic = n_samples * math.log(mse) + 2 * r 
        if n_samples < 100:   # small sample size correction (AICc)
            aic += 2 * r * (r + 1) / (n_samples - r - 1)  
            
        if aic < best_score:
            best_score = aic
            best_tuple = (h['lambda'], r, h['rel_error'])
    return best_tuple

# def pick_aic(path_history: List[Dict], n_samples: int, alpha: float) -> tuple[float, int, float]:
#     """Select the best regularization parameter based on Akaike Information Criterion (AIC):
#     AIC = e + (α * k * log(n_samples) / n_samples)
#     where:
#         - e is the relative error ||x - x_hat|| / ||x||
#         - k is the number of non-zero coefficients
#         - n_samples is the number of samples used in the model fitting
#         - α is a significance level (typically small, e.g., 0.1)

#     Args:
#         path_history (List[Dict]): List of dictionaries containing path history with keys:
#             - 'lambda': Regularization parameter
#             - 'rel_error': Relative error
#             - 'nonzero_count': Number of non-zero coefficients or
#             - 'k': Number of non-zero coefficients (alternative key)
#         n_samples (int): Number of samples used in the model fitting.
#         alpha (float): Significance level for AIC calculation, typically a small value like 0.1.

#     Raises:
#         ValueError: If the path history does not contain 'nonzero_count' or 'k'.

#     Returns:
#         tuple[float, int, float]: A tuple containing:
#             - Best regularization parameter (lambda)
#             - Number of non-zero coefficients (k)
#             - Best relative error
#     """
#     # Accept both 'nonzero_count' and 'k'
#     get_r = lambda h: h['r']
#     best_score = math.inf
#     best_tuple = (path_history[0]['lambda'], get_r(path_history[0]), path_history[0]['rel_error'])
#     for h in path_history:
#         r = get_r(h)
#         mse = h['val_error']  # the MSE
#         aic = mse + (alpha * r * math.log(n_samples) / n_samples)
#         if aic < best_score:
#             best_score = aic
#             best_tuple = (h['lambda'], r, mse)
#     return best_tuple

# def pick_aic(path_history: List[Dict], n_samples: int, alpha: float) -> tuple[float, int, float]:
#     # Accept both 'nonzero_count' and 'k'
#     if 'nonzero_count' in path_history[0]:
#         get_k = lambda h: h['nonzero_count']
#     elif 'k' in path_history[0]:
#         get_k = lambda h: h['k']
#     else:
#         raise ValueError("Path history missing 'nonzero_count' or 'k'.")
#     best_score = math.inf
#     best_tuple = (path_history[0]['lambda'], get_k(path_history[0]), path_history[0]['rel_error'])
#     for h in path_history:
#         k = get_k(h)
#         e = h['rel_error']
#         aic = e + (alpha * k * math.log(n_samples) / n_samples)
#         if aic < best_score:
#             best_score = aic
#             best_tuple = (h['lambda'], k, e)
#     return best_tuple