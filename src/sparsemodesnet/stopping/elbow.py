import numpy as np
from typing import List, Dict

def pick_elbow(path_history: List[Dict]) -> tuple[float, int, float]:
    """
    Given a list of dicts with keys 'lambda', 'nonzero_count', and 'rel_error',
    find the index i (1 <= i <= len-2) where the second‐difference of rel_error is largest.
    Returns (lambda_elbow, k_elbow, err_elbow).
    """
    # Extract arrays of k and e
    # Accept both 'nonzero_count' and 'k' as the key for number of nonzeros
    e_list = [h['rel_error'] for h in path_history]
    if 'nonzero_count' in path_history[0]:
        k_list = [h['nonzero_count'] for h in path_history]
    elif 'k' in path_history[0]:
        k_list = [h['k'] for h in path_history]
    else:
        raise ValueError("Path history missing 'nonzero_count' or 'k'.")
    T = len(e_list)
    if T < 3:
        h0 = path_history[0]
        return h0['lambda'], h0['nonzero_count'], h0['rel_error']

    # Compute second differences: Δ²E(i) = (E[i+1]-E[i]) - (E[i]-E[i-1]) for i = 1..T-2
    second_diff = []
    for i in range(1, T-1):
        dd = abs((e_list[i+1] - e_list[i]) - (e_list[i] - e_list[i-1]))
        second_diff.append(dd)
    idx0 = int(np.argmax(second_diff)) + 1  # offset by 1 because second_diff[0] corresponds to i=1
    h_best = path_history[idx0]
    lam = h_best['lambda']
    k = h_best['nonzero_count'] if 'nonzero_count' in h_best else h_best['k']
    err = h_best['rel_error']
    return lam, k, err
