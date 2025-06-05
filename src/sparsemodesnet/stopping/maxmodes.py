from typing import List, Dict

def pick_max_modes(path_history: List[Dict], K_max: int) -> tuple[float, int, float]:
    """
    Find the first λ in path_history such that nonzero_count <= K_max.
    If none satisfies, return the last entry.
    Returns (lambda_c, k_c, err_c).
    """
    # Accept both 'nonzero_count' and 'k'
    if 'nonzero_count' in path_history[0]:
        get_k = lambda h: h['nonzero_count']
    elif 'k' in path_history[0]:
        get_k = lambda h: h['k']
    else:
        raise ValueError("Path history missing 'nonzero_count' or 'k'.")
    for h in path_history:
        if get_k(h) <= K_max:
            return h['lambda'], get_k(h), h['rel_error']
    # fallback to last
    h_last = path_history[-1]
    return h_last['lambda'], get_k(h_last), h_last['rel_error']
