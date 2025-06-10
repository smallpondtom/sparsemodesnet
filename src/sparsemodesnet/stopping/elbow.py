import numpy as np
from typing import List, Dict

def pick_elbow(path_history: List[Dict]) -> tuple[float, int, float]:
    """Given a list of dicts with keys 'l1_b', 'lambda', 'nonzero_count', and 
    'rel_error', find the index i (1 <= i <= len-2) where the L-curve elbow 
    is located.

    Arguments
    ---------
    path_history : (List[Dict])
        List of dictionaries containing the path history.

    Returns
    -------
    tuple[float, int, float] 
        - lam_star (float): The optimal regularization parameter λ*.
        - r_star (int): The number of non-zero features k*.
        - err_star (float): The relative reconstruction error E* at λ*.
    """
    # Extract arrays of solution norm and relative reconstruction error
    P_list = [h['l1_b'] for h in path_history]
    E_list = [h['rel_error'] for h in path_history]
    
    # Obtain the index of the elbow using Castellanos's algorithm 
    idx_star = find_lcurve_corner_triangle(E_list, P_list)
    
    # Select the best path history entry 
    h_best = path_history[idx_star] 
    lam_star = h_best['lambda'] 
    r_star = h_best['nonzero_count'] 
    err_star = h_best['rel_error']
    return lam_star, r_star, err_star

# def pick_elbow(path_history: List[Dict]) -> tuple[float, int, float]:
#     """
#     Given a list of dicts with keys 'lambda', 'nonzero_count', and 'rel_error',
#     find the index i (1 <= i <= len-2) where the second-difference of rel_error is largest.
#     Returns (lambda_elbow, k_elbow, err_elbow).
#     """
#     # Extract arrays of 'r' and 'e'
#     # Accept both 'nonzero_count' and 'r' as the key for number of nonzeros
#     e_list = [h['rel_error'] for h in path_history]
#     if 'nonzero_count' in path_history[0]:
#         r_list = [h['nonzero_count'] for h in path_history]
#     elif 'r' in path_history[0]:
#         r_list = [h['r'] for h in path_history]
#     else:
#         raise ValueError("Path history missing 'nonzero_count' or 'r'.")
#     T = len(e_list)
#     if T < 3:
#         h0 = path_history[0]
#         return h0['lambda'], h0['nonzero_count'], h0['rel_error']

#     # Compute second differences: Δ²E(i) = (E[i+1]-E[i]) - (E[i]-E[i-1]) for i = 1..T-2
#     second_diff = []
#     for i in range(1, T-1):
#         dd = abs((e_list[i+1] - e_list[i]) - (e_list[i] - e_list[i-1]))
#         second_diff.append(dd)
#     idx0 = int(np.argmax(second_diff)) + 1  # offset by 1 because second_diff[0] corresponds to i=1
#     h_best = path_history[idx0]
#     lam = h_best['lambda']
#     k = h_best['nonzero_count'] if 'nonzero_count' in h_best else h_best['k']
#     err = h_best['rel_error']
#     return lam, k, err


def find_lcurve_corner_triangle(r, x):
    """
    Triangle-Method Corner Finder for an L-curve (Castellanos et al. 2002).

    Arguments
    ---------
    r : array-like, shape (n,)
        The residual norms ‖r_k‖ for k = 1..n (must be positive).
    x : array-like, shape (n,)
        The solution norms (penalty) ‖x_k‖ for k = 1..n (must be positive).

    Returns
    -------
    corner_idx : int
        The 0-based index k* ∈ [0, n-1] corresponding to the corner point of 
        the L-curve.
        
    References
    ----------
    [1] J. L. Castellanos, S. Gómez, and V. Guerra, “The triangle method for 
    finding the corner of the L-curve,” Applied Numerical Mathematics, 
    vol. 43, no. 4, pp. 359-373, Dec. 2002, doi: 10.1016/S0168-9274(01)00179-9.
    """
    r = np.asarray(r, dtype=float)
    x = np.asarray(x, dtype=float)
    if r.shape != x.shape:
        raise ValueError("`r` and `x` must have the same length.")
    n = len(r)
    if n < 3:
        return 0  # trivial
    
    # 1) Log–log transform
    lr = np.log(r)
    lx = np.log(x)

    # 2) Affine‐scale each coordinate into the interval [-10, 10]
    lnew, unew = -10.0, 10.0
    lr_min, lr_max = lr.min(), lr.max()
    lx_min, lx_max = lx.min(), lx.max()
    r_new = (lr - lr_min) / (lr_max - lr_min) * (unew - lnew) + lnew
    x_new = (lx - lx_min) / (lx_max - lx_min) * (unew - lnew) + lnew

    # 3) Triangle‐method loops
    cte = np.cos(7 * np.pi / 8)  # threshold for sufficiently sharp corner
    cos_max = -2.0               # initial minimum cosine
    corner = n - 1               # default to last index if none found
    C = np.array([r_new[-1], x_new[-1]])  # fixed "C" vertex at last point

    for k in range(0, n - 2):
        B = np.array([r_new[k], x_new[k]])
        for j in range(k, n - 2):
            A = np.array([r_new[j + 1], x_new[j + 1]])
            BA = B - A
            AC = A - C

            # cosine of angle at A between vectors BA and AC
            dot = BA.dot(-AC)
            denom = np.linalg.norm(BA) * np.linalg.norm(AC)
            if denom == 0:
                continue
            cos_val = dot / denom

            # oriented area of triangle (BA, AC)
            area = 0.5 * (BA[0] * AC[1] - BA[1] * AC[0])

            # check sharpness and concavity
            if cos_val > cte and cos_val > cos_max and area < 0:
                corner = j + 1
                cos_max = cos_val

    return corner


# def find_lambda_lcurve_max_curvature(lambdas, feature_counts, errors,
#                                      log_k=False, log_e=False,
#                                      smooth_method=None,
#                                      window_length=7,
#                                      polyorder=3):
#     """
#     Identify the regularization parameter λ* on an L-curve (feature_counts vs errors)
#     by finding the point of maximum discrete curvature, with optional smoothing.

#     Parameters
#     ----------
#     lambdas : array-like, shape (N,)
#         The grid of λ values corresponding to each point on the L-curve.
#     feature_counts : array-like, shape (N,)
#         Number of selected features (k) at each λ.
#     errors : array-like, shape (N,)
#         Reconstruction error (E) at each λ.
#     log_k : bool, default False
#         If True, compute curvature in log(k) space.
#     log_e : bool, default False
#         If True, compute curvature in log(E) space.
#     smooth_method : {'savgol', None}, default None
#         If 'savgol', apply Savitzky-Golay filter.
#     window_length : int, default 7
#         Window length for Savitzky-Golay filter (must be odd and >= polyorder+2).
#     polyorder : int, default 3
#         Polynomial order for Savitzky-Golay filter (must be < window_length).

#     Returns
#     -------
#     lambda_star : float
#         The λ corresponding to the point of maximum curvature.
#     idx_star : int
#         Index of the chosen point in the input arrays.
#     curvatures : ndarray, shape (N,)
#         Discrete curvature values (NaN at endpoints).
#     """
#     lam = np.asarray(lambdas, dtype=float)
#     k   = np.asarray(feature_counts, dtype=float)
#     e   = np.asarray(errors, dtype=float)

#     # Optional smoothing
#     if smooth_method == 'savgol':
#         if window_length % 2 == 0:
#             window_length += 1  # ensure odd
#         if window_length <= polyorder:
#             raise ValueError("window_length must be > polyorder")
#         k = savgol_filter(k, window_length=window_length, polyorder=polyorder, mode='interp')
#         e = savgol_filter(e, window_length=window_length, polyorder=polyorder, mode='interp')

#     # Optional log‐scale
#     eps = 1e-16
#     if log_k:
#         k = np.log(k + eps)
#     if log_e:
#         e = np.log(e + eps)

#     N = len(lam)
#     if N < 3:
#         raise ValueError("Need at least 3 points to compute curvature")

#     # Compute discrete curvature via triangle area
#     curvatures = np.full(N, np.nan, dtype=float)
#     for i in range(1, N-1):
#         x1, y1 = k[i-1], e[i-1]
#         x2, y2 = k[i],   e[i]
#         x3, y3 = k[i+1], e[i+1]
#         a = np.hypot(x2 - x1, y2 - y1)
#         b = np.hypot(x3 - x2, y3 - y2)
#         c = np.hypot(x3 - x1, y3 - y1)
#         if a * b * c != 0:
#             area = abs((x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)) / 2.0
#             curvatures[i] = 4.0 * area / (a * b * c)
#         else:
#             curvatures[i] = 0.0

#     # Pick λ at max curvature
#     idx_star = int(np.nanargmax(curvatures))
#     lambda_star = lam[idx_star]

#     return lambda_star, idx_star, curvatures