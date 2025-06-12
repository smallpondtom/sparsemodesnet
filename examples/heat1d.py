import numpy as np

# def generate_heat_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0):
#     """
#     Simulate 1D heat equation u_t = alpha * u_xx on [0, x_max], t ∈ [0, t_max].
#     Dirichlet BCs (u=0 at boundaries). Returns data ∈ R^{nx x nt}.
#     """
#     x = np.linspace(0, x_max, nx)
#     dx = x[1] - x[0]
#     dt = t_max / (nt - 1)
#     if dt > dx**2 / (2 * alpha):
#         print("Warning: dt may be too large for stability (heat eq).")

#     u = np.exp(-((x - x_max/2)**2 * 50.0))
#     snapshots = [u.copy()]

#     for k in range(1, nt):
#         u_new = u.copy()
#         u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[0:-2] - 2*u[1:-1] + u[2:])
#         u_new[0] = 0.0
#         u_new[-1] = 0.0
#         snapshots.append(u_new)
#         u = u_new

#     data = np.column_stack(snapshots)  # shape (nx, nt)
#     return data, x, np.linspace(0, t_max, nt)

def generate_heat_data(
    nx=100,
    nt=200,
    alpha=0.01,
    x_max=1.0,
    t_max=1.0,
    method='crank-nicolson'  # 'explicit', 'implicit', or 'crank-nicolson'
):
    """
    Simulate 1D heat equation u_t = α u_xx on [0, x_max], t∈[0, t_max].
    Dirichlet BCs (u=0 at boundaries).
    method: time-stepping scheme
      - 'explicit'         : forward Euler (CFL r ≤ 0.5 required)
      - 'implicit'         : backward Euler (unconditionally stable)
      - 'crank-nicolson'   : θ=1/2 scheme (2nd order in time, unconditionally stable)
    Returns data ∈ R^{nx x nt}, grid x, times t.
    """
    import numpy as np

    x = np.linspace(0, x_max, nx, dtype=np.float64)
    dx = x[1] - x[0]
    t = np.linspace(0, t_max, nt, dtype=np.float64)
    dt = t[1] - t[0]
    r = alpha * dt / dx**2

    if method == 'explicit' and r > 0.5:
        print(f"Warning: explicit scheme unstable (r={r:.3f} > 0.5).")

    # Prebuild implicit matrices for interior points 1..nx-2
    N = nx - 2
    if method in ('implicit', 'crank-nicolson'):
        # second‐difference matrix D_int
        diag = -2.0 * np.ones(N)
        off  = 1.0 * np.ones(N - 1)
        D = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        I = np.eye(N)
        if method == 'implicit':
            A = I - r * D
            B = None
        else:  # crank-nicolson
            A = I - 0.5 * r * D
            B = I + 0.5 * r * D

    # initial condition
    u = np.exp(-((x - x_max/2)**2) * 50.0)
    u[0] = u[-1] = 0.0
    snapshots = [u.copy()]

    for k in range(1, nt):
        if method == 'explicit':
            u_new = u.copy()
            u_new[1:-1] = (
                u[1:-1]
                + alpha * dt / dx**2 * (u[0:-2] - 2*u[1:-1] + u[2:])
            )
            u_new[0] = u_new[-1] = 0.0

        else:
            # solve for interior points
            u_int = u[1:-1]
            if method == 'implicit':
                u_int_new = np.linalg.solve(A, u_int)
            else:  # crank-nicolson
                rhs = B.dot(u_int)
                u_int_new = np.linalg.solve(A, rhs)

            u_new = u.copy()
            u_new[1:-1] = u_int_new
            u_new[0] = u_new[-1] = 0.0

        snapshots.append(u_new)
        u = u_new

    data = np.column_stack(snapshots).astype(np.float64)  # shape (nx, nt)
    return data, x, t