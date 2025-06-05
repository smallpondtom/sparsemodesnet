import numpy as np

def generate_heat_data(nx=100, nt=200, alpha=0.01, x_max=1.0, t_max=1.0):
    """
    Simulate 1D heat equation u_t = alpha * u_xx on [0, x_max], t ∈ [0, t_max].
    Dirichlet BCs (u=0 at boundaries). Returns data ∈ R^{nx x nt}.
    """
    x = np.linspace(0, x_max, nx)
    dx = x[1] - x[0]
    dt = t_max / (nt - 1)
    if dt > dx**2 / (2 * alpha):
        print("Warning: dt may be too large for stability (heat eq).")

    u = np.exp(-((x - x_max/2)**2 * 50.0))
    snapshots = [u.copy()]

    for k in range(1, nt):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + alpha * dt / dx**2 * (u[0:-2] - 2*u[1:-1] + u[2:])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        snapshots.append(u_new)
        u = u_new

    data = np.column_stack(snapshots)  # shape (nx, nt)
    return data, x, np.linspace(0, t_max, nt)
