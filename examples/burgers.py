import numpy as np
from scipy.integrate import odeint

def generate_burgers_data(nx=100, nt=200, nu=0.01, x_max=1.0, t_max=0.5):
    """
    Simulate 1D viscous Burgers' equation u_t + u u_x = nu u_xx using FFT method.
    Uses spectral differentiation for high accuracy and stability.
    Returns data ∈ R^{nx x nt}.
    """
    
    # Spatial discretization
    dx = x_max / nx
    x = np.linspace(0, x_max, nx, endpoint=False, dtype=np.float64)  # Periodic domain
    
    # Temporal discretization
    t = np.linspace(0, t_max, nt, dtype=np.float64)
    
    # Wave number discretization for FFT
    k = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    
    # Initial condition - smooth Gaussian-like profile
    u0 = np.exp(-((x - x_max/2)**2) / (2 * (x_max/10)**2))
    
    # Define the Burgers system using FFT for spatial derivatives
    def burgers_system(u, t, k, nu):
        # Compute spatial derivatives in Fourier domain
        u_hat = np.fft.fft(u)
        u_hat_x = 1j * k * u_hat
        u_hat_xx = -k**2 * u_hat
        
        # Transform back to spatial domain
        u_x = np.fft.ifft(u_hat_x)
        u_xx = np.fft.ifft(u_hat_xx)
        
        # Burgers equation: u_t = nu*u_xx - u*u_x
        u_t = nu * u_xx - u * u_x
        return u_t.real
    
    # Solve the PDE system using adaptive ODE solver
    U = odeint(burgers_system, u0, t, args=(k, nu), mxstep=5000).T
    
    return U, x, t

