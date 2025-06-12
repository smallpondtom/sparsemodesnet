import numpy as np

def generate_kse_data(nx=256, nt=200, L=32*np.pi, t_max=150.0):
    """
    Simulate 1D Kuramoto-Sivashinsky equation u_t + u*u_x + u_xx + u_xxxx = 0 using ETDRK4.
    Uses Exponential Time Differencing Runge-Kutta 4th order for high accuracy and stability.
    Returns data ∈ R^{nx x nt}.
    """
    # Spatial grid setup (normalized to [0,1] then scaled)
    x = np.arange(1, nx+1, dtype=np.float64) / nx
    x_scaled = x * L  # Scale to actual domain [0, L]
    
    # Time discretization
    h = t_max / (nt - 1)  # Time step
    
    # Initial condition
    u = np.cos(x/16) * (1 + np.sin(x/16))
    v = np.fft.fft(u)
    
    # Wave numbers
    k = np.concatenate(
        (
            np.arange(0, nx//2, dtype=np.float64), 
            [0.0], 
            np.arange(-nx//2+1, 0, dtype=np.float64)
        )
    ) / 16
    
    # Linear operator for KS equation: L = k^2 - k^4
    L_op = k**2 - k**4
    
    # ETDRK4 coefficients
    E = np.exp(h * L_op)
    E_2 = np.exp(h * L_op / 2)
    
    # Contour integral parameters for ETDRK4 coefficients
    Mcnt = 16
    r = np.exp(1j * np.pi * (np.arange(1, Mcnt+1, dtype=np.float64) - 0.5) / Mcnt)
    LR = h * np.outer(L_op, np.ones(Mcnt, dtype=np.float64)) + np.outer(np.ones(nx), r)
    
    # ETDRK4 coefficients computed via contour integrals
    Q  = h * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=1))
    
    # Handle potential division by zero at k=0
    zero_idx = np.where(np.abs(L_op) < 1e-14)[0]
    if len(zero_idx) > 0:
        Q[zero_idx]  = h
        f1[zero_idx] = h
        f2[zero_idx] = h/2
        f3[zero_idx] = h
    
    # Nonlinear operator
    g = -0.5j * k
    
    # Storage for solution
    uu = np.zeros((nx, nt), dtype=np.float64)
    uu[:, 0] = u
    
    # Time stepping with ETDRK4
    for n in range(1, nt):
        # Stage 1
        Nv = g * np.fft.fft(np.real(np.fft.ifft(v))**2)
        a  = E_2 * v + Q * Nv
        
        # Stage 2
        Na = g * np.fft.fft(np.real(np.fft.ifft(a))**2)
        b  = E_2 * v + Q * Na
        
        # Stage 3
        Nb = g * np.fft.fft(np.real(np.fft.ifft(b))**2)
        c  = E_2 * a + Q * (2*Nb - Nv)
        
        # Stage 4
        Nc = g * np.fft.fft(np.real(np.fft.ifft(c))**2)
        
        # Final update
        v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        
        # Store solution
        u = np.real(np.fft.ifft(v))
        uu[:, n] = u
    
    # Time array
    t = np.linspace(0, t_max, nt, dtype=np.float64)
    
    return uu, x_scaled, t

