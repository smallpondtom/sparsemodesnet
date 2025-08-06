"""
Simple neural network to learn an advecting Gaussian pulse.
"""

# Modules
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Gaussian pulse data generate (OPTIONAL)
def gaussian_pulse(x: np.ndarray, pulse_width: float = 2.0e-4, 
                   pulse_shift: float = 1.0e-1) -> np.ndarray:
    return (1 / np.sqrt(pulse_width * np.pi) * 
            np.exp(-((x - pulse_shift)**2) / pulse_width))

def generate_advecting_pulse(
    pulse_width: float = 5.0e-4,
    pulse_shift: float = 1.0e-1,
    speed: float = 5.0,
    final_time: float = 0.15,
    n_time_samples: int = 1000,
    n_space_samples: int = 1024,
    x_min: float = 0.0,
    x_max: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Create spatial and temporal grids
    x = np.linspace(x_min, x_max, n_space_samples, dtype=np.float64)
    t = np.linspace(0, final_time, n_time_samples, dtype=np.float64)
    
    # Initialize data matrix
    data_matrix = np.zeros((n_space_samples, n_time_samples), dtype=np.float64)
    
    # Generate pulse at each time step
    for i, ti in enumerate(t):
        # Pulse position at time ti
        pulse_position = pulse_shift + speed * ti
        
        # Handle periodic boundary conditions (optional)
        # Uncomment if you want the pulse to wrap around
        # pulse_position = pulse_position % (x_max - x_min)
        
        # Generate pulse at current time
        data_matrix[:, i] = gaussian_pulse(x, pulse_width, pulse_position)
    return data_matrix, x, t


# MLP
class MLP(nn.Module):
    def __init__(self, hidden_units=[350, 400], dropout=0.1, bias=False):
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_prob = dropout
        self.bias = bias
        self.layers = None
        self.dropout = nn.Dropout(self.dropout_prob)

    def initialize(self, input_dim, mapping_dim, output_dim, Ubar):
        # Store POD basis
        self.register_buffer('Ubar', torch.tensor(Ubar, dtype=torch.float64))
        assert Ubar.shape[1] == mapping_dim
        
        # Build sequence of linear layers
        sizes = [input_dim] + self.hidden_units + [mapping_dim]
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1], bias=self.bias, dtype=torch.float64)
            for i in range(len(sizes) - 1)
        ])
        
        # Initialize projection matrix
        self.projection = nn.Parameter(
            torch.randn(mapping_dim, output_dim, dtype=torch.float64) * 0.01
        )

    def orthogonalize_projection(self):
        """Apply Gram-Schmidt orthogonalization to ensure projection ⊥ Ur"""
        if self.Ur is not None:
            with torch.no_grad():
                # Project out the POD basis components
                # P_orth = P - Ur @ (Ur.T @ P)
                proj_on_Ur = self.Ur @ (self.Ur.T @ self.projection.T)  # (d, mapping_dim)
                self.projection.data = (self.projection.T - proj_on_Ur).T

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        # Final projection
        x = x @ self.projection @ self.Ubar.T 
        return x


# Main
if __name__== "__main__":
    # # Load Gaussian pulse data
    # X = np.load('X_pulse.npy')                      # original data (d, n)
    # U = np.load('X_pulse_svd.npz')['U']             # POD basis (d, n)
    # selected_modes = np.load('selected_modes.npy')  # selected r modes (r,) by GreedyQM
    # Ur = U[:, selected_modes]                       # reduced POD basis (d, r)
    # Z = np.load('Z_pulse.npy')                      # reduced data (r, n)

    # Generating the data instead and preprocessing (centering + normalizing)
    X, xspan, tspan = generate_advecting_pulse()
    X_mean = np.mean(X, axis=1, keepdims=True)
    X = X - X_mean  # Centering
    X_max = np.max(X, axis=1, keepdims=True)
    X_min = np.min(X, axis=1, keepdims=True)
    X = (X - X_min) / (X_max - X_min)  # Normalization to [0,1]
    U = np.linalg.svd(X, full_matrices=False)[0]
    selected_modes = np.load('selected_modes.npy')  
    Ur = U[:, selected_modes]
    Z = Ur.T @ X

    # Regression || x - Ur @ z - fnn(z) ||^2_2
    # Compute B = X - Ur @ Z
    B = X - Ur @ Z

    # Initialize MLP
    d, r = Ur.shape
    p = 125
    mlp = MLP(hidden_units=[500, 500, 500], dropout=0.0, bias=True)
    mlp.initialize(input_dim=r, mapping_dim=p, output_dim=d, Ur=Ur)

    # Train MLP
    num_epochs = 10000
    lr = 1e-2
    optimizer = optim.Adam(mlp.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.8, patience=100)

    # Convert data to torch tensors 
    Z_tensor = torch.tensor(Z.T, dtype=torch.float64)  # (n, r)
    B_tensor = torch.tensor(B.T, dtype=torch.float64)  # (n, d)

    mlp.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        # Forward pass
        fnn_output = mlp(Z_tensor)  
        # Compute loss
        loss = mse_loss(fnn_output, B_tensor)
        # Backward pass
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"LR: {optimizer.param_groups[0]['lr']:.4e}, "
                  f"Loss: {loss.item():.6e}")

        lr_scheduler.step(loss)

    # Compute reconstruction error 
    mlp.eval()
    with torch.no_grad():
        fnn_output = mlp(Z_tensor)
        fnn_output = fnn_output.numpy() 
    recon_err = np.linalg.norm(B - fnn_output.T, ord='fro') / np.linalg.norm(X, ord='fro')
    print("\n")
    print(f"Relative reconstruction error: {recon_err:.6e}")
    print("GreedyQM's relative reconstruction error is on the order of 1e-9 ~ 1e-8.")