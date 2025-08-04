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
        """
        Parameters:
        -----------
        hidden_units : list of int
            Sizes of each hidden layer in sequence. e.g. [256, 512]
        dropout : float
            Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.dropout_prob = dropout
        self.bias = bias
        self.layers = None
        self.dropout = nn.Dropout(self.dropout_prob)

    def initialize(self, input_dim, output_dim):
        """
        Initialize the MLP with input and output sizes.

        Parameters:
        -----------
        input_dim : int
            Size of the input features.
        output_dim : int
            Size of the output features.
        """
        # Build sequence of linear layers: input->hidden...->output
        sizes = [input_dim] + self.hidden_units + [output_dim]
        # Create a list of linear layers for each hidden layer
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1], bias=self.bias, dtype=torch.float64)
            for i in range(len(sizes) - 1)
        ])

    def forward(self, x):
        """
        Forward pass through MLP.

        Applies ReLU + dropout after each hidden layer; no activation on output.
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation & dropout for all but the final layer
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x



# Main
if __name__== "__main__":
    # Load Gaussian pulse data
    X = np.load('X_pulse.npy')                      # original data (d, n)
    U = np.load('X_pulse_svd.npz')['U']             # POD basis (d, n)
    selected_modes = np.load('selected_modes.npy')  # selected r modes (r,) by GreedyQM
    Ur = U[:, selected_modes]                       # reduced POD basis (d, r)
    Z = np.load('Z_pulse.npy')                      # reduced data (r, n)

    # # Generating the data instead
    # X, xspan, tspan = generate_advecting_pulse()
    # U = np.linalg.svd(X, full_matrices=False)[0] 
    # Ur = U[:, selected_modes]
    # Z = Ur.T @ X

    # Regression || x - Ur @ z - fnn(z) ||^2_2
    # Compute B = X - Ur @ Z
    B = X - Ur @ Z

    # Initialize MLP
    d, r = Ur.shape
    mlp = MLP(hidden_units=[2000, 2000], dropout=0.0, bias=True)
    mlp.initialize(input_dim=r, output_dim=d)

    # Train MLP
    num_epochs = 1000
    lr = 1e-3
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
        fnn_output = mlp(torch.tensor(Z.T, dtype=torch.float64))
        fnn_output = fnn_output.numpy() 
    recon_err = np.linalg.norm(fnn_output - B.T, ord='fro') / np.linalg.norm(X, ord='fro')
    print("\n")
    print(f"Reconstruction error: {recon_err:.6e}")
    print("GreedyQM reconstruction error is on the order of 1e-9 ~ 1e-8.")