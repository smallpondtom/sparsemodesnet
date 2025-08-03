#%% Imports and data generation (same as before)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Use your existing pulse generator
from examples.pulse import generate_advecting_pulse

device = 'cpu'
torch.manual_seed(42)
np.random.seed(42)

# Parameters
n_grids = 2**10
n_time_samples = 1000
r_latent = 15        # latent dimension of the autoencoder
batch_size = 100
num_epochs = 500     # reduce/raise as needed
learning_rate = 1e-2

# Generate advecting pulse data
X_pulse, xspan_p, tspan_p = generate_advecting_pulse(
    pulse_width=5.0e-4,
    pulse_shift=0.1,
    speed=5.0,
    final_time=0.15,
    n_time_samples=n_time_samples,
    n_space_samples=n_grids
)
# Ensure double precision
X_pulse = X_pulse.astype(np.float64)

#%% Data normalization (same as your previous code)
X_pulse_mean = X_pulse.mean(axis=1, keepdims=True)
X_pulse_ = X_pulse - X_pulse_mean
X_pulse_min, X_pulse_max = X_pulse_.min(axis=1), X_pulse_.max(axis=1)
X_pulse_shift = X_pulse_min.reshape(-1, 1)
X_pulse_scale = (X_pulse_max - X_pulse_min).reshape(-1, 1)
X_pulse_norm = (X_pulse_ - X_pulse_shift) / X_pulse_scale

# Convert to tensors; note we use float64
X_pulse_norm_tensor = torch.tensor(X_pulse_norm, dtype=torch.float64)
# The autoencoder will take each time snapshot as a single training example,
# so transpose to shape (n_samples, d)
dataset = TensorDataset(X_pulse_norm_tensor.T)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#%% Define the autoencoder network
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(512, 256, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(256, 128, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(128, latent_dim, dtype=torch.float64),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(128, 256, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(256, 512, dtype=torch.float64),
            nn.LeakyReLU(),
            nn.Linear(512, input_dim, dtype=torch.float64),
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        latent = self.encoder(x.double())
        reconstructed = self.decoder(latent)
        return reconstructed
    
class ConvAutoencoder(nn.Module):
    def __init__(self, input_len, latent_len):
        super().__init__()
        # 1D input of shape (batch, channels=1, length=input_len)
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4, dtype=torch.float64),  # downsample by 2
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=7, stride=2, padding=3, dtype=torch.float64),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, dtype=torch.float64),
            nn.ReLU()
        )
        # Flatten and map to latent
        # Compute the length after downsampling: input_len // 8 (three stride‑2 conv layers)
        enc_len = input_len // 8
        self.fc_enc = nn.Linear(64 * enc_len, latent_len, dtype=torch.float64)
        self.fc_dec = nn.Linear(latent_len, 64 * enc_len, dtype=torch.float64)
        # Decoder mirrors the encoder with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1, dtype=torch.float64),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=7, stride=2, padding=3, output_padding=1, dtype=torch.float64),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=9, stride=2, padding=4, output_padding=1, dtype=torch.float64),
            # Optional tanh/sigmoid depending on the data range; here we leave it linear
        )

    def forward(self, x):
        # x: (batch, input_dim), reshape to (batch, 1, input_dim)
        x = x.double().unsqueeze(1)
        enc = self.encoder(x)
        latent = self.fc_enc(enc.flatten(start_dim=1))
        dec_flat = self.fc_dec(latent)
        dec = dec_flat.view(x.size(0), 64, x.size(2) // 8)
        recon = self.decoder(dec)
        return recon.squeeze(1)

#%% Instantiate the model
input_dim = X_pulse_norm.shape[0]  # number of spatial grid points
# autoencoder = Autoencoder(input_dim=input_dim, latent_dim=r_latent).to(device)
autoencoder = ConvAutoencoder(input_len=input_dim, latent_len=r_latent).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

#%% Training loop
autoencoder.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for (batch,) in dataloader:
        batch = batch.to(device)           # (batch_size, input_dim)
        optimizer.zero_grad()
        outputs = autoencoder(batch)
        loss = criterion(outputs, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= len(dataloader)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1:4d}, Avg Loss = {epoch_loss:.4e}")

#%% Reconstruction and error computation
autoencoder.eval()
with torch.no_grad():
    # Reconstruct all samples
    X_recon_norm = autoencoder(X_pulse_norm_tensor.T.to(device)).cpu().numpy().T
    # Undo normalization
    X_recon = X_recon_norm * X_pulse_scale + X_pulse_shift + X_pulse_mean

rel_error = np.linalg.norm(X_recon - X_pulse) / np.linalg.norm(X_pulse)
print(f"Relative reconstruction error: {rel_error:.3e}")
# %%
