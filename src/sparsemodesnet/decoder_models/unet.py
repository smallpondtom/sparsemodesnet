import torch
import torch.nn as nn
from .abstract_decoder import AbstractDecoder

"""
This was sampled from the original work of the following reference: 
https://github.com/pyshred-dev/pyshred/blob/main/pyshred/models/decoder_models/unet_model.py
"""

class UNET(AbstractDecoder):
    """
    1D U-Net for spatio-temporal reconstruction.
    
    Takes POD coefficients and reconstructs spatial structure using
    1D convolutions that respect spatial locality.
    """
    
    def __init__(self, conv1: int = 256, conv2: int = 1024, bias: bool = False):
        super().__init__()
        self.c1 = conv1
        self.c2 = conv2
        self.bias = bias
        
        # Will be initialized later
        self.input_projection = None
        self.conv1 = None
        self.conv2 = None
        self.conv3 = None
        self.output_projection = None
        self.first_layer = None

    def initialize(self, input_dim, output_dim):
        """
        Initialize for POD coefficient -> spatial field reconstruction.
        
        Parameters
        ----------
        input_dim : int
            Number of POD modes (r)
        output_dim : int
            Spatial dimension (d)
        """
        super().initialize(input_dim)
        
        # Project POD coefficients to a spatial-like representation
        # We'll treat the POD coefficients as "features" at each spatial point
        spatial_size = min(output_dim, 512)  # Reasonable spatial resolution
        
        # Linear projection: r POD modes -> spatial features
        self.first_layer = nn.Linear(input_dim, spatial_size, bias=self.bias)
        
        # 1D convolutions on spatial dimension
        self.conv1 = nn.Conv1d(1, self.c1, kernel_size=3, padding=1, bias=self.bias)
        self.conv2 = nn.Conv1d(self.c1, self.c2, kernel_size=3, padding=1, bias=self.bias)
        self.conv3 = nn.Conv1d(self.c2, 1, kernel_size=3, padding=1, bias=self.bias)
        
        # Final projection to correct spatial dimension
        self.output_projection = nn.Linear(spatial_size, output_dim, bias=self.bias)
        
        self.gelu = nn.LeakyReLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: POD coefficients -> spatial field
        
        Parameters
        ----------
        z : torch.Tensor
            POD coefficients of shape (batch_size, r)
            
        Returns
        -------
        torch.Tensor
            Spatial field of shape (batch_size, d)
        """
        # Project to spatial representation
        x = self.first_layer(z)  # (batch, spatial_size)
        
        # Reshape for 1D convolution: treat as single channel spatial signal
        x = x.unsqueeze(1)  # (batch, 1, spatial_size)
        
        # Apply 1D convolutions
        x = self.gelu(self.conv1(x))  # (batch, c1, spatial_size)
        x = self.gelu(self.conv2(x))  # (batch, c2, spatial_size)
        x = self.conv3(x)             # (batch,  1, spatial_size)
        
        # Remove channel dimension and project to final spatial dimension
        x = x.squeeze(1)              # (batch, spatial_size)
        x = self.output_projection(x) # (batch, d)
        
        return x

    @property
    def model_name(self):
        return "UNET"