import torch
import torch.nn as nn
from .abstract_decoder import AbstractDecoder

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
        self.input_projection = nn.Linear(input_dim, spatial_size, bias=self.bias)
        self.first_layer = self.input_projection
        
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
        x = self.input_projection(z)  # (batch, spatial_size)
        
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

# from .abstract_decoder import AbstractDecoder
# import torch
# import torch.nn as nn

# """
# Reference: https://github.com/pyshred-dev/pyshred/blob/main/pyshred/models/decoder_models/unet_model.py
# """


# class UNET(AbstractDecoder):
#     """
#     1D U-Net style convolutional decoder for SHRED.
    
#     Uses a series of 1D convolutions to decode latent representations
#     back to the physical state space. Particularly suitable for 
#     spatially-structured data.

#     Parameters
#     ----------
#     conv1 : int, optional
#         Number of channels in first convolutional layer. Defaults to 256.
#     conv2 : int, optional
#         Number of channels in second convolutional layer. Defaults to 1024.

#     Attributes
#     ----------
#     c1 : int
#         First convolution layer channel count.
#     c2 : int
#         Second convolution layer channel count.
#     """
    
#     def __init__(self, conv1: int = 256, conv2: int = 1024, bias: bool = False):
#         """
#         Initialize the UNET decoder.

#         Parameters
#         ----------
#         conv1 : int, optional
#             Number of channels in first convolutional layer. Defaults to 256.
#         conv2 : int, optional
#             Number of channels in second convolutional layer. Defaults to 1024.
#         """
#         super().__init__()
#         self.c1 = conv1
#         self.c2 = conv2
#         self.bias = bias
#         # self.dropout = dropout


#     def initialize(self, input_dim, output_dim):
#         """
#         Initialize the UNET decoder with input and output sizes.

#         Parameters
#         ----------
#         input_dim : int
#             Size of the input latent features.
#         output_dim : int
#             Size of the output physical state.
#         """
#         super().initialize(input_dim)
#         self.conv1 = nn.Conv1d(input_dim, self.c1, kernel_size=2, padding=1, bias=self.bias)
#         self.conv2 = nn.Conv1d(self.c1, self.c2, kernel_size=4, padding=1, bias=self.bias)
#         self.conv3 = nn.Conv1d(self.c2, output_dim, kernel_size=2, padding=1, bias=self.bias)
#         self.gelu = nn.LeakyReLU()


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass through the UNET decoder.

#         Parameters
#         ----------
#         x : torch.Tensor
#             Input latent tensor of shape (batch_size, input_dim, sequence_length).

#         Returns
#         -------
#         torch.Tensor
#             Decoded output tensor of shape (batch_size, output_dim).
#         """
#         x = self.gelu(self.conv1(x))
#         x = self.gelu(self.conv2(x))
#         x = self.gelu(self.conv3(x))
#         x = x.permute(0, 2, 1)  # Change shape back to [batch_size, sequence_length, d_model]
#         x = torch.mean(x, dim=1)
#         return x

#     @property
#     def model_name(self):
#         """
#         Name of the decoder model.

#         Returns
#         -------
#         str
#             Returns "UNET".
#         """
#         return "UNET"