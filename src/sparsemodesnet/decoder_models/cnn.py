from .abstract_decoder import AbstractDecoder
import torch.nn as nn

class SpatialCNN(AbstractDecoder):
    """
    CNN for spatial reconstruction: R^s → R^d
    Maps POD coefficients to nonlinear correction in original space
    """
    def __init__(self, hidden_units=[64, 3], bias=False, dropout=0.1):
        """
        Parameters:
        -----------
        hidden_units : list of int
            CNN architecture specification: [num_filters, kernel_size, ...]
            First element is number of filters, second is kernel size,
            additional elements can specify more conv layers
        bias : bool
            Whether to use bias in conv layers
        dropout : float
            Dropout probability applied after each conv layer
        """
        super().__init__()
        self.hidden_units = hidden_units
        self.bias = bias
        self.dropout_prob = dropout
        
        # Will be initialized later
        self.conv_layers = None
        self.projection = None
        
    def initialize(self, input_dim, output_dim):
        """
        Initialize the CNN with input and output sizes.

        Parameters:
        -----------
        input_dim : int
            Size of the input features (s - POD dimension).
        output_dim : int
            Size of the output features (d - spatial dimension).
        """
        super().initialize(input_dim)
        
        self.input_dim = input_dim    # s (POD dimension)
        self.output_dim = output_dim  # d (original spatial dimension)
        
        assert len(self.hidden_units) >= 2, "Need [num_filters, kernel_size, ...]"
        
        num_filters = self.hidden_units[0]
        kernel_size = self.hidden_units[1]
        padding = (kernel_size - 1) // 2
        
        # Strategy: Treat POD coefficients as a 1D "spatial" signal
        # Apply 1D convolutions to extract nonlinear patterns
        # Then map to full spatial dimension
        
        layers = []
        
        # Input layer: 1 channel (treating s coefficients as 1D signal)
        first_conv = nn.Conv1d(1, num_filters, kernel_size, padding=padding, bias=self.bias)
        layers.extend([
            first_conv,
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_prob)
        ])
        
        current_filters = num_filters
        
        # Additional conv layers
        for i in range(2, len(self.hidden_units)):
            next_filters = self.hidden_units[i]
            # Apply activation, dropout, & normalization for all but the final layer
            if i == len(self.hidden_units) - 1:
                layers.extend([
                    nn.Conv1d(current_filters, next_filters, kernel_size, 
                            padding=padding, bias=self.bias),
                ])
            else:
                layers.extend([
                    nn.Conv1d(current_filters, next_filters, kernel_size, 
                            padding=padding, bias=self.bias),
                    nn.BatchNorm1d(next_filters),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=self.dropout_prob)
                ])

            current_filters = next_filters
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final projection to spatial dimension d
        # Need to handle the dimension mismatch: (batch, filters, s) → (batch, d)
        # Fully convolutional (better for spatial patterns)
        self.projection = nn.Sequential(
            nn.Conv1d(current_filters, output_dim, 
                      kernel_size=1, 
                      bias=self.bias),  # (batch, filters, s) → (batch, d, s)
            nn.AdaptiveAvgPool1d(1),    # (batch, d, s)       → (batch, d, 1)
            nn.Flatten()                # (batch, d, 1)       → (batch, d)
        )
    
    def forward(self, pod_coeffs):
        """
        Forward pass through CNN.
        
        Args:
            pod_coeffs: POD coefficients (batch_size, s)
            
        Returns:
            spatial_correction: Nonlinear correction (batch_size, d)
        """
        # Reshape for 1D convolution: (batch, s) → (batch, 1, s)
        x = pod_coeffs.unsqueeze(1)  # Treat POD coeffs as 1D signal
        
        # Apply convolutions
        x = self.conv_layers(x)      # (batch, filters, s)
        
        # Project to spatial dimension
        x = self.projection(x)
        
        return x

    @property
    def model_name(self):
        return "SpatialCNN"



# import torch
# import torch.nn as nn

# class SpatialCNN(nn.Module):
#     """
#     CNN for spatial reconstruction: R^s → R^d
#     Maps POD coefficients to nonlinear correction in original space
#     """
#     def __init__(self, input_dim, output_dim, hidden_units, bias=False):
#         super(SpatialCNN, self).__init__()
        
#         self.input_dim = input_dim    # s (POD dimension)
#         self.output_dim = output_dim  # d (original spatial dimension)
        
#         assert len(hidden_units) >= 2, "Need [num_filters, kernel_size, ...]"
        
#         num_filters = hidden_units[0]
#         kernel_size = hidden_units[1]
#         padding = (kernel_size - 1) // 2
        
#         # Strategy: Treat POD coefficients as a 1D "spatial" signal
#         # Apply 1D convolutions to extract nonlinear patterns
#         # Then map to full spatial dimension
        
#         layers = []
        
#         # Input layer: 1 channel (treating s coefficients as 1D signal)
#         layers.extend([
#             nn.Conv1d(1, num_filters, kernel_size, padding=padding, bias=bias),
#             nn.BatchNorm1d(num_filters),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.1)
#         ])
        
#         current_filters = num_filters
        
#         # Additional conv layers
#         for i in range(2, len(hidden_units)):
#             next_filters = hidden_units[i]
#             layers.extend([
#                 nn.Conv1d(current_filters, next_filters, kernel_size, 
#                          padding=padding, bias=bias),
#                 nn.BatchNorm1d(next_filters),
#                 nn.ReLU(inplace=True),
#                 nn.Dropout(p=0.1)
#             ])
#             current_filters = next_filters
        
#         self.conv_layers = nn.Sequential(*layers)
        
#         # Final projection to spatial dimension d
#         # Need to handle the dimension mismatch: (batch, filters, s) → (batch, d)
#         # Fully convolutional (better for spatial patterns)
#         self.projection = nn.Sequential(
#             nn.Conv1d(current_filters, output_dim, 
#                       kernel_size=1, 
#                       bias=bias),     # (batch, filters, s) → (batch, d, s)
#             nn.AdaptiveAvgPool1d(1),  # (batch, d, s)       → (batch, d, 1)
#             nn.Flatten()              # (batch, d, 1)       → (batch, d)
#         )
    
#     def forward(self, pod_coeffs):
#         """
#         Args:
#             pod_coeffs: POD coefficients (batch_size, s)
            
#         Returns:
#             spatial_correction: Nonlinear correction (batch_size, d)
#         """
#         # Reshape for 1D convolution: (batch, s) → (batch, 1, s)
#         x = pod_coeffs.unsqueeze(1)  # Treat POD coeffs as 1D signal
        
#         # Apply convolutions
#         x = self.conv_layers(x)      # (batch, filters, s)
        
#         # Project to spatial dimension
#         x = self.projection(x)
        
#         return x
