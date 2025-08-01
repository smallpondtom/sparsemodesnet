import torch
import torch.nn as nn

class SpatialCNN(nn.Module):
    """
    CNN for spatial reconstruction: R^s → R^d
    Maps POD coefficients to nonlinear correction in original space
    """
    def __init__(self, input_dim, output_dim, hidden_units):
        super(SpatialCNN, self).__init__()
        
        self.input_dim = input_dim    # s (POD dimension)
        self.output_dim = output_dim  # d (original spatial dimension)
        
        assert len(hidden_units) >= 2, "Need [num_filters, kernel_size, ...]"
        
        num_filters = hidden_units[0]
        kernel_size = hidden_units[1]
        padding = (kernel_size - 1) // 2
        
        # Strategy: Treat POD coefficients as a 1D "spatial" signal
        # Apply 1D convolutions to extract nonlinear patterns
        # Then map to full spatial dimension
        
        layers = []
        
        # Input layer: 1 channel (treating s coefficients as 1D signal)
        layers.extend([
            nn.Conv1d(1, num_filters, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(num_filters),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1)
        ])
        
        current_filters = num_filters
        
        # Additional conv layers
        for i in range(2, len(hidden_units)):
            next_filters = hidden_units[i]
            layers.extend([
                nn.Conv1d(current_filters, next_filters, kernel_size, 
                         padding=padding, bias=False),
                nn.BatchNorm1d(next_filters),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1)
            ])
            current_filters = next_filters
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final projection to spatial dimension d
        # Need to handle the dimension mismatch: (batch, filters, s) → (batch, d)
        # Fully convolutional (better for spatial patterns)
        self.projection = nn.Sequential(
            nn.Conv1d(current_filters, output_dim, 
                      kernel_size=1, 
                      bias=False),    # (batch, filters, s) → (batch, d, s)
            nn.AdaptiveAvgPool1d(1),  # (batch, d, s) → (batch, d, 1)
            nn.Flatten()              # (batch, d, 1) → (batch, d)
        )
    
    def forward(self, pod_coeffs):
        """
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
