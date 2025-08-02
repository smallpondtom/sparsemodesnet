from .abstract_decoder import AbstractDecoder
import torch.nn as nn
import torch.nn.functional as F

"""
Reference: https://github.com/pyshred-dev/pyshred/blob/main/pyshred/models/decoder_models/mlp_model.py
"""

class MLP(AbstractDecoder):
    """
    Flexible Multilayer Perceptron (MLP) implementation.

    A fully connected decoder that maps a low-dimensional latent space
    back to a high-dimensional state with user-defined hidden layers.
    """

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
        self.first_layer = None  # Expose first layer for proximal operations
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
        super().initialize(input_dim)

        # Build sequence of linear layers: input->hidden...->output
        sizes = [input_dim] + self.hidden_units + [output_dim]
        # Create a list of linear layers for each hidden layer
        self.layers = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1], bias=self.bias)
            for i in range(len(sizes) - 1)
        ])

        # Expose the first layer for compatibility with SparseModesNet
        if len(self.layers) > 0:
            self.first_layer = self.layers[0]

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

    @property
    def model_name(self):
        return "MLP"