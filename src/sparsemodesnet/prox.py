"""
@author: Tomoki Koike
@date: 5/28/2025
@description: This code is the direct copy of the prox.py file of LassoNet
from (https://github.com/lasso-net/lassonet/blob/master/lassonet/prox.py). It 
contains the hierarchical proximal optimization function used in LassoNet. For 
more information, refer to the origin paper: https://arxiv.org/abs/1907.12207
"""

import torch
from torch.nn import functional as F

def soft_threshold(l: float, x: torch.Tensor) -> torch.Tensor:
    """
    Apply soft thresholding to the input tensor x with threshold l.
    
    Args:
        l (float): The threshold value.
        x (torch.Tensor): The input tensor to apply soft thresholding on.
        
    Returns:
        torch.Tensor: The tensor after applying soft thresholding.
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - l) if l > 0 else x

def sign_binary(x: torch.Tensor) -> torch.Tensor:
    """
    Convert the input tensor x to a binary tensor with values -1 or 1.
    
    Args:
        x (torch.Tensor): The input tensor to convert.
        
    Returns:
        torch.Tensor: The binary tensor with values -1 or 1.
    """
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)
