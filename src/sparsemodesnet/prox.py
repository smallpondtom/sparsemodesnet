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