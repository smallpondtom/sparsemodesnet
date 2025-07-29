import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PiNetCCP(nn.Module):
    """
    Coupled CP Π-net (CCP)
    ----------------------
    Π-net via a coupled CP decomposition (CCP).
    Recursion (for n=2..N), Eq. (6) of [1]:
        x_n = (U_n z) ⊙ x_{n-1} + x_{n-1}
    Final output:  x_hat = C x_N + β
    
    Notes
    ----- 
    - if drop_constant=True, then β is dropped that is C's bias is dropped.
    - Due to the specific recursion, the linear part cannot be dropped.
    
    Reference
    ---------
    [1] G. G. Chrysos, S. Moschoglou, G. Bouritsas, J. Deng, Y. Panagakis, and
    S. Zafeiriou, “Deep Polynomial Neural Networks,” IEEE Transactions on 
    Pattern Analysis and Machine Intelligence, vol. 44, no. 8, pp. 4021-4034, 
    Aug. 2022, doi: 10.1109/TPAMI.2021.3058891.
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int,
                 drop_constant: bool = False,
                 normalize: None | str = None):
        """
        Arguments
        ---------
        in_dim : input dim
        out_dim : output dim
        inter_dim : intermediate rank
        poly_order : polynomial order
        drop_constant : whether to drop the constant term (bias in C)
        normalize : normalization method for U_n
            - None: no normalization
            - 'all': normalize all U_n
            - 'last': normalize only the highest/last order U_n
            - 'last2': normalize the last two highest order U_n
        """
        super().__init__()
        # Let s = in_dim, d = out_dim, k = inter_dim, N = poly_order
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N

        # U_n : s → k for n=1..N
        # Note: default initialization for Linear is Kaiming
        self.U = nn.ModuleList([
            nn.Linear(s, k, bias=False) 
            for _ in range(N)
        ])
        
        # Batch normalization for each U_n to prevent exploding gradients
        # due to the polynomial structure
        if normalize is not None:
            if normalize == 'all':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True)
                    for _ in range(N)
                ])
            elif normalize == 'last2':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n >= N-2 else nn.Identity(k)
                    for n in range(N)
                ])
            elif normalize == 'last':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n == N-1 else nn.Identity(k)
                    for n in range(N)
                ])
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")
        
        # C : k → d  (includes bias β)
        # Note: default initialization for Linear is Kaiming
        self.C = nn.Linear(k, d, bias=not drop_constant)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch_size, s)
        returns x_hat : (batch_size, d)
        """
        # level-1
        x = self.U[0](z)                    # (B, k)
        x = self.batch_norms[0](x) if hasattr(self, 'batch_norms') else x
        
        # levels 2..N via Eq. (6): 
        # x_n = (U_n z) ⊙ x + x
        for n in range(1, self.N):
            u_n = self.U[n](z)              # (B, k)
            x = u_n * x + x                 # Hadamard + (previous iter)
            x = self.batch_norms[n](x) if hasattr(self, 'batch_norms') else x

        # final linear map
        x_hat = self.C(x)                   # (B, d)
        return x_hat

class PiNetNCP(nn.Module):
    """
    Nested CP Π-net (NCP)
    ---------------------
    Recursion relation of Nested CP Π-net (NCP), Eq. (9) of [1]:
        x_1 = (A_1 z) ⊙ (B_1 b_1)
        x_n = (A_n z) ⊙ (S_n x_{n-1} + B_n b_n) for n=2..N
        x_hat = C x_N + β
        
    Notes
    -----
    - if drop_linear=True, then the first-order term is dropped by
        - dropping B_N and b_N which means `self.A` and `self.S` are size N
          and `self.B` and `self.b` are of size N-1.
        - For more details on how the linear part is dropped, see Eq. (8) of [1]
    - if drop_constant=True, then β is dropped that is C's bias is dropped.
    
    Reference
    ---------
    [1] G. G. Chrysos, S. Moschoglou, G. Bouritsas, J. Deng, Y. Panagakis, and
    S. Zafeiriou, “Deep Polynomial Neural Networks,” IEEE Transactions on
    Pattern Analysis and Machine Intelligence, vol. 44, no. 8, pp. 4021-4034,
    Aug. 2022, doi: 10.1109/TPAMI.2021.3058891.
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int, 
                 drop_linear: bool = False,
                 drop_constant: bool = False,
                 normalize: None | str = None):
        """
        Arguments
        ---------
        in_dim : input dimension (s)
        out_dim : output dimension (d)
        inter_dim : intermediate dimension (k)
        poly_order : polynomial order (N)
        drop_linear : whether to drop the linear term (first-order)
        drop_constant : whether to drop the constant term (bias in C) 
        normalize : normalization method for U_n
            - None: no normalization
            - 'all': normalize all U_n
            - 'last': normalize only the highest/last order U_n
            - 'last2': normalize the last two highest order U_n
        """ 
        
        super().__init__()
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N
        self.drop_linear = drop_linear
        
        # drop_linear=True, we don't need the N-th module for B and b
        # drop_linear=False, we need modules all levels 1..N for all
        num_bB_modules = N - 1 if drop_linear else N

        # A_n : R^s → R^k  
        self.A = nn.ModuleList([
            nn.Linear(s, k, bias=False) 
            for _ in range(N)
        ])
        # S_n : R^k → R^k  
        self.S = nn.ModuleList([
            nn.Linear(k, k, bias=False) 
            for _ in range(N)
        ])
        # b_n : R^k hyper-parameters
        self.b = nn.ParameterList([
            nn.Parameter(torch.ones(k)) 
            for _ in range(num_bB_modules)
        ])
        # B_n : maps b_n (k) → R^k
        self.B = nn.ModuleList([
            nn.Linear(k, k, bias=False) 
            for _ in range(num_bB_modules)
        ])
        # Final C: R^k → R^d  (with bias β)
        self.C = nn.Linear(k, d, bias=not drop_constant)
        
        # Normalizations 
        if normalize is not None:
            if normalize == 'all':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True)
                    for _ in range(N)
                ])
            elif normalize == 'last2':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n >= N-2 else nn.Identity(k)
                    for n in range(N)
                ])
            elif normalize == 'last':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n == N-1 else nn.Identity(k)
                    for n in range(N)
                ])
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch_size, s)  input to the first polynomial
        returns x_hat : (batch_size, d)  output of the last polynomial
        """
        # level-1
        b0 = self.B[0](self.b[0])                    # (k,)
        x = self.A[0](z) * b0.unsqueeze(0)           # (batch, k)
        x = self.batch_norms[0](x) if hasattr(self, 'batch_norms') else x
            
        # levels 2..N 
        for n in range(1, self.N):
            u = self.A[n](z)                         # (batch, k)
            if n == self.N-1 and self.drop_linear:   # last level, drop linear
                v = self.S[n](x)                     # (batch, k)
            else:                                    
                bn = self.B[n](self.b[n])            # (k,)
                v = self.S[n](x) + bn.unsqueeze(0)   # (batch, k)
            x = u * v                                # (batch, k)
            x = self.batch_norms[n](x) if hasattr(self, 'batch_norms') else x
             
        # final output
        return self.C(x)                             # (batch, d)

class PiNetNCPSkip(nn.Module):
    """
    Nested CP Π-net with skip connections (NCP-Skip)
    ------------------------------------------------
    Recursion relation of Nested CP Π-net with skip (NCP-Skip), Eq. (10) of [1]:
        x_1 = (A_1 z) ⊙ (B_1 b_1)
        x_n = (A_n z) ⊙ (S_n x_{n-1} + B_n b_n) + V_n x_{n-1}
        x_hat = C x_N + β
        
    Notes
    -----
    - if drop_linear=True, then the first-order term is dropped by
        - dropping B_N and b_N which means `self.A` and `self.S` are size N
          and `self.B` and `self.b` are of size N-1.
        - For more details on how the linear part is dropped, see Eq. (8) of [1]
    - if drop_constant=True, then β is dropped that is C's bias is dropped.
    
    Reference
    ---------
    [1] G. G. Chrysos, S. Moschoglou, G. Bouritsas, J. Deng, Y. Panagakis, and
    S. Zafeiriou, “Deep Polynomial Neural Networks,” IEEE Transactions on
    Pattern Analysis and Machine Intelligence, vol. 44, no. 8, pp. 4021-4034,
    Aug. 2022, doi: 10.1109/TPAMI.2021.3058891.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 inter_dim: int, poly_order: int, 
                 drop_linear: bool = False,
                 drop_constant: bool = False,
                 normalize: None | str = None):
        """
        Arguments
        ---------
        in_dim : input dimension (s)
        out_dim : output dimension (d)
        inter_dim : intermediate dimension (k)
        poly_order : polynomial order (N)
        drop_linear : whether to drop the linear term (first-order)
        drop_constant : whether to drop the constant term (bias in C) 
        normalize : normalization method for U_n
            - None: no normalization
            - 'all': normalize all U_n
            - 'last': normalize only the highest/last order U_n
            - 'last2': normalize the last two highest order U_n
        """ 
        
        super().__init__()
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N
        self.drop_linear = drop_linear
        
        # drop_linear=True, we don't need the N-th module for B and b
        # drop_linear=False, we need modules all levels 1..N for all
        num_bB_modules = N - 1 if drop_linear else N

        # Same as PiNetNCP, but with V_n skip connections
        self.A = nn.ModuleList([
            nn.Linear(s, k, bias=False) 
            for _ in range(N)
        ])
        self.S = nn.ModuleList([
            nn.Linear(k, k, bias=False) 
            for _ in range(N)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.ones(k)) 
            for _ in range(num_bB_modules)
        ])
        self.B = nn.ModuleList([
            nn.Linear(k, k, bias=False) 
            for _ in range(num_bB_modules)
        ])
        # V_n: R^k → R^k skip on x_{n-1}
        self.V = nn.ModuleList([
            nn.Linear(k, k, bias=False) 
            for _ in range(N)
        ])
        self.C = nn.Linear(k, d, bias=not drop_constant)
        
        # Normalizations
        if normalize is not None:
            if normalize == 'all':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True)
                    for _ in range(N)
                ])
            elif normalize == 'last2':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n >= N-2 else nn.Identity(k)
                    for n in range(N)
                ])
            elif normalize == 'last':
                self.batch_norms = nn.ModuleList([
                    nn.BatchNorm1d(k, affine=True) 
                    if n == N-1 else nn.Identity(k)
                    for n in range(N)
                ])
            else:
                raise ValueError(f"Unknown normalization method: {normalize}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch_size, s)  input to the first polynomial
        returns x_hat : (batch_size, d)  output of the last polynomial
        """
        # level-1
        b0 = self.B[0](self.b[0])                     # (k,)
        x = self.A[0](z) * b0.unsqueeze(0)            # (batch, k)
        x = self.batch_norms[0](x) if hasattr(self, 'batch_norms') else x
        
        # recursion
        for n in range(1, self.N):
            u = self.A[n](z)                          # (batch, k)
            if n == self.N-1 and self.drop_linear:    # last level, drop linear
                v = self.S[n](x)                      # (batch, k)
            else:
                bn = self.B[n](self.b[n])             # (k,)
                v = self.S[n](x) + bn.unsqueeze(0)    # (batch, k)
            x = u * v + self.V[n](x)                  # (batch, k)
            x = self.batch_norms[n](x) if hasattr(self, 'batch_norms') else x
        
        # final output 
        return self.C(x)                              # (batch, d)

class ProdPoly(nn.Module):
    """
    Product-of-Polynomials wrapper (Sec. 3.2 [1]): chains P polynomial blocks.
    
    Each block is itself a Π-net (e.g. PiNetCCP, PiNetNCP, PiNetNCPSkip)
    that maps R^s → R^s (or more generally R^{in_dim} → R^{out_dim}).
    The overall approximation is the successive product of these polynomials,
    yielding a final order = ∏ block_orders.
    """
    def __init__(self,
                 pinet_class: type,
                 in_dim: int,
                 out_dim: int,
                 inter_dim: int,
                 poly_order: int,
                 num_polys: int,
                 drop_linear: bool,
                 drop_constant: bool,
                 normalize: None | str):
        """
        pinet_class   : Pi-Net class (PiNetCCP, PiNetNCP, PiNetNCPSkip)
        in_dim        : input dimension (s)
        inter_dim     : intermediate dimension (k) 
        out_dim       : output dimension (d)
        poly_order    : polynomial order for each block
        num_polys     : number of polynomial blocks
        drop_linear   : whether to drop linear terms
        drop_constant : whether to drop constant terms (bias in C)
        normalize     : normalization method for parameters
        """
        super().__init__()
        
        self.num_polys = num_polys
        self.blocks = nn.ModuleList() 
         
        for i in range(num_polys):
            if i == 0:
                # First block: in_dim → inter_dim
                block_in_dim = in_dim
                block_out_dim = inter_dim if num_polys > 1 else out_dim
                block_inter_dim = inter_dim
            elif i == num_polys - 1:
                # Last block: inter_dim → out_dim
                block_in_dim = inter_dim
                block_out_dim = out_dim
                block_inter_dim = inter_dim 
            else:
                # Intermediate blocks: inter_dim → inter_dim
                block_in_dim = inter_dim
                block_out_dim = inter_dim
                block_inter_dim = inter_dim
        
            # Create the Pi-Net block with appropriate dimensions
            if pinet_class is PiNetCCP:
                block = pinet_class(
                    in_dim=block_in_dim,
                    out_dim=block_out_dim,
                    inter_dim=block_inter_dim,
                    poly_order=poly_order,
                    drop_constant=drop_constant,
                    normalize=normalize
                )
            else:
                block = pinet_class(
                    in_dim=block_in_dim,
                    out_dim=block_out_dim,
                    inter_dim=block_inter_dim,
                    poly_order=poly_order,
                    drop_linear=drop_linear,
                    drop_constant=drop_constant,
                    normalize=normalize
                )
            self.blocks.append(block)

    def forward(self, z):
        """
        z : (batch, s)  input to first polynomial
        returns G(z) : output of last polynomial
        """
        x = z
        for poly in self.blocks:
            x = poly(x)
        return x
    