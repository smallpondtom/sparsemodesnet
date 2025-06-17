import torch
import torch.nn as nn
import torch.nn.functional as F

class PiNetCCP(nn.Module):
    """
    Π-net via a coupled CP decomposition (CCP).
    Recursion (for n=2..N):
        x_n = (U_n z) ⊙ x_{n-1} + x_{n-1}
    Final output:  x_hat = C x_N + β
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int):
        """
        in_dim : input dim
        out_dim : output dim
        inter_dim : intermediate rank
        poly_order : polynomial order
        """
        super().__init__()
        # Let s = in_dim, d = out_dim, k = inter_dim, N = poly_order
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N

        # U_n : s → k for n=1..N
        self.U = nn.ModuleList([
            nn.Linear(s, k, bias=False) 
            for _ in range(poly_order)
        ])
        # C : k → d  (includes bias β)
        self.C = nn.Linear(k, d, bias=True)

        # initialize U[0] as the "first level" factor if desired
        # e.g. orthonormal init or Xavier:
        for u in self.U:
            nn.init.xavier_uniform_(u.weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z : (batch_size, s)
        returns x_hat : (batch_size, d)
        """
        # level-1
        x = self.U[0](z)                    # (B, k)
        # levels 2..N via Eq. (6): x_n = (U_n z) ⊙ x + x
        for n in range(1, self.N):
            u_n = self.U[n](z)              # (B, k)
            x = u_n * x + x                 # Hadamard + skip

        # final linear map
        x_hat = self.C(x)                   # (B, d)
        return x_hat
    
class PiNetNCP(nn.Module):
    """
    Nested CP Π-net (NCP), Eq. (9):
        x_1 = (A_1 z) ⊙ b_1
        x_n = (A_n z) ⊙ (S_n x_{n-1} + b_n)   for n=2..N
        x_hat = C x_N + β
    See Eq. (9) and Fig. 3  
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int):
        """
        in_dim : input dim
        out_dim : output dim
        inter_dim : intermediate “rank”
        poly_order : total polynomial order
        """
        super().__init__()
        # Let s = in_dim, d = out_dim, k = inter_dim, N = poly_order
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N

        # A_n : R^s → R^k  for n=1..N
        self.A = nn.ModuleList(
            [nn.Linear(s, k, bias=False) for _ in range(N)])
        # S_n : R^k → R^k  for n=2..N (we’ll ignore index 0)
        self.S = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        # b_n : R^k bias vectors
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(k)) for _ in range(N)])
        # Final C: R^k → R^d  (with bias β)
        self.C = nn.Linear(k, d, bias=True)

        # (optional) initialize A and S
        for m in range(N):
            nn.init.xavier_uniform_(self.A[m].weight)
            nn.init.xavier_uniform_(self.S[m].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, s)
        # 1) first order
        x = self.A[0](z) * self.b[0]              # (B, k)
        # 2) higher orders
        for n in range(1, self.N):
            u = self.A[n](z)                      # (B, k)
            v = self.S[n](x) + self.b[n]         # (B, k)
            x = u * v                             # Hadamard
        # 3) final linear map + bias
        return self.C(x)                          # (B, d)

class PiNetNCPSkip(nn.Module):
    """
    Nested CP Π-net with skip (NCP-Skip), Eq. (10):
        x_1 = (A_1 z) ⊙ b_1
        x_n = (A_n z) ⊙ (S_n x_{n-1} + b_n) + V_n x_{n-1}   for n=2..N
        x_hat = C x_N + β
    See Eq. (10) and Fig. 4  
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int):
        super().__init__()
        # Let s = in_dim, d = out_dim, k = inter_dim, N = poly_order
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order 
        self.N = N

        self.A = nn.ModuleList([nn.Linear(s, k, bias=False) for _ in range(N)])
        self.S = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        self.b = nn.ParameterList(
            [nn.Parameter(torch.zeros(k)) for _ in range(N)])
        # V_n: R^k → R^k skip on x_{n-1}, used for n>=1 (we’ll ignore V[0])
        self.V = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        self.C = nn.Linear(k, d, bias=True)

        for m in range(N):
            nn.init.xavier_uniform_(self.A[m].weight)
            nn.init.xavier_uniform_(self.S[m].weight)
            nn.init.xavier_uniform_(self.V[m].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # 1) initial
        x = self.A[0](z) * self.b[0]
        # 2) recursion
        for n in range(1, self.N):
            u = self.A[n](z)
            v = self.S[n](x) + self.b[n]
            x = u * v + self.V[n](x)
        # 3) final
        return self.C(x)
    
class ProdPoly(nn.Module):
    """
    Product-of-Polynomials wrapper (Sec. 3.2): chains P polynomial blocks.
    
    Each block is itself a Π-net (e.g. PiNetCCP, PiNetNCP, PiNetNCPSkip)
    that maps R^s → R^s (or more generally R^{in_dim} → R^{out_dim}).
    The overall approximation is the successive product of these polynomials,
    yielding a final order = ∏ block_orders.
    """
    def __init__(self,
                 block_cls: type,
                 num_polys: int,
                 *block_args,
                 **block_kwargs):
        """
        block_cls   : class of a single polynomial block (must be nn.Module)
        num_polys   : number of successive polynomials P
        *block_args : positional args passed to each block constructor
        **block_kwargs: keyword args passed to each block constructor
        """
        super().__init__()
        # Create P independent polynomial blocks
        self.blocks = nn.ModuleList([
            block_cls(*block_args, **block_kwargs)
            for _ in range(num_polys)
        ])

    def forward(self, z):
        """
        z : (batch, s)  input to first polynomial
        returns G(z) : output of last polynomial
        """
        x = z
        for poly in self.blocks:
            x = poly(x)
        return x

