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
            nn.Linear(s, k, bias=False) for _ in range(poly_order)
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
        x_1 = (A_1 z) ⊙ (B_1 b_1)
        x_n = (A_n z) ⊙ (S_n x_{n-1} + B_n b_n) for n=2..N
        x_hat = C x_N + β
    """

    def __init__(self, in_dim: int, out_dim: int, 
                 inter_dim: int, poly_order: int, drop_linear: bool = False):
        super().__init__()
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N
        self.drop_linear = drop_linear

        # A_n : R^s → R^k  for n=1..N
        self.A = nn.ModuleList([nn.Linear(s, k, bias=False) for _ in range(N)])
        # S_n : R^k → R^k  for n=2..N
        self.S = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        # b_n : R^k hyper-parameters
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(k)) for _ in range(N)])
        # B_n : maps b_n (k) → R^k
        self.B = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        # Final C: R^k → R^d  (with bias β)
        self.C = nn.Linear(k, d, bias=True)

        # initialize weights
        for m in range(N):
            nn.init.xavier_uniform_(self.A[m].weight)
            nn.init.xavier_uniform_(self.S[m].weight)
            nn.init.xavier_uniform_(self.B[m].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (batch, s)
        
        if self.drop_linear:
            # level-1: no first-order term → start from zero
            batch, k = z.shape[0], self.b[0].shape[0]
            x = z.new_zeros(batch, k)
        else:
            # level-1: x1 = A1(z) * B1(b1)
            b0 = self.B[0](self.b[0])              # (k,)
            x = self.A[0](z) * b0.unsqueeze(0)     # (batch, k)
        # levels 2..N
        for n in range(1, self.N):
            u = self.A[n](z)                   # (batch, k)
            bn = self.B[n](self.b[n])          # (k,)
            v = self.S[n](x) + bn.unsqueeze(0) # (batch, k)
            x = u * v                          # (batch, k)
        # final output
        return self.C(x)                       # (batch, d)


class PiNetNCPSkip(nn.Module):
    """
    Nested CP Π-net with skip (NCP-Skip), Eq. (10):
        x_1 = (A_1 z) ⊙ (B_1 b_1)
        x_n = (A_n z) ⊙ (S_n x_{n-1} + B_n b_n) + V_n x_{n-1}
        x_hat = C x_N + β
    """

    def __init__(self, in_dim: int, out_dim: int,
                 inter_dim: int, poly_order: int, drop_linear: bool = False):
        super().__init__()
        s, d, k, N = in_dim, out_dim, inter_dim, poly_order
        self.N = N
        self.drop_linear = drop_linear

        self.A = nn.ModuleList([nn.Linear(s, k, bias=False) for _ in range(N)])
        self.S = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(k)) for _ in range(N)])
        self.B = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        # V_n: R^k → R^k skip on x_{n-1}
        self.V = nn.ModuleList([nn.Linear(k, k, bias=False) for _ in range(N)])
        self.C = nn.Linear(k, d, bias=True)

        # initialize weights
        for m in range(N):
            nn.init.xavier_uniform_(self.A[m].weight)
            nn.init.xavier_uniform_(self.S[m].weight)
            nn.init.xavier_uniform_(self.B[m].weight)
            nn.init.xavier_uniform_(self.V[m].weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.drop_linear:
            # no first-order term → start from zero
            batch, k = z.shape[0], self.b[0].shape[0]
            x = z.new_zeros(batch, k)
        else:
            # initial: x1 = A1(z) * B1(b1)
            b0 = self.B[0](self.b[0])              # (k,)
            x = self.A[0](z) * b0.unsqueeze(0)     # (batch, k)
        # recursion
        for n in range(1, self.N):
            u = self.A[n](z)                   # (batch, k)
            bn = self.B[n](self.b[n])          # (k,)
            v = self.S[n](x) + bn.unsqueeze(0) # (batch, k)
            x = u * v + self.V[n](x)           # (batch, k)
        return self.C(x)                       # (batch, d)

class ProdPoly(nn.Module):
    """
    Product-of-Polynomials wrapper (Sec. 3.2): chains P polynomial blocks.
    
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
                 drop_linear: bool):
        """
        pinet_class : Pi-Net class (PiNetCCP, PiNetNCP, PiNetNCPSkip)
        in_dim      : input dimension (s)
        inter_dim   : intermediate dimension (k) 
        out_dim     : output dimension (d)
        poly_order  : polynomial order for each block
        num_polys   : number of polynomial blocks
        drop_linear : whether to drop linear terms
        use_batch_norm : whether to use batch normalization
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
                )
            else:
                block = pinet_class(
                    in_dim=block_in_dim,
                    out_dim=block_out_dim,
                    inter_dim=block_inter_dim,
                    poly_order=poly_order,
                    drop_linear=drop_linear,
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

