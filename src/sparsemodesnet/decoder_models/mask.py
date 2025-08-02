import torch

class MaskedLayer(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,    # e.g., 6
        out_features: int,   # e.g., 2
        mask: torch.Tensor,  # e.g., shape(2,6)
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,   # no need to use a bias in our case
        )
        self.register_buffer('mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
        return x
