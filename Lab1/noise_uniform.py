import torch
from torchvision.transforms import v2


class UniformNoise(v2.Transform):
    """Add uniform random noise to an image tensor in [0, 1]."""

    def __init__(self, amount: float = 0.05, inplace: bool = False):
        super().__init__()
        self.amount = float(amount)
        self.inplace = inplace

    def transform(self, inpt: torch.Tensor, params: dict):
        # inpt: Tensor image, shape [C,H,W], dtype float32, in [0,1]
        if not torch.is_floating_point(inpt):
            raise TypeError("UniformNoise expects a float tensor input in [0, 1].")

        if self.inplace:
            out = inpt
        else:
            out = inpt.clone()

        noise = torch.empty_like(out).uniform_(-self.amount, self.amount)
        out.add_(noise).clamp_(0.0, 1.0)
        return out