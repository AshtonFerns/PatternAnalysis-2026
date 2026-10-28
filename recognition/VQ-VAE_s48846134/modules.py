import torch
import torch.nn as nn

def norm(ch):
    # cap the number of channel groups to 32 for at least some channel coupling
    return nn.GroupNorm(min(32, ch // 4), ch)

# residual block used in VAE
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x):
        return x + self.body(x)