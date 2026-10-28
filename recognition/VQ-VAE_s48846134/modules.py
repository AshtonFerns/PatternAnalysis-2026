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

class Encoder(nn.Module):
    # channels is the number of channels in each layer
    # n_res is number of residual blocks in each layer
    # out_ch is the number of channels in the output layer
    def __init__(self, channels=(32, 64, 128, 256), n_res=2, out_ch=64):
        super().__init__()
        layers = [nn.Conv2d(1, channels[0], 3, padding=1)]
        for i, ch in enumerate(channels):
            layers += [ResBlock(ch) for _ in range(n_res)]
            if i < len(channels) - 1:
                layers.append(nn.Conv2d(ch, channels[i + 1], 4, stride=2, padding=1))

        layers += [norm(channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), n_res=2, in_ch=64):
        super().__init__()
        rev = channels[::-1]
        layers = [nn.Conv2d(in_ch, rev[0], 3, padding=1)]
        for i, ch in enumerate(rev):
            layers += [ResBlock(ch) for _ in range(n_res)]
            if i < len(rev) - 1:
                layers += [nn.Upsample(scale_factor=2, mode="nearest"),
                           nn.Conv2d(ch, rev[i + 1], 3, padding=1)]
        layers += [norm(rev[-1]), nn.SiLU(), nn.Conv2d(rev[-1], 1, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)