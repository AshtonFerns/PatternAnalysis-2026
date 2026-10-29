import torch
import torch.nn as nn
import torch.nn.functional as F

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
    # see above
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

class VectorQuantizer(nn.Module):
    # num_codes is the number of codebook vectors
    # dim is their dimensionality, which should match the encoder output
    # beta is the encoder penalty for being far from the code it snaps to
    # decay is how much previous updates are weighted in the moving average
    # restart_below is the minimum usage count for a code during one batch before it is replaced
    def __init__(self, num_codes=512, dim=64, beta=0.25, decay=0.99, restart_below=1.0):
        super().__init__()
        self.num_codes, self.dim, self.beta, self.decay = num_codes, dim, beta, decay
        self.restart_below = restart_below
        self.register_buffer("codebook", torch.zeros(num_codes, dim))
        self.register_buffer("ema_count", torch.zeros(num_codes))
        self.register_buffer("ema_sum", torch.zeros(num_codes, dim))
        self.register_buffer("ready", torch.tensor(False))

    def nearest(self, flat):
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        return d.argmin(1)

    @torch.no_grad()
    def update(self, flat):
        if not self.ready:
            pick = torch.randint(0, len(flat), (self.num_codes,), device=flat.device)
            self.codebook.copy_(flat[pick])
            self.ema_sum.copy_(flat[pick])
            self.ema_count.fill_(len(flat) / self.num_codes)
            self.ready.fill_(True)
        idx = self.nearest(flat)
        count = torch.bincount(idx, minlength=self.num_codes).float()
        total = torch.zeros_like(self.ema_sum).index_add_(0, idx, flat)
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(total, alpha=1 - self.decay)
        self.codebook.copy_(self.ema_sum / self.ema_count.clamp_min(1e-5)[:, None])
        dead = self.ema_count < self.restart_below
        if dead.any():
            new = flat[torch.randint(0, len(flat), (int(dead.sum()),), device=flat.device)]
            self.codebook[dead] = new
            self.ema_sum[dead] = new * self.restart_below
            self.ema_count[dead] = self.restart_below
        return int(dead.sum())

    def lookup(self, idx):
        return self.codebook[idx].permute(0, 3, 1, 2).contiguous()

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            B, D, h, w = z.shape
            flat = z.permute(0, 2, 3, 1).reshape(-1, D)
            restarts = self.update(flat.detach()) if self.training else 0
            idx = self.nearest(flat)
            q = self.lookup(idx.view(B, h, w))
            commit = self.beta * F.mse_loss(z, q.detach())
            q = z + (q - z).detach()
            p = torch.bincount(idx, minlength=self.num_codes).float() / idx.numel()
            perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
        return q, commit, idx.view(B, h, w), {"perplexity": perplexity.item(), "restarts": restarts}


class VQVAE(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), n_res=2, dim=64, num_codes=512, beta=0.25):
        super().__init__()
        self.encoder = Encoder(channels, n_res, dim)
        self.quantizer = VectorQuantizer(num_codes, dim, beta)
        self.decoder = Decoder(channels, n_res, dim)

    def forward(self, x):
        q, commit, idx, stats = self.quantizer(self.encoder(x))
        return self.decoder(q), commit, idx, stats

    @torch.no_grad()
    def encode(self, x):
        return self.quantizer(self.encoder(x))[2]

    @torch.no_grad()
    def decode(self, idx):
        return self.decoder(self.quantizer.lookup(idx))


def ssim(x, y, reduction="mean"):
    x, y = x.float(), y.float()
    # gaussian with length 11, centre at 5, std of 1.5
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    win = (g[:, None] * g[None, :])[None, None]
    # k1, k2, and L values used in most papers
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = F.conv2d(x, win), F.conv2d(y, win)
    vx = F.conv2d(x * x, win) - mx ** 2
    vy = F.conv2d(y * y, win) - my ** 2
    cxy = F.conv2d(x * y, win) - mx * my
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    s = s.flatten(1).mean(1)
    return s.mean() if reduction == "mean" else s