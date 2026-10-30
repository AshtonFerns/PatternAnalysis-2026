import torch
import torch.nn as nn
import torch.nn.functional as F

def norm(ch):
    # cap the number of channel groups to 32 for at least some channel coupling
    # results in groups of 4 channels
    return nn.GroupNorm(min(32, ch // 4), ch)

# residual block used in VAE
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
            norm(ch), nn.SiLU(), nn.Conv2d(ch, ch, 3, padding=1),
        )
        # starts at zero so the block begins as a plain skip
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
                # halves the height and width
                layers.append(nn.Conv2d(ch, channels[i + 1], 4, stride=2, padding=1))

        layers += [norm(channels[-1]), nn.SiLU(), nn.Conv2d(channels[-1], out_ch, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    # see above
    def __init__(self, channels=(32, 64, 128, 256), n_res=2, in_ch=64):
        super().__init__()
        # reverse the widths so upsampling mirrors the encoder
        rev = channels[::-1]
        layers = [nn.Conv2d(in_ch, rev[0], 3, padding=1)]
        for i, ch in enumerate(rev):
            layers += [ResBlock(ch) for _ in range(n_res)]
            if i < len(rev) - 1:
                # nearest upsample followed by a conv to avoid checkerboard patterns
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
        # calculate the squared distance between the flat vectors and the codebook
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.codebook.t() + self.codebook.pow(2).sum(1)
        return d.argmin(1)

    @torch.no_grad()
    def update(self, flat):
        if not self.ready:
            # pick random encoder vectors from this batch as the starting codes
            pick = torch.randint(0, len(flat), (self.num_codes,), device=flat.device)
            self.codebook.copy_(flat[pick])
            self.ema_sum.copy_(flat[pick])
            self.ema_count.fill_(len(flat) / self.num_codes)
            self.ready.fill_(True)
        idx = self.nearest(flat)
        # how many vectors in this batch chose each code
        count = torch.bincount(idx, minlength=self.num_codes).float()
        # sum of the vectors that chose each code
        total = torch.zeros_like(self.ema_sum).index_add_(0, idx, flat)
        # decay the old stats and mix in this batch
        self.ema_count.mul_(self.decay).add_(count, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(total, alpha=1 - self.decay)
        # each code becomes the average of the vectors assigned to it
        # 1e-5 keeps an unused code from dividing by zero
        self.codebook.copy_(self.ema_sum / self.ema_count.clamp_min(1e-5)[:, None])
        dead = self.ema_count < self.restart_below
        if dead.any():
            # replaces a dead code with a new random one
            new = flat[torch.randint(0, len(flat), (int(dead.sum()),), device=flat.device)]
            self.codebook[dead] = new
            # matches the stored sum to the new vector so its average doesn't jump
            self.ema_sum[dead] = new * self.restart_below
            self.ema_count[dead] = self.restart_below
        return int(dead.sum())

    def lookup(self, idx):
        # rearrange to channels-first for the decoder
        return self.codebook[idx].permute(0, 3, 1, 2).contiguous()

    def forward(self, z):
        # uses full precision
        with torch.autocast(z.device.type, enabled=False):
            z = z.float()
            B, D, h, w = z.shape
            flat = z.permute(0, 2, 3, 1).reshape(-1, D)
            # no gradient through the codebook update and don't update when testing
            restarts = self.update(flat.detach()) if self.training else 0
            idx = self.nearest(flat)
            q = self.lookup(idx.view(B, h, w))
            # stop the gradient at the code so only the encoder is affected
            commit = self.beta * F.mse_loss(z, q.detach())
            q = z + (q - z).detach()
            # share of positions that landed on each code
            p = torch.bincount(idx, minlength=self.num_codes).float() / idx.numel()
            # 1e-10 to avoid log(0)
            perplexity = torch.exp(-(p * (p + 1e-10).log()).sum())
        # returns perplexity and the restart count
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
        # index 2 of the quantiser output is the grid of code ids
        return self.quantizer(self.encoder(x))[2]

    @torch.no_grad()
    def decode(self, idx):
        return self.decoder(self.quantizer.lookup(idx))


def ssim(x, y, reduction="mean"):
    x, y = x.float(), y.float()
    # gaussian with length 11, centre at 5, std of 1.5
    g = torch.exp(-(torch.arange(11, device=x.device) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    # makes a 2d window
    win = (g[:, None] * g[None, :])[None, None]
    # k1, k2, and L values used in most papers
    # L is 1 because the slices are normalised to 0 to 1
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mx, my = F.conv2d(x, win), F.conv2d(y, win)
    vx = F.conv2d(x * x, win) - mx ** 2
    vy = F.conv2d(y * y, win) - my ** 2
    cxy = F.conv2d(x * y, win) - mx * my
    # product of the local brightness and contrast scores
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    # averages to one number per image
    s = s.flatten(1).mean(1)
    return s.mean() if reduction == "mean" else s

class MaskedConv(nn.Conv2d):
    def __init__(self, *args, mask, **kw):
        super().__init__(*args, **kw)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding)

class GatedLayer(nn.Module):
    def __init__(self, dim, k=5, first=False, dropout=0.0):
        super().__init__()
        p = k // 2 # k = 5 so each layer sees 2 rows above and 2 columns to the left
        # output is split into a value and a gate
        vm, hm = torch.ones(2 * dim, dim, p + 1, k), torch.ones(2 * dim, dim, 1, p + 1)
        if first:
            vm[:, :, -1] = 0 # ignore rows before the first row
            hm[:, :, :, -1] = 0  # ignore columns before the first column
        self.first = first
        self.v = MaskedConv(dim, 2 * dim, (p + 1, k), padding=(p, p), mask=vm)
        self.h = MaskedConv(dim, 2 * dim, (1, p + 1), padding=(0, p), mask=hm)
        self.v2h = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.out = nn.Conv2d(dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def gate(x):
        a, b = x.chunk(2, 1)
        return torch.tanh(a) * torch.sigmoid(b)

    def forward(self, xv, xh):
        H, W = xv.shape[-2:]
        v = self.v(xv)[:, :, :H] # ignore rows below the image
        h = self.h(xh)[:, :, :, :W] # ignore columns to the right
        hout = self.drop(self.out(self.gate(h + self.v2h(v))))
        # skip after the first layer
        return self.gate(v), hout if self.first else xh + hout

class GatedPixelCNN(nn.Module):
    def __init__(self, num_codes=512, dim=128, layers=18, k=5, dropout=0.1):
        super().__init__()
        self.num_codes = num_codes
        # 18 layers * 2 rows each + the centre = 37 rows, which covers the 32-row code grid
        self.embed = nn.Embedding(num_codes, dim)
        # skip the rows above for the first layer
        self.layers = nn.ModuleList([GatedLayer(dim, k, first=(i == 0), dropout=dropout) for i in range(layers)])
        # hidden width of 512 then one logit per code
        self.head = nn.Sequential(nn.ReLU(), nn.Conv2d(dim, 512, 1), nn.ReLU(), nn.Conv2d(512, num_codes, 1))

    def forward(self, idx):
        # idx is integer codes (B, H, W), output is a score for each of the 512 codes at every position
        # moves the embedding axis to the front
        x = self.embed(idx).permute(0, 3, 1, 2)
        xv = xh = x
        for layer in self.layers:
            xv, xh = layer(xv, xh)
        return self.head(xh)

    @torch.no_grad()
    def sample(self, n, h, w, device, temperature=1.0, generator=None):
        # zeros stand in for codes that have not been chosen yet
        idx = torch.zeros(n, h, w, dtype=torch.long, device=device)
        for i in range(h):
            for j in range(w):
                # leaves out rows below
                logits = self(idx[:, :i + 1])[:, :, i, j].float() / temperature
                idx[:, i, j] = torch.multinomial(logits.softmax(1), 1, generator=generator)[:, 0]
        return idx


class VAE(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256, 256), n_res=2, z_dim=256, ch=16, hw=(256, 128)):
        super().__init__()
        f = 2 ** (len(channels) - 1)
        self.shape = (ch, hw[0] // f, hw[1] // f) # 16 x 16 x 8 feature map
        n = ch * self.shape[1] * self.shape[2]
        self.z_dim = z_dim
        self.encoder = Encoder(channels, n_res, ch)
        # puts the mean and log variance into one linear map
        self.to_stats = nn.Linear(n, 2 * z_dim) 
        # expands the sampled vector back onto the feature map
        self.from_z = nn.Linear(z_dim, n)
        self.decoder = Decoder(channels, n_res, ch)

    def decode(self, z):
        return self.decoder(self.from_z(z).view(-1, *self.shape))

    def forward(self, x):
        mu, logvar = self.to_stats(self.encoder(x).flatten(1)).float().chunk(2, 1)
        # standard deviation sits between exp(-6) and exp(4) at these limits
        logvar = logvar.clamp(-12, 8) # keep the random spread from exploding or vanishing
        # only add noise during training
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        # average KL divergence over batch
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean()
        return self.decode(z), kl, mu

    @torch.no_grad()
    def sample(self, n, device, generator=None):
        return self.decode(torch.randn(n, self.z_dim, device=device, generator=generator))
