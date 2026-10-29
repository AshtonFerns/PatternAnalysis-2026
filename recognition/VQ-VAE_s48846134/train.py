import os

# needed for my AMD GPU 
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn.functional as F
from dataset import get_loaders
from modules import VQVAE, ssim

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vqvae_loss(model, x):
    recon, commit, _, stats = model(x)
    recon = recon.float()
    l1 = F.l1_loss(recon, x)
    loss = l1 + 0.5 * (1 - ssim(recon, x)) + commit
    return loss, stats

if __name__ == "__main__":
    model = VQVAE().to(DEV).train()
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4)
    x, _ = next(iter(get_loaders(32)["train"]))
    x = x.to(DEV)
    opt.zero_grad(set_to_none=True)
    loss, stats = vqvae_loss(model, x)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()