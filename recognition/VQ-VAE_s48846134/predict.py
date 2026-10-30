import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import load_split
from modules import ssim
from train import device, load, root

fig_dir = os.path.join(root, "figures")

def show(img):
    return img.detach().float().cpu().numpy()

def save_grid(imgs, path, title):
    fig, axes = plt.subplots(2, 8, figsize=(12, 5))
    for ax, img in zip(axes.flat, imgs):
        ax.imshow(show(img), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)

@torch.no_grad()
def main():
    os.makedirs(fig_dir, exist_ok=True)
    vq, _, _ = load("vqvae")
    prior, pcfg, _ = load("prior")
    vae, _, _ = load("vae")
    h, w = pcfg["grid"]
    test = load_split("test")
    images = test["images"].float()
    pick = torch.linspace(5, len(images) - 6, 6).long()
    recon_vq = torch.cat([vq(images[i:i + 1].to(device))[0].cpu() for i in pick])
    recon_vae = torch.cat([vae(images[i:i + 1].to(device))[0].cpu() for i in pick])
    fig, axes = plt.subplots(3, 6, figsize=(12, 8))
    for j, i in enumerate(pick):
        rows = (
            (images[i, 0], "case {} w{} s{}".format(*test["keys"][i])),
            (recon_vq[j, 0], f"VQ-VAE {ssim(recon_vq[j:j + 1], images[i:i + 1]).item():.3f}"),
            (recon_vae[j, 0], f"VAE {ssim(recon_vae[j:j + 1], images[i:i + 1]).item():.3f}"),
        )
        for r, (img, title) in enumerate(rows):
            axes[r, j].imshow(show(img), cmap="gray", vmin=0, vmax=1)
            axes[r, j].set_title(title, fontsize=8)
            axes[r, j].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "test_reconstructions.png"), dpi=100)
    plt.close(fig)
    g = torch.Generator(device=device).manual_seed(0)
    codes = prior.sample(16, h, w, device, generator=g)
    new = vq.decode(codes).cpu()
    vae_new = vae.sample(16, device, torch.Generator(device=device).manual_seed(0)).cpu()
    save_grid([x[0] for x in new], os.path.join(fig_dir, "samples_vqvae_seed0.png"), "VQ-VAE + PixelCNN, seed 0")
    save_grid([x[0] for x in vae_new], os.path.join(fig_dir, "samples_vae.png"), "VAE baseline, z ~ N(0, I)")

if __name__ == "__main__":
    main()