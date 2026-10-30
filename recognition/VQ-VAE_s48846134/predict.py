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

def save_rows(rows, path):
    n = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), n, figsize=(1.5 * n, 2.6 * len(rows)))
    for r, (title, imgs) in enumerate(rows):
        for c, img in enumerate(imgs):
            axes[r, c].imshow(show(img), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(title, fontsize=9)
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
    n = 8
    train = load_split("train")
    # sample slices from random patients evenly distributed through a 3D scan
    levels = torch.linspace(0, 59, n).round().long().tolist()
    patients = sorted({k[0] for k in train["keys"]})
    chosen = [patients[i] for i in torch.randperm(len(patients), generator=torch.Generator().manual_seed(0))[:n]]
    real = []
    for patient, level in zip(chosen, levels):
        i = next(i for i, k in enumerate(train["keys"]) if k[0] == patient and k[2] == int(level))
        real.append(train["images"][i, 0])
    pixelcnn = vq.decode(prior.sample(n, h, w, device, generator=torch.Generator(device=device).manual_seed(0))).cpu()
    random_codes = torch.randint(0, prior.num_codes, (n, h, w), generator=torch.Generator().manual_seed(0))
    random_vq = vq.decode(random_codes.to(device)).cpu()
    random_vae = vae.sample(n, device, torch.Generator(device=device).manual_seed(0)).cpu()
    save_rows([
        ("Real slices", real),
        ("VQ-VAE + PixelCNN", [x[0] for x in pixelcnn]),
        ("VQ-VAE, random codes", [x[0] for x in random_vq]),
        ("VAE, random latent", [x[0] for x in random_vae]),
    ], os.path.join(fig_dir, "samples.png"))

if __name__ == "__main__":
    main()