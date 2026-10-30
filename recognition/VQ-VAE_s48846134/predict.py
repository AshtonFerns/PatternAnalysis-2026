import json
import os
import time

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import load_split
from modules import ssim
from train import device, load, root

fig_dir = os.path.join(root, "figures")
res_dir = os.path.join(root, "results")

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

@torch.no_grad()
def nearest(queries, bank, chunk=64):
    bank = bank.flatten(1).float().to(device)
    bank_sq = bank.pow(2).sum(1)
    dist, idx = [], []
    for i in range(0, len(queries), chunk):
        q = queries[i:i + chunk].flatten(1).float().to(device)
        dist2 = q.pow(2).sum(1, keepdim=True) - 2 * q @ bank.t() + bank_sq
        v, j = dist2.min(1)
        dist.append((v.clamp_min(0) / bank.shape[1]).sqrt().cpu())
        idx.append(j.cpu())
    return torch.cat(dist), torch.cat(idx)

def nearest_rmse(queries, bank, chunk=64):
    return nearest(queries, bank, chunk)[0]

def summarise(dist):
    return {"median": dist.median().item(), "minimum": dist.min().item()}

@torch.no_grad()
def memorisation():
    os.makedirs(res_dir, exist_ok=True)
    vq, _, _ = load("vqvae")
    prior, pcfg, _ = load("prior")
    vae, _, _ = load("vae")
    h, w = pcfg["grid"]
    train = load_split("train")["images"].float()
    test = load_split("test")["images"].float()
    pixelcnn, vae_imgs = [], []
    for seed in range(8):
        g = torch.Generator(device=device).manual_seed(seed)
        pixelcnn.append(vq.decode(prior.sample(125, h, w, device, generator=g)).cpu())
        vae_imgs.append(vae.sample(125, device, g).cpu())
    saved = {"pixelcnn": torch.cat(pixelcnn), "vae": torch.cat(vae_imgs)}
    torch.save(saved, os.path.join(res_dir, "samples.pt"))
    report = {
        "real test patients": summarise(nearest_rmse(test, train)),
        "PixelCNN samples": summarise(nearest_rmse(saved["pixelcnn"].float(), train)),
        "VAE samples": summarise(nearest_rmse(saved["vae"].float(), train)),
    }
    json.dump(report, open(os.path.join(res_dir, "memorisation.json"), "w"), indent=1)

def median_seconds(fn, warm=1, runs=3):
    for _ in range(warm):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]

@torch.no_grad()
def resources():
    os.makedirs(res_dir, exist_ok=True)
    vq, _, _ = load("vqvae")
    prior, pcfg, _ = load("prior")
    vae, _, _ = load("vae")
    h, w = pcfg["grid"]
    n = 32
    report = {
        "PixelCNN samples": {"seconds_per_slice": median_seconds(lambda: vq.decode(prior.sample(n, h, w, device))) / n},
        "VAE samples": {"seconds_per_slice": median_seconds(lambda: vae.sample(n, device)) / n},
    }
    json.dump(report, open(os.path.join(res_dir, "resources.json"), "w"), indent=1)

@torch.no_grad()
def interpolation():
    os.makedirs(fig_dir, exist_ok=True)
    vq, _, _ = load("vqvae")
    vae, _, _ = load("vae")
    test = load_split("test")
    images = test["images"].float()
    a = next(i for i, k in enumerate(test["keys"]) if k[0] == test["keys"][0][0] and k[2] == 30)
    b = next(i for i, k in enumerate(test["keys"]) if k[0] != test["keys"][0][0] and k[2] == 30)
    xa, xb = images[a:a + 1].to(device), images[b:b + 1].to(device)
    steps = torch.linspace(0, 1, 8, device=device)
    za, zb = vq.encoder(xa), vq.encoder(xb)
    vq_row = vq.decoder(vq.quantizer(torch.cat([(1 - t) * za + t * zb for t in steps]))[0])
    ma, mb = vae(xa)[2], vae(xb)[2]
    vae_row = vae.decode(torch.cat([(1 - t) * ma + t * mb for t in steps]))
    fig, axes = plt.subplots(2, 10, figsize=(14, 6))
    for r, (name, seq) in enumerate((("VQ-VAE", vq_row), ("VAE", vae_row))):
        cols = [xa[0, 0]] + [img[0] for img in seq] + [xb[0, 0]]
        for c, img in enumerate(cols):
            axes[r, c].imshow(show(img), cmap="gray", vmin=0, vmax=1)
            axes[r, c].axis("off")
            if c == 0:
                title = "patient A"
            elif c == 9:
                title = "patient B"
            else:
                title = f"{name} t={steps[c - 1].item():.2f}"
            axes[r, c].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "interpolation.png"), dpi=100)
    plt.close(fig)

def same_patient_rmse(images, patients, n=500):
    pick = torch.randperm(len(images), generator=torch.Generator().manual_seed(2))[:n]
    bank = images.flatten(1).float().to(device)
    bank_sq = bank.pow(2).sum(1)
    out = []
    for i in range(0, n, 64):
        idx = pick[i:i + 64]
        q = images[idx].flatten(1).float().to(device)
        dist2 = q.pow(2).sum(1, keepdim=True) - 2 * q @ bank.t() + bank_sq
        forbid = (patients[idx, None] != patients[None]) | (idx[:, None] == torch.arange(len(patients)))
        out.append((dist2.masked_fill(forbid.to(device), float("inf")).min(1).values.clamp_min(0) / bank.shape[1]).sqrt().cpu())
    return torch.cat(out)

@torch.no_grad()
def memorisation_figure():
    os.makedirs(fig_dir, exist_ok=True)
    vq, _, _ = load("vqvae")
    train = load_split("train")
    test = load_split("test")["images"].float()
    images = train["images"].float()
    saved = torch.load(os.path.join(res_dir, "samples.pt"), weights_only=False)
    recon = torch.cat([vq(test[i:i + 32].to(device))[0].cpu() for i in range(0, len(test), 32)])
    self_rmse = []
    for i in range(0, len(images), 32):
        x = images[i:i + 32].to(device)
        y = vq(x)[0].float()
        self_rmse.append((y - x.float()).flatten(1).pow(2).mean(1).sqrt().cpu())
    patients = torch.tensor([k[0] for k in train["keys"]])
    groups = (
        ("Final generated samples", nearest_rmse(saved["pixelcnn"].float(), images)),
        ("VAE samples", nearest_rmse(saved["vae"].float(), images)),
        ("test slices", nearest_rmse(test, images)),
        ("VQ-VAE reconstructions", nearest_rmse(recon, images)),
        ("train slice to same patient", same_patient_rmse(images, patients)),
        ("train slice to reconstruction", torch.cat(self_rmse)),
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = torch.linspace(0, 0.35, 400)
    bandwidth = 0.005
    for name, dist in groups:
        z = (xs[:, None] - dist.float()[None, :]) / bandwidth
        y = torch.exp(-0.5 * z.pow(2)).mean(1) / (bandwidth * (2 * torch.pi) ** 0.5)
        line, = ax.plot(xs.numpy(), y.numpy(), linewidth=1.6, label=name)
        ax.fill_between(xs.numpy(), y.numpy(), color=line.get_color(), alpha=0.15)
    ax.axvline(groups[2][1].median().item() / 2, color="black", linestyle="--", linewidth=1, label="half the test median")
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, None)
    ax.set_xlabel("RMSE to the nearest training slice")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "memorisation_distances.png"), dpi=100)
    plt.close(fig)

def memorisation_pairs():
    os.makedirs(fig_dir, exist_ok=True)
    train = load_split("train")
    images = train["images"].float()
    samples = torch.load(os.path.join(res_dir, "samples.pt"), weights_only=False)["pixelcnn"].float()
    dist, idx = nearest(samples, images)
    chosen = list(range(8)) + dist.argsort()[:4].tolist()
    fig, axes = plt.subplots(3, 8, figsize=(14, 8))
    for n, i in enumerate(chosen):
        r, p = divmod(n, 4)
        c = p * 2
        neighbour = images[idx[i]:idx[i] + 1]
        score = ssim(samples[i:i + 1], neighbour, "none").item()
        for col, img, title in (
            (c, samples[i, 0], f"sample {i}"),
            (c + 1, neighbour[0, 0], "nearest case {} w{} s{}\nRMSE {:.3f} SSIM {:.2f}".format(*train["keys"][idx[i]], dist[i], score)),
        ):
            axes[r, col].imshow(show(img), cmap="gray", vmin=0, vmax=1)
            axes[r, col].set_title(title, fontsize=7)
            axes[r, col].axis("off")
    fig.suptitle("Generated slice vs nearest training slice")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "memorisation_pairs.png"), dpi=100)
    plt.close(fig)

if __name__ == "__main__":
    main()
    memorisation()
    resources()
    interpolation()
    memorisation_figure()
    memorisation_pairs()