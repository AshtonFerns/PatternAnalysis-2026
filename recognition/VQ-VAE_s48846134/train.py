import os

# needed for my AMD GPU 
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn.functional as F
from dataset import get_loaders
from modules import VQVAE, ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vqvae_loss(recon, x, commit):
    recon = recon.float()
    l1 = F.l1_loss(recon, x)
    s = ssim(recon, x)
    loss = l1 + 0.5 * (1 - s) + commit
    return loss, l1, s

def psnr(a, b):
    mse = F.mse_loss(a.float(), b.float(), reduction="none").flatten(1).mean(1).clamp_min(1e-10)
    return 10 * torch.log10(1 / mse)

def lr_at(step, total, base, warmup=500):
    # ramp up learning rate then smoothly drop back down
    if step < warmup:
        return base * (step + 1) / warmup
    return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup))))

import argparse
import json
import math
import time
root = os.path.dirname(os.path.abspath(__file__))
amp = dict(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
# channels for each downsampling factor 
channels_by_f = {4: (32, 64, 128), 8: (32, 64, 128, 256), 16: (32, 64, 128, 256, 256)}

def build(cfg):
    return VQVAE(channels_by_f[cfg["f"]], cfg["n_res"], cfg["dim"], cfg["codes"], cfg["beta"])

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    scores, mses, psnrs = [], [], []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast(**amp):
            recon, _, _, _ = model(x)
        recon = recon.float()
        scores.append(ssim(recon, x, reduction="none"))
        mses.append(F.mse_loss(recon, x, reduction="none").flatten(1).mean(1))
        psnrs.append(psnr(recon, x))
    s, m, p = torch.cat(scores), torch.cat(mses), torch.cat(psnrs)
    return {
        "ssim": s.mean().item(),
        "ssim_min": s.min().item(),
        "frac_ssim_above_0.6": (s > 0.6).float().mean().item(),
        "mse": m.mean().item(),
        "psnr": p.mean().item(),
    }

def train(cfg):
    out = os.path.join(root, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    loaders = get_loaders(cfg["batch"])
    model = build(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{cfg['name']}] {cfg['stage']} with {n_params / 1e6:.2f} M parameters on {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, best_epoch = -1.0, 0
    history = []
    step = 0
    total = cfg["epochs"] * len(loaders["train"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for x, _ in loaders["train"]:
            x = x.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg["lr"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**amp):
                recon, commit, _, stats = model(x)
            loss, l1, s = vqvae_loss(recon, x, commit)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            # reg is the commitment loss and perplexity is how many codebook entries are in use (512 max)
            # restarts is how many dead codes were replaced each step
            row = {"loss": loss.item(), "l1": l1.item(), "ssim": s.item(), "reg": commit.item(),
                   "grad_norm": grad_norm.item() if torch.isfinite(grad_norm) else 0.0, **stats}
            for k, v in row.items():
                sums[k] = sums.get(k, 0) + v
            n += 1
            step += 1
        val = evaluate(model, loaders["validate"])
        rec = {"epoch": epoch, **{f"train_{k}": v / n for k, v in sums.items()},
               **{f"val_{k}": v for k, v in val.items()}, "epoch_s": time.time() - t0}
        history.append(rec)
        tag = ""
        if val["ssim"] > best:
            best, best_epoch, tag = val["ssim"], epoch, " *"
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                os.path.join(out, "best.pt"),
            )
        print(f"epoch {epoch:3d} | loss {rec['train_loss']:.4f} | val SSIM {val['ssim']:.4f} PSNR {val['psnr']:.2f} | "
              f"perplexity {rec['train_perplexity']:.0f} | {rec['epoch_s']:.0f}s{tag}", flush=True)
        json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, loaders["test"])
    test.update(best_epoch=best_epoch, params=n_params, train_minutes=(time.time() - t_start) / 60,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None)
    json.dump(test, open(os.path.join(out, "test.json"), "w"), indent=1)
    print(f"[{cfg['name']}] TEST: " + ", ".join(f"{k}={v:.4f}" for k, v in test.items() if isinstance(v, float)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["vqvae"])
    p.add_argument("--name", default="vqvae")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--f", type=int, default=8, choices=[4, 8, 16]) # downsampling factor
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    cfg = {
        "stage": a.stage,
        "name": a.name,
        "seed": a.seed,
        "epochs": a.epochs,
        "batch": 32,
        "lr": 4e-4,
        "f": a.f,
        "n_res": a.n_res,
        "dim": 64,
        "codes": 512,
        "beta": 0.25,
    }
    train(cfg)
