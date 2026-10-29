import os

# needed for my AMD GPU 
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn.functional as F
from dataset import get_loaders
from modules import VQVAE, ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vqvae_loss(model, x):
    recon, commit, _, stats = model(x)
    recon = recon.float()
    l1 = F.l1_loss(recon, x)
    loss = l1 + 0.5 * (1 - ssim(recon, x)) + commit
    return loss, stats

import argparse
import json
root = os.path.dirname(os.path.abspath(__file__))
amp = dict(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")
# channels for each downsampling factor 
channels_by_f = {4: (32, 64, 128), 8: (32, 64, 128, 256), 16: (32, 64, 128, 256, 256)}

def build(cfg):
    return VQVAE(channels_by_f[cfg["f"]], cfg["n_res"], cfg["dim"], cfg["codes"], cfg["beta"])

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    scores = []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast(**amp):
            recon, _, _, _ = model(x)
        scores.append(ssim(recon.float(), x, reduction="none"))
    s = torch.cat(scores)
    return {
        "ssim": s.mean().item(), 
        "ssim_min": s.min().item(), 
        "frac_ssim_above_0.6": (s > 0.6).float().mean().item()
    }

def train(cfg):
    out = os.path.join(root, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    loaders = get_loaders(cfg["batch"])
    model = build(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.99), weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, best_epoch = -1.0, 0
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for x, _ in loaders["train"]:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**amp):
                loss, stats = vqvae_loss(model, x)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        val = evaluate(model, loaders["validate"])
        rec = {"epoch": epoch, **{f"val_{k}": v for k, v in val.items()}}
        history.append(rec)
        if val["ssim"] > best:
            best, best_epoch = val["ssim"], epoch
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                os.path.join(out, "best.pt"),
            )
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, loaders["test"])
    test.update(best_epoch=best_epoch, best_val_ssim=best) 
    json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    json.dump(test, open(os.path.join(out, "test.json"), "w"), indent=1)

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
