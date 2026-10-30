import os

# needed for my AMD GPU 
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import torch
import torch.nn.functional as F
from dataset import check_patient_leakage, get_loaders, load_split
from modules import GatedPixelCNN, VAE, VQVAE, ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vqvae_loss(recon, x, commit):
    recon = recon.float()
    l1 = F.l1_loss(recon, x)
    s = ssim(recon, x)
    loss = l1 + 0.5 * (1 - s) + commit
    return loss, l1, s

def psnr(a, b):
    # per-image squared error
    mse = F.mse_loss(a.float(), b.float(), reduction="none").flatten(1).mean(1).clamp_min(1e-10)
    # peak value in the formula is 1 because the slices sit between 0 and 1
    return 10 * torch.log10(1 / mse)

def lr_at(step, total, base, warmup=500):
    # ramp up learning rate then smoothly drop back down
    if step < warmup:
        return base * (step + 1) / warmup
    # cosine falloff toward 5 percent of the base rate
    # the max keeps the divisor at least 1 when the run is shorter than the warmup
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
    if cfg["stage"] == "prior":
        return GatedPixelCNN(cfg["codes"], cfg["prior_dim"], cfg["prior_layers"], 5, cfg["dropout"])
    if cfg["stage"] == "vae":
        # 16x downsampling, latent vector of 256 numbers
        return VAE(channels_by_f[16], cfg["n_res"], cfg["z_dim"])
    return VQVAE(channels_by_f[cfg["f"]], cfg["n_res"], cfg["dim"], cfg["codes"], cfg["beta"])

def load(name):
    # load the weights and their config
    ck = torch.load(os.path.join(root, "runs", name, "best.pt"), map_location=device, weights_only=False)
    model = build(ck["cfg"]).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), ck["cfg"], ck

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    scores, mses, psnrs = [], [], []
    for x, _ in loader:
        x = x.to(device)
        with torch.autocast(**amp):
            recon = model(x)[0]
        recon = recon.float()
        # keep one score per slice for the worst case and the fraction above 0.6
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
    # one step per optimiser update
    total = cfg["epochs"] * len(loaders["train"])
    if device.type == "cuda":
        # start the peak memory measurement from the training loop
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, sums, n = time.time(), {}, 0
        for x, _ in loaders["train"]:
            x = x.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, cfg["lr"])
            # drop the stored gradients
            opt.zero_grad(set_to_none=True)
            with torch.autocast(**amp):
                if cfg["stage"] == "vae":
                    recon, kl, _ = model(x)
                else:
                    recon, kl, _, stats = model(x)
            if cfg["stage"] == "vae":
                recon_f = recon.float()
                l1 = F.l1_loss(recon_f, x)
                s = ssim(recon_f, x)
                # warm the KL up over 3000 steps. 0.05 matches a Laplace likelihood of scale 0.05.
                # divide by the number of pixels so the KL is on the same scale as L1.
                loss = l1 + 0.5 * (1 - s) + min(1, step / 3000) * cfg["kl_weight"] * kl / x[0].numel()
                stats = {"kl": kl.item()}
            else:
                loss, l1, s = vqvae_loss(recon, x, kl)
                # kl is the commitment loss here. perplexity is codes in use. restarts are dead codes replaced.
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            row = {"loss": loss.item(), "l1": l1.item(), "ssim": s.item(), "reg": kl.item(),
                   # store non-finite norms as zero for history plotting
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
            # keep the weights with the best validation structure score
            best, best_epoch, tag = val["ssim"], epoch, " *"
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                os.path.join(out, "best.pt"),
            )
        extra = f"perplexity {rec['train_perplexity']:.0f}" if cfg["stage"] == "vqvae" else f"KL {rec['train_kl']:.0f}"
        print(f"epoch {epoch:3d} | loss {rec['train_loss']:.4f} | val SSIM {val['ssim']:.4f} PSNR {val['psnr']:.2f} | "
              f"{extra} | {rec['epoch_s']:.0f}s{tag}", flush=True)
        json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
    ck = torch.load(os.path.join(out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    test = evaluate(model, loaders["test"])
    test.update(best_epoch=best_epoch, params=n_params, train_minutes=(time.time() - t_start) / 60,
                peak_vram_gb=torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None)
    json.dump(test, open(os.path.join(out, "test.json"), "w"), indent=1)
    print(f"[{cfg['name']}] TEST: " + ", ".join(f"{k}={v:.4f}" for k, v in test.items() if isinstance(v, float)))

@torch.no_grad()
def encode_all(vq, images, bs=64):
    out = []
    for i in range(0, len(images), bs):
        with torch.autocast(**amp):
            out.append(vq.encode(images[i:i + bs].float().to(device)).cpu())
    return torch.cat(out)

@torch.no_grad()
def nll(prior, codes, bs=64):
    # negative log likelihood in nats averaged over every code position
    prior.eval()
    total = 0.0
    for i in range(0, len(codes), bs):
        z = codes[i:i + bs].to(device)
        with torch.autocast(**amp):
            total += F.cross_entropy(prior(z).float(), z, reduction="sum").item()
    return total / codes.numel()

def train_prior(cfg):
    out = os.path.join(root, "runs", cfg["name"])
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(cfg["seed"])
    vq, vq_cfg, _ = load(cfg["vq_run"])
    cfg["codes"] = vq_cfg["codes"]
    check_patient_leakage()
    # train prior on frozen code grids from VQ-VAE
    codes = {s: encode_all(vq, load_split(s)["images"]) for s in ("train", "validate", "test")}
    torch.save(codes, os.path.join(out, "codes.pt"))
    cfg["grid"] = list(codes["train"].shape[1:])
    prior = build(cfg).to(device)
    n_params = sum(p.numel() for p in prior.parameters())
    print(f"[{cfg['name']}] PixelCNN with {n_params / 1e6:.2f} M parameters, code grid {cfg['grid']}")
    opt = torch.optim.AdamW(prior.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    steps = len(codes["train"]) // cfg["batch"]
    # start with loss as infinite
    total, step, best, bad, history = cfg["epochs"] * steps, 0, float("inf"), 0, []
    # separate generator so this does not affect the global random stream
    g = torch.Generator().manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        prior.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(codes["train"]), generator=g)
        for b in range(steps):
            z = codes["train"][perm[b * cfg["batch"]:(b + 1) * cfg["batch"]]].to(device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(step, total, cfg["lr"])
            with torch.autocast(**amp):
                loss = F.cross_entropy(prior(z).float(), z)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            step += 1
        val = nll(prior, codes["validate"])
        history.append({"epoch": epoch, "train_nll": tot / steps, "val_nll": val, "epoch_s": time.time() - t0})
        tag = ""
        if val < best:
            best, bad, tag = val, 0, " *"
            torch.save({"model": prior.state_dict(), "cfg": cfg, "epoch": epoch}, os.path.join(out, "best.pt"))
        else:
            bad += 1
        print(f"epoch {epoch:3d} | train NLL {tot / steps:.4f} | val NLL {val:.4f} ({val / math.log(2):.3f} bits/code) | "
              f"{history[-1]['epoch_s']:.0f}s{tag}", flush=True)
        json.dump(history, open(os.path.join(out, "history.json"), "w"), indent=1)
        # stop once validation has not improved for 8 epochs
        if bad >= cfg["patience"]:
            print(f"early stop: no validation improvement for {bad} epochs")
            break
    prior, _, ck = load(cfg["name"])
    t = nll(prior, codes["test"])
    test = {"nll": t, "bits_per_code": t / math.log(2), "best_epoch": ck["epoch"], "params": n_params,
            # epoch numbers start at 1 and the history list starts at 0
            "train_nll_at_best": history[ck["epoch"] - 1]["train_nll"], "val_nll_at_best": history[ck["epoch"] - 1]["val_nll"],
            "train_minutes": (time.time() - t_start) / 60,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else None}
    json.dump(test, open(os.path.join(out, "test.json"), "w"), indent=1)
    print(f"[{cfg['name']}] TEST NLL {t:.4f} nats/code = {test['bits_per_code']:.3f} bits/code")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["vqvae", "prior", "vae"])
    p.add_argument("--name")
    p.add_argument("--epochs", type=int)
    p.add_argument("--f", type=int, default=8, choices=[4, 8, 16]) # downsampling factor
    p.add_argument("--n-res", type=int, default=2)
    p.add_argument("--vq-run", default="vqvae")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    if a.stage == "prior":
        cfg = {
            "stage": "prior",
            "name": a.name or "prior",
            "seed": a.seed,
            "epochs": a.epochs or 60, 
            "batch": 64,
            "lr": 3e-4,
            "codes": 512,
            "prior_dim": 128, # width of each gated layer
            "prior_layers": 18,
            "dropout": 0.1,
            "patience": 8, # consecutive epochs allowed with no loss improvement
            "vq_run": a.vq_run,
        }
        print(f"device {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
        train_prior(cfg)
    elif a.stage == "vae":
        cfg = {
            "stage": "vae",
            "name": a.name or "vae",
            "seed": a.seed,
            "epochs": a.epochs or 30,
            "batch": 32,
            "lr": 4e-4,
            "n_res": a.n_res,
            "z_dim": 256, # length of the gaussian vector
            "kl_weight": 0.05,
        }
        print(f"device {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
        train(cfg)
    else:
        cfg = {
            "stage": a.stage,
            "name": a.name or "vqvae",
            "seed": a.seed,
            "epochs": a.epochs or 30,
            "batch": 32,
            "lr": 4e-4,
            "f": a.f,
            "n_res": a.n_res,
            "dim": 64, # width of each code vector
            "codes": 512, # number of vectors in the codebook
            "beta": 0.25, # weight on the commitment term
        }
        print(f"device {device}: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
        train(cfg)
