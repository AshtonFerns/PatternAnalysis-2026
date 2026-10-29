import os

os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from dataset import load_split
from modules import ssim
from train import amp, build, device, root

@torch.no_grad()
def reconstruct(name, x):
    ck = torch.load(os.path.join(root, "runs", name, "best.pt"), map_location=device, weights_only=False)
    model = build(ck["cfg"]).to(device).eval()
    model.load_state_dict(ck["model"])
    with torch.autocast(**amp):
        recon, _, _, _ = model(x)
    return recon.float()

def main():
    data = load_split("test")
    i = next(i for i, key in enumerate(data["keys"]) if key[2] == 30)
    x = data["images"][i:i + 1].to(device)
    patient, week, slice_i = data["keys"][i]
    panels = [("original", x)]
    for f in (4, 8, 16):
        panels.append((f"f={f}", reconstruct(f"sweep_f{f}", x)))
    fig, axes = plt.subplots(1, 4, figsize=(8, 5))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        if title == "original":
            ax.set_title(f"case {patient} week {week} slice {slice_i}")
        else:
            score = ssim(img, x).item()
            ax.set_title(f"{title}\nSSIM {score:.3f}")
    fig.tight_layout()
    out = os.path.join(root, "figures", "downsample_testing_reconstructions.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

if __name__ == "__main__":
    main()
