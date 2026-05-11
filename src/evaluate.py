import os
import sys
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders
from models  import Generator

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    print("scikit-image not installed, skipping SSIM")


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = cfg.get("checkpoint_path")
    if not ckpt:
        raise ValueError("Set checkpoint_path in config.yaml before evaluating.")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    G = Generator(in_channels=cfg["seismic_channels"]).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    _, val_loader = get_dataloaders(cfg)

    total_l1, total_mse, total_ssim, n = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for seismic, vel_real in val_loader:
            seismic  = seismic.to(device)
            vel_real = vel_real.to(device)
            vel_pred = G(seismic)
            B = seismic.size(0)

            total_l1  += nn.functional.l1_loss(vel_pred, vel_real, reduction="sum").item()
            total_mse += nn.functional.mse_loss(vel_pred, vel_real, reduction="sum").item()

            if SSIM_AVAILABLE:
                pred_np = vel_pred.cpu().numpy()
                real_np = vel_real.cpu().numpy()
                for i in range(B):
                    total_ssim += ssim_fn(real_np[i, 0], pred_np[i, 0], data_range=2.0)

            n += B

    print(f"\nSamples: {n}")
    print(f"L1  (normalised): {total_l1 / n:.6f}")
    print(f"MSE (normalised): {total_mse / n:.6f}")
    if SSIM_AVAILABLE:
        print(f"SSIM:             {total_ssim / n:.4f}")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    evaluate(load_config(cfg_path))