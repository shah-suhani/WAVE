import os
import sys
import yaml
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from dataset import FlatVelDataset
from models  import Generator


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def denorm_velocity(v, v_min=1500.0, v_max=4500.0):
    return (v + 1.0) / 2.0 * (v_max - v_min) + v_min


def visualize(cfg, n_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = cfg.get("checkpoint_path")
    if not ckpt or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    G = Generator(in_channels=cfg["seismic_channels"]).to(device)
    G.load_state_dict(torch.load(ckpt, map_location=device))
    G.eval()

    data_root = cfg["data_root"]
    dataset = FlatVelDataset(
        seismic_dir  = os.path.join(data_root, "val", "seismic"),
        velocity_dir = os.path.join(data_root, "val", "velocity"),
    )

    os.makedirs(cfg["output_dir"], exist_ok=True)

    for idx in range(min(n_samples, len(dataset))):
        seismic_t, vel_real_t = dataset[idx]

        with torch.no_grad():
            vel_pred_t = G(seismic_t.unsqueeze(0).to(device))

        seismic_np  = seismic_t[0].numpy()
        vel_real_np = denorm_velocity(vel_real_t[0].numpy())
        vel_pred_np = denorm_velocity(vel_pred_t[0, 0].cpu().numpy())

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Sample {idx}")

        axes[0].imshow(seismic_np, aspect="auto", cmap="gray")
        axes[0].set_title("Seismic (source 0)")

        im = axes[1].imshow(vel_real_np, aspect="auto", cmap="jet", vmin=1500, vmax=4500)
        axes[1].set_title("Ground Truth")
        fig.colorbar(im, ax=axes[1], label="m/s")

        im = axes[2].imshow(vel_pred_np, aspect="auto", cmap="jet", vmin=1500, vmax=4500)
        axes[2].set_title("Predicted")
        fig.colorbar(im, ax=axes[2], label="m/s")

        plt.tight_layout()
        path = os.path.join(cfg["output_dir"], f"sample_{idx:04d}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    visualize(load_config(cfg_path), n_samples=4)