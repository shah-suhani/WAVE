import os
import sys
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataloaders
from models  import Generator, Discriminator


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    train_loader, val_loader = get_dataloaders(cfg)

    G = Generator(in_channels=cfg["seismic_channels"]).to(device)
    D = Discriminator().to(device)

    criterion_adv = nn.BCEWithLogitsLoss()
    criterion_rec = nn.L1Loss()

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg["learning_rate_g"], betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg["learning_rate_d"], betas=(0.5, 0.999))

    for epoch in range(1, cfg["num_epochs"] + 1):
        G.train()
        D.train()
        total_G, total_D = 0.0, 0.0

        for seismic, vel_real in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg['num_epochs']}", leave=False):
            seismic  = seismic.to(device)
            vel_real = vel_real.to(device)
            B = seismic.size(0)

            real_label = torch.ones(B, 1, device=device) * 0.9
            fake_label = torch.zeros(B, 1, device=device)

            opt_D.zero_grad()
            loss_D = (criterion_adv(D(vel_real), real_label) +
                      criterion_adv(D(G(seismic).detach()), fake_label)) * 0.5
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            vel_fake = G(seismic)
            loss_G = criterion_adv(D(vel_fake), real_label) + cfg["lambda_l1"] * criterion_rec(vel_fake, vel_real)
            loss_G.backward()
            opt_G.step()

            total_G += loss_G.item()
            total_D += loss_D.item()

        avg_G = total_G / len(train_loader)
        avg_D = total_D / len(train_loader)

        G.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for seismic, vel_real in val_loader:
                seismic  = seismic.to(device)
                vel_real = vel_real.to(device)
                val_l1 += nn.functional.l1_loss(G(seismic), vel_real).item()
        val_l1 /= len(val_loader)

        print(f"Epoch {epoch:3d} | Loss_G: {avg_G:.4f} | Loss_D: {avg_D:.4f} | Val L1: {val_l1:.4f}")

        if epoch % cfg["checkpoint_save_every"] == 0:
            torch.save(G.state_dict(), os.path.join(cfg["checkpoint_dir"], f"generator_epoch{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(cfg["checkpoint_dir"], f"discriminator_epoch{epoch}.pth"))
            print(f"  -> checkpoint saved at epoch {epoch}")

    print("Done.")


if __name__ == "__main__":
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    train(load_config(cfg_path))