import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class FlatVelDataset(Dataset):
    VEL_MIN = 1500.0
    VEL_MAX = 4500.0

    def __init__(self, seismic_dir, velocity_dir, time_downsample=14):
        self.seismic_dir = seismic_dir
        self.velocity_dir = velocity_dir
        self.time_downsample = time_downsample

        self.seismic_files = sorted([f for f in os.listdir(seismic_dir) if f.endswith(".npy")])
        self.velocity_files = sorted([f for f in os.listdir(velocity_dir) if f.endswith(".npy")])

        assert len(self.seismic_files) == len(self.velocity_files), \
            f"Mismatch: {len(self.seismic_files)} seismic vs {len(self.velocity_files)} velocity files"

    def __len__(self):
        return len(self.seismic_files)

    def __getitem__(self, idx):
        seismic  = np.load(os.path.join(self.seismic_dir,  self.seismic_files[idx]))
        velocity = np.load(os.path.join(self.velocity_dir, self.velocity_files[idx]))

        # seismic: (5, 1, 1000, 70) -> (5, 70, 70)
        seismic = seismic[:, 0, :, :]
        seismic = seismic[:, ::self.time_downsample, :]
        seismic = seismic[:, :70, :]
        seismic = (seismic - seismic.mean()) / (seismic.std() + 1e-8)

        # velocity: (1, 70, 70), normalize to [-1, 1]
        velocity = (velocity - self.VEL_MIN) / (self.VEL_MAX - self.VEL_MIN)
        velocity = velocity * 2.0 - 1.0

        return torch.from_numpy(seismic.astype(np.float32)), \
               torch.from_numpy(velocity.astype(np.float32))


def get_dataloaders(cfg):
    data_root = cfg["data_root"]

    train_dataset = FlatVelDataset(
        seismic_dir  = os.path.join(data_root, "train", "seismic"),
        velocity_dir = os.path.join(data_root, "train", "velocity"),
    )
    val_dataset = FlatVelDataset(
        seismic_dir  = os.path.join(data_root, "val", "seismic"),
        velocity_dir = os.path.join(data_root, "val", "velocity"),
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg["batch_size"], shuffle=False, num_workers=2)

    return train_loader, val_loader