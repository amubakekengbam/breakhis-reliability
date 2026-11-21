"""Model, dataset and a simple training loop (PyTorch). Designed for a Colab T4 GPU."""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BreakHisDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        aug = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
               transforms.ColorJitter(0.1, 0.1, 0.1, 0.02)] if train else []
        self.tf = transforms.Compose([transforms.Resize((img_size, img_size)), *aug,
                                      transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("RGB")
        return self.tf(img), torch.tensor(row["label"], dtype=torch.float32)


def make_model(arch: str = "resnet18") -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
    elif arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int,
                      arch: str = "resnet18", epochs: int = 5, batch_size: int = 64,
                      lr: float = 3e-4, num_workers: int = 2) -> np.ndarray:
    """Train on train_df for a fixed number of epochs; return P(malignant) for test_df."""
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(arch).to(device)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(BreakHisDataset(train_df, train=True), batch_size=batch_size,
                          shuffle=True, num_workers=num_workers, generator=g)
    test_dl = DataLoader(BreakHisDataset(test_df, train=False), batch_size=batch_size,
                         num_workers=num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Weight the loss so the smaller benign class isn't ignored.
    pos = train_df["label"].mean()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=device))
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = loss_fn(model(x).squeeze(1), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(y)
        print(f"  epoch {epoch + 1}/{epochs}  train loss {total / len(train_df):.4f}")

    model.eval()
    probs = []
    with torch.no_grad():
        for x, _ in test_dl:
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                probs.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu())
    return torch.cat(probs).numpy()
