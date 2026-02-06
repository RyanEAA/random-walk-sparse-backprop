import argparse
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Import your masker
from randomwalkmask import RandomWalkMasker


# ----------------------------
# Models
# ----------------------------

class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallCNN(nn.Module):
    """
    CIFAR-10 friendly CNN. We'll apply RW masking only to the Linear classifier head.
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
        )
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ----------------------------
# Data
# ----------------------------

def get_loaders(dataset: str, batch_size: int, num_workers: int = 2):
    dataset = dataset.lower()

    if dataset in ("mnist", "fashionmnist"):
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        ds_cls = datasets.MNIST if dataset == "mnist" else datasets.FashionMNIST
        train_ds = ds_cls(root="./data", train=True, download=True, transform=tfm)
        test_ds = ds_cls(root="./data", train=False, download=True, transform=tfm)

    elif dataset == "cifar10":
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])

        train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm_train)
        test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm_test)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def build_model(dataset: str) -> nn.Module:
    dataset = dataset.lower()
    if dataset in ("mnist", "fashionmnist"):
        # 28x28 grayscale
        return SimpleMLP(input_dim=28 * 28, num_classes=10)
    elif dataset == "cifar10":
        return SmallCNN(num_classes=10)
    else:
        raise ValueError(dataset)


# ----------------------------
# Train / Eval
# ----------------------------

@dataclass
class EpochStats:
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    density: Optional[float] = None
    epoch_time_sec: Optional[float] = None


@torch.no_grad()
def eval_epoch(model: nn.Module, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_epoch(
    model: nn.Module,
    loader,
    optimizer,
    device,
    mode: str,
    masker: Optional[RandomWalkMasker] = None,
) -> Tuple[float, float, Optional[float]]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    densities = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        if mode != "full":
            if masker is None:
                raise ValueError("masker is required for mode != 'full'")
            stats = masker.apply(model, labels=y if "target" in mode else None)
            densities.append(stats.density)

        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    acc = correct / total
    avg_density = float(sum(densities) / len(densities)) if densities else None
    return avg_loss, acc, avg_density


# ----------------------------
# Main experiment runner
# ----------------------------

def run_one(dataset: str, mode: str, num_paths: int, epochs: int, batch_size: int, lr: float, device: str):
    train_loader, test_loader = get_loaders(dataset, batch_size=batch_size)

    model = build_model(dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Build masker for sparse modes
    masker = None
    if mode != "full":
        masker = RandomWalkMasker(
            mode=mode,  # "rw_random" or "rw_target"
            num_paths=num_paths,
            include_bias=True,
            device=torch.device(device),
        )

    if mode == "full":
        print(f"\n=== Dataset={dataset.upper()} | Mode={mode} | epochs={epochs} ===")
    else:
        print(f"\n=== Dataset={dataset.upper()} | Mode={mode} | num_paths={num_paths} | epochs={epochs} ===")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, density = train_epoch(model, train_loader, optimizer, device, mode, masker)
        te_loss, te_acc = eval_epoch(model, test_loader, device)
        t1 = time.time()

        if density is None:
            print(f"[Epoch {ep}] Train loss {tr_loss:.4f}, acc {tr_acc:.4f} | "
                  f"Test loss {te_loss:.4f}, acc {te_acc:.4f} | time {t1-t0:.1f}s")
        else:
            print(f"[Epoch {ep}] Train loss {tr_loss:.4f}, acc {tr_acc:.4f}, density {density:.6f} | "
                  f"Test loss {te_loss:.4f}, acc {te_acc:.4f} | time {t1-t0:.1f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist", "cifar10", "all"])
    p.add_argument("--mode", type=str, default="full", choices=["full", "rw_random", "rw_target"])
    p.add_argument("--num_paths", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.dataset == "all":
        for ds in ["mnist", "fashionmnist", "cifar10"]:
            run_one(ds, args.mode, args.num_paths, args.epochs, args.batch_size, args.lr, args.device)
    else:
        run_one(args.dataset, args.mode, args.num_paths, args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
