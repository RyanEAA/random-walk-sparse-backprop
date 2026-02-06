#!/usr/bin/env python3
"""
run_experiments.py

Runs training for multiple num_paths values and logs results to metrics.csv.

Design:
- Training is script-based, reproducible.
- Plotting is separate (see plot_results.py).
- Logs per-epoch metrics:
    test_acc, train_loss, elapsed_s, cumulative_param_updates, avg_step_ms

Works with:
- Option B (row-sparse backward) in randomwalk.py:
    - RandomWalkRowSampler
    - RandomWalkConfig
    - SparseMLPWrapper

Notes:
- This script uses MNIST/FashionMNIST by default.
- CIFAR-10 later once we wrap conv layers or use sparse only on classifier head.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# torchvision is typical for MNIST/FashionMNIST
from torchvision import datasets, transforms

# Import your Option B components
from randomwalk import (
    RandomWalkConfig,
    RandomWalkRowSampler,
    SparseMLPWrapper,
)


# ----------------------------
# Simple MLP (matches randomwalk.py demo expectations)
# ----------------------------

class SimpleMLP(nn.Module):
    def __init__(self, in_dim=784, h1=256, h2=128, out_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ----------------------------
# Utils
# ----------------------------

def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return torch.device("cuda")
    if device_str == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


def count_param_updates_for_selected_rows(
    sparse_wrapper: SparseMLPWrapper,
    selected_rows_fwd: List[torch.Tensor],
) -> int:
    """
    Row-sparse Option B update count per step.

    For each Linear layer with weight shape [out, in]:
      - weights updated: |I| * in
      - bias updated: |I| (if bias present)
    """
    total = 0
    for lin, I in zip(sparse_wrapper.linears_fwd, selected_rows_fwd):
        k = int(I.numel())
        total += k * lin.in_features
        if lin.bias is not None:
            total += k
    return total


def maybe_cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Training (Option B)
# ----------------------------

def train_one_run(
    *,
    run_dir: Path,
    dataset_name: str,
    num_paths: int,
    seed: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    epsilon_greedy: float,
    weight_temperature: float,
    min_rows_per_layer: int,
    log_every_steps: int,
    mode: str, # "full" or "rw_sparse
) -> Path:
    """
    Trains one configuration and appends metrics rows to run_dir/metrics.csv.
    Returns metrics path.
    """
    ensure_dir(run_dir)
    metrics_path = run_dir / "metrics.csv"
    config_path = run_dir / "config.json"

    # Data
    tfm = transforms.Compose([transforms.ToTensor()])
    if dataset_name.lower() == "mnist":
        train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    elif dataset_name.lower() in ("fashionmnist", "fashion_mnist", "fmnist"):
        train_ds = datasets.FashionMNIST(root="./data", train=True, download=True, transform=tfm)
        test_ds = datasets.FashionMNIST(root="./data", train=False, download=True, transform=tfm)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    # Model
    model = SimpleMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Sparse wrapper + sampler
    # We trace with an example input shaped like a batch element:
    example_x, _ = next(iter(train_loader))
    example_x = example_x[:1].to(device)
    sparse_wrapper = None
    sampler = None

    if mode == "rw_sparse":
        sparse_wrapper = SparseMLPWrapper(model, example_input=example_x)

        rw_cfg = RandomWalkConfig(
            num_paths=num_paths,
            start_uniform=True,
            epsilon_greedy=epsilon_greedy,
            weight_temperature=weight_temperature,
            min_rows_per_layer=min_rows_per_layer,
            device=device,
        )
        sampler = RandomWalkRowSampler(rw_cfg)

    config_payload = {
        "timestamp": now_iso(),
        "dataset": dataset_name,
        "mode": mode,
        "num_paths": num_paths if mode == "rw_sparse" else None,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
    }

    if mode == "rw_sparse":
        rw_cfg_dict = asdict(rw_cfg)
        rw_cfg_dict["device"] = str(rw_cfg_dict.get("device"))
        config_payload["randomwalk_config"] = rw_cfg_dict

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)



    # Prepare CSV (append-safe)
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "dataset",
                "mode",
                "num_paths",
                "seed",
                "epoch",
                "step",
                "train_loss",
                "test_acc",
                "elapsed_s",
                "avg_step_ms",
                "cumulative_param_updates",
            ],
        )
        if write_header:
            writer.writeheader()

        run_id = run_dir.name

        cumulative_updates = 0
        global_step = 0

        t0 = time.perf_counter()

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            # step timing window
            step_times: List[float] = []
            window_start = None

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                maybe_cuda_sync(device)
                t_step_start = time.perf_counter()

                opt.zero_grad(set_to_none=True)

                if mode == "full":
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    opt.step()

                    # FULL updates all params
                    updates_this_step = sum(p.numel() for p in model.parameters())

                elif mode == "rw_sparse":
                    selected_rows = sampler.sample_selected_rows(sparse_wrapper.linears_fwd)
                    logits = sparse_wrapper(xb.view(xb.size(0), -1), selected_rows)
                    loss = criterion(logits, yb)
                    loss.backward()
                    opt.step()

                    updates_this_step = count_param_updates_for_selected_rows(
                        sparse_wrapper, selected_rows
                    )

                else:
                    raise ValueError(f"Unknown mode: {mode}")

                cumulative_updates += updates_this_step

                running_loss += loss.item()
                n_batches += 1
                global_step += 1

                maybe_cuda_sync(device)
                t_step_end = time.perf_counter()
                step_times.append((t_step_end - t_step_start) * 1000.0)

                # Optional: log intra-epoch (not required for plots; useful for long runs)
                if log_every_steps > 0 and (global_step % log_every_steps == 0):
                    elapsed = time.perf_counter() - t0
                    avg_step_ms = sum(step_times[-min(len(step_times), 50):]) / max(min(len(step_times), 50), 1)
                    # quick test acc occasionally is expensive; skip
                    writer.writerow(
                        {
                            "run_id": run_id,
                            "dataset": dataset_name,
                            "mode": mode,
                            "num_paths": num_paths,
                            "seed": seed,
                            "epoch": epoch,
                            "step": global_step,
                            "train_loss": running_loss / max(n_batches, 1),
                            "test_acc": "",
                            "elapsed_s": elapsed,
                            "avg_step_ms": avg_step_ms,
                            "cumulative_param_updates": cumulative_updates,
                        }
                    )
                    f.flush()

            # Epoch end eval
            elapsed = time.perf_counter() - t0
            train_loss_epoch = running_loss / max(n_batches, 1)
            test_acc = evaluate(model, test_loader, device)

            avg_step_ms_epoch = sum(step_times) / max(len(step_times), 1)

            writer.writerow(
                {
                    "run_id": run_id,
                    "dataset": dataset_name,
                    "mode": mode,
                    "num_paths": num_paths,
                    "seed": seed,
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": train_loss_epoch,
                    "test_acc": test_acc,
                    "elapsed_s": elapsed,
                    "avg_step_ms": avg_step_ms_epoch,
                    "cumulative_param_updates": cumulative_updates,
                }
            )
            f.flush()

            print(
                f"[{run_id}] epoch {epoch:02d}/{epochs} | "
                f"mode={mode} | num_paths={num_paths if mode=='rw_sparse' else 'FULL'} | "
                f"train_loss={train_loss_epoch:.4f} | test_acc={test_acc:.4f} | "
                f"elapsed_s={elapsed:.1f} | avg_step_ms={avg_step_ms_epoch:.2f} | "
                f"cum_updates={cumulative_updates}"
            )

    return metrics_path


# ----------------------------
# Main
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist"])
    p.add_argument("--num-paths", type=str, default="10,50,200,500,1000",
                   help="Comma-separated list, e.g. '10,50,200,500,1000'")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--epsilon-greedy", type=float, default=0.0)
    p.add_argument("--weight-temperature", type=float, default=1.0)
    p.add_argument("--min-rows-per-layer", type=int, default=1)
    p.add_argument("--log-every-steps", type=int, default=0,
                   help="If >0, appends extra rows (without test_acc) every N steps")
    return p.parse_args()


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = get_device(args.device)

    num_paths_list = [int(s.strip()) for s in args.num_paths.split(",") if s.strip()]
    results_root = Path(args.results_dir)
    ensure_dir(results_root)

    # ------------------
    # FULL BASELINE (run once)
    # ------------------
    run_id = f"{args.dataset}_FULL_seed{args.seed}_{uuid.uuid4().hex[:8]}"
    run_dir = results_root / run_id

    train_one_run(
        run_dir=run_dir,
        dataset_name=args.dataset,
        num_paths=0,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        epsilon_greedy=0.0,
        weight_temperature=1.0,
        min_rows_per_layer=0,
        log_every_steps=args.log_every_steps,
        mode="full",
    )

    # ------------------
    # RW-SPARSE runs
    # ------------------
    for num_paths in num_paths_list:
        run_id = f"{args.dataset}_rw_sparse_np{num_paths}_seed{args.seed}_{uuid.uuid4().hex[:8]}"
        run_dir = results_root / run_id

        train_one_run(
            run_dir=run_dir,
            dataset_name=args.dataset,
            num_paths=num_paths,
            seed=args.seed,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            epsilon_greedy=args.epsilon_greedy,
            weight_temperature=args.weight_temperature,
            min_rows_per_layer=args.min_rows_per_layer,
            log_every_steps=args.log_every_steps,
            mode="rw_sparse",
        )

    print("\nDone. Plot with:\n  python plot_results.py --results-dir results\n")
    print("\nDone. Save Plots with:\n  python plot_results.py --results-dir results --save-dir plots --speed-bars\n")

if __name__ == "__main__":
    main()
