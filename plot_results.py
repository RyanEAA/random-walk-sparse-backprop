#!/usr/bin/env python3
"""
plot_results.py

Reads one or many metrics.csv files under a results directory and generates:
- accuracy vs cumulative updates (log x)
- accuracy vs elapsed time
- optional speed bars (avg_step_ms) at final epoch

Usage:
  python plot_results.py --results-dir results
  python plot_results.py --results-dir results --save-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt


def find_metrics_csvs(results_dir: Path) -> List[Path]:
    return sorted(results_dir.rglob("metrics.csv"))


def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    csvs = find_metrics_csvs(results_dir)
    if not csvs:
        raise FileNotFoundError(f"No metrics.csv files found under: {results_dir}")

    dfs = []
    for p in csvs:
        df = pd.read_csv(p)
        df["metrics_path"] = str(p)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)

    # Clean numeric columns
    for col in ["num_paths", "seed", "epoch", "step", "train_loss", "test_acc", "elapsed_s", "avg_step_ms", "cumulative_param_updates"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def plot_acc_vs_updates(df: pd.DataFrame, save_path: Path | None = None):
    # use only rows with test_acc present
    d = df.dropna(subset=["test_acc", "cumulative_param_updates"]).copy()

    plt.figure()
    for (mode, num_paths), g in d.groupby(["mode", "num_paths"]):
        g = g.sort_values("cumulative_param_updates")
        label = f"{mode} ({int(num_paths)} paths)"
        plt.plot(g["cumulative_param_updates"], g["test_acc"], marker="o", linewidth=1.5, label=label)

    plt.xscale("log")
    plt.xlabel("Cumulative number of parameter updates (log scale)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Cumulative Parameter Updates")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_acc_vs_time(df: pd.DataFrame, save_path: Path | None = None):
    d = df.dropna(subset=["test_acc", "elapsed_s"]).copy()

    plt.figure()
    for (mode, num_paths), g in d.groupby(["mode", "num_paths"]):
        g = g.sort_values("elapsed_s")
        label = f"{mode} ({int(num_paths)} paths)"
        plt.plot(g["elapsed_s"], g["test_acc"], marker="o", linewidth=1.5, label=label)

    plt.xlabel("Elapsed training time (s)")
    plt.ylabel("Test accuracy")
    plt.title("Accuracy vs Training Time")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def plot_speed_bars(df: pd.DataFrame, save_path: Path | None = None):
    """
    Bar chart of avg_step_ms at the *final* epoch for each (mode, num_paths).
    """
    d = df.dropna(subset=["avg_step_ms", "epoch"]).copy()

    # Keep only final epoch per run_id
    d = d.sort_values(["run_id", "epoch"])
    final_rows = d.groupby("run_id").tail(1)

    # Aggregate across runs with same (mode, num_paths): median is robust
    agg = final_rows.groupby(["mode", "num_paths"])["avg_step_ms"].median().reset_index()
    agg = agg.sort_values(["mode", "num_paths"])

    labels = [f"{m}\n{int(np)}" for m, np in zip(agg["mode"], agg["num_paths"])]
    values = agg["avg_step_ms"].tolist()

    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels)
    plt.ylabel("Avg step time (ms)")
    plt.title("Speed (Median avg_step_ms at final epoch)")
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    else:
        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--save-dir", type=str, default="", help="If set, saves plots to this folder instead of showing.")
    p.add_argument("--speed-bars", action="store_true", help="Also generate speed bar chart.")
    return p.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    df = load_all_metrics(results_dir)

    save_dir = Path(args.save_dir) if args.save_dir else None

    plot_acc_vs_updates(df, save_path=(save_dir / "acc_vs_updates.png") if save_dir else None)
    plot_acc_vs_time(df, save_path=(save_dir / "acc_vs_time.png") if save_dir else None)

    if args.speed_bars:
        plot_speed_bars(df, save_path=(save_dir / "speed_bars.png") if save_dir else None)


if __name__ == "__main__":
    main()
