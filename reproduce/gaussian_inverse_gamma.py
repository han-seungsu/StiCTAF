
"""
Gaussian-Inverse-Gamma experiment utilities for the StiCTAF paper.

This script is a cleaned extraction of the original notebook
`GaussinInverseGamma.ipynb`.  It focuses on reproducing the
benchmark figure and percentile table from saved sample CSV files.

Expected CSV exports
--------------------
By default the script looks for the following files under `--exports-dir`.

    Gaussian_NIG_*.csv
    Gaussian_Mixture_NIG_*.csv
    TAF_NIG_*.csv
    ATAF_NIG_*.csv
    StiCTAF_NIG_*.csv
    gTAF/gTAF_NIG_*.csv
    gTAF/gTAF_group_NIG_*.csv
    gTAF_mix/gTAF_Mixture_NIG_*.csv

Each CSV should contain N x 2 samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
import pandas as pd
import torch

from HeavyTarget import GaussianInverseGamma


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_target(device: torch.device) -> GaussianInverseGamma:
    target = GaussianInverseGamma(mu=0.0, sigma=1.0, alpha=3.0, beta=1.0)
    return target.to(device)


def load_latest_csv(glob_pattern: str, directory: str | Path) -> Tuple[np.ndarray, Path]:
    directory = Path(directory)
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match: {directory / glob_pattern}")
    df = pd.read_csv(files[0])
    return df.to_numpy(), files[0]


def load_benchmark_samples(exports_dir: str | Path, device: torch.device) -> Dict[str, torch.Tensor]:
    exports_dir = Path(exports_dir)
    patterns = {
        "Gaussian": "Gaussian_NIG_*.csv*",
        "Gaussian_mix": "Gaussian_Mixture_NIG_*.csv*",
        "TAF": "TAF_NIG_*.csv*",
        "gTAF_group": "gTAF/gTAF_group_NIG_*.csv*",
        "gTAF": "gTAF/gTAF_NIG_*.csv*",
        "gTAF_mix": "gTAF_mix/gTAF_Mixture_NIG_*.csv*",
        "ATAF": "ATAF_NIG_*.csv*",
        "StiCTAF": "StiCTAF_NIG_*.csv*",
    }
    samples: Dict[str, torch.Tensor] = {}
    for name, pattern in patterns.items():
        arr, path = load_latest_csv(pattern, exports_dir)
        print(f"Loaded {name:>12s}: {path} {arr.shape}")
        samples[name] = torch.tensor(arr, dtype=torch.float32, device=device)
    return samples


def plot_benchmark(
    target_samples: np.ndarray,
    benchmark_samples: np.ndarray,
    label: str,
    output_path: str | Path | None = None,
    xy=(-5, 5, -0.5, 7),
    q=0.999,
    size=5,
) -> None:
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(size, size))

    x_min, x_max, y_min, y_max = xy
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\beta$", fontsize=18)
    ax.set_ylabel(r"$\sigma^2$", fontsize=18, rotation=0, labelpad=10)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    fig.subplots_adjust(left=0.20)

    labels = [r"Gaussian $\times$ Inverse-Gamma", label]
    samples_list = [target_samples, benchmark_samples]

    for samp, name in zip(samples_list, labels):
        ax.scatter(samp[:, 0], samp[:, 1], alpha=0.4, s=1.2, label=name)

    tx = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for ref_idx, ref in enumerate(samples_list):
        thr_q = np.quantile(ref[:, 1], q)
        x_q1 = np.quantile(ref[:, 0], 1 - q)
        x_q2 = np.quantile(ref[:, 0], q)

        color = f"C{ref_idx}"
        ax.axhline(thr_q, ls=":", lw=1.2, color=color)
        ax.axvline(x_q1, ls=":", lw=1.2, color=color)
        ax.axvline(x_q2, ls=":", lw=1.2, color=color)
        ax.text(x_max, thr_q, f" {q*100:.1f}%", va="center", ha="left", color=color, fontsize=14)
        ax.text(x_q2, -0.08 - 0.08 * ref_idx, f"{q*100:.1f}%", transform=tx,
                ha="center", va="top", color=color, fontsize=14)
        ax.text(x_q1, -0.08 - 0.08 * ref_idx, f"{(1-q)*100:.1f}%", transform=tx,
                ha="center", va="top", color=color, fontsize=14)

    ax.set_aspect("equal", "box")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.01, fontsize=14, markerscale=3)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    plt.close(fig)


def percentile_table(
    samples: Dict[str, torch.Tensor],
    target,
    num_target_samples: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    set_seed(seed)
    true_target = torch.tensor(target.sample(num_target_samples), dtype=torch.float32, device=next(iter(samples.values())).device)

    ordered = {
        "TrueTarget": true_target,
        **samples,
    }

    q_lo = 0.001
    q_hi = 0.999
    rows = []
    for name, x in ordered.items():
        x1 = x[:, 0]
        x2 = x[:, 1]
        rows.append(
            {
                "model": name,
                "x1_p0.1": torch.quantile(x1, q_lo).item(),
                "x1_p99.9": torch.quantile(x1, q_hi).item(),
                "x2_p99.9": torch.quantile(x2, q_hi).item(),
            }
        )
    df = pd.DataFrame(rows).set_index("model")
    return df.round(6)


def save_percentile_table(df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        df.to_csv(output_path)
    else:
        df.to_latex(output_path)
    print(f"Saved table to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", type=str, default="exports")
    parser.add_argument("--output-dir", type=str, default="outputs/gaussian_inverse_gamma")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-target-samples", type=int, default=10_000)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--make-table", action="store_true")
    args = parser.parse_args()

    if not args.make_plots and not args.make_table:
        args.make_plots = True
        args.make_table = True

    device = get_device(args.device)
    target = make_target(device)
    samples = load_benchmark_samples(args.exports_dir, device=device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.make_plots:
        set_seed(args.seed)
        target_np = target.sample(args.num_target_samples).cpu().detach().numpy()
        plot_labels = {
            "Gaussian": "Gaussian",
            "Gaussian_mix": "Gaussian Mixture",
            "TAF": "TAF",
            "gTAF": "gTAF",
            "gTAF_group": "gTAF",
            "gTAF_mix": "gTAF Mixture",
            "ATAF": "ATAF",
            "StiCTAF": "StiCTAF",
        }
        for key, tensor in samples.items():
            plot_benchmark(
                target_samples=target_np,
                benchmark_samples=tensor.detach().cpu().numpy(),
                label=plot_labels[key],
                output_path=output_dir / f"{key}_benchmark.png",
            )

    if args.make_table:
        df = percentile_table(samples=samples, target=target, num_target_samples=args.num_target_samples, seed=args.seed)
        print(df)
        save_percentile_table(df, output_dir / "percentiles.csv")


if __name__ == "__main__":
    main()
