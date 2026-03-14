
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import gaussian_kde

try:
    import pyreadr
except Exception:
    pyreadr = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_latest_csv(glob_pattern: str, directory: Path) -> Tuple[np.ndarray, Path]:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match: {directory / glob_pattern}")
    df = pd.read_csv(files[0])
    return df.to_numpy(), files[0]


def load_optional_latest_csv(glob_pattern: str, directory: Path) -> Tuple[np.ndarray | None, Path | None]:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        return None, None
    df = pd.read_csv(files[0])
    return df.to_numpy(), files[0]


def load_mcmc_samples(rds_path: Path) -> np.ndarray:
    if pyreadr is None:
        raise ImportError("pyreadr is required to read the MCMC .rds file. Please install pyreadr.")
    result = pyreadr.read_r(str(rds_path))
    mcmc_obj = next(iter(result.values()))
    mcmc_arr = getattr(mcmc_obj, "values", np.array(mcmc_obj))
    return np.asarray(mcmc_arr)


def to_torch(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float()
    return torch.tensor(x, dtype=torch.float32)


def fixed_bw_kde(grid: np.ndarray, arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
    all_data = np.concatenate(arrays)
    n = all_data.size
    std = np.std(all_data, ddof=1) if n > 1 else 1.0
    bw = max(1.06 * std * (n ** (-1 / 5)), 1e-6)
    out: List[np.ndarray] = []
    inv = 1.0 / (bw * np.sqrt(2 * np.pi))
    for data in arrays:
        dif = (grid[:, None] - data[None, :]) / bw
        dens = np.exp(-0.5 * dif**2).sum(axis=1) * inv / data.size
        out.append(dens)
    return out


def style(i: int) -> Dict[str, object]:
    if i == 0:
        return dict(color="black", ls="-", lw=1.8)
    if i == 1:
        return dict(color="C3", ls="--", lw=1.8)
    return dict(color=f"C{i+3}", ls=":", lw=1.6)


def _to_1d(a: np.ndarray | torch.Tensor, idx: int) -> np.ndarray:
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    else:
        a = np.asarray(a)
    if a.ndim == 2:
        a = a[:, idx]
    return a.ravel()


def plot_param_kde(
    *samples: np.ndarray | torch.Tensor,
    idx: int,
    xlabel: str,
    labels: Sequence[str],
    q_tail: float = 0.95,
    kde_points: int = 800,
    figsize: Tuple[float, float] = (7.5, 3.5),
    x_major: float | None = None,
    y_major: float | None = None,
    inset_left: bool = False,
    inset_loc: str = "lower left",
    inset_bbox: Tuple[float, float, float, float] = (0.08, 0.14, 0.9, 0.9),
    save: Path | None = None,
) -> None:
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif"})
    arrs = [_to_1d(s, idx) for s in samples]
    mcmc = arrs[0]

    lo, hi = mcmc.min(), mcmc.max()
    pad = 0.05 * (hi - lo + 1e-12)
    xlim = (lo - pad, hi + pad)

    grid = np.linspace(xlim[0], xlim[1], kde_points)
    densities = fixed_bw_kde(grid, arrs)

    all_concat = np.concatenate(arrs)
    cutL = np.quantile(all_concat, 1.0 - q_tail)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for i, (d, lab) in enumerate(zip(densities, labels)):
        ax.plot(grid, d, label=lab, **style(i))

    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=18)
    if y_major is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major))
    if x_major is not None:
        ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=10)

    if inset_left:
        axL = inset_axes(
            ax,
            width="50%",
            height="80%",
            loc=inset_loc,
            bbox_to_anchor=inset_bbox,
            bbox_transform=ax.transAxes,
            borderpad=0.5,
        )
        maskL = grid <= cutL
        for i, d in enumerate(densities):
            axL.plot(grid[maskL], d[maskL], **style(i))
        axL.set_xlim(grid[maskL].min(), grid[maskL].max())
        axL.tick_params(labelsize=8)

    fig.tight_layout()
    if save is not None:
        ensure_dir(save.parent)
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_20param_kde_grid(
    *samples: np.ndarray | torch.Tensor,
    labels: Sequence[str],
    q_tail: float = 0.95,
    kde_points: int = 200,
    figsize: Tuple[float, float] = (12, 13),
    save: Path | None = None,
) -> None:
    plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm", "font.family": "serif"})

    def to_np2d(a: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        else:
            a = np.asarray(a)
        if a.ndim != 2 or a.shape[1] < 20:
            raise ValueError(f"Each sample must be (N,20). got {a.shape}")
        return a[:, :20]

    arrs2d = [to_np2d(s) for s in samples]
    xlabels = [
        r"$\gamma^{(\sigma)}_{1}$", r"$\gamma^{(\sigma)}_{2}$", r"$\gamma^{(\sigma)}_{3}$", r"$\gamma^{(\sigma)}_{4}$",
        r"$\epsilon^{(\sigma)}_{1}$", r"$\epsilon^{(\sigma)}_{2}$", r"$\epsilon^{(\sigma)}_{3}$", r"$\epsilon^{(\sigma)}_{4}$",
        r"$\gamma^{(\eta)}_{1}$", r"$\gamma^{(\eta)}_{2}$", r"$\gamma^{(\eta)}_{3}$", r"$\gamma^{(\eta)}_{4}$",
        r"$\epsilon^{(\eta)}_{1}$", r"$\epsilon^{(\eta)}_{2}$", r"$\epsilon^{(\eta)}_{3}$", r"$\epsilon^{(\eta)}_{4}$",
        r"$\alpha^{*}_{1}$", r"$\alpha^{*}_{2}$", r"$\alpha^{*}_{3}$", r"$\alpha^{*}_{4}$",
    ]

    fig, axes = plt.subplots(5, 4, figsize=figsize)
    axes = axes.reshape(-1)

    for ax in axes:
        ax.set_facecolor("#f7f7f7")
        ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.8)

    shifts = [5, 5, 5, 5, 5, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]

    for j in range(20):
        ax = axes[j]
        cols = [a[:, j] for a in arrs2d]
        mcmc_col = cols[0]
        lo, hi = mcmc_col.min(), mcmc_col.max()
        if j < 8:
            lo -= shifts[j]
        pad = 0.05 * (hi - lo + 1e-12)
        xlim = (lo - pad, hi + pad)
        grid = np.linspace(xlim[0], xlim[1], kde_points)
        dens_list = fixed_bw_kde(grid, cols)

        for i, d in enumerate(dens_list):
            ax.plot(grid, d, **style(i))

        ax.set_yticks([])
        ax.set_xlabel(xlabels[j], fontsize=14)

        if j >= 16:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
        elif j >= 8:
            ax.xaxis.set_major_locator(MultipleLocator(2))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(5))

    handles = [plt.Line2D([0], [0], **style(i)) for i in range(len(labels))]
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save is not None:
        ensure_dir(save.parent)
        fig.savefig(save, dpi=300, bbox_inches="tight")
    plt.close(fig)


def summarize_samples(x: torch.Tensor, ci: float = 0.99) -> pd.DataFrame:
    assert x.ndim == 2, "x must be (N, D)"
    x = x.detach()
    alpha = 1.0 - ci
    q = torch.tensor([alpha / 2, 1 - alpha / 2], device=x.device, dtype=x.dtype)
    q_low, q_high = torch.quantile(x, q, dim=0)
    out = pd.DataFrame({"q_low": q_low.cpu().numpy(), "q_high": q_high.cpu().numpy()})
    out.index = [f"param_{i+1}" for i in range(x.shape[1])]
    return out


def marginal_modes_from_tensor(
    X: torch.Tensor, grid: int = 512, trim_q: Tuple[float, float] = (0.001, 0.999), bw_method: str | float = "scott"
) -> pd.DataFrame:
    assert X.ndim == 2, "X must be (N, D)"
    x = X.detach().cpu().numpy()
    n, d = x.shape
    modes = np.zeros(d, dtype=float)
    peaks = np.zeros(d, dtype=float)

    for j in range(d):
        col = x[:, j]
        lo, hi = np.quantile(col, trim_q)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            modes[j] = np.median(col)
            peaks[j] = np.nan
            continue
        kde = gaussian_kde(col, bw_method=bw_method)
        xs = np.linspace(lo, hi, grid)
        pdf = kde(xs)
        k = int(np.argmax(pdf))
        modes[j] = xs[k]
        peaks[j] = pdf[k]

    return pd.DataFrame(
        {"mode": modes, "peak_density": peaks},
        index=[f"param_{i+1}" for i in range(d)],
    )


def build_summary_table(datasets: Dict[str, torch.Tensor]) -> pd.DataFrame:
    rows = []
    for name, X in datasets.items():
        ci = summarize_samples(X, ci=0.99)
        modes = marginal_modes_from_tensor(X, grid=512, trim_q=(0.001, 0.999), bw_method="scott")
        row = pd.DataFrame({
            "method": name,
            "parameter": ci.index,
            "mode": modes["mode"].values,
            "q_low": ci["q_low"].values,
            "q_high": ci["q_high"].values,
        })
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce real-data figures/tables from saved samples.")
    parser.add_argument("--exports-dir", type=Path, default=Path("exports"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gpd"))
    parser.add_argument("--mcmc-rds", type=Path, required=True)
    args = parser.parse_args()

    ensure_dir(args.output_dir)

    samples_mcmc = load_mcmc_samples(args.mcmc_rds)

    arrs: Dict[str, np.ndarray] = {"MCMC": samples_mcmc}
    patterns = {
        "StickTAF": "StickTAF_samples_*.csv*",
        "NF (Gaussian)": "GaussianNF_samples_*.csv*",
        "NF (Gaussian Mixture)": "GMMNF_samples_*.csv*",
        "TAF": "TAF(3)_samples_*.csv*",
        "gTAF": "gTAF/gTAF_group_gpd_*.csv*",
        "gTAF Mixture": "gTAF_mix/gTAF_Mixture_gpd_*.csv*",
        "ATAF": "ATAF_samples_*.csv*",
    }

    for name, pattern in patterns.items():
        arr, file = load_optional_latest_csv(pattern, args.exports_dir)
        if arr is not None:
            arrs[name] = arr
            print(f"Loaded {name}: {file} {arr.shape}")
        else:
            print(f"Missing {name}: {pattern}")

    labels = list(arrs.keys())
    sample_list = [to_torch(arrs[k]) for k in labels]

    # Representative figures
    if len(sample_list) >= 2:
        plot_param_kde(
            *sample_list,
            idx=13,
            xlabel=r"$\epsilon^{(\eta)}_2$",
            labels=labels,
            q_tail=0.95,
            kde_points=100,
            figsize=(7.5, 3.5),
            x_major=4,
            y_major=0.1,
            inset_left=True,
            inset_loc="lower left",
            inset_bbox=(0.08, 0.14, 0.9, 0.9),
            save=args.output_dir / "epsilon_eta_2.png",
        )
        plot_param_kde(
            *sample_list,
            idx=18,
            xlabel=r"$\alpha_4^*$",
            labels=labels,
            q_tail=0.95,
            kde_points=50,
            figsize=(7.5, 3.5),
            x_major=0.5,
            y_major=0.5,
            inset_left=False,
            save=args.output_dir / "alpha_star_4.png",
        )
        plot_20param_kde_grid(
            *sample_list,
            labels=labels,
            q_tail=0.95,
            kde_points=200,
            save=args.output_dir / "full_gpd.png",
        )

    datasets = {name: to_torch(arrs[name]) for name in labels}
    summary = build_summary_table(datasets)
    summary.to_csv(args.output_dir / "gpd_summary_table.csv", index=False)
    print(f"Saved summary table to {args.output_dir / 'gpd_summary_table.csv'}")


if __name__ == "__main__":
    main()
