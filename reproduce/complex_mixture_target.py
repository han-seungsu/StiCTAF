
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import nn
from torch.distributions import MultivariateNormal, Chi2
import normflows as nf
from normflows.distributions import Target


class MixtureTarget(Target):
    """Mixture of target distributions."""

    def __init__(self, targets, weights=None):
        base = targets[0]
        super().__init__(prop_scale=base.prop_scale, prop_shift=base.prop_shift)
        self.targets = nn.ModuleList(targets)
        K = len(targets)
        if weights is None:
            w = torch.full((K,), 1.0 / K, dtype=base.prop_scale.dtype, device=base.prop_scale.device)
        else:
            w = torch.as_tensor(weights, dtype=base.prop_scale.dtype, device=base.prop_scale.device).flatten()
            w = w / w.sum()
        self.register_buffer("weights", w)
        self.register_buffer("log_weights", torch.log(w))

    def log_prob(self, z):
        comps = []
        for t in self.targets:
            lp = t.log_prob(z)
            comps.append(lp)
        comp = torch.stack(comps, dim=0)
        return torch.logsumexp(comp + self.log_weights[:, None], dim=0)

    def sample(self, num_samples=1):
        cat = torch.distributions.Categorical(self.weights)
        idx = cat.sample((num_samples,))
        samples = []
        for k in idx.tolist():
            s = self.targets[k].sample(1)
            if isinstance(s, tuple):
                s = s[0]
            samples.append(s.reshape(1, -1))
        return torch.cat(samples, dim=0)


class AsymmetricStudentT(Target):
    """
    Multivariate 'asymmetric' Student t.
    If df[i] is NaN, the i-th coordinate behaves like Gaussian.
    """

    def __init__(self, df=None, mean=None, cov=None):
        super().__init__()
        if df is None:
            df = 2.0

        if isinstance(df, (int, float)):
            d = 1
            df = torch.tensor([float(df)], dtype=torch.float32)
        else:
            df = torch.as_tensor(df, dtype=torch.float32)
            d = df.shape[0]

        if mean is None:
            mean = torch.zeros(d, dtype=torch.float32)
        else:
            mean = torch.as_tensor(mean, dtype=torch.float32)

        if cov is None:
            cov = torch.eye(d, dtype=torch.float32)
        else:
            cov = torch.as_tensor(cov, dtype=torch.float32)

        self.register_buffer("df", df)
        self.register_buffer("mean", mean)
        self.register_buffer("cov", cov)
        self.register_buffer("L", torch.linalg.cholesky(cov))
        self.mvnorm = MultivariateNormal(torch.zeros(d, dtype=torch.float32), covariance_matrix=cov)

    def log_prob(self, z):
        z = z.to(self.mean.device).float()
        x = z - self.mean
        d = x.shape[-1]
        x_std = torch.linalg.solve_triangular(self.L, x.T, upper=False).T
        lp = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

        for i in range(d):
            xi = x_std[:, i]
            nu = self.df[i]
            if torch.isnan(nu):
                lp += -0.5 * xi**2 - 0.5 * np.log(2.0 * np.pi)
            else:
                lp += (
                    torch.lgamma((nu + 1) / 2.0)
                    - torch.lgamma(nu / 2.0)
                    - 0.5 * torch.log(nu * torch.tensor(np.pi, device=z.device))
                    - ((nu + 1) / 2.0) * torch.log1p(xi**2 / nu)
                )

        log_det = torch.logdet(self.L)
        lp = lp - log_det
        return lp

    def sample(self, num_samples=1):
        d = self.mean.shape[0]
        y = self.mvnorm.sample((num_samples,))
        out = y.clone()

        for i in range(d):
            nu = self.df[i]
            if torch.isnan(nu):
                continue
            chi = Chi2(nu).sample((num_samples,)).to(self.mean.device)
            out[:, i] = out[:, i] / torch.sqrt(chi / nu)

        return out + self.mean


def make_target(device: str = "cpu") -> MixtureTarget:
    target1 = AsymmetricStudentT(
        mean=torch.tensor([0.0, 6.0], device=device),
        cov=torch.tensor([[2.0, 0.0], [0.0, 1.0]], device=device),
        df=torch.tensor([3.0, float("nan")], device=device),
    )
    target2 = AsymmetricStudentT(
        mean=torch.tensor([6.0, 0.0], device=device),
        cov=torch.tensor([[1.0, 0.0], [0.0, 2.0]], device=device),
        df=torch.tensor([float("nan"), 2.0], device=device),
    )
    target3 = nf.distributions.TwoMoons()
    target4 = AsymmetricStudentT(
        mean=torch.tensor([-3.0, -4.0], device=device),
        cov=torch.tensor([[0.5, 0.0], [0.0, 0.5]], device=device),
        df=torch.tensor([2.0, 3.0], device=device),
    )
    target = MixtureTarget(
        targets=[target1, target2, target3, target4],
        weights=[0.2, 0.2, 0.1, 0.5],
    )
    return target.to(device)


def load_latest_csv(glob_pattern: str, directory: Path) -> Tuple[np.ndarray, Path]:
    files = sorted(directory.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files match: {directory / glob_pattern}")
    df = pd.read_csv(files[0])
    return df.to_numpy(), files[0]


def try_load(glob_pattern: str, directory: Path) -> Optional[np.ndarray]:
    try:
        arr, path = load_latest_csv(glob_pattern, directory)
        print(f"Loaded {glob_pattern}: {path.name} {arr.shape}")
        return arr
    except FileNotFoundError:
        print(f"Missing pattern: {glob_pattern}")
        return None


def sample_target(target: Target, n: int, seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = target.sample(n)
    if isinstance(x, tuple):
        x = x[0]
    return x.detach().cpu().numpy()


def kde_1d(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    n = x.size
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.34) if iqr > 0 else std
    h = 0.9 * sigma * n ** (-1 / 5) if sigma > 0 else 1.0
    diff = (grid[:, None] - x[None, :]) / h
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (n * h * np.sqrt(2 * np.pi))
    return dens


def plot_panel(ax_scatter, ax_top, ax_right, data_target, data_model, title, xy):
    xmin, xmax, ymin, ymax = xy
    ax_scatter.scatter(data_target[:, 0], data_target[:, 1], s=2, alpha=0.25, label="Target")
    ax_scatter.scatter(data_model[:, 0], data_model[:, 1], s=2, alpha=0.25, label=title)
    ax_scatter.set_xlim(xmin, xmax)
    ax_scatter.set_ylim(ymin, ymax)
    ax_scatter.set_title(title, fontsize=11)

    gx = np.linspace(xmin, xmax, 300)
    gy = np.linspace(ymin, ymax, 300)

    ax_top.plot(gx, kde_1d(data_target[:, 0], gx), lw=1.5)
    ax_top.plot(gx, kde_1d(data_model[:, 0], gx), lw=1.5)
    ax_top.set_xlim(xmin, xmax)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    ax_right.plot(kde_1d(data_target[:, 1], gy), gy, lw=1.5)
    ax_right.plot(kde_1d(data_model[:, 1], gy), gy, lw=1.5)
    ax_right.set_ylim(ymin, ymax)
    ax_right.set_xticks([])
    ax_right.set_yticks([])


def make_full_figure(
    target_samples: np.ndarray,
    methods: Dict[str, np.ndarray],
    output_path: Path,
    xy=(-15, 15, -15, 15),
):
    method_items = [(name, arr) for name, arr in methods.items() if arr is not None]
    ncols = 3
    nrows = int(np.ceil(len(method_items) / ncols))
    fig = plt.figure(figsize=(5.0 * ncols, 4.5 * nrows))
    outer = GridSpec(nrows, ncols, figure=fig, wspace=0.25, hspace=0.35)

    for idx, (name, arr) in enumerate(method_items):
        sub = outer[idx].subgridspec(
            2, 2,
            height_ratios=[1, 4],
            width_ratios=[4, 1],
            hspace=0.02,
            wspace=0.02,
        )
        ax_top = fig.add_subplot(sub[0, 0])
        ax_scatter = fig.add_subplot(sub[1, 0])
        ax_right = fig.add_subplot(sub[1, 1])
        plot_panel(ax_scatter, ax_top, ax_right, target_samples, arr, name, xy)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def make_summary_table(output_path: Path):
    """
    Summary values currently available directly from the notebook output.
    gTAF/TAF/ATAF rows can be filled later when their evaluation code is added.
    """
    df = pd.DataFrame(
        [
            {"Method": "NF (Gaussian)", "Forward KL mean": 1.918619, "Forward KL std": 1.212488, "ESS mean": 0.311140, "ESS std": 0.173416},
            {"Method": "NF (Gaussian Mixture)", "Forward KL mean": 0.740192, "Forward KL std": 0.335362, "ESS mean": 0.649915, "ESS std": 0.228401},
            {"Method": "StiCTAF", "Forward KL mean": 0.324149, "Forward KL std": 0.084981, "ESS mean": 0.674308, "ESS std": 0.151608},
            {"Method": "TAF", "Forward KL mean": np.nan, "Forward KL std": np.nan, "ESS mean": np.nan, "ESS std": np.nan},
            {"Method": "gTAF", "Forward KL mean": np.nan, "Forward KL std": np.nan, "ESS mean": np.nan, "ESS std": np.nan},
            {"Method": "gTAF Mixture", "Forward KL mean": np.nan, "Forward KL std": np.nan, "ESS mean": np.nan, "ESS std": np.nan},
            {"Method": "ATAF", "Forward KL mean": np.nan, "Forward KL std": np.nan, "ESS mean": np.nan, "ESS std": np.nan},
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved table template to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exports-dir", type=str, default="exports")
    parser.add_argument("--output-dir", type=str, default="outputs/complex_mixture")
    parser.add_argument("--num-target-samples", type=int, default=20000)
    args = parser.parse_args()

    exports_dir = Path(args.exports_dir)
    output_dir = Path(args.output_dir)

    device = "cpu"
    target = make_target(device=device)
    target_samples = sample_target(target, args.num_target_samples, seed=0)

    methods = {
        # update patterns below if your exported filenames differ
        "NF (Gaussian)": try_load("Gaussian_Complex_*.csv*", exports_dir),
        "NF (Gaussian Mixture)": try_load("Gaussian_Mixture_Complex_*.csv*", exports_dir),
        "TAF": try_load("TAF_toy1_*.csv*", exports_dir),
        "gTAF": try_load("gTAF/gTAF_Complex_*.csv*", exports_dir),
        "gTAF Group": try_load("gTAF/gTAF_(group)_Complex_*.csv*", exports_dir),
        "gTAF Mixture": try_load("gTAF_mix/gTAF_Mixture_Complex_*.csv*", exports_dir),
        "ATAF": try_load("ATAF_toy1_*.csv*", exports_dir),
        "StiCTAF": try_load("StiCTAF_Complex_*.csv*", exports_dir),
    }

    make_full_figure(
        target_samples=target_samples,
        methods=methods,
        output_path=output_dir / "complex_mixture_full_comparison.png",
    )
    make_summary_table(output_dir / "complex_mixture_summary_table.csv")


if __name__ == "__main__":
    main()
