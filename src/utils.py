import torch
import numpy as np
import normflows as nf
from matplotlib import pyplot as plt
from scipy.stats import gaussian_kde
from core import MixtureBaseNormalizingFlow

def _print_tensor(name, tensor, max_entries=20):
    data = tensor.cpu().detach().numpy().ravel()
    if data.size > max_entries:
        summary = ", ".join(f"{x:.4f}" for x in data[:max_entries]) + ", …"
    else:
        summary = ", ".join(f"{x:.4f}" for x in data)
    print(f"  {name}: [{summary}]")

def print_model_parameters(q0):
    # 1) Top‐level stick-breaking params
    if hasattr(q0, 'log_a'):
        print("--- stick-breaking params ---")
        _print_tensor("a", torch.exp(q0.log_a))
        _print_tensor("b", torch.exp(q0.log_b))
        _print_tensor("pi (expected)", q0.pi)
        print()

    for idx, comp in enumerate(getattr(q0, 'components', [q0])):
        print(f"--- component #{idx} ({comp.__class__.__name__}) ---")
        # parameters
        for name, param in comp.named_parameters(recurse=False):
            val = param
            if name == 'log_scale':
                name = 'scale'
                val = torch.exp(param)
            _print_tensor(name, val)
        # buffers (e.g. df)
        for name, buf in comp.named_buffers(recurse=False):
            _print_tensor(name, buf)
        print()

def plot_samples(*dists,
                 num_samples=10000,
                 show=1,
                 save_img=False,
                 size=5,
                 two_d=True,
                 contour=False,
                 grid_size=200,
                 density=True,
                 vmin=None,
                 vmax=None,
                 kde=False):
    """
    Samples from one or more distributions and plots them on the same axes.
    If you pass a single dist, behavior is unchanged. If you pass multiple,
    all will be drawn together (scatter/contour for 2D, histogram/KDE for 1D).
    """

    # --- draw samples for each dist ---
    samples_list = []
    labels = []
    for i, dist in enumerate(dists):
        if isinstance(dist, nf.distributions.BaseDistribution):
            samp, _ = dist.forward(num_samples=num_samples)
            samp = samp.cpu().detach().numpy()
        elif isinstance(dist, nf.distributions.Target):
            samp = dist.sample(num_samples=num_samples).cpu().detach().numpy()
        elif isinstance(dist, (nf.NormalizingFlow, MixtureBaseNormalizingFlow)):
            samp, _ = dist.sample(num_samples=num_samples)
            samp = samp.cpu().detach().numpy()
        else:
            raise ValueError(f"Unsupported distribution type: {type(dist)}")
        samples_list.append(samp)
        labels.append(getattr(dist, '__class__', type(dist)).__name__ + f'[{i+1}]')

    # --- 2D case ---
    if two_d and samples_list[0].shape[1] == 2:
        plt.figure(figsize=(size, size))
        x_min = vmin if vmin is not None else -size
        x_max = vmax if vmax is not None else  size
        plt.xlim(x_min, x_max)
        plt.ylim(x_min, x_max)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Samples from distributions")

        for samp, label in zip(samples_list, labels):
            x, y = samp[:,0], samp[:,1]
            if contour:
                # contour for each dist
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, grid_size),
                    np.linspace(x_min, x_max, grid_size),
                    indexing='xy'
                )
                pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
                try:
                    lp = dist.log_prob(torch.tensor(pts, dtype=torch.float32))
                    zz = torch.exp(lp).cpu().view(grid_size, grid_size).numpy()
                except Exception:
                    from scipy.stats import gaussian_kde
                    kde2d = gaussian_kde(np.vstack([x, y]))
                    zz = kde2d(np.vstack([xx.ravel(), yy.ravel()])).reshape(grid_size, grid_size)
                CS = plt.contour(xx, yy, zz, levels=5, alpha=0.7)
                CS.collections[0].set_label(label)
            else:
                plt.scatter(x, y, alpha=0.3, s=1, label=label)

        plt.gca().set_aspect('equal', 'box')
        plt.legend()
        if save_img:
            plt.savefig("figure/samples_2d.png", dpi=300, bbox_inches='tight')
        plt.show()

    # --- 1D case ---
    else:
        marginal = show - 1
        plt.figure(figsize=(6,6))
        plt.title(f'Marginal Distribution x{show}')
        plt.xlabel(f'x{show}')
        plt.ylabel('Density')

        # determine common bin edges if histogram
        if not kde:
            all_vals = np.concatenate([s[:, marginal] for s in samples_list])
            if vmin is not None and vmax is not None:
                bins = np.linspace(vmin, vmax, 50)
            else:
                bins = 100

        for samp, label in zip(samples_list, labels):
            x = samp[:, marginal]
            if kde:
                from scipy.stats import gaussian_kde
                kde_func = gaussian_kde(x)
                xmin = vmin if vmin is not None else x.min()
                xmax = vmax if vmax is not None else x.max()
                xs = np.linspace(xmin, xmax, 200)
                plt.plot(xs, kde_func(xs), label=label)
            else:
                plt.hist(x, bins=bins, density=density, alpha=0.5, edgecolor='black', label=label)

        plt.grid(True)
        plt.legend()
        if save_img:
            suffix = '_kde' if kde else ''
            plt.savefig(f"figure/marginal_x{show}{suffix}.png", dpi=300, bbox_inches='tight')
        plt.show()
