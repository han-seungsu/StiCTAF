import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Chi2, StudentT, Normal
import numpy as np

from normflows.distributions import BaseDistribution, Target


class AsymmetricStudentT(Target):

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

        assert mean.shape == (d,), "mean must be shape (d,)"
        assert cov.shape  == (d, d), "cov must be shape (d,d)"

        self.register_buffer('df', df)
        self.register_buffer('mean', mean)
        self.register_buffer('cov', cov)
        self.d    = d

    def log_prob(self, z):
        # z: (batch, d)
        batch_size, d = z.shape
        device = z.device
        dtype  = z.dtype

        mean  = self.mean.to(device=device, dtype=dtype)
        cov   = self.cov.to(device=device, dtype=dtype)
        df    = self.df.to(device=device, dtype=dtype)

        scales = torch.sqrt(torch.diagonal(cov))  # (d,)
        total_logp = torch.zeros(batch_size, device=device, dtype=dtype)

        for i in range(d):
            if torch.isnan(df[i]):
                dist_i = Normal(loc=mean[i], scale=scales[i])
            else:
                dist_i = StudentT(df=df[i], loc=mean[i], scale=scales[i])
            total_logp = total_logp + dist_i.log_prob(z[:, i])

        return total_logp

    def sample(self, num_samples=1):
        device = self.cov.device
        dtype  = self.cov.dtype

        d     = self.d
        mean  = self.mean.to(device=device, dtype=dtype)
        cov   = self.cov.to(device=device, dtype=dtype)
        df    = self.df.to(device=device, dtype=dtype)

        # 1) MVN(0, cov) 샘플
        mvn = MultivariateNormal(loc=torch.zeros(d, device=device, dtype=dtype),
                                 covariance_matrix=cov)
        mv_samples = mvn.sample((num_samples,))  # (N, d)

        # 2) 차원별 스케일 팩터 (Gaussian 차원은 1.0)
        scale_factors = torch.ones((num_samples, d), device=device, dtype=dtype)
        for i in range(d):
            if torch.isnan(df[i]):
                continue  # Gaussian -> scale=1
            chi = Chi2(df[i]).sample((num_samples,)).to(device=device, dtype=dtype)
            scale_factors[:, i] = torch.sqrt(chi / df[i])

        t_samples = mv_samples / scale_factors  # t 차원만 heavy-tail
        return t_samples + mean

class MultStudentT(Target):
    """
    Multivariate Student's t-distribution with full covariance.

    log p(z) = log Gamma((nu + d)/2) - log Gamma(nu/2)
             - (d/2) * log(nu * pi) - 0.5 * log|Sigma|
             - ((nu + d)/2) * log(1 + M/nu)
    where M = (z - mean)^T Sigma^{-1} (z - mean).
    """
    def __init__(self, df=None, mean=None, Sigma=None):
        """
        Args:
            df: degrees of freedom (scalar). Defaults to 1.
            mean: tensor-like of shape (d,). Defaults to zero vector.
            Sigma: covariance matrix of shape (d, d). Defaults to identity.
        """
        super().__init__()
        # defaults
        if df is None:
            df = 1.0
        self.df = float(df)
        # mean
        if mean is None:
            self.mean = None
        else:
            self.mean = torch.as_tensor(mean, dtype=torch.float32)
        # covariance
        if Sigma is None:
            if self.mean is None:
                d = 1
                self.Sigma = torch.eye(1, dtype=torch.float32)
            else:
                d = self.mean.shape[0]
                self.Sigma = torch.eye(d, dtype=torch.float32)
        else:
            self.Sigma = torch.as_tensor(Sigma, dtype=torch.float32)
            d = self.Sigma.shape[0]
        # infer d from mean or Sigma
        if self.mean is None:
            self.mean = torch.zeros(d, dtype=torch.float32)
        self.d = d
        # precompute inverse and logdet
        self.Sigma_inv = torch.inverse(self.Sigma)
        sign, logdet = torch.slogdet(self.Sigma)
        assert sign > 0, "Sigma must be positive-definite"
        self.log_det_Sigma = logdet
        # normalization constant
        self.const = (
            torch.lgamma(torch.tensor((self.df + self.d) / 2.0))
            - torch.lgamma(torch.tensor(self.df / 2.0))
            - (self.d / 2.0) * torch.log(torch.tensor(self.df) * torch.tensor(np.pi))
            - 0.5 * self.log_det_Sigma
        )

    def log_prob(self, z):
        """
        Args:
            z: Tensor of shape (batch_size, d)
        Returns:
            Tensor of shape (batch_size,)
        """
        # center
        x = z - self.mean
        # Mahalanobis distance
        M = torch.sum((x @ self.Sigma_inv) * x, dim=1)
        # log-density
        logp = self.const - ((self.df + self.d) / 2.0) * torch.log1p(M / self.df)
        return logp
    
    def sample(self, num_samples=1):
        """
        Samples from the multivariate Student's t-distribution.

        Steps:
        1) Sample X ~ MVN(0, Sigma) of shape (num_samples, d)
        2) Sample W ~ Chi2(df) of shape (num_samples,)
        3) Return mean + X * sqrt(df / W)[:, None]
        """
        # 1) MVN samples
        mvn = torch.distributions.MultivariateNormal(
            loc=torch.zeros(self.d, dtype=self.mean.dtype, device=self.mean.device),
            covariance_matrix=self.Sigma
        )
        X = mvn.sample((num_samples,))  # (num_samples, d)
        # 2) Chi-squared samples
        W = torch.distributions.Chi2(self.df).sample((num_samples,)).to(X)
        # 3) scale and shift
        scale = torch.sqrt(self.df / W).unsqueeze(1)  # (num_samples, 1)
        samples = self.mean + X * scale  # (num_samples, d)
        return samples
    
class SymmetricParetoMixture(Target):
    """Mixture of symmetric Pareto in R^dim with per-dimension alpha."""
    def __init__(self, n_mode=2, weight=None, alpha=None, mean=None, dim=2):
        super().__init__()
        self.n_mode = int(n_mode)
        self.dim = int(dim)

        # weights
        if weight is None:
            w = torch.ones(self.n_mode, dtype=torch.float32) / float(self.n_mode)
        else:
            w = torch.as_tensor(weight, dtype=torch.float32)
            assert w.shape == (self.n_mode,), "weight must be (n_mode,)"
            w = w / (w.sum() + 1e-12)
        self.register_buffer("weight", w)

        # alpha (scalar or (dim,))
        if alpha is None:
            a = torch.full((self.dim,), 2.0, dtype=torch.float32)
        elif isinstance(alpha, (int, float)):
            a = torch.full((self.dim,), float(alpha), dtype=torch.float32)
        else:
            a = torch.as_tensor(alpha, dtype=torch.float32)
            assert a.shape == (self.dim,), "alpha must be scalar or (dim,)"
        self.register_buffer("alpha", a)

        # means (n_mode, dim)
        if mean is None:
            m = torch.randn(self.n_mode, self.dim, dtype=torch.float32)
        else:
            m = torch.as_tensor(mean, dtype=torch.float32)
            assert m.shape == (self.n_mode, self.dim), "mean must be (n_mode, dim)"
        self.register_buffer("mean", m)

        # constants
        const = torch.sum(torch.log(self.alpha) - np.log(2.0))
        self.register_buffer("log_const", const)

    @staticmethod
    def _icdf_core(u, alpha):
        # u, alpha broadcast to (..., d)
        p1 = -((2.0 * u) ** (-1.0 / alpha) - 1.0)
        p2 =  (2.0 * (1.0 - u)) ** (-1.0 / alpha) - 1.0
        return torch.where(u < 0.5, p1, p2)

    def _icdf(self, u):
        return self._icdf_core(u, self.alpha.view(1, -1))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.mean.device, self.mean.dtype)           # (N, d)
        X = x[:, None, :] - self.mean[None, :, :]             # (N, K, d)
        a = self.alpha.view(1, 1, -1)                         # (1,1,d)
        sum_term = torch.sum((a + 1.0) * torch.log1p(torch.abs(X)), dim=-1)  # (N, K)
        log_comp = self.log_const - sum_term                  # (N, K)
        log_w = torch.log(self.weight + 1e-40).view(1, -1)    # (1, K)
        return torch.logsumexp(log_comp + log_w, dim=1)       # (N,)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        device, dtype = self.mean.device, self.mean.dtype
        k_idx = torch.multinomial(self.weight, num_samples, replacement=True).to(device)
        u = torch.rand((num_samples, self.dim), device=device, dtype=dtype)
        x = self._icdf(u) + self.mean[k_idx, :]
        return x
    
class MixtureTarget(Target):
    def __init__(self, target1, target2, weight=None, truncation=None):
        super().__init__(prop_scale=target1.prop_scale, prop_shift=target1.prop_shift)
        self.target1 = target1
        self.target2 = target2
        self.weight = weight if weight is not None else [0.5, 0.5]
        
        if truncation is not None:
            self.truncation = torch.tensor(truncation, dtype=target1.prop_scale.dtype, device=target1.prop_scale.device)
        else:
            self.truncation = None
        

    def log_prob(self, z):
        if self.truncation is not None:
            mask = (z >= self.truncation).all(dim=-1)
        else:
            mask = torch.ones(z.shape[0], dtype=torch.bool, device=z.device)
    
        log_p1 = self.target1.log_prob(z)
        log_p2 = self.target2.log_prob(z)
    
        log_mix = torch.logsumexp(torch.stack([
            torch.log(torch.tensor(self.weight[0], device=z.device)) + log_p1,
            torch.log(torch.tensor(self.weight[1], device=z.device)) + log_p2,
        ], dim=0), dim=0)
    
        log_mix = torch.where(mask, log_mix, torch.tensor(float('-inf'), device=z.device, dtype=log_mix.dtype))
        return log_mix
        
    def sample(self, num_samples=1):
        num_samples_1 = int(num_samples * self.weight[0])
        num_samples_2 = num_samples - num_samples_1

        samples_1 = self.target1.sample(num_samples_1)
        samples_2 = self.target2.sample(num_samples_2)

        samples = torch.cat([samples_1, samples_2], dim=0)

        if self.truncation is not None:
            mask = (samples >= self.truncation).all(dim=-1)
            samples = samples[mask]

        return samples

class ConstantNormal(Target):
    def __init__(self, n_dims, const=2.0):
        """
        Args:
          n_dims: 차원 수
          const: log_prob에 더할 상수 값 (기본값 2.0)
        """
        super().__init__()
        self.n_dims = n_dims
        self.const = const
        self.standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)

    def log_prob(self, z):
        """
        Args:
          z: [batch_size, n_dims] 텐서

        Returns:
          log_prob: 표준 정규분포 log_prob + 상수
        """
        logp = self.standard_normal.log_prob(z).sum(dim=-1)
        return logp + self.const
    
class GaussianTarget(Target):
    def __init__(self, shape, mean=None, scale=None):
        super().__init__()
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        d = int(np.prod(shape))
        if mean is None:
            mean = torch.zeros(d)
        if scale is None:
            log_scale = torch.zeros(d)
        else:
            log_scale = torch.log(torch.tensor(scale, dtype=torch.float32))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('log_scale', torch.as_tensor(log_scale, dtype=torch.float32))

    def sample(self, num_samples=1):
        eps = torch.randn(num_samples, *self.shape, device=self.mean.device)
        scale = torch.exp(self.log_scale).view(1, *self.shape)
        samples = eps * scale + self.mean.view(1, *self.shape)
        log_p = self.log_prob(samples)
        return samples

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], -1)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        lp = -0.5 * ((z_flat - mu)/sigma)**2 - torch.log(sigma) - 0.5 * np.log(2*np.pi)
        return lp.sum(dim=1)

class GaussianInverseGamma(Target):
    """
    Two-dimensional target distribution:
      • z[...,0] :  Normal(mu, sigma²)
      • z[...,1] : InverseGamma(alpha, beta)
    """
    def __init__(self, mu=0.0, sigma=1.0, alpha=3.0, beta=2.0):
        super().__init__()
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.alpha = float(alpha)
        self.beta = float(beta)

        # precompute constant part of Gaussian log-pdf
        self._gauss_const = -0.5 * torch.log(torch.tensor(2 * torch.pi * self.sigma**2))

        # precompute constant part of InverseGamma log-pdf
        # log [beta^alpha / Gamma(alpha)]
        self._ig_const = self.alpha * torch.log(torch.tensor(self.beta)) - torch.lgamma(torch.tensor(self.alpha))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: Tensor of shape (..., 2)
        returns: Tensor of shape z.shape[:-1], the joint log-density
        """
        x = z[..., 0]
        y = z[..., 1]

        # Normal log-pdf: -0.5*((x-mu)/sigma)^2 + const
        logp_x = -0.5 * ((x - self.mu) / self.sigma)**2 + self._gauss_const.to(x.device)

        # InverseGamma log-pdf: alpha*log(beta) - lgamma(alpha) - (alpha+1)*log(y) - beta/y
        ig_const = self._ig_const.to(y.device)
        logp_y = ig_const - (self.alpha + 1) * torch.log(y) - self.beta / y

        return logp_x + logp_y

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        """
        Draw samples from the joint:
          returns Tensor of shape (num_samples, 2)
        """
        # Gaussian samples
        x = self.mu + self.sigma * torch.randn(num_samples)

        # InverseGamma samples via Gamma inverse
        gamma = torch.distributions.Gamma(concentration=self.alpha, rate=1.0 / self.beta)
        y = 1.0 / gamma.sample((num_samples,))

        return torch.stack([x, y], dim=-1)

class GaussianMixtureTarget(Target):
    def __init__(self, modes, dim=1, weights=None, means=None, covariances=None):
        if weights is not None:
            w = torch.tensor(weights, dtype=torch.float32)
            self.weights = w / w.sum()
            K = self.weights.numel()
        else:
            K = modes
            self.weights = torch.ones(K, dtype=torch.float32) / K
        if means is not None:
            self.means = [torch.tensor(m, dtype=torch.float32).view(dim) for m in means]
        else:
            self.means = [torch.randn(dim) for _ in range(K)]
        if covariances is not None:
            self.covs = [torch.tensor(c, dtype=torch.float32).view(dim, dim) for c in covariances]
        else:
            self.covs = [torch.eye(dim) for _ in range(K)]
        self.comps = [MultivariateNormal(loc=m, covariance_matrix=S) for m, S in zip(self.means, self.covs)]

    def log_prob(self, x):
        log_ps = torch.stack([comp.log_prob(x) for comp in self.comps], dim=1)
        log_ps = log_ps + torch.log(self.weights)
        return torch.logsumexp(log_ps, dim=1)

    def sample(self, num_samples=1):
        idx = torch.multinomial(self.weights, num_samples, replacement=True)
        dim = self.means[0].shape[0]
        out = torch.zeros(num_samples, dim)
        for i, comp in enumerate(self.comps):
            mask = idx == i
            count = mask.sum().item()
            if count > 0:
                out[mask] = comp.sample((count,))
        return out

class UnivariateGaussian(Target): 
    """
    Univariate Gaussian with mean and std.
    Returns samples of shape (N, 1) to match other 1D targets.
    """
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('std',  torch.as_tensor(std,  dtype=torch.float32))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        z = z.to(device=self.mean.device, dtype=self.mean.dtype)
        logp = -0.5 * ((z - self.mean) / self.std) ** 2 \
               - torch.log(self.std) \
               - 0.5 * np.log(2 * np.pi)
        return logp.squeeze(-1)   # (N,)

    def sample(self, num_samples: int = 1) -> torch.Tensor:
        eps = torch.randn(num_samples, 1, device=self.mean.device, dtype=self.mean.dtype)
        return self.mean + self.std * eps  # (N,1)