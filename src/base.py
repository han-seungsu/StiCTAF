import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from normflows.distributions import BaseDistribution
from torch.distributions import StudentT
from math import lgamma, pi


class GaussianDistribution(BaseDistribution):
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
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, *self.shape, device=self.mean.device)
        scale = torch.exp(self.log_scale).view(1, *self.shape)
        samples = eps * scale + self.mean.view(1, *self.shape)
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], -1)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        lp = -0.5 * ((z_flat - mu)/sigma)**2 - torch.log(sigma) - 0.5 * np.log(2*pi)
        return lp.sum(dim=1)

class MultivariateTDistribution(BaseDistribution):
    def __init__(self, shape, mean=None, scale=None, df=None):
        super().__init__()
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        self.d = int(np.prod(shape))
        if mean is None:
            mean = torch.zeros(self.d)
        if scale is None:
            log_scale = torch.zeros(self.d)
        else:
            log_scale = torch.log(torch.tensor(scale, dtype=torch.float32))
        if df is None:
            df_val = float(self.d + 1)
        else:
            df_val = float(df)
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)
        self.register_buffer('df', torch.tensor(df_val, dtype=torch.float32))

    def forward(self, num_samples=1):
        eps = torch.randn((num_samples, self.d),
                          device=self.mean.device,
                          dtype=self.mean.dtype)

        scale_vec = torch.exp(self.log_scale).view(1, -1)  # (1, d)
        X = eps * scale_vec                              # (num_samples, d)

        df_val = float(self.df)  
        W = torch.distributions.Chi2(df_val).sample((num_samples,)).to(X)  # (num_samples,)

        scale_t = torch.sqrt(df_val / W).unsqueeze(1)     # (num_samples, 1)
        Z = self.mean.view(1, -1) + X * scale_t           # (num_samples, d)

        samples = Z.view(num_samples, *self.shape)
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        z_flat = z.view(z.shape[0], self.d)
        mu = self.mean.view(1, -1)
        sigma = torch.exp(self.log_scale).view(1, -1)
        df = self.df
        x = (z_flat - mu) / sigma
        m = torch.sum(x**2, dim=1)
        d = float(self.d)
        lp = lgamma((df + d)/2) - lgamma(df/2)
        lp = lp - 0.5*(d*torch.log(df*pi) + 2*torch.sum(torch.log(sigma)))
        lp = lp - ((df + d)/2)*torch.log1p(m/df)
        return lp

class TProductDistribution(BaseDistribution):
    def __init__(self, shape, mean=None, scale=None, df=None, train_df = False):
        super().__init__()
        # 1) record event‐shape and flatten‐dim
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.d = int(torch.tensor(shape).prod().item())

        # 2) mean parameter (d,)
        if mean is None:
            mean = torch.zeros(self.d)
        mean = torch.as_tensor(mean, dtype=torch.float32).view(self.d)
        self.mean = nn.Parameter(mean)

        # 3) log‐scale parameter (d,)
        if scale is None:
            log_scale = torch.zeros(self.d)
        else:
            log_scale = torch.log(torch.as_tensor(scale, dtype=torch.float32).view(self.d))
        self.log_scale = nn.Parameter(log_scale)

        # 4) df buffer (d,)
        if df is None:
            df_buf = torch.full((self.d,), 2.0)
        else:
            df_buf = torch.as_tensor(df, dtype=torch.float32).view(self.d)
        
        if train_df:
            self.df = nn.Parameter(df_buf)
        else:
            self.register_buffer("df", df_buf)

    def forward(self, num_samples=1):
        # a) create a batch of StudentT’s over each of d dims
        #    StudentT(df) has batch_shape=(d,), event_shape=()
        student = StudentT(self.df)

        # b) rsample yields (num_samples, d)
        eps = student.rsample((num_samples,))

        # c) scale & shift, then reshape back to (N,*shape)
        scale = torch.exp(self.log_scale)
        samples_flat = eps * scale + self.mean     # (N,d)
        samples = samples_flat.view(num_samples, *self.shape)

        # d) compute log‐density
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z):
        # flatten to (N,d)
        N = z.shape[0]
        z_flat = z.view(N, -1)

        # broadcast parameters to (N,d)
        mu    = self.mean.unsqueeze(0)                 # (1,d)
        sigma = torch.exp(self.log_scale).unsqueeze(0)  # (1,d)
        df    = self.df.unsqueeze(0)                    # (1,d)

        # standardize
        x = (z_flat - mu) / sigma                       # (N,d)

        # log‐pdf of StudentT:
        #   lgamma((ν+1)/2) − lgamma(ν/2) − ½[ ln(νπ) + 2 ln σ ]
        #   − (ν+1)/2 · ln(1 + x²/ν)
        coef = (
            torch.lgamma((df + 1.0) / 2.0)
            - torch.lgamma(df / 2.0)
            - 0.5 * (torch.log(df * np.pi) + 2.0 * torch.log(sigma))
        )                                               # (1,d) → broadcast

        lp = coef - ((df + 1.0) / 2.0) * torch.log1p(x * x / df)  # (N,d)
        return lp.sum(dim=1)                            # (N,)

class DirichletProcessMixture(BaseDistribution):
    def __init__(
        self,
        shape,
        components=None,
        T=None,
        alpha=1.0,
        train_alpha=True,
        mean_scale = 1.0
    ):
        super().__init__()
        # record shape for sampling
        if isinstance(shape, int): shape = (shape,)
        self.shape = shape
        # default T=30 if no components provided
        if components is None:
            T = T or 30
            comps = []
            for _ in range(T):
                mean = torch.randn(int(np.prod(self.shape))) * 2.0 * mean_scale
                comps.append(GaussianDistribution(self.shape, mean=mean))
            self.components = nn.ModuleList(comps)
            self.T = T
        else:
            self.components = nn.ModuleList(components)
            self.T = len(self.components) if T is None else T
            assert self.T == len(self.components), "T must match number of components"
        
        # concentration alpha
        init_alpha = torch.tensor(alpha, dtype=torch.float32)
        if train_alpha:
            self.log_alpha = nn.Parameter(torch.log(init_alpha))
        else:
            self.register_buffer("log_alpha", torch.log(init_alpha))

        # variational stick-breaking parameters
        self.log_a = nn.Parameter(torch.zeros(self.T - 1))
        self.log_b = nn.Parameter(torch.log(torch.ones(self.T - 1) * alpha))

        # buffers for expected weights (no grad)
        self.register_buffer("pi", torch.zeros(self.T))
        self.register_buffer("log_pi", torch.zeros(self.T))
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

    def _compute_expected_pi(self):
        if self.T == 1:
            return torch.tensor([1.0]), torch.tensor([0.0])
        a = torch.exp(self.log_a)
        b = torch.exp(self.log_b)
        v_mean = a / (a + b)
        pis = []
        remaining = torch.ones((), device=a.device)
        for k in range(self.T - 1):
            pis.append(v_mean[k] * remaining)
            remaining = remaining * (1 - v_mean[k])
        pis.append(remaining)
        return torch.stack(pis, dim=0), torch.log(torch.stack(pis, dim=0) + 1e-12)

    def forward(self, num_samples=1, return_component = False):
        device = self.log_alpha.device

        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

        if self.T == 1:
            z, _ = self.components[0].forward(num_samples)       # [N,*shape]
            log_q = self.components[0].log_prob(z)               # [N]
            if return_component:
                return z, log_q, torch.zeros_like(z)
            return z, log_q
        
        a = torch.exp(self.log_a).unsqueeze(0)                   # (1, T-1)
        b = torch.exp(self.log_b).unsqueeze(0)                   # (1, T-1)
        U = torch.rand(num_samples, self.T - 1, device=device)   # (N, T-1)
        V = (1 - (1 - U) ** (1.0 / b)) ** (1.0 / a)
        V = V.clamp(min=1e-6, max=1-1e-6)

        one_minus_V = 1 - V                                        # (N, T-1)
        cumprod_om = torch.cumprod(one_minus_V, dim=1)             # (N, T-1), 

        # 이전까지의 “remaining” 계산
        prod_prev = torch.cat([
            torch.ones(num_samples, 1, device=device),            # initial remaining=1
            cumprod_om[:, :-1]                                     # for k>1
        ], dim=1)                                                   # (N, T-1)

        # **여기서 잘못된 부분 수정**: 마지막 weight → cumprod_om[:, -1:]
        remaining = cumprod_om[:, -1:].clone()                     # (N, 1)

        # 각 컴포넌트 weight 계산
        pis = torch.cat([V * prod_prev, remaining], dim=1)         # (N, T)
        # (사실 이 합은 항상 1이므로 normalize 불필요하지만 안전하게)
        pi_mat = pis / pis.sum(dim=1, keepdim=True)                # (N, T)

        # 3) 샘플별 모드 선택
        modes = torch.multinomial(pi_mat, num_samples=1, replacement=True).squeeze(1)

        # 4) log π 샘플별로 미리 계산
        log_pi_mat = torch.log(pi_mat + 1e-12)  # (N, T)

        # 5) 최종 출력 텐서 초기화
        z = torch.zeros((num_samples, *self.shape), device=device)
        log_q = torch.zeros(num_samples,            device=device)
        
        # 6) 선택된 모드별로 개별 컴포넌트에서만 forward
        for k, comp in enumerate(self.components):
            idx = (modes == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            z_k, _ = comp.forward(idx.numel())  # 샘플 & (여기선 log_q_k 무시)
            z[idx] = z_k


        log_probs = torch.stack([comp.log_prob(z) for comp in self.components], dim=1)

        log_q = torch.logsumexp(log_probs + log_pi_mean.unsqueeze(0), dim=1)

        if return_component:
            return z, log_q, modes
        return z, log_q

    def log_prob(self, z):
        if self.T == 1:
            return self.components[0].log_prob(z)
        pi_mean, log_pi_mean = self._compute_expected_pi()
        self.pi.detach().copy_(pi_mean)
        self.log_pi.detach().copy_(log_pi_mean)

        log_probs = [comp.log_prob(z) for comp in self.components]
        log_probs = torch.stack(log_probs, dim=1)
        return torch.logsumexp(log_probs + log_pi_mean.unsqueeze(0), dim=1)
    
class tDist(BaseDistribution):
    """
    Independent multivariate Student's t-distribution with per-dimension degrees of freedom.
    """
    def __init__(self, shape=1, mean=None, scale=None, df=None, train_df = True, device=None):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        self.shape = shape
        self.d = int(np.prod(shape))

        # Device
        self.device = torch.device(device) if device is not None else torch.device('cpu')

        # Mean
        if mean is None:
            mean = torch.zeros(self.d, device=self.device)
        else:
            mean = torch.tensor(mean, dtype=torch.float32, device=self.device).view(self.d)

        # Scale parameter (log-scale internally)
        if scale is None:
            log_scale = torch.zeros(self.d, device=self.device)
        else:
            scale = torch.tensor(scale, dtype=torch.float32, device=self.device).view(self.d)
            log_scale = torch.log(scale)

        # Degrees of freedom per dimension
        if df is None:
            df_init = torch.full((self.d,), 2.0, dtype=torch.float32, device=self.device)
        else:
            df_tensor = torch.tensor(df, dtype=torch.float32, device=self.device)
            if df_tensor.numel() == 1:
                df_init = df_tensor.repeat(self.d)
            else:
                assert df_tensor.numel() == self.d, (
                    f"df must be scalar or have length {self.d}, got {df_tensor.numel()}"
                )
                df_init = df_tensor

        # Register parameters
        self.mean = nn.Parameter(mean)
        self.log_scale = nn.Parameter(log_scale)
        if train_df:
            self.df = nn.Parameter(df_init)
        else:
            self.register_buffer("df", df_init)
        
    def forward(self, num_samples=1):
        # Create StudentT distribution with batch_shape = (d,) and event_shape = ()
        t_dist = torch.distributions.StudentT(
            df=self.df,                 # shape: (d,)
            loc=self.mean,              # shape: (d,)
            scale=torch.exp(self.log_scale)  # shape: (d,)
        )
        # Sample: output shape (num_samples, d)
        samples = t_dist.rsample((num_samples,))
        # Reshape to (num_samples, *shape)
        samples = samples.view((num_samples,) + self.shape)
        # Compute joint log-prob
        log_p = self.log_prob(samples)
        return samples, log_p

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        # z: (num_samples, *shape)
        z_flat = z.view(z.shape[0], self.d)
        df = self.df.view(1, self.d)
        mu = self.mean.view(1, self.d)
        sigma = torch.exp(self.log_scale).view(1, self.d)

        # Standardized variable
        x = (z_flat - mu) / sigma

        # Compute per-dimension log-pdf terms
        # lgamma((df+1)/2) - lgamma(df/2) - 0.5*(log(df*pi) + 2*log(sigma)) - ((df+1)/2)*log1p(x^2/df)
        term1 = torch.lgamma((df + 1.0) / 2.0) - torch.lgamma(df / 2.0)
        term2 = -0.5 * (torch.log(df * np.pi) + 2.0 * torch.log(sigma))
        term3 = -((df + 1.0) / 2.0) * torch.log1p(x * x / df)
        log_per_dim = term1 + term2 + term3

        # Sum across dimensions for joint log-prob
        lp = log_per_dim.sum(dim=1)
        return lp
