import torch
import normflows as nf
from torch import nn
from normflows.flows import Flow
import torch.nn.functional as F

class Sigmoid(Flow):
    def forward(self, z):
        x = torch.sigmoid(z)
        log_det = -torch.nn.functional.softplus(-z) - torch.nn.functional.softplus(z)
        log_det = log_det.sum(dim=1)  # batch-wise sum
        return x, log_det

    def inverse(self, x):
        z = torch.log(x) - torch.log1p(-x)
        log_det = -torch.log(x) - torch.log1p(-x)
        log_det = log_det.sum(dim=1)
        return z, log_det

class ScaleTransform(Flow):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", scale)

    def forward(self, z):
        x = z * self.scale
        log_det = torch.log(torch.abs(self.scale)).sum()
        log_det = z.new_ones(z.shape[0]) * log_det
        return x, log_det

    def inverse(self, x):
        z = x / self.scale
        log_det = -torch.log(torch.abs(self.scale)).sum()
        log_det = x.new_ones(x.shape[0]) * log_det
        return z, log_det
    
class ShiftTransform(Flow):
    def __init__(self, shift):
        super().__init__()
        self.register_buffer("shift", shift)

    def forward(self, z):
        x = z + self.shift
        log_det = z.new_zeros(z.shape[0])
        return x, log_det

    def inverse(self, x):
        z = x - self.shift
        log_det = x.new_zeros(x.shape[0])
        return z, log_det
    
class SelectiveSoftplus(Flow):
    """
    z → x = softplus(z) only on selected dims, identity elsewhere.
    Enforces x[:, indices] > 0 smoothly.
    """
    def __init__(self, indices, eps=1e-6):
        super().__init__()
        # indices: list or 1D tensor of int positions to apply softplus to
        self.register_buffer("indices", torch.tensor(indices, dtype=torch.long))
        self.eps = eps

    def forward(self, z):
        # z: (batch, latent_size)
        x = z.clone()
        z_sel = z[:, self.indices]           # (batch, n_sel)
        x_sel = F.softplus(z_sel) + self.eps  # ensure strictly >0
        x[:, self.indices] = x_sel
        # log|det J| = sum_d log(sigmoid(z_d)) over selected dims
        log_det = torch.log(torch.sigmoid(z_sel)).sum(dim=1)
        return x, log_det

    def inverse(self, x):
        # x: (batch, latent_size)
        z = x.clone()
        x_sel = x[:, self.indices]
        # inverse softplus: z = log(exp(x) - 1)
        z_sel = torch.log(torch.expm1(x_sel).clamp_min(self.eps))
        z[:, self.indices] = z_sel
        # log|det J⁻¹| = sum_d log(1 + exp(-x_d)) over selected dims
        log_det = torch.log1p(torch.exp(-x_sel)).sum(dim=1)
        return z, log_det
    
import math
import torch
import torch.nn.functional as F

SQRT_2 = math.sqrt(2)
LOG_2_OVER_PI = math.log(2.0 / math.pi)

class TailTransformFlow(Flow):
    def __init__(self, features, shift_init=None, scale_init=None,
                 pos_tail_init=None, neg_tail_init=None, fix=False):
        super().__init__()
        self.fix = fix
        shift_init = torch.zeros(features) if shift_init is None else shift_init
        scale_init = torch.ones(features)  if scale_init is None else scale_init
        pos_init   = torch.ones(features)  if pos_tail_init is None else pos_tail_init
        neg_init   = torch.ones(features)  if neg_tail_init is None else neg_tail_init

        self.register_buffer('pos_id_mask', torch.isnan(pos_init))
        self.register_buffer('neg_id_mask', torch.isnan(neg_init))

        self.shift = torch.nn.Parameter(shift_init)
        self._unc_scale = torch.nn.Parameter(torch.log(torch.exp(scale_init - 1e-3) - 1.0))

        if not self.fix:
            up = pos_init.clone(); un = neg_init.clone()
            up[self.pos_id_mask] = 0.0
            un[self.neg_id_mask] = 0.0
            self._unc_pos_tail = torch.nn.Parameter(torch.log(torch.exp(up) - 1.0))
            self._unc_neg_tail = torch.nn.Parameter(torch.log(torch.exp(un) - 1.0))
            
        else:
            #self.register_buffer('shift', shift_init)
            #self.register_buffer('_unc_scale', torch.log(torch.exp(scale_init - 1e-3) - 1.0))
            self.register_buffer('pos_tail_const', pos_init)
            self.register_buffer('neg_tail_const', neg_init)

    @property
    def scale(self):
        return 1e-3 + F.softplus(self._unc_scale)

    @property
    def pos_tail(self):
        return self.pos_tail_const if self.fix else F.softplus(self._unc_pos_tail)

    @property
    def neg_tail(self):
        return self.neg_tail_const if self.fix else F.softplus(self._unc_neg_tail)

    def _dbg_stats(self, tag, u, g, lad, lam=None, upper=1.0):
        with torch.no_grad():
            tol = 1e-4
            near_lo = (g < tol).float().mean().item()
            near_hi = (g > (upper - tol)).float().mean().item()
            lad_hi = (lad > 50).float().mean().item()
            lad_lo = (lad < -50).float().mean().item()
            lam_hi = (lam > 5).float().mean().item() if lam is not None else float('nan')
            if max(near_lo, near_hi, lad_hi, lad_lo) > 0.0 or (lam is not None and lam_hi > 0.0):
                print(f"[TTF/{tag}] g[min,max]={g.min().item():.3e},{g.max().item():.3e} "
                      f"near_lo%={100*near_lo:.2f} near_hi%={100*near_hi:.2f} "
                      f"lad[min,max]={lad.min().item():.2f},{lad.max().item():.2f} "
                      f"|lad|>50%={100*max(lad_hi, lad_lo):.2f} "
                      f"|u|max={u.abs().max().item():.2f} "
                      f"lam>5%={(100*lam_hi if lam is not None else float('nan')):.2f}")

    def forward(self, z):
        u = (z - self.shift) / self.scale
        pos_mask = (u >= 0) & self.pos_id_mask.view(1, -1)
        neg_mask = (u <  0) & self.neg_id_mask.view(1, -1)
        mask = pos_mask | neg_mask

        lam = torch.where(u >= 0, self.pos_tail, self.neg_tail)
        lam_eff = lam.masked_fill(mask, 1.0)               # identity 방향 안정화
        s = torch.sign(u); m = u.abs()

        g = torch.erfc(m / SQRT_2).clamp(min=1e-6, max=2.0)
        x_ext = (g.pow(-lam_eff) - 1.0) / lam_eff
        y = s * x_ext
        x = self.shift + self.scale * y

        log_abs_dgdu = 0.5*LOG_2_OVER_PI - 0.5*(u*u)
        lad = (-(lam_eff + 1.0)) * torch.log(g) + log_abs_dgdu

        #self._dbg_stats("fwd", u, g, lad, lam=lam_eff, upper=2.0)

        x  = torch.where(mask, z, x)
        ld = torch.where(mask, torch.zeros_like(lad), lad) # scale은 정규화로 소거
        return x, ld.sum(dim=1) 

    def inverse(self, x):
        u_out = (x - self.shift) / self.scale
        pos_mask = (u_out >= 0) & self.pos_id_mask.view(1, -1)
        neg_mask = (u_out <  0) & self.neg_id_mask.view(1, -1)
        mask = pos_mask | neg_mask

        s = torch.sign(u_out); m = u_out.abs()
        lam = torch.where(u_out >= 0, self.pos_tail, self.neg_tail)
        lam_eff = lam.masked_fill(mask, 1.0)

        inner = (1.0 + lam_eff * m)
        g = inner.pow(-1.0 / lam_eff).clamp(min=1e-6, max=1.0 - 1e-6)
        u_in = s * (SQRT_2 * torch.erfinv(1.0 - g))
        z = self.shift + self.scale * u_in

        g_in = torch.erfc(u_in.abs() / SQRT_2).clamp(min=1e-6, max=2.0)
        log_abs_dgdu = 0.5*LOG_2_OVER_PI - 0.5*(u_in*u_in)
        lad = (-(lam_eff + 1.0)) * torch.log(g_in) + log_abs_dgdu

        #self._dbg_stats("inv", u_out, g, lad, lam=lam_eff, upper=1.0)

        ld = torch.where(mask, torch.zeros_like(lad), -lad)
        log_det = ld.sum(dim=1)
        z = torch.where(mask, x, z)
        return z, log_det
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, StudentT
from scipy.special import stdtrit as stdtrit_cpu

# --- Helper Functions ---
def inv_student_t_cdf(p: torch.Tensor, df: torch.Tensor) -> torch.Tensor:
    """
    Inverse CDF (quantile) of Student-t via CuPy (GPU) or SciPy (CPU).
    """
    if p.device.type == "cuda":
        try:
            import cupy as cp
            from cupyx.scipy.special import stdtrit as _stdtrit_gpu
            
            p_cp = cp.asarray(p.detach())
            df_cp = cp.asarray(df.detach())
            y_cp = _stdtrit_gpu(df_cp, p_cp)
            return torch.as_tensor(y_cp, device=p.device)
        except ImportError:
            pass # Fallback to CPU if cupy is not installed or failed

    # CPU fallback
    p_np = p.detach().cpu().numpy()
    df_np = df.detach().cpu().numpy()
    y_np = stdtrit_cpu(df_np, p_np)
    return torch.from_numpy(y_np).to(p.device)

class StudentTTailFlow(nn.Module):
    def __init__(self, features,
                 shift_init=None, scale_init=None,
                 init_df_pos=None, init_df_neg=None,
                 train_df: bool = False):
        super().__init__()
        
        # 1. 기본값 설정
        shift_init = torch.zeros(features) if shift_init is None else shift_init
        scale_init = torch.ones(features)  if scale_init is None else scale_init
        pos_init = torch.ones(features)  if init_df_pos is None else init_df_pos
        neg_init = torch.ones(features)  if init_df_neg is None else init_df_neg

        # 2. NaN 마스크 생성 (이 부분이 Identity 여부를 결정)
        # NaN이면 True (Identity), 숫자면 False (Transform)
        self.register_buffer('pos_id_mask', torch.isnan(init_df_pos))
        self.register_buffer('neg_id_mask', torch.isnan(init_df_neg))

        self.shift = torch.nn.Parameter(shift_init)
        self._unc_scale = torch.nn.Parameter(torch.log(torch.exp(scale_init - 1e-3) - 1.0))

        # 3. 안전한 초기화 값 생성 (핵심 수정 사항)
        # NaN인 곳을 100.0 같은 안전한 숫자로 바꿔야 StudentT 생성 시 에러가 안 남
        # 또한 df가 1.0 이하면 softplus 역산이 터지므로 최소 1.001로 클램핑
        # safe_nu_pos = torch.where(torch.isnan(init_nu_pos), torch.tensor(100.0), init_nu_pos).clamp(min=1.001)
        # safe_nu_neg = torch.where(torch.isnan(init_nu_neg), torch.tensor(100.0), init_nu_neg).clamp(min=1.001)

        # 4. 파라미터 등록
        self.register_buffer('pos_df', torch.nan_to_num(pos_init, 10.0))
        self.register_buffer('neg_df', torch.nan_to_num(neg_init, 10.0))

    @property
    def scale(self) -> torch.Tensor:
        return 1e-3 + F.softplus(self._unc_scale)

    @property
    def df_pos(self) -> torch.Tensor:
        return self.pos_df

    @property
    def df_neg(self) -> torch.Tensor:
        return self.neg_df

    def forward(self, z: torch.Tensor):
        u = (z - self.shift) / self.scale
        pos_mask = (u >= 0) & self.pos_id_mask.view(1, -1)
        neg_mask = (u <  0) & self.neg_id_mask.view(1, -1)
        mask = pos_mask | neg_mask


        u_abs = u.abs()
        p_tail = Normal(0, 1).cdf(-u_abs)


        y_pos_tail = inv_student_t_cdf(p_tail, self.df_pos)
        y_neg_tail = inv_student_t_cdf(p_tail, self.df_neg)

        y = torch.where(u >= 0, -y_pos_tail, y_neg_tail)


        pos_mask_bc = self.pos_id_mask.unsqueeze(0).expand_as(u)
        neg_mask_bc = self.neg_id_mask.unsqueeze(0).expand_as(u)
        
        y = torch.where((u >= 0) & pos_mask_bc, u, y)
        y = torch.where((u < 0)  & neg_mask_bc, u, y)

        x = self.shift + self.scale * y

        # --- 2. Log Determinant ---
        log_phi = Normal(0, 1).log_prob(u)
        

        log_t_pos = StudentT(self.df_pos).log_prob(y)
        log_t_neg = StudentT(self.df_neg).log_prob(y)
        
        log_t = torch.where(u >= 0, log_t_pos, log_t_neg)

        log_t = torch.where((u >= 0) & pos_mask_bc, log_phi, log_t)
        log_t = torch.where((u < 0)  & neg_mask_bc, log_phi, log_t)

        log_det = (log_phi - log_t).sum(-1)
        
        return x, log_det

    def inverse(self, x: torch.Tensor):
        mu, sigma = self.shift, self.scale
        y = (x - mu) / sigma


        y_np = y.detach().cpu().numpy()
        df_p_np = self.df_pos.detach().cpu().numpy()
        df_n_np = self.df_neg.detach().cpu().numpy()
        
        from scipy.special import stdtr as _stdtr_cpu
        y_abs = np.abs(y_np)
        

        cdf_p_tail = _stdtr_cpu(df_p_np, -y_abs)
        cdf_n_tail = _stdtr_cpu(df_n_np, -y_abs)
        
        p_tail_pos = torch.from_numpy(cdf_p_tail).to(x.device)
        p_tail_neg = torch.from_numpy(cdf_n_tail).to(x.device)

        u_tail_pos = Normal(0,1).icdf(p_tail_pos)
        u_tail_neg = Normal(0,1).icdf(p_tail_neg)
        
        u = torch.where(y >= 0, -u_tail_pos, u_tail_neg)

        # [Identity Masking]
        pos_mask_bc = self.pos_id_mask.unsqueeze(0).expand_as(y)
        neg_mask_bc = self.neg_id_mask.unsqueeze(0).expand_as(y)
        u = torch.where((y >= 0) & pos_mask_bc, y, u)
        u = torch.where((y < 0)  & neg_mask_bc, y, u)

        z = mu + sigma * u

        # --- 2. Log Determinant (Inverse) ---
        log_phi = Normal(0, 1).log_prob(u)
        

        log_t_pos = StudentT(self.df_pos).log_prob(y)
        log_t_neg = StudentT(self.df_neg).log_prob(y)
        
        log_t = torch.where(y >= 0, log_t_pos, log_t_neg)

        log_t = torch.where((y >= 0) & pos_mask_bc, log_phi, log_t)
        log_t = torch.where((y < 0)  & neg_mask_bc, log_phi, log_t)

        inv_log_det = (log_t - log_phi).sum(-1)

        return z, inv_log_det

class Softplus(Flow):
    """
    Softplus flow that only transforms the specified dimensions of z,
    leaving all other dims as identity.

    x_i = log(1 + exp(z_i))       for i in self.dims
    x_j = z_j                     for j not in self.dims
    """
    def __init__(self, dims):
        """
        dims: int or sequence of ints
          the indices of the latent dimensions to transform
        """
        super().__init__()
        if isinstance(dims, int):
            self.dims = [dims]
        else:
            self.dims = list(dims)

    def forward(self, z: torch.Tensor):
        """
        z: (batch, D)
        returns:
          x: (batch, D)
          log_det: (batch,) sum of log |dx/dz| over transformed dims
        """
        # make a copy so we can do identity on the other dims
        x = z.clone()

        # extract only the dims we want to Softplus
        z_t = z[:, self.dims]
        x_t = F.softplus(z_t)                  # log(1+e^z)

        # put transformed dims back into the full vector
        x[:, self.dims] = x_t

        # Jacobian diagonal entries for Softplus: sigmoid(z_t)
        # so log_det = sum log(sigmoid(z_t)) = -softplus(-z_t)
        log_det = (-F.softplus(-z_t)).sum(dim=1)
        return x, log_det

    def inverse(self, x: torch.Tensor):
        """
        x: (batch, D)
        returns:
          z: (batch, D)
          log_det: (batch,) sum of log |dz/dx| over transformed dims
        """
        z = x.clone()
        x_t = x[:, self.dims]

        # invert softplus: z = log(exp(x)-1)
        z_t = torch.log(torch.expm1(x_t))
        z[:, self.dims] = z_t

        # inverse Jacobian log-det = - (forward log-det at z_t)
        inv_log_det = (+F.softplus(-z_t)).sum(dim=1)
        return z, inv_log_det

class SafeAutoregressiveRQS(nf.flows.Flow):
    def __init__(
        self,
        num_input_channels,
        num_blocks,
        num_hidden_channels,
        num_context_channels=None,
        num_bins=8,
        tail_bound=8.0,              # 더 크게
        activation=nn.ReLU,
        dropout_probability=0.0,
        permute_mask=False,
        init_identity=True,
        min_bin_width=1e-2,          # 여유 있게
        min_bin_height=1e-2,
        min_derivative=1e-2,
    ):
        super().__init__()
        self.mprqat = MaskedPiecewiseRationalQuadraticAutoregressive2(
            features=num_input_channels,
            hidden_features=num_hidden_channels,
            context_features=num_context_channels,
            num_bins=num_bins,
            tails="linear",
            tail_bound=tail_bound,
            num_blocks=num_blocks,
            use_residual_blocks=True,
            random_mask=False,
            permute_mask=permute_mask,
            activation=activation(),
            dropout_probability=dropout_probability,
            use_batch_norm=False,
            init_identity=init_identity,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
        )

    def forward(self, z, context=None):
        z, log_det = self.mprqat.inverse(z, context=context)
        return z, log_det.view(-1)

    def inverse(self, z, context=None):
        z, log_det = self.mprqat(z, context=context)
        return z, log_det.view(-1)

"""
Implementations of autoregressive transforms.
Code taken from https://github.com/bayesiains/nsf
"""

import numpy as np
import torch
from torch.nn import functional as F

from normflows.nets import made as made_module
from normflows.utils import splines
from normflows.utils.nn import PeriodicFeaturesElementwise

class Autoregressive(Flow):
    """Transforms each input variable with an invertible elementwise transformation.

    The parameters of each invertible elementwise transformation can be functions of previous input
    variables, but they must not depend on the current or any following input variables.

    **NOTE** Calculating the inverse transform is D times slower than calculating the
    forward transform, where D is the dimensionality of the input to the transform.
    """

    def __init__(self, autoregressive_net):
        super(Autoregressive, self).__init__()
        self.autoregressive_net = autoregressive_net

    def forward(self, inputs, context=None):
        autoregressive_params = self.autoregressive_net(inputs, context)
        outputs, logabsdet = self._elementwise_forward(inputs, autoregressive_params)
        return outputs, logabsdet

    def inverse(self, inputs, context=None):
        num_inputs = np.prod(inputs.shape[1:])
        outputs = torch.zeros_like(inputs)
        logabsdet = None
        for _ in range(num_inputs):
            autoregressive_params = self.autoregressive_net(outputs, context)
            outputs, logabsdet = self._elementwise_inverse(
                inputs, autoregressive_params
            )
        return outputs, logabsdet

    def _output_dim_multiplier(self):
        raise NotImplementedError()

    def _elementwise_forward(self, inputs, autoregressive_params):
        raise NotImplementedError()

    def _elementwise_inverse(self, inputs, autoregressive_params):
        raise NotImplementedError()
    
class MaskedPiecewiseRationalQuadraticAutoregressive2(Autoregressive):
    def __init__(
        self,
        features,
        hidden_features,
        context_features=None,
        num_bins=10,
        tails=None,
        tail_bound=1.0,
        num_blocks=2,
        use_residual_blocks=True,
        random_mask=False,
        permute_mask=False,
        activation=F.relu,
        dropout_probability=0.0,
        use_batch_norm=False,
        init_identity=True,
        min_bin_width=splines.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.DEFAULT_MIN_DERIVATIVE,
    ):
        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tails = tails

        if isinstance(self.tails, list) or isinstance(self.tails, tuple):
            ind_circ = []
            for i in range(features):
                if self.tails[i] == "circular":
                    ind_circ += [i]
            if torch.is_tensor(tail_bound):
                scale_pf = np.pi / tail_bound[ind_circ]
            else:
                scale_pf = np.pi / tail_bound
            preprocessing = PeriodicFeaturesElementwise(features, ind_circ, scale_pf)
        else:
            preprocessing = None

        autoregressive_net = made_module.MADE(
            features=features,
            hidden_features=hidden_features,
            context_features=context_features,
            num_blocks=num_blocks,
            output_multiplier=self._output_dim_multiplier(),
            use_residual_blocks=use_residual_blocks,
            random_mask=random_mask,
            permute_mask=permute_mask,
            activation=activation,
            dropout_probability=dropout_probability,
            use_batch_norm=use_batch_norm,
            preprocessing=preprocessing,
        )

        if init_identity:
            torch.nn.init.constant_(autoregressive_net.final_layer.weight, 0.0)
            torch.nn.init.constant_(
                autoregressive_net.final_layer.bias,
                np.log(np.exp(1 - min_derivative) - 1),
            )

        super().__init__(autoregressive_net)

        if torch.is_tensor(tail_bound):
            self.register_buffer("tail_bound", tail_bound)
        else:
            self.tail_bound = tail_bound

    def _output_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        elif self.tails == "circular":
            return self.num_bins * 3
        else:
            return self.num_bins * 3 + 1

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline2
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline2
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        return outputs, sum_except_batch(logabsdet)

    def _elementwise_forward(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params)

    def _elementwise_inverse(self, inputs, autoregressive_params):
        return self._elementwise(inputs, autoregressive_params, inverse=True)

def sum_except_batch(x, num_batch_dims=1):
    """Sums all elements of `x` except for the first `num_batch_dims` dimensions."""
    reduce_dims = list(range(num_batch_dims, x.ndimension()))
    return torch.sum(x, dim=reduce_dims)

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def unconstrained_rational_quadratic_spline2(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., 0] = constant
        unnormalized_derivatives_[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif tails == "circular":
        unnormalized_derivatives_ = F.pad(unnormalized_derivatives, pad=(0, 1))
        unnormalized_derivatives_[..., -1] = unnormalized_derivatives_[..., 0]

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    elif isinstance(tails, list) or isinstance(tails, tuple):
        unnormalized_derivatives_ = unnormalized_derivatives.clone()
        ind_lin = [t == "linear" for t in tails]
        ind_circ = [t == "circular" for t in tails]
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives_[..., ind_lin, 0] = constant
        unnormalized_derivatives_[..., ind_lin, -1] = constant
        unnormalized_derivatives_[..., ind_circ, -1] = unnormalized_derivatives_[
            ..., ind_circ, 0
        ]
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    if torch.is_tensor(tail_bound):
        tail_bound_ = torch.broadcast_to(tail_bound, inputs.shape)
        left = -tail_bound_[inside_interval_mask]
        right = tail_bound_[inside_interval_mask]
        bottom = -tail_bound_[inside_interval_mask]
        top = tail_bound_[inside_interval_mask]
    else:
        left = -tail_bound
        right = tail_bound
        bottom = -tail_bound
        top = tail_bound

    (
        outputs_masked,
        logabsdet_masked
    ) = rational_quadratic_spline2(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives_[inside_interval_mask, :],
        inverse=inverse,
        left=left,
        right=right,
        bottom=bottom,
        top=top,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    if outputs.dtype == outputs_masked.dtype and logabsdet.dtype == logabsdet_masked.dtype:
        outputs[inside_interval_mask] = outputs_masked
        logabsdet[inside_interval_mask] = logabsdet_masked
    else:
        outputs[inside_interval_mask] = outputs_masked.to(outputs.dtype)
        logabsdet[inside_interval_mask] = logabsdet_masked.to(logabsdet.dtype)

    return outputs, logabsdet

def rational_quadratic_spline2(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    num_bins = unnormalized_widths.shape[-1]

    if torch.is_tensor(left):
        lim_tensor = True
    else:
        lim_tensor = False

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumwidths = (right[..., None] - left[..., None]) * cumwidths + left[..., None]
    else:
        cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    if lim_tensor:
        cumheights = (top[..., None] - bottom[..., None]) * cumheights + bottom[
            ..., None
        ]
    else:
        cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        if discriminant < 0:
            print("discriminant: ", discriminant)
            print("b: ", b)
            print("a: ", a)
            print("c: ", c)
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet