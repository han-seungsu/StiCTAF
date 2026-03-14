import torch
import torch.nn as nn
import numpy as np
import normflows as nf
from torch.distributions import Multinomial

from matplotlib import pyplot as plt
from tqdm import tqdm
from base import DirichletProcessMixture, MultivariateTDistribution
from scipy.stats import gaussian_kde
from utils import _print_tensor, print_model_parameters
from transforms import TailTransformFlow, StudentTTailFlow
from core import MixtureBaseNormalizingFlow

def reverse_kld(model, num_samples=1024, type = None):
    if type == 'reverse':
        return model.reverse_kld(num_samples)
    elif type == 'stratified':
        return stratified_reverse_kld(model, num_samples)
    elif type == 'siw':
        return siw_reverse_kld(model, num_samples)
    elif type =='componentwise':
        return componentwise_reverse_kld(model, num_samples)
    else:
        raise ValueError(f"Unknown mode: {type}")
'''
def componentwise_reverse_kld(model, num_samples=1024):
    device = next(model.parameters()).device
    T = model.q0.T
    pi_mean, _ = model.q0._compute_expected_pi()
    alphas = pi_mean.to(device)
    n = num_samples // T
    
    if device.type == 'cuda':
        streams = [torch.cuda.Stream(device=device) for _ in range(T)]
        comp_losses = [torch.zeros((), device=device) for _ in range(T)]

        for k, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                z, _ = model.q0.components[k].forward(n)
                log_q = model.q0.log_prob(z)
                for flow in model.flows[k]:
                    z, log_det = flow(z)
                    log_q = log_q - log_det
                log_p = model.p.log_prob(z)
                comp_losses[k] = alphas[k] * (log_q - log_p).mean()

        main_stream = torch.cuda.current_stream()
        for stream in streams:
            main_stream.wait_stream(stream)

        total_loss = torch.stack(comp_losses).sum()
    else:
        total_loss = 0.0
        for k in range(T):
            z, _ = model.q0.components[k].forward(n)
            log_q = model.q0.log_prob(z)
            for flow in model.flows[k]:
                z, log_det = flow(z)
                log_q = log_q - log_det
            log_p = model.p.log_prob(z)
            total_loss = total_loss + alphas[k] * torch.mean(log_q - log_p)

    return total_loss
'''
def componentwise_reverse_kld(model, num_samples=1024):
    device = next(model.parameters()).device
    T = model.q0.T
    pi_mean, _ = model.q0._compute_expected_pi()
    alphas = pi_mean.to(device)
    n = num_samples // T

    # 1) 컴포넌트별 샘플링
    zs = []
    for k in range(T):
        z, _ = model.q0.components[k].forward(n)
        zs.append(z)
    z_all = torch.cat(zs, dim=0)  # (T*n, d)

    # 2) base log_prob
    log_q = model.q0.log_prob(z_all)

    # 3) shared flows
    for flow in model.shared_flows:
        z_all, log_det = flow(z_all)
        log_q = log_q - log_det

    # 4) component-wise flows
    for k in range(T):
        start, end = k*n, (k+1)*n
        z_block = z_all[start:end]
        log_q_block = log_q[start:end]

        # component_flows[k] 안의 모든 flow 순회
        for flow in model.component_flows[k]:
            z_block, log_det = flow(z_block)
            log_q_block = log_q_block - log_det

        # 변환 결과를 원배치에 다시 저장
        z_all[start:end] = z_block
        log_q[start:end] = log_q_block

    # 5) target log_prob & weighted loss
    log_p = model.p.log_prob(z_all)
    total_loss = 0.0
    for k in range(T):
        start, end = k*n, (k+1)*n
        total_loss += alphas[k] * torch.mean(
            log_q[start:end] - log_p[start:end]
        )

    return total_loss

def stratified_reverse_kld(model, num_samples=1024):
    device = next(model.parameters()).device
    T = model.q0.T
    pi_mean, _ = model.q0._compute_expected_pi()
    alphas = pi_mean.to(device)
    n = num_samples // T
    total_samples = n * T

    z_all = []
    for k in range(T):
        z, _ = model.q0.components[k].forward(n)
        z_all.append(z)
    z_all = torch.cat(z_all, dim=0)

    log_q = model.q0.log_prob(z_all)
    for flow in model.flows:
        z_all, log_det = flow(z_all)
        log_q = log_q - log_det

    log_p = model.p.log_prob(z_all)

    total_loss = 0.0
    for k in range(T):
        start = k * n
        end = (k + 1) * n
        total_loss += alphas[k] * torch.mean(log_q[start:end] - log_p[start:end])

    return total_loss

def siw_reverse_kld(model, num_samples=1024):
    device = next(model.parameters()).device
    T = model.q0.T
    pi_mean, _ = model.q0._compute_expected_pi()
    alphas = pi_mean.to(device)
    n = num_samples // T

    zs = [comp.forward(n)[0].to(device) for comp in model.q0.components]
    z_all = torch.cat(zs, dim=0)

    log_q = model.q0.log_prob(z_all)
    for flow in model.flows:
        z_all, log_det = flow(z_all)
        log_q = log_q - log_det

    log_p = model.p.log_prob(z_all)

    w = (log_q - log_p).view(T, n)                   # [T, n]
    log_alpha = torch.log(alphas)                    # [T]
    comp_terms = log_alpha + torch.logsumexp(w, dim=1) - np.log(n)  # [T]

    total_loss = torch.logsumexp(comp_terms, dim=0)   # scalar
    return total_loss

def train(model, args):

    flow_params = list(model.flows.parameters())
    weight_params = []
    
    if isinstance(model.q0, DirichletProcessMixture):
        weight_params = [model.q0.log_a, model.q0.log_b]
    
    flow_ids   = {id(p) for p in flow_params}  
    weight_ids = {id(p) for p in weight_params}

    base_params = [
        p for p in model.parameters()
        if id(p) not in flow_ids and id(p) not in weight_ids
    ]
    saved_flows = None
    if args.freeze_flow:
        saved_flows      = model.flows
        model.flows      = nn.ModuleList()
        flow_params = list()
    
    optimizer = torch.optim.Adam([
        {'params': base_params,   'lr': args.base_lr},
        {'params': flow_params,   'lr': args.flow_lr},
        {'params': weight_params, 'lr': args.weight_lr},
    ], weight_decay=args.weight_decay)


    loss_hist = []
    if args.freeze_flow:
        for param in model.flows.parameters():
            param.requires_grad = False
    

    for it in tqdm(range(args.max_iter)):
        optimizer.zero_grad()
        loss = reverse_kld(model, num_samples = args.num_samples, type=args.loss_type)
        loss.backward()
        if isinstance(model.q0, DirichletProcessMixture):
            for n, p in model.shared_flows[0].mprqat.autoregressive_net.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    print("[GRAD NaN/Inf]", n)
                    raise RuntimeError("MADE grad NaN/Inf")
        optimizer.step()
        loss_hist.append(loss.item())
        if it % args.log_interval == 0:
            print(f"Iter {it}, Loss: {loss.item():.4f}")
            if isinstance(model.q0, DirichletProcessMixture):
                _print_tensor("pi (expected)", model.q0.pi)
                print()
    
        if it == int(args.max_iter * args.freeze_ratio) and args.freeze_flow:
            model.flows = saved_flows

            for param in model.flows.parameters():
                param.requires_grad = True
            
            if isinstance(model.q0, DirichletProcessMixture):
                model.q0.log_a.requires_grad = False
                model.q0.log_b.requires_grad = False
                for t in range(model.q0.T):
                    model.q0.components[t].mean.requires_grad = False
                    model.q0.components[t].log_scale.requires_grad = False
            
            if args.add_ttf and args.loss_type == 'componentwise':
                if args.add_StudentTttf:
                    print_model_parameters(model.q0)
                    device = model.q0.components[0].mean.device
                    mean_vec = model.q0.components[0].mean.to(device)   # shape (D,)
                    D = mean_vec.size(0)
                    pi = model.q0._compute_expected_pi()[0]
                    for t in range(model.q0.T):
                        if (pi[t].item() < args.ttf_threshold):
                            continue
                        index_inv = estimate_tail_index(
                            mean=model.q0.components[t].mean,
                            scale=torch.exp(model.q0.components[t].log_scale),
                            target=model.p,
                            model = model,
                            component = t,
                            df = 2.0,
                            num_samples=100
                            )
                        pos_init = torch.as_tensor(index_inv, dtype=torch.float32, device=device)[0]
                        neg_init = torch.as_tensor(index_inv, dtype=torch.float32, device=device)[1]
                        shift_init = model.q0.components[t].mean.clone().to(device)
                        scale_init = torch.exp(model.q0.components[t].log_scale).clone().to(device)

                        has_number = (
                            torch.any(~torch.isnan(pos_init)) or 
                            torch.any(~torch.isnan(neg_init))
                        )
                        if has_number:
                            ttf = StudentTTailFlow(features=D,
                                                   shift_init= shift_init,
                                                   scale_init= scale_init * args.ttf_init_scale, 
                                                init_df_pos=pos_init,
                                                init_df_neg=neg_init).to(device)

                            model.component_flows[t].insert(0, ttf)
                            print(f"Added TTF to component {t}, "
                                f"pos_init={pos_init.tolist()}, "
                                f"neg_init={neg_init.tolist()}")


                else:
                    print_model_parameters(model.q0)
                    device = model.q0.components[0].mean.device
                    mean_vec = model.q0.components[0].mean.to(device)   # shape (D,)
                    D = mean_vec.size(0)
                    pi = model.q0._compute_expected_pi()[0]


                    for t in range(model.q0.T):
                        if (pi[t].item() < args.ttf_threshold):
                            continue
                        index_inv = estimate_tail_index_inverse(
                            mean=model.q0.components[t].mean,
                            scale=torch.exp(model.q0.components[t].log_scale),
                            target=model.p,
                            model = model,
                            component = t,
                            df = 2.0,
                            num_samples=args.tail_nsamples
                            )
                        pos_init = torch.as_tensor(index_inv, dtype=torch.float32, device=device)[0]
                        neg_init = torch.as_tensor(index_inv, dtype=torch.float32, device=device)[1]

                        shift_init = model.q0.components[t].mean.clone().to(device)
                        scale_init = torch.exp(model.q0.components[t].log_scale).clone().to(device)
                        model.eval()
                        with torch.no_grad():
                            for flows in model.shared_flows:
                                shift_init, _ = flows(shift_init.view(1, -1))
                                scale_init, _ = flows(scale_init.view(1, -1))
                        model.train()

                        has_number = (
                            torch.any(~torch.isnan(pos_init)) or 
                            torch.any(~torch.isnan(neg_init))
                        )
                        if has_number:
                            ttf = TailTransformFlow(features=D, 
                                            pos_tail_init=pos_init,
                                            neg_tail_init=neg_init,
                                            fix=True,
                                            shift_init=shift_init,
                                            scale_init=scale_init * args.ttf_init_scale
                            )
                            model.component_flows[t].insert(0, ttf)
                            print(f"Added TTF to component {t}, "
                                f"pos_init={pos_init.tolist()}, "
                                f"neg_init={neg_init.tolist()}")

                flow_params = list(model.flows.parameters())
                optimizer = torch.optim.Adam([
                    {'params': base_params,   'lr': args.base_lr},
                    {'params': flow_params,   'lr': args.flow_lr},
                    {'params': weight_params, 'lr': args.weight_lr},
                ], weight_decay=args.weight_decay)
    # Save the model
    if args.save_model:
        torch.save(model.state_dict(), f"{args.save_path}/{args.file_name}.pth")
        print(f"Model saved to {args.save_path}/{args.file_name}.pth")
    
    # Plot loss
    plt.figure(figsize=(6, 6))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.show()

def estimate_tail_index_inverse(
    mean,
    scale,
    target,
    num_samples: int = 5000,
    k: int = 5,
    df: float = 2.0,
    model=None,
    component = None
):
    """
    Estimate tail index reciprocals along each coordinate axis (positive and negative directions).

    Returns a tensor of shape (2, d):
      - [0, i] = 1/nu_hat for positive direction of dim i, or NaN if nu_hat not in (0,30]
      - [1, i] = 1/nu_hat for negative direction of dim i, or NaN otherwise
    """
    device = mean.device
    
    if model is not None:
        model.eval()
        with torch.no_grad():
            for flows in model.shared_flows:
                mean, _ = flows(mean.view(1, -1))
                scale, _ = flows(scale.view(1, -1))
        model.train()
    mean = mean.squeeze(0)
    scale = scale.squeeze(0)
    mean = mean.to(device)
    sigma = scale.to(device)
    df_tensor = torch.as_tensor(df, dtype=torch.float32, device=device)
    d = mean.numel()

    # Helper to compute scalar tail index along unit vector u
    def _compute_nu(u):
        # compute scale factor sqrt(u^T Sigma^2 u)
        """for flow in model.shared_flows:
            u, _ = flow.inverse(u)"""
        
        scale_factor = torch.sqrt((sigma.pow(2) * u.pow(2)).sum())
        # sample t
        t_dist = torch.distributions.StudentT(df_tensor)
        t_samples = t_dist.rsample((num_samples,)).to(device)
        # lift to d-dim
        X = mean.view(1, -1) + t_samples.view(-1, 1) * (scale_factor * u).view(1, -1)
        # tail threshold
        sorted_t, _ = torch.sort(t_samples, descending=True)
        t_sk = sorted_t[k-1]
        mask = t_samples >= t_sk
        tail_t = t_samples[mask]
        tail_X = X[mask, :]
        # optionally apply flows
        if model is not None:
            model.eval()
            with torch.no_grad():
                z_sk, z_tail = mean.view(1, -1) + t_sk*(scale_factor*u).view(1, -1), tail_X
                if isinstance(model, MixtureBaseNormalizingFlow):
                    try:
                        for flow in model.component_flows[component]:
                            z_sk, _ = flow(z_sk)
                            z_tail, _ = flow(z_tail)
                    except Exception:
                        pass
                    
                else:
                    for flow in model.flows:
                        z_sk, _ = flow(z_sk)
                        z_tail, _ = flow(z_tail)
            model.train()
            log_sk = target.log_prob(z_sk).squeeze()
            log_tail = target.log_prob(z_tail)
        else:
            #print("u: ", u)
            #print("scale: ", (scale_factor*u).view(1, -1))
            #print("z_sk: ", mean.view(1, -1) + t_sk*(scale_factor*u).view(1, -1))
            #print("t_sk: ", t_sk)
            #print("z_tail: ", tail_X)
            #print("tail_t: ", tail_t)
            log_sk = target.log_prob(mean.view(1,-1) + t_sk*(scale_factor*u).view(1,-1)).squeeze()
            log_tail = target.log_prob(tail_X)
            #print("log_sk: ", log_sk)
            #print("log_tail: ", log_tail)
            #print("den: ", torch.log(tail_t / t_sk))

        num = (log_sk - log_tail).sum()
        den = torch.log(tail_t / t_sk).sum()
        nu_hat = (num / den - 1.0).item()
        #print("nu_hat: ", num / den - 1.0)
        return nu_hat

    # initialize result with NaN
    result = torch.full((2, d), float('nan'), device=device)
    for i in range(d):
        # positive direction
        u_pos = torch.zeros_like(mean)
        u_pos[i] = 1.0
        try:
            nu_p = _compute_nu(u_pos)
            if 1 <= nu_p <= 30:
                result[0, i] = 1.0 / nu_p
        except Exception:
            pass
        
        # negative direction
        u_neg = -u_pos
        try:
            nu_n = _compute_nu(u_neg)
            if 1 <= nu_n <= 30:
                result[1, i] = 1.0 / nu_n
        except Exception:
            pass

    return result
    
def estimate_tail_index(
    mean,
    scale,
    target,
    num_samples: int = 5000,
    k: int = 5,
    df: float = 2.0,
    model=None,
    component = None
):
    """
    Estimate tail index reciprocals along each coordinate axis (positive and negative directions).

    Returns a tensor of shape (2, d):
      - [0, i] = 1/nu_hat for positive direction of dim i, or NaN if nu_hat not in (0,30]
      - [1, i] = 1/nu_hat for negative direction of dim i, or NaN otherwise
    """
    device = mean.device
    if model is not None:
        model.eval()
        with torch.no_grad():
            for flows in model.shared_flows:
                mean, _ = flows.inverse(mean.view(1, -1))
                scale, _ = flows.inverse(scale.view(1, -1))
        model.train()
    mean = mean.squeeze(0)
    scale = scale.squeeze(0)
    mean = mean.to(device)
    sigma = scale.to(device)
    df_tensor = torch.as_tensor(df, dtype=torch.float32, device=device)
    d = mean.numel()

    # Helper to compute scalar tail index along unit vector u
    def _compute_nu(u):
        # compute scale factor sqrt(u^T Sigma^2 u)
        """for flow in model.shared_flows:
            u, _ = flow.inverse(u)"""
        
        scale_factor = torch.sqrt((sigma.pow(2) * u.pow(2)).sum())
        # sample t
        t_dist = torch.distributions.StudentT(df_tensor)
        t_samples = t_dist.rsample((num_samples,)).to(device)
        # lift to d-dim
        X = mean.view(1, -1) + t_samples.view(-1, 1) * (scale_factor * u).view(1, -1)
        # tail threshold
        sorted_t, _ = torch.sort(t_samples, descending=True)
        t_sk = sorted_t[k-1]
        mask = t_samples >= t_sk
        tail_t = t_samples[mask]
        tail_X = X[mask, :]
        # optionally apply flows
        if model is not None:
            model.eval()
            with torch.no_grad():
                z_sk, z_tail = mean.view(1, -1) + t_sk*(scale_factor*u).view(1, -1), tail_X
                if isinstance(model, MixtureBaseNormalizingFlow):
                    try:
                        for flow in model.component_flows[component]:
                            z_sk, _ = flow(z_sk)
                            z_tail, _ = flow(z_tail)
                    except Exception:
                        pass
                    
                else:
                    for flow in model.flows:
                        z_sk, _ = flow(z_sk)
                        z_tail, _ = flow(z_tail)
            model.train()
            log_sk = target.log_prob(z_sk).squeeze()
            log_tail = target.log_prob(z_tail)
        else:
            #print("u: ", u)
            #print("scale: ", (scale_factor*u).view(1, -1))
            #print("z_sk: ", mean.view(1, -1) + t_sk*(scale_factor*u).view(1, -1))
            #print("t_sk: ", t_sk)
            #print("z_tail: ", tail_X)
            #print("tail_t: ", tail_t)
            log_sk = target.log_prob(mean.view(1,-1) + t_sk*(scale_factor*u).view(1,-1)).squeeze()
            log_tail = target.log_prob(tail_X)
            #print("log_sk: ", log_sk)
            #print("log_tail: ", log_tail)
            #print("den: ", torch.log(tail_t / t_sk))

        num = (log_sk - log_tail).sum()
        den = torch.log(tail_t / t_sk).sum()
        nu_hat = (num / den - 1.0).item()
        #print("nu_hat: ", num / den - 1.0)
        return nu_hat

    # initialize result with NaN
    result = torch.full((2, d), float('nan'), device=device)
    for i in range(d):
        # positive direction
        u_pos = torch.zeros_like(mean)
        u_pos[i] = 1.0
        try:
            nu_p = _compute_nu(u_pos)
            if 1 <= nu_p <= 30:
                result[0, i] = nu_p
        except Exception:
            pass
        
        # negative direction
        u_neg = -u_pos
        try:
            nu_n = _compute_nu(u_neg)
            if 1 <= nu_n <= 30:
                result[1, i] = nu_n
        except Exception:
            pass

    return result
def global_mean(base):
    """
    Compute the global mean of a DirichletProcessMixture
    by taking the weighted average of its component means.
    """
    weights = base.pi 

    means = torch.stack([
        comp.mean.view(-1) for comp in base.components
    ], dim=0)

    global_mean_flat = (weights.unsqueeze(1) * means).sum(dim=0)

    return global_mean_flat.view(*base.shape)