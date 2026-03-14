import torch
import torch.nn as nn
import numpy as np
from normflows import NormalizingFlow

class MixtureBaseNormalizingFlow(nn.Module):
    def __init__(self, q0, shared_flows, per_component_flows=None, p=None):
        super().__init__()
        self.q0 = q0
        self.p = p
        self.shared_flows = nn.ModuleList(shared_flows)

        if per_component_flows is None:
            per_component_flows = [[] for _ in range(self.q0.T)]
        else:
            assert len(per_component_flows) == self.q0.T


        self.component_flows = nn.ModuleList([
            nn.ModuleList(row) for row in per_component_flows
        ])
        all_flows = list(self.shared_flows) \
            + [flow for comp in self.component_flows for flow in comp]
        self.flows = nn.ModuleList(all_flows)

    def sample(self, num_samples=1):
        z, log_q0, modes = self.q0(num_samples, return_component=True)

        for flow in self.shared_flows:
            z, log_det = flow(z)
            log_q0 = log_q0 - log_det

        for k in range(self.q0.T):
            idx = (modes == k).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            z_k = z[idx]
            for flow in self.component_flows[k]:
                z_k, log_det = flow(z_k)
                log_q0[idx] = log_q0[idx] - log_det
            z[idx] = z_k

        return z, log_q0

    def log_prob(self, x):
        # mixture weight
        pi_mean, _ = self.q0._compute_expected_pi()
        alphas = pi_mean.to(x.device)

        T = self.q0.T
        N = x.size(0)
        log_probs = []

        # loop over components
        for k in range(T):
            z = x
            total_log_det = torch.zeros(N, device=x.device)

            # 1) component-specific inverse flow
            for flow in reversed(self.component_flows[k]):
                z, log_det = flow.inverse(z)
                total_log_det += log_det

            # 2) shared inverse flow
            for flow in reversed(self.shared_flows):
                z, log_det = flow.inverse(z)
                total_log_det += log_det

            # 3) base log prob
            log_qk = self.q0.components[k].log_prob(z)  # (N,)

            # 4) mixture component contribution
            log_component = torch.log(alphas[k]) + log_qk + total_log_det
            log_probs.append(log_component)

        # 5) log-sum-exp over components
        log_probs = torch.stack(log_probs, dim=1)  # (N, T)
        log_q = torch.logsumexp(log_probs, dim=1)  # (N,)

        return log_q


    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))
