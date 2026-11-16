import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.part1.models.common import CausalDepthwiseConv


def selective_scan(u, delta, A, B, C, D):
    with torch.autocast(device_type=u.device.type, enabled=False):
        u, delta, B, C = u.float(), delta.float(), B.float(), C.float()
        decays = torch.exp(delta.unsqueeze(-1) * A).unbind(dim=1)
        drives = (delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)).unbind(dim=1)
        readouts = C.unsqueeze(2).unbind(dim=1)
        state = u.new_zeros(u.size(0), u.size(2), A.size(1))
        outputs = []
        for decay, drive, readout in zip(decays, drives, readouts):
            state = decay * state + drive
            outputs.append((state * readout).sum(dim=-1))
        return torch.stack(outputs, dim=1) + u * D


class Mamba(nn.Module):
    def __init__(self, d_model, d_inner, d_state, d_conv, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = CausalDepthwiseConv(d_inner, d_conv)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.dt_proj.custom_init = True
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)))
        self.A_log.no_weight_decay = True
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D.no_weight_decay = True
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = F.silu(self.conv(u))
        dt, B, C = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt).float())
        A = -torch.exp(self.A_log.float())
        y = selective_scan(u, delta, A, B, C, self.D.float())
        return self.out_proj((y * F.silu(z.float())).to(x.dtype))

    def extra_flops(self, length):
        return length * self.d_inner * self.d_state * 9 + length * self.d_inner * 4
