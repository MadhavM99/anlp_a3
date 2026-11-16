import torch
import torch.nn as nn
import torch.nn.functional as F

from src.part1.models.common import CausalDepthwiseConv, RMSNorm


def delta_rule_chunkwise(q, k, v, beta, chunk):
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, beta = q.float(), k.float(), v.float(), beta.float()
        batch, heads, length, d_k = q.shape
        d_v = v.size(-1)
        pad = (-length) % chunk
        if pad:
            q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
            beta = F.pad(beta, (0, pad))
        n_chunks = (length + pad) // chunk
        q, k, v = (t.view(batch, heads, n_chunks, chunk, -1) for t in (q, k, v))
        beta = beta.view(batch, heads, n_chunks, chunk, 1)
        k_beta, v_beta = k * beta, v * beta
        eye = torch.eye(chunk, device=q.device)
        system = eye + torch.tril(k_beta @ k.transpose(-1, -2), diagonal=-1)
        w = torch.linalg.solve_triangular(system, k_beta, upper=False, unitriangular=True)
        u = torch.linalg.solve_triangular(system, v_beta, upper=False, unitriangular=True)
        causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=q.device))
        state = q.new_zeros(batch, heads, d_k, d_v)
        outputs = []
        for i in range(n_chunks):
            q_i, k_i = q[:, :, i], k[:, :, i]
            u_i = u[:, :, i] - w[:, :, i] @ state
            scores = (q_i @ k_i.transpose(-1, -2)).masked_fill(~causal, 0.0)
            outputs.append(q_i @ state + scores @ u_i)
            state = state + k_i.transpose(-1, -2) @ u_i
        return torch.stack(outputs, dim=2).view(batch, heads, n_chunks * chunk, d_v)[:, :, :length]


class DeltaNet(nn.Module):
    def __init__(self, d_model, n_heads, d_conv, chunk):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk = chunk
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads)
        self.q_conv = CausalDepthwiseConv(d_model, d_conv, bias=False)
        self.k_conv = CausalDepthwiseConv(d_model, d_conv, bias=False)
        self.v_conv = CausalDepthwiseConv(d_model, d_conv, bias=False)
        self.o_norm = RMSNorm(self.head_dim)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def heads(self, x):
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x):
        batch, length, d_model = x.shape
        q = F.normalize(self.heads(F.silu(self.q_conv(self.q_proj(x)))).float(), p=2, dim=-1)
        k = F.normalize(self.heads(F.silu(self.k_conv(self.k_proj(x)))).float(), p=2, dim=-1)
        v = self.heads(F.silu(self.v_conv(self.v_proj(x))))
        beta = torch.sigmoid(self.b_proj(x).float()).transpose(1, 2)
        o = delta_rule_chunkwise(q, k, v, beta, self.chunk)
        o = self.o_norm(o).transpose(1, 2).reshape(batch, length, d_model)
        return self.o_proj(o.to(x.dtype))

    def extra_flops(self, length):
        n_chunks = -(-length // self.chunk)
        return 2 * n_chunks * self.n_heads * self.chunk * self.chunk * self.head_dim
