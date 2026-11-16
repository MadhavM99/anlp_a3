import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x):
    first, second = x.chunk(2, dim=-1)
    return torch.cat([-second, first], dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim)), persistent=False)

    def forward(self, length, device):
        positions = torch.arange(length, device=device).float()
        freqs = torch.outer(positions, self.inv_freq)
        angles = torch.cat([freqs, freqs], dim=-1)
        return angles.cos()[None, None], angles.sin()[None, None]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def forward(self, x):
        batch, length, d_model = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        cos, sin = self.rotary(length, x.device)
        q = (q * cos + rotate_half(q) * sin).to(v.dtype)
        k = (k * cos + rotate_half(k) * sin).to(v.dtype)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.out(attended.transpose(1, 2).reshape(batch, length, d_model))
