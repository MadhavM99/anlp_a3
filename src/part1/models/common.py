import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x_float = x.float()
        normed = x_float * torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(x.dtype)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class CausalDepthwiseConv(nn.Module):
    def __init__(self, channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(channels, channels, kernel_size, groups=channels, padding=kernel_size - 1, bias=bias)

    def forward(self, x):
        length = x.size(1)
        return self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)


class Block(nn.Module):
    def __init__(self, mixer, d_model, d_ff, dropout, with_mlp):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = mixer
        self.dropout = nn.Dropout(dropout)
        self.norm2 = RMSNorm(d_model) if with_mlp else None
        self.mlp = MLP(d_model, d_ff, dropout) if with_mlp else None

    def forward(self, x):
        x = x + self.dropout(self.mixer(self.norm1(x)))
        if self.mlp is not None:
            x = x + self.dropout(self.mlp(self.norm2(x)))
        return x
