import torch.nn as nn

from src.part1.models.common import Block, RMSNorm
from src.part1.models.deltanet import DeltaNet
from src.part1.models.mamba import Mamba
from src.part1.models.transformer import CausalSelfAttention

KINDS = {"transformer": "attention", "mamba": "mamba", "deltanet": "deltanet"}


def layout(cfg):
    if cfg["name"] == "hybrid":
        return [("mamba", "deltanet")[index % 2] for index in range(cfg["n_layers"])]
    return [KINDS[cfg["name"]]] * cfg["n_layers"]


def make_block(kind, cfg):
    d_model, dropout = cfg["d_model"], cfg["dropout"]
    if kind == "attention":
        mixer = CausalSelfAttention(d_model, cfg["n_heads"], dropout)
    elif kind == "mamba":
        mixer = Mamba(d_model, cfg["mamba_d_inner"], cfg["d_state"], cfg["d_conv"])
    else:
        mixer = DeltaNet(d_model, cfg["n_heads"], cfg["d_conv"], cfg["delta_chunk"])
    return Block(mixer, d_model, cfg["d_ff"], dropout, with_mlp=kind != "mamba")


def init_weights(module):
    if getattr(module, "custom_init", False):
        return
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=0.02)


class CipherDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.kinds = layout(cfg)
        self.embedding = nn.Embedding(cfg["input_vocab"], cfg["d_model"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.layers = nn.ModuleList(make_block(kind, cfg) for kind in self.kinds)
        self.norm = RMSNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["output_vocab"])
        self.apply(init_weights)

    def forward(self, x):
        h = self.dropout(self.embedding(x))
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))

    def extra_flops(self, length):
        return sum(layer.mixer.extra_flops(length) for layer in self.layers if hasattr(layer.mixer, "extra_flops"))


def build_model(cfg):
    return CipherDecoder(cfg)
