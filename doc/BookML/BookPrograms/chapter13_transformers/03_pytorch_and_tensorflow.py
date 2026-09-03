"""Chapter 13: listing 3, from the section on pytorch and tensorflow.

Extracted from doc/BookML/chapter13.tex.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """Eq. (13.block), pre-norm, written out rather than assembled."""
    def __init__(self, d=64, H=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)                  # Eq. (13.layernorm)
        self.attn = nn.MultiheadAttention(d, H, dropout=dropout,
                                          batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, d),
            nn.Dropout(dropout))

    def forward(self, x, mask=None):
        h = self.ln1(x)
        a, A = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        x = x + a                                   # residual: see Section 13.block
        x = x + self.mlp(self.ln2(x))
        return x, A


def causal_mask(n, device=None):
    """Eq. (13.mask): True where attention is forbidden."""
    return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), 1)


class Transformer(nn.Module):
    def __init__(self, n_vocab, d=64, H=4, d_ff=256, n_blocks=2, n_ctx=64):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d)
        self.pos = nn.Parameter(torch.zeros(n_ctx, d))   # learned; or Eq. (13.posenc)
        self.blocks = nn.ModuleList(
            [Block(d, H, d_ff) for _ in range(n_blocks)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_vocab)

    def forward(self, idx, causal=True):
        n = idx.shape[1]
        x = self.emb(idx) + self.pos[:n]
        m = causal_mask(n, idx.device) if causal else None
        for blk in self.blocks:
            x, _ = blk(x, m)
        return self.head(self.ln_f(x))
