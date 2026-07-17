"""
Attention Head Reweighting (AHR) — minimal reimplementation.

Paper: arXiv:2607.13425, Eq. 3:
    z_att^(l+1) = z_r^l + sum_{h in H_(l+1)} (1 + gamma_h) * h(z_r^l)

One learnable scalar per attention head, zero-initialized, everything else frozen.

The official repo (tuomaso/attention-head-reweighting) contains only a README, so this
is written from the paper's equations. See ../hyperparams.md for provenance.

Implementation point: each attention block sums its heads into the residual stream via the
output projection W_O (`c_proj` in GPT-2, `o_proj` in Qwen2). Because that projection is
linear and its input is the concatenation of per-head outputs, scaling head h's slice of
the *input* by (1 + gamma_h) is exactly equivalent to scaling head h's block of W_O --
which is exactly equivalent to scaling head h's additive contribution to the residual
stream. So a forward_pre_hook on the output projection implements Eq. 3 exactly.
"""

import torch
import torch.nn as nn


def _iter_attn_out_projections(model):
    """Yield (layer_idx, out_projection_module) for every attention block, in order.

    Works for both GPT-2 (Conv1D `c_proj`) and Qwen2/Llama-style (nn.Linear `o_proj`).
    """
    found = []
    for name, module in model.named_modules():
        if not module.__class__.__name__.endswith("Attention"):
            continue
        proj = getattr(module, "o_proj", None) or getattr(module, "c_proj", None)
        if proj is None:
            raise RuntimeError(f"attention module {name} has neither o_proj nor c_proj")
        found.append((name, proj))
    if not found:
        raise RuntimeError("no attention modules found")
    return found


def get_head_geometry(model):
    """Return (n_layers, n_heads_per_layer, head_dim).

    n_heads is the number of *query* heads: the output projection's input is
    num_attention_heads * head_dim wide, one slice per query head. Under GQA the KV heads
    are fewer, but the residual-stream contributions -- the thing Eq. 3 reweights -- are
    per query head.
    """
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
    n_layers = len(_iter_attn_out_projections(model))
    return n_layers, n_heads, head_dim


class AHR:
    """Attaches one gamma scalar per head to a frozen model via forward pre-hooks."""

    def __init__(self, model):
        self.model = model
        self.n_layers, self.n_heads, self.head_dim = get_head_geometry(model)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        # gamma: (n_layers, n_heads), zero-init => model at init IS the frozen model.
        self.gamma = nn.Parameter(
            torch.zeros(self.n_layers, self.n_heads, device=device, dtype=dtype)
        )
        self._handles = []
        self._attach()

    @property
    def n_params(self):
        return self.gamma.numel()

    def _make_hook(self, layer_idx):
        def hook(module, args):
            # args[0]: (batch, seq, n_heads * head_dim) -- concatenated per-head outputs
            x = args[0]
            b, s, d = x.shape
            assert d == self.n_heads * self.head_dim, (
                f"layer {layer_idx}: proj input {d} != n_heads*head_dim "
                f"{self.n_heads}*{self.head_dim}"
            )
            scale = 1.0 + self.gamma[layer_idx]           # (n_heads,)
            x = x.view(b, s, self.n_heads, self.head_dim)
            x = x * scale.view(1, 1, self.n_heads, 1)     # Eq. 3
            return (x.view(b, s, d),) + args[1:]
        return hook

    def _attach(self):
        for i, (_, proj) in enumerate(_iter_attn_out_projections(self.model)):
            self._handles.append(
                proj.register_forward_pre_hook(self._make_hook(i), with_kwargs=False)
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def penalty(self, lam, p=2):
        """Regularizer on gamma. Paper Eq. 4 writes  lambda * (sum_h |gamma_h|^p)^(1/p).

        DEVIATION (deliberate, documented in ../hyperparams.md and ../LAB-NOTES.md):
        Eq. 4 as literally written is the Lp *norm*, and the Lp norm is not differentiable at
        the origin -- while the paper also specifies gamma is ZERO-initialized. For p=2,
        d/dgamma sqrt(sum gamma^2) -> inf as gamma -> 0, so the very first backward pass from
        the paper's own stated initialization produces a NaN gradient. We hit exactly this:
        step 0 CE was finite, every subsequent step was NaN. See LAB-NOTES 2026-07-17.

        So we use the standard convention the paper almost certainly means:
          p=2 -> squared L2, lambda * sum(gamma^2)   (smooth everywhere, = weight decay)
          p=1 -> L1, lambda * sum|gamma|             (subgradient 0 at origin; fine)
        Both are still a pull toward gamma=0, i.e. toward the unmodified pretrained model,
        which is the property the paper's text relies on. With lambda=0.01 the numerical
        difference is negligible; the difference between "runs" and "NaN" is not.
        """
        if lam == 0:
            return torch.zeros((), device=self.gamma.device, dtype=self.gamma.dtype)
        if p == 1:
            return lam * self.gamma.abs().sum()
        return lam * self.gamma.pow(2).sum()
