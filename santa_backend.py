from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass
class SantaConfig:
    mode: str = "dense"              # dense | fixed | adaptive
    alpha: float = 1.0
    fixed_s: int = 128
    candidates: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
    seed: int = 0
    sample_decode_only: bool = True


class SantaController:
    def __init__(self):
        self.config = SantaConfig()
        self._generators: dict[str, torch.Generator] = {}
        self.reset_stats()

    def reset_stats(self):
        self.s_hist = Counter()
        self.sampled_head_queries = 0
        self.z_sum = 0.0
        self.z_count = 0

    def configure(self, mode: str, *, alpha=None, fixed_s=None,
                  candidates: Sequence[int] | None = None, seed: int = 0):
        if mode not in {"dense", "fixed", "adaptive"}:
            raise ValueError(mode)
        self.config.mode = mode
        if alpha is not None:
            self.config.alpha = float(alpha)
        if fixed_s is not None:
            self.config.fixed_s = int(fixed_s)
        if candidates is not None:
            self.config.candidates = tuple(sorted({int(x) for x in candidates}))
        self.config.seed = int(seed)
        self._generators = {}
        self.reset_stats()

    def generator(self, device: torch.device):
        key = str(device)
        if key not in self._generators:
            g = torch.Generator(device=device)
            g.manual_seed(self.config.seed)
            self._generators[key] = g
        return self._generators[key]

    def summary(self):
        n = self.sampled_head_queries
        return {
            "sampled_head_queries": n,
            "mean_S": (sum(s*c for s,c in self.s_hist.items()) / n) if n else None,
            "S_hist": dict(sorted(self.s_hist.items())),
            "mean_Z": (self.z_sum / self.z_count) if self.z_count else None,
        }


CONTROLLER = SantaController()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, t, d).reshape(b, h*n_rep, t, d)


def _select_s(z: torch.Tensor) -> torch.Tensor:
    cfg = CONTROLLER.config
    cand = torch.tensor(cfg.candidates, device=z.device, dtype=torch.long)
    if cfg.mode == "fixed":
        if cfg.fixed_s not in cfg.candidates:
            raise ValueError("fixed_s must be in candidates")
        return torch.full(z.shape, cfg.fixed_s, device=z.device, dtype=torch.long)
    target = (z * cfg.alpha).clamp_min(0)
    which = torch.searchsorted(cand.to(target.dtype), target, right=False)
    which = which.clamp_max(len(cfg.candidates)-1)
    return cand[which]


def santa_attention_forward(module, query, key, value, attention_mask, scaling,
                            dropout=0.0, **kwargs):
    """Dense SDPA for prefill; systematic SANTA only for q_len==1 decode."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cfg = CONTROLLER.config
    q_len = query.shape[-2]
    if cfg.mode == "dense" or (cfg.sample_decode_only and q_len != 1):
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs
        )

    key_states = _repeat_kv(key, module.num_key_value_groups)
    value_states = _repeat_kv(value, module.num_key_value_groups)

    # Decode q_len=1, so materializing scores is O(context length), not O(L^2).
    scores = torch.matmul(query, key_states.transpose(2, 3)) * float(scaling)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        else:
            scores = scores + attention_mask.to(scores.dtype)

    scores32 = scores.float()
    max_score = scores32.max(dim=-1, keepdim=True).values
    weights = torch.exp(scores32 - max_score)
    z = weights.sum(dim=-1)                         # [B,H,Q]
    cdf = weights.cumsum(dim=-1)
    selected_s = _select_s(z)

    b, h, q, k = weights.shape
    d = value_states.shape[-1]
    n = b*h*q
    cdf_f = cdf.reshape(n, k).contiguous()
    z_f = z.reshape(n)
    s_f = selected_s.reshape(n)
    values_f = value_states.unsqueeze(2).expand(b,h,q,k,d).reshape(n,k,d)
    out_f = torch.empty((n,d), device=value_states.device, dtype=value_states.dtype)
    gen = CONTROLLER.generator(weights.device)

    for s in cfg.candidates:
        rows = torch.nonzero(s_f == s, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        count = rows.numel()
        u = torch.rand((count,1), device=weights.device, generator=gen, dtype=torch.float32)
        j = torch.arange(s, device=weights.device, dtype=torch.float32).unsqueeze(0)
        thresholds = (j + u) * (z_f[rows].unsqueeze(1) / float(s))
        idx = torch.searchsorted(cdf_f[rows], thresholds, right=True).clamp_max(k-1)
        gather_idx = idx.unsqueeze(-1).expand(-1,-1,d)
        samples = torch.gather(values_f[rows], 1, gather_idx)
        out_f[rows] = samples.mean(dim=1)
        CONTROLLER.s_hist[int(s)] += int(count)
        CONTROLLER.sampled_head_queries += int(count)

    CONTROLLER.z_sum += float(z.sum().item())
    CONTROLLER.z_count += int(z.numel())
    return out_f.reshape(b,h,q,d).transpose(1,2).contiguous(), None


def register_backend():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register("santa_systematic", santa_attention_forward)
    AttentionMaskInterface.register("santa_systematic", sdpa_mask)
