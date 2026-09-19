from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch


# Every sampled head/query contributes one row with these fields.  The runner
# keeps a bounded uniform reservoir, so layer/head analysis stays manageable
# even for long-context RULER runs.
DIAGNOSTIC_COLUMNS = (
    "z",
    "q",
    "n_eff",
    "count_ge_0p5",
    "count_ge_0p25",
    "selected_s",
    "unique_rows",
    "duplicates",
    "layer",
    "head",
    "context_len",
)
DIAGNOSTIC_INDEX = {name: i for i, name in enumerate(DIAGNOSTIC_COLUMNS)}


# Where the systematic grid of S equal-mass thresholds is anchored.
#
#   random   -- u ~ U[0,1) drawn per decode head/query.  Thresholds
#               (j + u) * Z / S.  This is systematic sampling proper:
#               unbiased, with one source of randomness per head/query
#               regardless of S.
#   midpoint -- u held at the constant 0.5.  Thresholds (j + 0.5) * Z / S,
#               i.e. the midpoint of every equal-mass stratum.  No rng is
#               consumed, the index set is a deterministic function of the
#               attention weights, the variance across repeats is exactly
#               zero, and the error is entirely bias.  It is the midpoint
#               quadrature rule on the inverse CDF, not a sampler.
#
# The grid is identical either way -- draws are still spread evenly over the
# CDF.  Only the anchoring changes, which separates the two things systematic
# sampling does: spreading is what reduces error, randomizing the anchor is
# what buys unbiasedness and the ability to state a confidence interval.
OFFSET_MODES = ("random", "midpoint")


@dataclass
class SantaConfig:
    mode: str = "dense"              # dense | fixed | adaptive
    offset: str = "random"           # random | midpoint
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
        self.attention_head_queries = 0
        self.total_samples = 0
        # "logical_row_accesses" counts distinct sampled token indices within
        # each attention head/query after duplicate compression.  It is a
        # hardware-relevant logical V-row metric, not a claim about DRAM/cache
        # transactions for a particular physical layout.
        self.logical_row_accesses = 0
        # Number of logical rows dense decode would read over the same
        # head/query events: sum(context_len) for every head/query.
        self.dense_equivalent_row_accesses = 0
        self._diagnostic_chunks: list[torch.Tensor] = []

    def configure(self, mode: str, *, offset=None, alpha=None, fixed_s=None,
                  candidates: Sequence[int] | None = None, seed: int = 0):
        if mode not in {"dense", "fixed", "adaptive"}:
            raise ValueError(mode)
        self.config.mode = mode
        # Offset is a property of the sampler, not of the budget policy, so it
        # composes with both fixed-S and adaptive-Z.  An omitted offset resets
        # to "random" so a midpoint method cannot leak into the next method of
        # a matrix run.
        offset = "random" if offset is None else str(offset)
        if offset not in OFFSET_MODES:
            raise ValueError(f"offset must be one of {OFFSET_MODES}, got {offset!r}")
        self.config.offset = offset
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

    def record_dense_decode(self, query: torch.Tensor, key: torch.Tensor):
        b, h, q, _ = query.shape
        k = int(key.shape[-2])
        n = int(b * h * q)
        rows = n * k
        self.attention_head_queries += n
        self.logical_row_accesses += rows
        self.dense_equivalent_row_accesses += rows

    def record_sampled_events(self, diagnostics_cpu: torch.Tensor):
        if diagnostics_cpu.ndim != 2 or diagnostics_cpu.shape[1] != len(DIAGNOSTIC_COLUMNS):
            raise ValueError("Unexpected diagnostic tensor shape")
        n = int(diagnostics_cpu.shape[0])
        if n == 0:
            return
        self.attention_head_queries += n
        self.total_samples += int(diagnostics_cpu[:, DIAGNOSTIC_INDEX["selected_s"]].sum().item())
        self.logical_row_accesses += int(diagnostics_cpu[:, DIAGNOSTIC_INDEX["unique_rows"]].sum().item())
        self.dense_equivalent_row_accesses += int(diagnostics_cpu[:, DIAGNOSTIC_INDEX["context_len"]].sum().item())
        self._diagnostic_chunks.append(diagnostics_cpu)

    def consume_diagnostic_events(self) -> torch.Tensor:
        if not self._diagnostic_chunks:
            return torch.empty((0, len(DIAGNOSTIC_COLUMNS)), dtype=torch.float32)
        out = torch.cat(self._diagnostic_chunks, dim=0)
        self._diagnostic_chunks = []
        return out

    def summary(self):
        n = self.attention_head_queries
        logical = self.logical_row_accesses
        dense_rows = self.dense_equivalent_row_accesses
        total_samples = self.total_samples
        return {
            "offset": self.config.offset,
            "sampled_head_queries": self.sampled_head_queries,
            "attention_head_queries": n,
            "mean_S": (total_samples / self.sampled_head_queries) if self.sampled_head_queries else None,
            "S_hist": dict(sorted(self.s_hist.items())),
            "total_samples": total_samples if self.sampled_head_queries else None,
            "logical_row_accesses": logical,
            "dense_equivalent_row_accesses": dense_rows,
            "mean_unique_rows": (logical / n) if n else None,
            "mean_context_rows": (dense_rows / n) if n else None,
            "row_reduction_vs_dense": (dense_rows / logical) if logical else None,
            "sample_to_row_ratio": (total_samples / logical) if total_samples and logical else None,
            "duplicate_fraction": (1.0 - logical / total_samples) if total_samples else None,
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
    """Dense SDPA for prefill; systematic SANTA only for q_len==1 decode.

    The threshold grid is the same in both offset modes; only where it is
    anchored differs (see ``OFFSET_MODES``).  The registered backend name
    stays ``santa_systematic`` because the mechanism is unchanged -- a
    midpoint run is still a systematic grid, just not a randomized one.
    """
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    cfg = CONTROLLER.config
    q_len = query.shape[-2]

    # Prefill is always exact and is intentionally excluded from decode-only
    # sampling/memory statistics.
    if cfg.sample_decode_only and q_len != 1:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            scaling=scaling, dropout=dropout, **kwargs
        )

    if cfg.mode == "dense":
        CONTROLLER.record_dense_decode(query, key)
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
    weights = torch.exp(scores32 - max_score)       # w_i, max(w)=1
    z = weights.sum(dim=-1)                         # [B,H,Q]
    q_stat = (weights * weights).sum(dim=-1)        # Q = sum_i w_i^2
    count_ge_0p5 = (weights >= 0.5).sum(dim=-1)
    count_ge_0p25 = (weights >= 0.25).sum(dim=-1)
    cdf = weights.cumsum(dim=-1)
    selected_s = _select_s(z)

    b, h, q, k = weights.shape
    d = value_states.shape[-1]
    n = b*h*q
    cdf_f = cdf.reshape(n, k).contiguous()
    z_f = z.reshape(n)
    q_f = q_stat.reshape(n)
    c05_f = count_ge_0p5.reshape(n)
    c025_f = count_ge_0p25.reshape(n)
    s_f = selected_s.reshape(n)
    values_f = value_states.unsqueeze(2).expand(b,h,q,k,d).reshape(n,k,d)
    out_f = torch.empty((n,d), device=value_states.device, dtype=value_states.dtype)
    unique_f = torch.empty((n,), device=weights.device, dtype=torch.long)
    gen = CONTROLLER.generator(weights.device)

    for s in cfg.candidates:
        rows = torch.nonzero(s_f == s, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        count = rows.numel()
        if cfg.offset == "midpoint":
            # Deterministic: the generator is not touched, so the seed never
            # reaches the sampler and repeated runs return the same indices.
            u = torch.full((count,1), 0.5, device=weights.device, dtype=torch.float32)
        else:
            u = torch.rand((count,1), device=weights.device, generator=gen, dtype=torch.float32)
        j = torch.arange(s, device=weights.device, dtype=torch.float32).unsqueeze(0)
        thresholds = (j + u) * (z_f[rows].unsqueeze(1) / float(s))
        idx = torch.searchsorted(cdf_f[rows], thresholds, right=True).clamp_max(k-1)
        gather_idx = idx.unsqueeze(-1).expand(-1,-1,d)
        samples = torch.gather(values_f[rows], 1, gather_idx)
        out_f[rows] = samples.mean(dim=1)

        # Thresholds are monotonically increasing, therefore searchsorted indices
        # are nondecreasing and adjacent changes count distinct sampled V rows.
        if s == 1:
            unique = torch.ones((count,), device=weights.device, dtype=torch.long)
        else:
            unique = 1 + (idx[:, 1:] != idx[:, :-1]).sum(dim=1)
        unique_f[rows] = unique

        CONTROLLER.s_hist[int(s)] += int(count)
        CONTROLLER.sampled_head_queries += int(count)

    n_eff_f = (z_f * z_f) / q_f.clamp_min(torch.finfo(torch.float32).tiny)
    head_f = torch.arange(h, device=weights.device, dtype=torch.float32).view(1, h, 1)
    head_f = head_f.expand(b, h, q).reshape(n)
    layer_idx = float(getattr(module, "layer_idx", -1))

    diagnostics = torch.stack(
        [
            z_f,
            q_f,
            n_eff_f,
            c05_f.float(),
            c025_f.float(),
            s_f.float(),
            unique_f.float(),
            (s_f - unique_f).float(),
            torch.full((n,), layer_idx, device=weights.device, dtype=torch.float32),
            head_f,
            torch.full((n,), float(k), device=weights.device, dtype=torch.float32),
        ],
        dim=1,
    ).detach().cpu()
    CONTROLLER.record_sampled_events(diagnostics)

    return out_f.reshape(b,h,q,d).transpose(1,2).contiguous(), None


def register_backend():
    from transformers import AttentionInterface, AttentionMaskInterface
    from transformers.masking_utils import sdpa_mask
    AttentionInterface.register("santa_systematic", santa_attention_forward)
    AttentionMaskInterface.register("santa_systematic", sdpa_mask)
