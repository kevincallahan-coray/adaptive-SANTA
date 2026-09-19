import argparse
import csv
import json
import math
import os
import pathlib
import time
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ruler_metrics import score_records
from santa_backend import CONTROLLER, DIAGNOSTIC_COLUMNS, DIAGNOSTIC_INDEX, register_backend

TASK_TOKENS = {"fwe": 50, "niah_multivalue": 128, "qa_1": 32, "qa_2": 32}
# Appended to a method name to swap systematic sampling's random offset for a
# constant 0.5.  See parse_method and santa_backend.OFFSET_MODES.
OFFSET_SUFFIX = "-mid"
DEFAULT_METHODS = "dense,fixed8,fixed16,fixed32,fixed64,fixed128,fixed256,adaptive4,adaptive8,adaptive16,adaptive32"
# Both offset arms over the same budgets, interleaved so a job killed part way
# through still leaves matched pairs rather than a complete random arm with
# nothing to compare it against.
OFFSET_PAIRED_METHODS = (
    "dense,"
    "fixed8,fixed8-mid,fixed16,fixed16-mid,fixed32,fixed32-mid,"
    "fixed64,fixed64-mid,fixed128,fixed128-mid,fixed256,fixed256-mid"
)
PERCENTILES = (10, 25, 50, 75, 90, 95, 99)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", choices=sorted(TASK_TOKENS), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--methods", default=DEFAULT_METHODS,
                   help="Comma-separated method names. Append '-mid' to run that "
                        "method with a constant 0.5 systematic offset instead of a "
                        "random one, e.g. 'fixed64,fixed64-mid'. The paired offset "
                        f"sweep is: {OFFSET_PAIRED_METHODS}")
    p.add_argument("--candidates", default="8,16,32,64,128,256",
                   help="Comma-separated allowed S values used by fixed/adaptive sampling")
    p.add_argument("--max-examples", type=int, default=10)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=1690)
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16"],
                   help="Model dtype. 'auto' picks bfloat16 where the GPU supports it "
                        "and float16 otherwise, which is what lets this run on Turing. "
                        "Both offset arms share one dtype because they share a process, "
                        "so a paired comparison stays valid either way -- but numbers "
                        "from a float16 run are NOT comparable across runs to bfloat16 ones.")
    p.add_argument("--diagnostic-sample-cap", type=int, default=None,
                   help="Maximum synchronized attention-event rows retained per method")
    p.add_argument("--z-sample-cap", type=int, default=200000,
                   help="Backward-compatible alias used as diagnostic cap when --diagnostic-sample-cap is omitted")
    return p.parse_args()


def parse_method(name):
    """`fixed64` -> randomized systematic; `fixed64-mid` -> constant 0.5 offset.

    The suffix selects the sampler's offset mode and is orthogonal to the
    budget policy, so `adaptive8-mid` is legal too. The full label including
    the suffix is returned, so output files and CSV rows stay distinct between
    the two arms and both can sit in one `summary.csv`.

    The suffix is `-mid` and not `-fixed` because "fixed" is already taken:
    `fixedS` is a fixed sample *budget*, and `fixed64-fixed` would be
    unreadable.
    """
    label = name.strip().lower()
    body = label
    offset = "random"
    if body.endswith(OFFSET_SUFFIX):
        body = body[: -len(OFFSET_SUFFIX)]
        offset = "midpoint"
    if body == "dense":
        if offset != "random":
            raise ValueError("dense decode does not sample, so '-mid' is meaningless on it")
        return label, "dense", None, None, None
    if body.startswith("fixed"):
        s = int(body[len("fixed"):])
        return label, "fixed", None, s, offset
    if body.startswith("adaptive"):
        alpha = float(body[len("adaptive"):])
        return label, "adaptive", alpha, None, offset
    raise ValueError(f"Unknown method: {name}")


def resolve_dtype(name):
    """Pick the model dtype, falling back off bfloat16 on pre-Ampere GPUs.

    Turing (compute 7.x) has no native bfloat16, and on that path an 8K BF16
    prefill can drop into a much more memory-hungry SDPA fallback and OOM a
    24 GiB card. float16 keeps it on the efficient path. The cost is that
    float16 numbers cannot be lined up against bfloat16 numbers from another
    run; within one process both offset arms see the same dtype, so the paired
    comparison this experiment exists for is unaffected.
    """
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_candidates(text):
    values = tuple(sorted({int(x.strip()) for x in text.split(",") if x.strip()}))
    if not values or any(x <= 0 for x in values):
        raise ValueError("--candidates must contain positive integers")
    return values


def make_prompt(rec):
    prompt = rec.get("input") or rec.get("prompt")
    if prompt is None:
        raise ValueError("RULER row needs an 'input' field")
    prefix = rec.get("answer_prefix")
    if prefix and prefix.strip() and prefix.strip() not in prompt[-max(256, len(prefix) + 16):]:
        prompt = prompt + prefix
    return prompt


def tokenize_prompt(tok, prompt, max_input_tokens):
    preformatted = "<|start_header_id|>" in prompt or prompt.startswith("<|begin_of_text|>")
    enc = tok(prompt, return_tensors="pt", add_special_tokens=not preformatted)
    n = int(enc["input_ids"].shape[-1])
    if n > max_input_tokens:
        raise ValueError(f"Prompt has {n} tokens, above --max-input-tokens={max_input_tokens}; refusing to truncate RULER data")
    return {k: v.cuda() for k, v in enc.items()}, n


class DiagnosticDistribution:
    """Exact moments plus one bounded synchronized reservoir of attention events."""

    def __init__(self, cap: int, seed: int):
        self.cap = max(0, int(cap))
        self.rng = np.random.default_rng(seed)
        self.count = 0
        width = len(DIAGNOSTIC_COLUMNS)
        self.sum = np.zeros((width,), dtype=np.float64)
        self.sumsq = np.zeros((width,), dtype=np.float64)
        self.min = np.full((width,), np.inf, dtype=np.float64)
        self.max = np.full((width,), -np.inf, dtype=np.float64)
        self.reservoir = np.empty((0, width), dtype=np.float32)
        self._reservoir_keys = np.empty((0,), dtype=np.float64)

    def update(self, events):
        if isinstance(events, torch.Tensor):
            arr = events.numpy()
        else:
            arr = np.asarray(events, dtype=np.float32)
        if arr.size == 0:
            return
        if arr.ndim != 2 or arr.shape[1] != len(DIAGNOSTIC_COLUMNS):
            raise ValueError(f"Unexpected diagnostics shape: {arr.shape}")
        arr = arr.astype(np.float32, copy=False)
        arr64 = arr.astype(np.float64, copy=False)
        self.count += int(arr.shape[0])
        self.sum += arr64.sum(axis=0)
        self.sumsq += np.square(arr64).sum(axis=0)
        self.min = np.minimum(self.min, arr64.min(axis=0))
        self.max = np.maximum(self.max, arr64.max(axis=0))

        if self.cap <= 0:
            return
        keys = self.rng.random(arr.shape[0])
        combined_values = np.concatenate((self.reservoir, arr), axis=0)
        combined_keys = np.concatenate((self._reservoir_keys, keys))
        if combined_values.shape[0] <= self.cap:
            self.reservoir = combined_values
            self._reservoir_keys = combined_keys
            return
        keep = np.argpartition(combined_keys, -self.cap)[-self.cap:]
        self.reservoir = combined_values[keep]
        self._reservoir_keys = combined_keys[keep]

    def _metric_summary(self, column: str, prefix: str):
        idx = DIAGNOSTIC_INDEX[column]
        if self.count == 0:
            out = {
                f"mean_{prefix}": None,
                f"std_{prefix}": None,
                f"min_{prefix}": None,
                f"max_{prefix}": None,
            }
            out.update({f"{prefix}_p{p}": None for p in PERCENTILES})
            return out
        mean = self.sum[idx] / self.count
        variance = max(0.0, self.sumsq[idx] / self.count - mean * mean)
        out = {
            f"mean_{prefix}": float(mean),
            f"std_{prefix}": math.sqrt(variance),
            f"min_{prefix}": float(self.min[idx]),
            f"max_{prefix}": float(self.max[idx]),
        }
        if self.reservoir.shape[0]:
            q = np.percentile(self.reservoir[:, idx], PERCENTILES)
            out.update({f"{prefix}_p{p}": float(v) for p, v in zip(PERCENTILES, q)})
        else:
            out.update({f"{prefix}_p{p}": None for p in PERCENTILES})
        return out

    def summary(self):
        out = {
            "diagnostic_event_count": self.count,
            "diagnostic_sample_count": int(self.reservoir.shape[0]),
            # Backward-compatible names used by the earlier 4K summaries.
            "Z_count": self.count,
            "Z_sample_count": int(self.reservoir.shape[0]),
        }
        if self.count == 0:
            # Preserve the older Z field names for easy concatenation with 4K CSVs.
            out.update(self._metric_summary("z", "Z"))
            out.update(self._metric_summary("q", "Q"))
            out.update(self._metric_summary("n_eff", "Neff"))
            out.update(self._metric_summary("count_ge_0p5", "C05"))
            out.update(self._metric_summary("count_ge_0p25", "C025"))
            out.update(self._metric_summary("unique_rows", "U"))
            out.update(self._metric_summary("duplicates", "duplicates"))
            return out
        out.update(self._metric_summary("z", "Z"))
        out.update(self._metric_summary("q", "Q"))
        out.update(self._metric_summary("n_eff", "Neff"))
        out.update(self._metric_summary("count_ge_0p5", "C05"))
        out.update(self._metric_summary("count_ge_0p25", "C025"))
        out.update(self._metric_summary("unique_rows", "U"))
        out.update(self._metric_summary("duplicates", "duplicates"))
        return out

    def save(self, path):
        payload = {
            name: self.reservoir[:, DIAGNOSTIC_INDEX[name]].astype(np.float32, copy=False)
            for name in DIAGNOSTIC_COLUMNS
        }
        payload["total_count"] = np.asarray([self.count], dtype=np.int64)
        payload["columns"] = np.asarray(DIAGNOSTIC_COLUMNS)
        np.savez_compressed(path, **payload)

    def save_z_compat(self, path):
        np.savez_compressed(
            path,
            z=self.reservoir[:, DIAGNOSTIC_INDEX["z"]].astype(np.float32, copy=False),
            total_count=np.asarray([self.count], dtype=np.int64),
        )

    def save_layer_head_csv(self, path):
        if self.reservoir.shape[0] == 0:
            return
        layer_idx = DIAGNOSTIC_INDEX["layer"]
        head_idx = DIAGNOSTIC_INDEX["head"]
        layer = self.reservoir[:, layer_idx].astype(np.int32)
        head = self.reservoir[:, head_idx].astype(np.int32)
        pairs = np.unique(np.stack((layer, head), axis=1), axis=0)
        value_cols = [
            "z", "q", "n_eff", "count_ge_0p5", "count_ge_0p25",
            "selected_s", "unique_rows", "duplicates", "context_len",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["layer", "head", "sample_count"] + [f"mean_{c}" for c in value_cols]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for layer_id, head_id in pairs:
                mask = (layer == layer_id) & (head == head_id)
                row = {
                    "layer": int(layer_id),
                    "head": int(head_id),
                    "sample_count": int(mask.sum()),
                }
                for col in value_cols:
                    row[f"mean_{col}"] = float(self.reservoir[mask, DIAGNOSTIC_INDEX[col]].mean())
                w.writerow(row)


def main():
    args = parse_args()
    candidates = parse_candidates(args.candidates)
    diagnostic_cap = args.diagnostic_sample_cap
    if diagnostic_cap is None:
        diagnostic_cap = args.z_sample_cap
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, encoding="utf-8") as f:
        data = [json.loads(line) for _, line in zip(range(args.max_examples), f)]
    if not data:
        raise ValueError("Dataset is empty")

    register_backend()
    dtype = resolve_dtype(args.dtype)
    print(f"model dtype: {dtype} (--dtype {args.dtype}); "
          f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)}",
          flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=dtype,
        attn_implementation="santa_systematic",
        low_cpu_mem_usage=True,
    ).cuda().eval()

    max_new_tokens = args.max_new_tokens or TASK_TOKENS[args.task]
    summaries = []

    for method_index, method_text in enumerate(args.methods.split(",")):
        method, mode, alpha, fixed_s, offset = parse_method(method_text)
        CONTROLLER.configure(
            mode,
            offset=offset,
            alpha=1.0 if alpha is None else alpha,
            fixed_s=128 if fixed_s is None else fixed_s,
            candidates=candidates,
            seed=args.seed,
        )

        rows = []
        total_hist = Counter()
        total_sampled_heads = 0
        total_attention_heads = 0
        total_samples = 0
        total_logical_rows = 0
        total_dense_rows = 0
        diagnostics = DiagnosticDistribution(diagnostic_cap, args.seed + 1000 * method_index)
        method_start = time.time()
        torch.cuda.reset_peak_memory_stats()

        out_path = out_dir / f"{args.task}_{method}.jsonl"
        with out_path.open("w", encoding="utf-8") as out_f:
            for i, rec in enumerate(data):
                prompt = make_prompt(rec)
                enc, n = tokenize_prompt(tok, prompt, args.max_input_tokens)
                CONTROLLER.reset_stats()

                torch.cuda.synchronize()
                t0 = time.time()
                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tok.eos_token_id,
                    )
                torch.cuda.synchronize()
                dt = time.time() - t0

                pred = tok.decode(out[0, n:], skip_special_tokens=True).strip()
                stats = CONTROLLER.summary()
                hist = Counter({int(k): int(v) for k, v in stats.get("S_hist", {}).items()})
                total_hist.update(hist)
                total_sampled_heads += int(stats.get("sampled_head_queries") or 0)
                total_attention_heads += int(stats.get("attention_head_queries") or 0)
                total_samples += int(stats.get("total_samples") or 0)
                total_logical_rows += int(stats.get("logical_row_accesses") or 0)
                total_dense_rows += int(stats.get("dense_equivalent_row_accesses") or 0)
                diagnostics.update(CONTROLLER.consume_diagnostic_events())

                row = {
                    **rec,
                    "index": rec.get("index", i),
                    "pred": pred,
                    "generation": pred,
                    "method": method,
                    "mode": mode,
                    "offset": offset,
                    "alpha": alpha,
                    "fixed_s": fixed_s,
                    "seed": args.seed,
                    "candidates": list(candidates),
                    "prompt_tokens": n,
                    "seconds": dt,
                    "santa_stats": stats,
                }
                rows.append(row)
                out_f.write(json.dumps(row) + "\n")
                print(
                    f"[{method} {i+1}/{len(data)}] tokens={n} sec={dt:.2f} "
                    f"rows={stats.get('logical_row_accesses')} pred={pred[:100]!r}",
                    flush=True,
                )

        score = score_records(rows, args.task)
        mean_s = (sum(s * c for s, c in total_hist.items()) / total_sampled_heads) if total_sampled_heads else None
        mean_unique_rows = (total_logical_rows / total_attention_heads) if total_attention_heads else None
        mean_context_rows = (total_dense_rows / total_attention_heads) if total_attention_heads else None
        row_reduction = (total_dense_rows / total_logical_rows) if total_logical_rows else None
        sample_to_row_ratio = (total_samples / total_logical_rows) if total_samples and total_logical_rows else None
        duplicate_fraction = (1.0 - total_logical_rows / total_samples) if total_samples else None
        diag_summary = diagnostics.summary()

        diagnostics.save(out_dir / f"{args.task}_{method}_diagnostics.npz")
        diagnostics.save_z_compat(out_dir / f"{args.task}_{method}_z_samples.npz")
        diagnostics.save_layer_head_csv(out_dir / f"{args.task}_{method}_layer_head_sample.csv")

        summary = {
            "task": args.task,
            "method": method,
            "mode": mode,
            "offset": offset,
            "dtype": str(dtype).replace("torch.", ""),
            "alpha": alpha,
            "fixed_s": fixed_s,
            "candidates": ",".join(str(x) for x in candidates),
            "candidate_min": min(candidates),
            "candidate_max": max(candidates),
            "examples": len(rows),
            "score": round(score, 4),
            "mean_S": mean_s,
            "attention_head_queries": total_attention_heads,
            "total_samples": total_samples if total_sampled_heads else None,
            "logical_row_accesses": total_logical_rows,
            "dense_equivalent_row_accesses": total_dense_rows,
            "mean_unique_rows": mean_unique_rows,
            "mean_context_rows": mean_context_rows,
            "row_reduction_vs_dense": row_reduction,
            "sample_to_row_ratio": sample_to_row_ratio,
            "duplicate_fraction": duplicate_fraction,
            **diag_summary,
            "S_hist": json.dumps(dict(sorted(total_hist.items()))),
            "wall_seconds": round(time.time() - method_start, 3),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3),
            "seed": args.seed,
        }
        summaries.append(summary)
        print("SUMMARY", summary, flush=True)

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        w.writeheader()
        w.writerows(summaries)
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
