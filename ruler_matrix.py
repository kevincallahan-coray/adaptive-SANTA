import argparse
import csv
import json
import os
import pathlib
import time
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ruler_metrics import score_records
from santa_backend import CONTROLLER, register_backend

TASK_TOKENS = {"fwe": 50, "niah_multivalue": 128, "qa_1": 32, "qa_2": 32}
DEFAULT_METHODS = "dense,fixed8,fixed16,fixed32,fixed64,fixed128,fixed256,adaptive0.5,adaptive1,adaptive2,adaptive4,adaptive8,adaptive16,adaptive32"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--task", choices=sorted(TASK_TOKENS), required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    p.add_argument("--methods", default=DEFAULT_METHODS)
    p.add_argument("--max-examples", type=int, default=10)
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=1690)
    return p.parse_args()


def parse_method(name):
    name = name.strip().lower()
    if name == "dense":
        return name, "dense", None, None
    if name.startswith("fixed"):
        s = int(name[len("fixed"):])
        return name, "fixed", None, s
    if name.startswith("adaptive"):
        alpha = float(name[len("adaptive"):])
        return name, "adaptive", alpha, None
    raise ValueError(f"Unknown method: {name}")


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


def aggregate_stats(total_hist, total_heads, total_z_sum, total_z_count, stats):
    hist = Counter({int(k): int(v) for k, v in stats.get("S_hist", {}).items()})
    total_hist.update(hist)
    n = int(stats.get("sampled_head_queries") or 0)
    total_heads += n
    mean_z = stats.get("mean_Z")
    if mean_z is not None and n:
        total_z_sum += float(mean_z) * n
        total_z_count += n
    return total_heads, total_z_sum, total_z_count


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.dataset, encoding="utf-8") as f:
        data = [json.loads(line) for _, line in zip(range(args.max_examples), f)]
    if not data:
        raise ValueError("Dataset is empty")

    register_backend()
    tok = AutoTokenizer.from_pretrained(args.model, token=os.getenv("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        token=os.getenv("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        attn_implementation="santa_systematic",
        low_cpu_mem_usage=True,
    ).cuda().eval()

    max_new_tokens = args.max_new_tokens or TASK_TOKENS[args.task]
    summaries = []

    for method_text in args.methods.split(","):
        method, mode, alpha, fixed_s = parse_method(method_text)
        CONTROLLER.configure(
            mode,
            alpha=1.0 if alpha is None else alpha,
            fixed_s=128 if fixed_s is None else fixed_s,
            candidates=(8, 16, 32, 64, 128, 256),
            seed=args.seed,
        )

        rows = []
        total_hist = Counter()
        total_heads = 0
        total_z_sum = 0.0
        total_z_count = 0
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
                total_heads, total_z_sum, total_z_count = aggregate_stats(
                    total_hist, total_heads, total_z_sum, total_z_count, stats
                )
                row = {
                    **rec,
                    "index": rec.get("index", i),
                    "pred": pred,
                    "generation": pred,
                    "method": method,
                    "mode": mode,
                    "alpha": alpha,
                    "fixed_s": fixed_s,
                    "seed": args.seed,
                    "prompt_tokens": n,
                    "seconds": dt,
                    "santa_stats": stats,
                }
                rows.append(row)
                out_f.write(json.dumps(row) + "\n")
                print(f"[{method} {i+1}/{len(data)}] tokens={n} sec={dt:.2f} pred={pred[:100]!r}", flush=True)

        score = score_records(rows, args.task)
        mean_s = (sum(s * c for s, c in total_hist.items()) / total_heads) if total_heads else None
        mean_z = (total_z_sum / total_z_count) if total_z_count else None
        summary = {
            "task": args.task,
            "method": method,
            "mode": mode,
            "alpha": alpha,
            "fixed_s": fixed_s,
            "examples": len(rows),
            "score": round(score, 4),
            "mean_S": mean_s,
            "mean_Z": mean_z,
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
