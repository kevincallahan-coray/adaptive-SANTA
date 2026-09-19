"""Randomized systematic offset vs a constant 0.5, scored on RULER accuracy.

`ruler_matrix.py` writes one JSONL per (task, method) holding the model's
actual generation for every example. This script re-scores those rows with
`ruler_metrics` and pairs `fixedS-mid` against `fixedS` **example by example**.

Why paired
----------
A 100-example RULER task gives a score whose standard error is several points.
Quoting two scores side by side and eyeballing the gap will "find" differences
that are not there. Both arms ran in the same pod over the same prompts in the
same order, so the per-example difference removes example difficulty outright
and its standard error is typically 3-5x tighter than the error on either
score alone.

Why the seed spread matters
---------------------------
The midpoint arm is deterministic -- it consumes no rng, so re-running it at a
different `--seed` returns the identical index set and the identical score. The
random arm does not have that property, and its seed-to-seed spread is the real
noise floor for an offset comparison. Pass results directories from runs that
differ only in seed and that spread is reported next to the delta. With one
seed it cannot be, and this script says so rather than implying the question is
settled.

Usage
-----
    python compare_offsets.py --results 'results/8192_100_midpoint/*' --out-dir summary

    python compare_offsets.py \
        --results results/8192_100_midpoint/fwe_seed1690 \
        --results results/8192_100_midpoint/fwe_seed991 \
        --out-dir summary
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import pathlib
from collections import defaultdict

from ruler_metrics import metric_for_task, score_prediction

# Longest first: task names contain underscores, so a prefix match against the
# known list is safer than splitting the filename on the separator.
TASKS = ("niah_multivalue", "qa_1", "qa_2", "fwe")
OFFSET_SUFFIX = "-mid"


def split_filename(stem: str):
    """`fwe_fixed64-mid` -> ('fwe', 'fixed64-mid')."""
    for task in TASKS:
        if stem.startswith(task + "_"):
            return task, stem[len(task) + 1:]
    return None, None


def arm(method: str):
    """`fixed64-mid` -> base `fixed64`, offset `midpoint`."""
    if method.endswith(OFFSET_SUFFIX):
        return method[: -len(OFFSET_SUFFIX)], "midpoint"
    return method, "random"


def load_run(path: pathlib.Path):
    """Score one (task, method) JSONL. Returns None for dense or an empty file."""
    task, method = split_filename(path.stem)
    if task is None or method == "dense":
        return None
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return None

    metric = metric_for_task(task)
    per_example, samples, logical_rows, head_queries = {}, 0, 0, 0
    for i, r in enumerate(rows):
        key = r.get("index", i)
        pred = r.get("pred", r.get("generation", ""))
        per_example[key] = 100.0 * score_prediction(pred, r.get("outputs", []), metric)
        st = r.get("santa_stats") or {}
        samples += int(st.get("total_samples") or 0)
        logical_rows += int(st.get("logical_row_accesses") or 0)
        head_queries += int(st.get("attention_head_queries") or 0)

    base, offset = arm(method)
    return {
        "task": task,
        "method": method,
        "base": base,
        # Trust the recorded offset over the filename when the run wrote one.
        "offset": (rows[0].get("offset") or offset),
        "seed": rows[0].get("seed"),
        "score": sum(per_example.values()) / len(per_example),
        "per_example": per_example,
        "mean_unique_rows": (logical_rows / head_queries) if head_queries else float("nan"),
        "source": str(path),
    }


def paired_delta(mid: dict, rand: dict):
    """Mean of (midpoint - random) over examples both arms actually ran."""
    shared = sorted(set(mid["per_example"]) & set(rand["per_example"]))
    d = [mid["per_example"][k] - rand["per_example"][k] for k in shared]
    n = len(d)
    if n == 0:
        return None
    mean = sum(d) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = float("nan")
    return {
        "n_paired": n,
        "delta": mean,
        "se": se,
        "ci_lo": mean - 1.96 * se,
        "ci_hi": mean + 1.96 * se,
        "wins": sum(1 for x in d if x > 0),
        "ties": sum(1 for x in d if x == 0),
        "losses": sum(1 for x in d if x < 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", action="append", required=True,
                   help="Results directory written by ruler_matrix.py. Repeatable, "
                        "and globs are expanded. Point it at several runs that "
                        "differ only in seed to get the random arm's spread.")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()

    dirs = []
    for pattern in args.results:
        hits = [pathlib.Path(x) for x in sorted(glob.glob(pattern))]
        dirs += [d for d in (hits or [pathlib.Path(pattern)]) if d.is_dir()]
    if not dirs:
        raise SystemExit(f"no results directories matched {args.results}")

    runs = []
    for d in dirs:
        for path in sorted(d.glob("*.jsonl")):
            run = load_run(path)
            if run is not None:
                runs.append(run)
    if not runs:
        raise SystemExit("no scorable per-example JSONL found; expected <task>_<method>.jsonl")
    print(f"loaded {len(runs)} (task, method, seed) runs from {len(dirs)} directories")

    by_arm = defaultdict(list)
    for r in runs:
        by_arm[(r["task"], r["base"], r["offset"])].append(r)

    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fields = ["task", "base_method", "n_paired", "score_random", "seeds_random",
              "random_seed_spread", "score_midpoint", "delta", "se", "ci_lo", "ci_hi",
              "wins", "ties", "losses", "unique_rows_random", "unique_rows_midpoint"]
    table = []

    for task, base in sorted({(r["task"], r["base"]) for r in runs}):
        mids = by_arm.get((task, base, "midpoint"), [])
        rands = by_arm.get((task, base, "random"), [])
        if not mids or not rands:
            print(f"  skipping {task}/{base}: no {'midpoint' if not mids else 'random'} arm to pair against")
            continue

        # The midpoint arm is deterministic, so repeats of it should agree
        # exactly. Say so if they do not -- that means something other than the
        # offset differs between those runs.
        if len({round(m["score"], 6) for m in mids}) > 1:
            print(f"  WARNING {task}/{base}: midpoint arm is not reproducible across runs "
                  f"({[round(m['score'], 2) for m in mids]}); the offset is not the only "
                  "thing that differs between them")
        mid = mids[0]

        rand_scores = [r["score"] for r in rands]
        spread = (max(rand_scores) - min(rand_scores)) if len(rand_scores) > 1 else float("nan")
        # Pair against the median seed, so the headline is not the luckiest one.
        rand = sorted(rands, key=lambda r: r["score"])[len(rands) // 2]

        d = paired_delta(mid, rand)
        if d is None:
            print(f"  skipping {task}/{base}: the two arms share no example indices")
            continue

        finite_se = d["se"] == d["se"]
        table.append({
            "task": task,
            "base_method": base,
            "n_paired": d["n_paired"],
            "score_random": round(sum(rand_scores) / len(rand_scores), 3),
            "seeds_random": len(rands),
            "random_seed_spread": round(spread, 3) if spread == spread else "",
            "score_midpoint": round(mid["score"], 3),
            "delta": round(d["delta"], 3),
            "se": round(d["se"], 3) if finite_se else "",
            "ci_lo": round(d["ci_lo"], 3) if finite_se else "",
            "ci_hi": round(d["ci_hi"], 3) if finite_se else "",
            "wins": d["wins"],
            "ties": d["ties"],
            "losses": d["losses"],
            "unique_rows_random": round(rand["mean_unique_rows"], 2),
            "unique_rows_midpoint": round(mid["mean_unique_rows"], 2),
        })

    if not table:
        raise SystemExit("nothing paired up; check that both arms ran in the same directories")

    csv_path = out / "offset_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(table)

    print("\npaired midpoint - random, in RULER score points "
          "(positive means the constant 0.5 offset scored higher)\n")
    head = (f"{'task':>16} {'method':>10} {'random':>8} {'midpoint':>9} "
            f"{'delta':>8} {'95% CI':>18} {'W/T/L':>12} {'rows rand':>10} {'rows mid':>9}")
    print(head)
    print("-" * len(head))
    for row in table:
        ci = f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]" if row["ci_lo"] != "" else "n/a"
        wtl = f"{row['wins']}/{row['ties']}/{row['losses']}"
        print(f"{row['task']:>16} {row['base_method']:>10} {row['score_random']:>8.2f} "
              f"{row['score_midpoint']:>9.2f} {row['delta']:>+8.2f} {ci:>18} {wtl:>12} "
              f"{row['unique_rows_random']:>10.1f} {row['unique_rows_midpoint']:>9.1f}")

    seeds = max(row["seeds_random"] for row in table)
    print()
    if seeds < 2:
        print("Only one seed of the random arm is present, so there is no measured noise\n"
              "floor to read these deltas against. The paired CI covers example-to-example\n"
              "variation, not offset-draw variation. Re-run the same jobs with a different\n"
              "seed and pass both directories here before calling any of these gaps real.")
    else:
        print(f"Random arm spans {seeds} seeds. A delta should clear both its own CI and the\n"
              "random_seed_spread column before it counts as a difference.")
    print("\nRead the two row columns too: the offsets can land on different numbers of\n"
          "distinct V rows at the same S, and accuracy at equal S is not the same claim\n"
          "as accuracy at equal memory traffic.")
    print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
