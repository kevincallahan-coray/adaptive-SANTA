# 8K RULER instrumentation run

This run is intended to answer two questions at once:

1. Does the adaptive-Z policy separate from fixed-S on a harder 8K benchmark?
2. Does sample count `S` actually translate into fewer logical V-row accesses after duplicate compression?

The run keeps **exact SDPA prefill** and only applies SANTA when `q_len == 1` during decode.

## Tasks and methods

Two 100-example RULER tasks are prepared at 8192 tokens and run in parallel:

- `fwe`
- `niah_multivalue`

Each GPU job runs:

```text
dense
fixed8 fixed16 fixed32 fixed64 fixed128 fixed256
adaptive4 adaptive8 adaptive16 adaptive32
```

Adaptive candidates remain `8,16,32,64,128,256`, so the minimum sample count is 8.

## Launch

From Windows Command Prompt or PowerShell:

```bat
scripts\run-ruler-8k-instrumented100.bat
```

The launcher only deletes/recreates the **8K** job names below. It does not touch earlier 4K jobs or result directories.

```text
ruler-prepare-8k-100
santa-ruler-8k-fwe-instrumented100
santa-ruler-8k-niah-multivalue-instrumented100
```

Check status:

```bat
kubectl get jobs,pods -o wide
```

Follow logs:

```bat
kubectl logs -f job/santa-ruler-8k-fwe-instrumented100
kubectl logs -f job/santa-ruler-8k-niah-multivalue-instrumented100
```

The jobs require a GPU with more than 20 GB and compute capability major > 6, which excludes the older Tesla/Pascal nodes that failed with `cudaErrorNoKernelImageForDevice`.

## Storage

The existing 50 GiB RWX PVC is reused:

```text
kevin-ruler-shared
```

Datasets:

```text
/shared/ruler/data/8192_100/fwe/validation.jsonl
/shared/ruler/data/8192_100/niah_multivalue/validation.jsonl
```

Results:

```text
/shared/ruler/results/8192_100_instrumented/fwe/
/shared/ruler/results/8192_100_instrumented/niah_multivalue/
```

The existing download helper still archives the entire `/shared/ruler/results` tree:

```bat
scripts\download-ruler-results.bat
```

## New memory-cost statistics

For every decode attention head/query under a sampled method, the backend records the systematic sample indices and counts **distinct indices after within-head duplicate compression**.

`logical_row_accesses` is the sum of those distinct sampled indices. It is a logical V-row metric; it is intentionally not called a DRAM transaction count because physical cache-line behavior depends on the eventual hardware layout.

The summary includes:

- `total_samples`
- `logical_row_accesses`
- `dense_equivalent_row_accesses`
- `mean_unique_rows`
- `mean_context_rows`
- `row_reduction_vs_dense`
- `sample_to_row_ratio`
- `duplicate_fraction`

`dense_equivalent_row_accesses` is the number of logical rows exact dense decode would read over the same head/query events, i.e. the sum of context length over all decode attention head/queries.

This lets the main plots become:

```text
benchmark accuracy vs mean S
benchmark accuracy vs mean_unique_rows
benchmark accuracy vs row_reduction_vs_dense
```

rather than relying on sample count alone.

## New attention-shape statistics

The same sampled attention event records:

```text
Z      = sum_i w_i
Q      = sum_i w_i^2
N_eff  = Z^2 / Q
C05    = count(w_i >= 0.5)
C025   = count(w_i >= 0.25)
```

where `w_i = exp(score_i - max(score))`, so the largest `w_i` is 1.

The two counters are deliberately simple hardware-oriented tail-shape features. `Q/N_eff` is kept as a software reference for whether those counters provide comparable information.

The summary CSV reports exact means/standard deviations/min/max and reservoir-estimated percentiles for these statistics.

## Layer/head logging

Every sampled diagnostic row also carries:

- layer index
- query-head index
- context length
- selected S
- unique row count
- duplicate count

A synchronized uniform reservoir of up to 200,000 complete attention events is saved for each method as:

```text
<TASK>_<METHOD>_diagnostics.npz
```

Arrays in that file are:

```text
z
q
n_eff
count_ge_0p5
count_ge_0p25
selected_s
unique_rows
duplicates
layer
head
context_len
```

Because all fields use the same reservoir decisions, correlations such as `Z` versus row accesses or `C025` versus required sample count are preserved.

For quick inspection, an approximate layer/head aggregate from the same uniform reservoir is also written as:

```text
<TASK>_<METHOD>_layer_head_sample.csv
```

The layer/head data is **diagnostic only** for now. The controller remains global; there is no per-layer policy in this run.

## Instrumentation overhead

`Q`, the two threshold counters, duplicate counting, and the diagnostic CPU copies add work that is not part of the intended final SANTA datapath. Do not interpret `wall_seconds` from this run as a clean implementation speed comparison. The accuracy, sample-count, distribution, and logical-row metrics are the intended outputs.

Dense mode does not materialize attention scores merely for diagnostics, so it has row-access accounting but no `Z/Q/counter` distribution. This preserves exact memory-efficient SDPA behavior for the dense baseline.

To download only this 8K experiment rather than the entire historical result tree:

```bat
scripts\download-ruler-8k-instrumented.bat
```

This creates `ruler-8k-instrumented.tar.gz` in the repository directory.
