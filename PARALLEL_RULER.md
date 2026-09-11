# Parallel 4K RULER sweep

This version keeps the existing `kevin-workspace` block PVC for private/single-node work and adds a separate 50 GiB shared volume:

```text
kevin-ruler-shared  (rook-cephfs, ReadWriteMany, 50 GiB)
  /shared/ruler/
    data/4096_100/
      fwe/
      niah_multivalue/
      qa_1/
      qa_2/
    results/4096_100/
      fwe/
      niah_multivalue/
      qa_1/
      qa_2/
```

Only RULER datasets and results are stored on CephFS. Python virtual environments and Hugging Face caches remain in `/tmp` inside each pod, so package installation does not use the shared filesystem.

## Run the four tasks

Commit/push this repo first because the GPU jobs clone it from GitHub. Then from Windows:

```bat
scripts\run-ruler-parallel100.bat
```

The script creates/confirms the 50 GiB RWX PVC, generates 100 4K examples for each task, and launches these four finite GPU jobs in parallel:

```text
santa-ruler-4k-fwe-100
santa-ruler-4k-niah-multivalue-100
santa-ruler-4k-qa1-100
santa-ruler-4k-qa2-100
```

Each job runs the same sweep:

```text
dense
fixed8 fixed16 fixed32 fixed64 fixed128 fixed256
adaptive0.5 adaptive1 adaptive2 adaptive4 adaptive8 adaptive16 adaptive32
```

Check status:

```bat
kubectl get jobs,pods -o wide
```

## Z distribution data

For sampled methods, `summary.csv` now includes exact:

```text
mean_Z, std_Z, min_Z, max_Z, Z_count
```

and reservoir-estimated:

```text
Z_p10, Z_p25, Z_p50, Z_p75, Z_p90, Z_p95, Z_p99
```

Each method also writes `TASK_METHOD_z_samples.npz`, containing up to 200,000 uniformly sampled Z observations. These files let us inspect the shape/tail of the Z distribution before choosing a nonlinear `S(Z)` rule. Dense rows have no Z statistics because dense SDPA intentionally does not materialize the decode score vector.

## Inspect shared storage

```bat
scripts\open-ruler-shared-shell.bat
```

This creates a small CPU-only pod, opens a shell in `/shared/ruler`, and deletes the pod automatically when you exit.

## Download all results

After the jobs finish:

```bat
scripts\download-ruler-results.bat
```

This creates a temporary read-only CPU pod and downloads `ruler-results.tar.gz` to the repo directory.

## RULER auxiliary data note

The RULER repository does not ship the generated `PaulGrahamEssays.json`, `squad.json`, or `hotpotqa.json` files. The preparation Job now runs RULER's own download scripts before generating the four tasks. It also removes any stale task output from a previously failed preparation run and requires exactly 100 rows per task before the GPU jobs are launched.

The message from Transformers saying that PyTorch/TensorFlow/Flax is unavailable during the CPU-only preparation job is expected; that job only needs the Hugging Face tokenizer, not a model runtime.

## Starvation add-on while the main sweep is still running

To probe the low-sample regime without touching the currently running jobs, use:

```bat
scripts\run-ruler-starvation100.bat
```

This launches four additional jobs on the same prepared 4K/100 datasets. They run only `fixed1`, `fixed2`, `fixed4`, and adaptive `alpha={0.5,1,2,4}` with allowed sample counts `1,2,4,8,16,32,64,128,256`. Results go under `/shared/ruler/results/4096_100_starve/`, so they do not overwrite the main sweep.
