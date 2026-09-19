# Constant 0.5 offset vs randomized systematic, on 8K RULER

Systematic SANTA draws one `u ~ U[0,1)` per decode head/query and lays S
equal-mass thresholds across the CDF at `(j + u) * Z / S`. This experiment
replaces that draw with a constant `u = 0.5`, so the thresholds sit at the
midpoint of every stratum:

```text
random    thr_j = (j + u)   * Z / S,   u ~ U[0,1) per head/query
midpoint  thr_j = (j + 0.5) * Z / S
```

The grid itself is unchanged — draws are still spread evenly over the CDF.
Only the anchoring changes, and that is what makes the comparison worth
running, because systematic sampling is doing two separable things:

- **Spreading draws evenly over the CDF** is what reduces error.
- **Randomizing where the grid lands** is what buys unbiasedness and the
  ability to state a confidence interval.

With `u = 0.5` the method stops being a sampler at all. It is the midpoint
quadrature rule applied to the inverse CDF: no rng is consumed, the index set
is a deterministic function of the attention weights, the variance across
repeats is exactly zero, and the error is entirely bias. The question here is
whether any of that shows up in downstream benchmark accuracy.

## Naming

`fixed` was already taken: in this repo `fixedS` is a fixed sample *budget*,
so `fixed64-fixed` would have been unreadable. The offset suffix is `-mid`:

```text
fixed64        S = 64, random offset
fixed64-mid    S = 64, constant 0.5 offset
adaptive8-mid  adaptive-Z at alpha = 8, constant 0.5 offset
dense-mid      rejected -- dense decode does not sample
```

The suffix sets the sampler's offset and is orthogonal to the budget policy,
so it composes with both `fixedS` and `adaptive<alpha>`. `summary.csv` and the
per-example JSONL both gain an `offset` column (`random` / `midpoint`, empty
for dense), and the full label including the suffix is used for output
filenames, so the two arms never collide.

## Tasks and methods

The same two 100-example 8K tasks as the instrumented run:

- `fwe`
- `niah_multivalue`

Each job runs both arms **in one pod**, interleaved over the budget sweep:

```text
dense
fixed8   fixed8-mid
fixed16  fixed16-mid
fixed32  fixed32-mid
fixed64  fixed64-mid
fixed128 fixed128-mid
fixed256 fixed256-mid
```

One pod matters. Both arms then share a model load, a dataset, an example order
and a GPU, which is what makes the per-example paired comparison valid. Pairing
the midpoint arm against the older `8192_100_instrumented` results instead
would confound the offset with everything else that differs between two jobs.
Interleaving matters for a smaller reason: a job killed part way through still
leaves matched pairs rather than a complete random arm and nothing to compare
it against.

## Launch

```bat
scripts\run-midpoint.bat            8K, seed 1690
scripts\run-midpoint.bat 4k         4K, seed 1690  -- widest node pool
scripts\run-midpoint.bat 8k seed2   8K, seed 991
scripts\run-midpoint.bat 4k seed2   4K, seed 991
```

Each of those four is a plain YAML file under `k8s/instrumented8k/`, so the
launcher only ever runs `kubectl delete` and `kubectl apply`. There is no text
substitution and nothing that depends on a particular shell.

### Getting scheduled

The job asks for **one** GPU and runs both tasks sequentially inside the pod,
rather than two GPUs at once. Each task's results are complete before the next
begins, so a preemption costs at most one task. Beyond that:

- `compute.major > 6` instead of `> 7`, matching the 4K jobs in `k8s/parallel`.
  Turing is back in the pool, and the bfloat16 hazard that motivated `> 7` is
  handled at runtime by `--dtype auto` (see below) instead of by the scheduler.
- Host cpu/memory cut to 1 / 12Gi. The model lives on the GPU and loads with
  `low_cpu_mem_usage`; the old 2 / 24Gi request was excluding usable nodes over
  RAM it never touched.
- A trimmed default sweep: three matched pairs (`fixed8`, `fixed32`, `fixed128`
  x both offsets), no dense. Less than half the wall time, which both fits
  opportunistic slots and reduces exposure to preemption. `dense` is already in
  the instrumented results.

The **20 GB VRAM floor is not relaxed**. Llama-3.1-8B is ~16 GB of weights
before any KV cache, so a smaller card fails after the queue rather than in it.

If 8K will not place, 4K is a real answer rather than a consolation prize: the
offset question is a property of the sampler, not of context length. It reuses
the `4096_100` data already staged and the node pool the parallel 4K jobs
already schedule on. It simply does not additionally tell you about long
context.

### dtype

`ruler_matrix.py --dtype auto` picks bfloat16 where the GPU supports it
natively and float16 otherwise. On Turing an 8K BF16 prefill can fall into a
much more memory-hungry SDPA path and OOM a 24 GiB card; float16 keeps it on
the efficient path. Two consequences, both recorded in the pod log and in every
`summary.csv` row so they cannot be mixed up later:

- Both offset arms share one process and therefore one dtype, so the paired
  comparison this experiment exists for is unaffected.
- Absolute scores from a float16 run are **not** comparable to the bfloat16
  numbers in `8192_100_instrumented`.

The preflight refuses pre-Ampere above 4K rather than letting the OOM surface
an hour in. Override with `SANTA_ALLOW_TURING_LONG=1`.

### Output

```text
/shared/ruler/results/8192_100_midpoint/fwe_seed1690/
/shared/ruler/results/4096_100_midpoint/niah_multivalue_seed991/
```

Nothing under `8192_100_instrumented` or `4096_100` is read or written.
Pull results with `scripts\download-midpoint.bat`.

## Reading the result

Each job runs the comparison on its own task before exiting, so the headline
table is already in the pod log. Across tasks or seeds, run it locally:

```bash
python compare_offsets.py --results '*_100_midpoint/*' --out-dir summary
```

It re-scores the per-example generations with `ruler_metrics` and pairs
`fixedS-mid` against `fixedS` **example by example**. Both arms saw identical
prompts in identical order, so the paired difference removes example difficulty
outright, and its standard error is typically 3–5× tighter than the standard
error on either score alone. Two scores quoted side by side on 100 examples
will "find" gaps that are not there.

Three things have to line up before a gap counts:

- **The paired CI excludes zero.** This covers example-to-example variation.
- **The gap clears `random_seed_spread`.** The midpoint arm is deterministic,
  so it has no seed spread; the random arm does, and that spread is the real
  noise floor for an offset comparison. With one seed the column is empty and
  the script says so rather than implying the question is settled. This is what
  the second seed buys, and it is the difference between a result and an
  anecdote.
- **`unique_rows_*` are comparable.** The two offsets can land on different
  numbers of distinct V rows at the same S, since duplicate compression depends
  on where the thresholds fall. Accuracy at equal S and accuracy at equal
  memory traffic are different claims, and the row-access columns are what
  tells them apart. `row_reduction_vs_dense` in `summary.csv` is the same
  quantity per method.

The comparison also warns if two runs of the midpoint arm disagree. They should
be identical; if they are not, something other than the offset differs between
those runs.

## What a result would mean

**No difference on either task.** The accuracy was coming from spreading draws
over the CDF, not from randomizing the grid. The randomization is then buying
unbiasedness and interval estimates rather than accuracy — a defensible thing
to pay for, but it should be argued on those terms.

**Midpoint loses, especially at small S.** The natural suspect is bias that
does not average away. With a deterministic grid, every decode step in every
layer picks its rows the same way, so a systematic miss compounds across the
generation instead of cancelling. `fwe` and `niah_multivalue` ask different
things of retrieval, so a split between them is informative on its own.

**Midpoint wins.** Check first that it is not winning by quietly collapsing
onto a point estimate. At S well above `N_eff` a deterministic grid can
concentrate on a few high-mass rows; `mean_unique_rows` and
`duplicate_fraction` show that directly, and a win that comes with a collapse
in unique rows is a different claim from a win at matched memory traffic.

## Files

```text
santa_backend.py            OFFSET_MODES, SantaConfig.offset, the branch on the grid anchor
ruler_matrix.py             '-mid' method suffix, --dtype, offset column in JSONL and summary.csv
compare_offsets.py          paired per-example comparison, seed-spread reporting

k8s/instrumented8k/job-ruler-midpoint100-8k.yaml        1 GPU, both tasks, 8K, seed 1690
k8s/instrumented8k/job-ruler-midpoint100-8k-s991.yaml   same, seed 991
k8s/instrumented8k/job-ruler-midpoint100-4k.yaml        same at 4K, widest node pool
k8s/instrumented8k/job-ruler-midpoint100-4k-s991.yaml   same, seed 991

k8s/instrumented8k/job-ruler-8k-fwe-midpoint100.yaml            2-GPU variant, one task per pod
k8s/instrumented8k/job-ruler-8k-niah-multivalue-midpoint100.yaml   (apply directly when the cluster is quiet)

scripts/run-midpoint.bat        launcher -- kubectl apply only
scripts/download-midpoint.bat   pulls every *_100_midpoint result
```
