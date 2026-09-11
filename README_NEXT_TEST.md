# Next SANTA validation + first RULER test

This package adds two steps:

1. Validate that the custom backend in `dense` mode matches native Hugging Face SDPA.
2. Run a very small 4K RULER FWE comparison: dense vs fixed S=128 vs adaptive alpha=1.

## Files to copy into your Git repo

Copy these to the root of `adaptive-SANTA`:

- `validate_dense.py`
- `ruler_metrics.py`
- `ruler_matrix.py`
- `requirements.txt`

Copy the YAML files into `k8s/`.

Commit and push before launching a Job because the Nautilus Jobs clone your GitHub repo.

## 1. Dense correctness validation

From Windows PowerShell:

```powershell
kubectl delete job santa-validate-dense --ignore-not-found
kubectl apply -f .\k8s\job-validate-dense.yaml
kubectl logs -f job/santa-validate-dense
```

Do not move to benchmark results unless the final line is:

```text
DENSE BACKEND VALIDATION: PASS
```

The check compares final-token logits, next-token argmax, and a short greedy generation for three prompts. It loads native `sdpa` first and then the custom `santa_systematic` backend configured in dense mode.

## 2. Generate the first RULER dataset

This first test intentionally uses the simple NVIDIA/RULER `main` data generator so we can validate the experiment with minimal infrastructure. It creates 10 FWE examples at 4096 tokens on the `santa-work` PVC.

```powershell
kubectl delete job ruler-prepare-4k-fwe --ignore-not-found
kubectl apply -f .\k8s\job-ruler-prepare-4k-fwe.yaml
kubectl logs -f job/ruler-prepare-4k-fwe
```

Expected final output includes:

```text
10 /work/data/ruler/4096/fwe/validation.jsonl
```

## 3. Run the first RULER comparison

```powershell
kubectl delete job santa-ruler-4k-fwe --ignore-not-found
kubectl apply -f .\k8s\job-ruler-4k-fwe.yaml
kubectl logs -f job/santa-ruler-4k-fwe
```

The first run intentionally uses only:

- dense
- fixed S=128
- adaptive alpha=1

on the same 10 examples. The job writes:

```text
/work/results/ruler_4096_fwe/summary.csv
```

and prints that CSV at the end of the log.

## After this passes

Expand `--methods` in `k8s/job-ruler-4k-fwe.yaml` to:

```text
dense,fixed32,fixed64,fixed128,fixed256,adaptive0.5,adaptive1,adaptive2,adaptive4
```

Then increase to 25 examples and add `niah_multivalue`, `qa_1`, and `qa_2`, followed by 8192-token datasets.

The RULER scorer in this package mirrors the benchmark's simple substring metrics for the four tasks we care about: `string_match_all` for FWE/NIAH and `string_match_part` for QA.
