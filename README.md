# Adaptive SANTA on NRP Nautilus

This is a starter project for closed-loop dense/fixed-S/adaptive-S experiments with
Meta-Llama-3.1-8B-Instruct on one Nautilus GPU.

## Important design choice

The custom backend delegates **all prompt prefill** to Hugging Face SDPA.  It only
materializes an explicit attention-score vector when q_len == 1 during decode.
This matters at 8k-32k context: eager dense prefill would materialize O(L^2)
attention matrices and is not the intended experiment.

## 0. Prerequisites

- Working `kubectl`/kubelogin access to Nautilus.
- A Nautilus namespace selected in your kube context.
- Hugging Face access to `meta-llama/Meta-Llama-3.1-8B-Instruct`.
- Put this directory in a Git repository reachable by Nautilus, then replace
  `REPLACE_WITH_YOUR_REPO_URL` in the job YAMLs.

## 1. Select your namespace

```bash
kubectl config set-context nautilus --namespace=YOUR_NAMESPACE
kubectl get pods
```

## 2. Store the Hugging Face token as a Kubernetes secret

Avoid putting the token in YAML or Git:

```bash
read -s HF_TOKEN
kubectl create secret generic hf-token --from-literal=token="$HF_TOKEN"
unset HF_TOKEN
```

## 3. Create persistent cache/results storage

```bash
kubectl apply -f k8s/pvc.yaml
kubectl get pvc kevin-workspace
```

Wait for `Bound`. This uses 100 GiB `rook-ceph-block` / ReadWriteOnce storage.
It is ideal for a single job at a time and caches the model at `/work/santa-adaptive-z/hf`.

## 4. Smoke test

Edit `k8s/job-smoke-a40.yaml` and replace the repo URL, then:

```bash
kubectl apply -f k8s/job-smoke-a40.yaml
kubectl get jobs,pods
kubectl logs -f job/santa-smoke-a40
```

The job runs dense, fixed S=128 and adaptive alpha=1 on three questions.

To rerun the same named job:

```bash
kubectl delete job santa-smoke-a40
kubectl apply -f k8s/job-smoke-a40.yaml
```

## 5. RULER data

The benchmark runner accepts JSONL rows with at least `input` (or `prompt`).
The NVIDIA RULER format and the SANTA tutorial format both use JSONL and can be
adapted easily. Put generated/evaluation data under `/work/santa-adaptive-z/data/ruler`.

If you have the exact JSONLs used in the SANTA paper, prefer those. Otherwise,
generate RULER v1 data and begin with the four paper-relevant tasks:

- fwe
- niah_multivalue
- qa_1
- qa_2

Start with 4k/8k and about 25 examples per task. Do not jump directly to the
full 32k evaluation until fixed-S behavior has been validated.

## 6. Run an 8k job

Place (for example) `/work/santa-adaptive-z/data/ruler/qa_1_8192.jsonl` on the PVC, edit the repo
URL, and run:

```bash
kubectl apply -f k8s/job-ruler-8k-a40.yaml
kubectl logs -f job/santa-ruler-8k-adaptive-a1
```

Results are written to `/work/santa-adaptive-z/results` on the PVC.

For the initial matrix, run these **serially** with the block PVC:

- dense
- fixed S=32
- fixed S=64
- fixed S=128
- adaptive alpha=0.5
- adaptive alpha=1
- adaptive alpha=2
- adaptive alpha=4

Then compare benchmark accuracy versus realized mean S.

## 7. GPU choice

The supplied YAML requests one `nvidia.com/a40` (48 GiB). This is a comfortable
starting point for Llama-3.1-8B BF16 at 4k/8k with SDPA prefill.

If A40s are unavailable, inspect cluster GPU labels and choose another 48 GiB+
GPU. A100/H100/H200/GH200 are quota-gated; opportunistic priority can access
spare cycles but may be preempted.

For 16k/32k, an 80 GiB A100 is preferable if you have quota. A 48 GiB GPU may
still fit batch-1 with memory-efficient prefill, but validate memory first.

## 8. Parallel jobs

`rook-ceph-block` is ReadWriteOnce, so use it for one GPU job at a time. If you
later want multiple jobs on different nodes sharing the same dataset/results,
use a `rook-cephfs` ReadWriteMany volume for large immutable datasets/results.
Do not install pip/conda environments on CephFS; NRP explicitly prohibits that.
Keep Python packages in the container image or local/block storage.

## 9. Useful commands

```bash
kubectl get jobs,pods -o wide
kubectl describe pod POD_NAME
kubectl logs -f job/JOB_NAME
kubectl get pvc
kubectl delete job JOB_NAME
```

If a job is Pending, inspect Events at the bottom of `kubectl describe pod`.
Common causes are GPU quota/type availability or storage/node locality.

## Kevin workspace layout

The current NRP setup uses the `kevin-workspace` RBD PVC and keeps this experiment under `/work/santa-adaptive-z`. See `WORKSPACE_AND_SWEEP.md` for migration, interactive inspection, and the 50-example FWE sweep.

## Parallel RULER experiment

For the current 4-task, 100-example parallel RULER sweep with the 50 GiB shared CephFS volume, see `PARALLEL_RULER.md` and run:

```bat
scripts\run-ruler-parallel100.bat
```

### RULER data prerequisites

The parallel RULER preparation job automatically downloads the Paul Graham essay corpus and the SQuAD/HotpotQA source files required by `niah_multivalue`, `qa_1`, and `qa_2`. Failed/stale generated task directories are cleared before regeneration, and each task must contain exactly 100 rows before the GPU jobs launch.

## 8K instrumented RULER run

The next long-context experiment is documented in `INSTRUMENTED_8K.md`.
It runs 100-example 8K FWE and NIAH-multivalue jobs in parallel with a minimum
S of 8 and adds logical V-row accesses, duplicate compression, `Q`, effective
support, 0.5/0.25 threshold counters, and layer/head diagnostic logging.

Launch from Windows with:

```bat
scripts\run-ruler-8k-instrumented100.bat
```

Results are isolated under `/shared/ruler/results/8192_100_instrumented/` so
existing 4K results are not overwritten.
