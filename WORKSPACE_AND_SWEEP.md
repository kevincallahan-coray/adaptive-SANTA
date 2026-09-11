# Kevin workspace migration and 50-example FWE sweep

The persistent volume is now named `kevin-workspace`. Project data lives under:

```text
/work/santa-adaptive-z/
  hf/
  data/
  results/
```

Python environments are still created under `/tmp`, not on the PVC. Only model cache, benchmark data, and results are persisted.

## 1. Migrate from the old `santa-work` PVC

From Windows PowerShell, from the repo root:

```powershell
.\scripts\migrate-workspace.ps1
```

The script:
1. creates/updates the `kevin-workspace` PVC;
2. waits for it to bind;
3. runs a CPU-only migration Job that mounts `santa-work` read-only and copies `data`, `results`, and `hf` into `/santa-adaptive-z` on the new PVC;
4. prints the migrated files and sizes;
5. asks you to type `DELETE` before deleting the old PVC.

The explicit confirmation is intentional because deleting a PVC destroys the old persistent storage.

## 2. Inspect the workspace interactively

```powershell
.\scripts\open-workspace-shell.ps1
```

Inside the pod:

```bash
cd /work/santa-adaptive-z
find . -maxdepth 3 -type f | sort
```

Exit with `exit`, then remove the CPU-only shell pod:

```powershell
kubectl delete pod kevin-workspace-shell
```

No GPU is requested by the shell pod.

## 3. Generate 50 FWE examples and launch the full 4K sweep

Commit/push this repo first because the GPU Job clones the GitHub repo. Then:

```powershell
.\scripts\run-fwe-sweep50.ps1
kubectl logs -f job/santa-ruler-4k-fwe-sweep50
```

The sweep compares:

```text
dense
fixed8
fixed16
fixed32
fixed64
fixed128
fixed256
adaptive0.5
adaptive1
adaptive2
adaptive4
adaptive8
adaptive16
adaptive32
```

on the same 50 4096-token FWE examples.

Results are stored at:

```text
/work/santa-adaptive-z/results/ruler_4096_fwe_50/summary.csv
```

The model cache is stored persistently at:

```text
/work/santa-adaptive-z/hf
```

This avoids redownloading Llama for every benchmark Job while keeping pip/venv files off persistent Ceph storage.
