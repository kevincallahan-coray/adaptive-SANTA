#!/usr/bin/env bash
set -euo pipefail
kubectl auth can-i create jobs
kubectl auth can-i create persistentvolumeclaims
kubectl get resourcequota || true
kubectl get nodes -L nvidia.com/gpu.product | head -n 20
kubectl get nodes -l 'nvidia.com/a40' 2>/dev/null | head || true
