$ErrorActionPreference = "Stop"
kubectl delete pod kevin-workspace-shell --ignore-not-found | Out-Null
kubectl apply -f .\k8s\workspace\workspace-shell.yaml
kubectl wait --for=condition=Ready pod/kevin-workspace-shell --timeout=2m
kubectl exec -it kevin-workspace-shell -- bash
