$ErrorActionPreference = "Stop"

kubectl delete job ruler-prepare-4k-fwe-50 --ignore-not-found | Out-Null
kubectl apply -f .\k8s\job-ruler-prepare-4k-fwe-50.yaml
kubectl wait --for=condition=complete job/ruler-prepare-4k-fwe-50 --timeout=30m
kubectl logs job/ruler-prepare-4k-fwe-50

kubectl delete job santa-ruler-4k-fwe-sweep50 --ignore-not-found | Out-Null
kubectl apply -f .\k8s\job-ruler-4k-fwe-sweep50.yaml
Write-Host "Sweep submitted. Follow it with:"
Write-Host "kubectl logs -f job/santa-ruler-4k-fwe-sweep50"
