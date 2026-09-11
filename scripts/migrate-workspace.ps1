$ErrorActionPreference = "Stop"

Write-Host "Creating/confirming PVC kevin-workspace..."
kubectl apply -f .\k8s\pvc.yaml

Write-Host "Waiting for kevin-workspace to become Bound..."
$deadline = (Get-Date).AddMinutes(5)
do {
    $phase = kubectl get pvc kevin-workspace -o jsonpath="{.status.phase}" 2>$null
    if ($phase -eq "Bound") { break }
    if ((Get-Date) -gt $deadline) { throw "Timed out waiting for kevin-workspace PVC to bind." }
    Start-Sleep -Seconds 2
} while ($true)

kubectl get pvc kevin-workspace santa-work

Write-Host "Running read-only migration from santa-work -> kevin-workspace/santa-adaptive-z..."
kubectl delete job santa-migrate-workspace --ignore-not-found | Out-Null
kubectl apply -f .\k8s\workspace\job-migrate-santa-work.yaml
kubectl wait --for=condition=complete job/santa-migrate-workspace --timeout=15m
kubectl logs job/santa-migrate-workspace

$answer = Read-Host "Migration completed. Type DELETE to permanently delete the old santa-work PVC"
if ($answer -eq "DELETE") {
    kubectl delete pvc santa-work
    Write-Host "Old PVC santa-work deleted."
} else {
    Write-Host "Old PVC retained. You can delete it later with: kubectl delete pvc santa-work"
}

Write-Host "Current PVCs:"
kubectl get pvc
