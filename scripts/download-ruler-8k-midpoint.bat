@echo off
setlocal
cd /d "%~dp0.."

REM Pulls only the midpoint-offset experiment, not the whole historical
REM result tree. Use scripts\download-ruler-results.bat for everything.

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\export-ruler-results.yaml
if errorlevel 1 exit /b 1
kubectl wait --for=condition=Ready pod/kevin-ruler-export --timeout=5m
if errorlevel 1 exit /b 1

kubectl exec kevin-ruler-export -- sh -c "cd /shared/ruler/results && ls -d *_100_midpoint >/dev/null 2>&1 && tar -czf /tmp/ruler-8k-midpoint.tar.gz *_100_midpoint"
if errorlevel 1 (
    echo ERROR: nothing at /shared/ruler/results/8192_100_midpoint to export.
    kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
    exit /b 1
)

kubectl cp kevin-ruler-export:/tmp/ruler-8k-midpoint.tar.gz .\ruler-8k-midpoint.tar.gz
if errorlevel 1 exit /b 1

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
echo Downloaded .\ruler-8k-midpoint.tar.gz
echo Unpack with:  tar -xzf ruler-8k-midpoint.tar.gz
endlocal
