@echo off
setlocal
cd /d "%~dp0.."

REM Pulls the midpoint-offset results (4K and 8K, every seed) as a tarball.
REM Use scripts\download-ruler-results.bat for the whole historical tree.

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\export-ruler-results.yaml
if errorlevel 1 exit /b 1
kubectl wait --for=condition=Ready pod/kevin-ruler-export --timeout=5m
if errorlevel 1 exit /b 1

kubectl exec kevin-ruler-export -- sh -c "cd /shared/ruler/results && ls -d *_100_midpoint >/dev/null 2>&1 && tar -czf /tmp/ruler-midpoint.tar.gz *_100_midpoint"
if errorlevel 1 (
    echo ERROR: no *_100_midpoint results found on the PVC yet.
    kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
    exit /b 1
)

kubectl cp kevin-ruler-export:/tmp/ruler-midpoint.tar.gz .\ruler-midpoint.tar.gz
if errorlevel 1 exit /b 1

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
echo.
echo Downloaded .\ruler-midpoint.tar.gz
echo Unpack:  tar -xzf ruler-midpoint.tar.gz
echo Compare: python compare_offsets.py --results "*_100_midpoint\*" --out-dir summary
endlocal
