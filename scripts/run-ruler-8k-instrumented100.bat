@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

echo === Ensuring shared 50Gi RULER PVC exists ===
kubectl apply -f k8s\pvc-ruler-shared.yaml
if errorlevel 1 exit /b 1

echo Waiting for kevin-ruler-shared to become Bound...
for /L %%N in (1,1,150) do (
    set "phase="
    for /F "delims=" %%P in ('kubectl get pvc kevin-ruler-shared -o jsonpath^="{.status.phase}" 2^>nul') do set "phase=%%P"
    if "!phase!"=="Bound" goto :pvc_ready
    timeout /t 2 /nobreak >nul
)
echo ERROR: Timed out waiting for kevin-ruler-shared PVC to bind.
exit /b 1

:pvc_ready
kubectl get pvc kevin-ruler-shared

echo.
echo === Preparing 100 examples for 8K FWE and NIAH-multivalue ===
kubectl delete job ruler-prepare-8k-100 --ignore-not-found >nul 2>&1
kubectl apply -f k8s\instrumented8k\job-ruler-prepare-8k-100.yaml
if errorlevel 1 exit /b 1

kubectl wait --for=condition=complete job/ruler-prepare-8k-100 --timeout=90m
if errorlevel 1 (
    echo ERROR: 8K RULER data preparation failed. Showing logs:
    kubectl logs job/ruler-prepare-8k-100
    exit /b 1
)
kubectl logs job/ruler-prepare-8k-100

echo.
echo === Launching two instrumented 8K GPU jobs in parallel ===
for %%J in (
    santa-ruler-8k-fwe-instrumented100
    santa-ruler-8k-niah-multivalue-instrumented100
) do kubectl delete job %%J --ignore-not-found >nul 2>&1

kubectl apply -f k8s\instrumented8k\job-ruler-8k-fwe-100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\instrumented8k\job-ruler-8k-niah-multivalue-100.yaml
if errorlevel 1 exit /b 1

echo.
echo Submitted. These jobs do not touch the earlier 4K result directories.
echo Check status with:
echo   kubectl get jobs,pods -o wide
echo.
echo Follow logs with:
echo   kubectl logs -f job/santa-ruler-8k-fwe-instrumented100
echo   kubectl logs -f job/santa-ruler-8k-niah-multivalue-instrumented100
endlocal
