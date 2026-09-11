@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

echo === Creating shared 50Gi RULER PVC ===
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
echo === Preparing 100 examples for all four 4K RULER tasks ===
kubectl delete job ruler-prepare-4k-100 --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\job-ruler-prepare-4k-100.yaml
if errorlevel 1 exit /b 1

kubectl wait --for=condition=complete job/ruler-prepare-4k-100 --timeout=60m
if errorlevel 1 (
    echo ERROR: RULER data preparation failed. Showing logs:
    kubectl logs job/ruler-prepare-4k-100
    exit /b 1
)
kubectl logs job/ruler-prepare-4k-100

echo.
echo === Launching four GPU jobs in parallel ===
for %%J in (
    santa-ruler-4k-fwe-100
    santa-ruler-4k-niah-multivalue-100
    santa-ruler-4k-qa1-100
    santa-ruler-4k-qa2-100
) do kubectl delete job %%J --ignore-not-found >nul 2>&1

kubectl apply -f k8s\parallel\job-ruler-4k-fwe-100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\parallel\job-ruler-4k-niah-multivalue-100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\parallel\job-ruler-4k-qa1-100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\parallel\job-ruler-4k-qa2-100.yaml
if errorlevel 1 exit /b 1

echo.
echo Submitted. Check status with:
echo   kubectl get jobs,pods -o wide
echo.
echo Follow individual jobs with:
echo   kubectl logs -f job/santa-ruler-4k-fwe-100
echo   kubectl logs -f job/santa-ruler-4k-niah-multivalue-100
echo   kubectl logs -f job/santa-ruler-4k-qa1-100
echo   kubectl logs -f job/santa-ruler-4k-qa2-100
endlocal
