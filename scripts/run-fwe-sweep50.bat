@echo off
setlocal
cd /d "%~dp0.."

kubectl delete job ruler-prepare-4k-fwe-50 --ignore-not-found >nul 2>&1
kubectl apply -f k8s\job-ruler-prepare-4k-fwe-50.yaml
if errorlevel 1 exit /b 1

kubectl wait --for=condition=complete job/ruler-prepare-4k-fwe-50 --timeout=30m
if errorlevel 1 (
    echo RULER preparation failed. Showing logs:
    kubectl logs job/ruler-prepare-4k-fwe-50
    exit /b 1
)

kubectl logs job/ruler-prepare-4k-fwe-50

kubectl delete job santa-ruler-4k-fwe-sweep50 --ignore-not-found >nul 2>&1
kubectl apply -f k8s\job-ruler-4k-fwe-sweep50.yaml
if errorlevel 1 exit /b 1

echo.
echo Sweep submitted. Follow it with:
echo kubectl logs -f job/santa-ruler-4k-fwe-sweep50
endlocal
