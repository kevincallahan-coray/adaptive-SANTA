@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo === Re-running only the FWE starvation job ===
echo Existing NIAH/QA jobs are NOT modified.
echo Requires GPU compute capability major version 7 or newer.
echo.

kubectl delete job santa-ruler-4k-fwe-starve100 --ignore-not-found
if errorlevel 1 exit /b 1

kubectl apply -f k8s\starvation\job-ruler-4k-fwe-starve100.yaml
if errorlevel 1 exit /b 1

echo.
echo Submitted santa-ruler-4k-fwe-starve100.
echo Watch status with:
echo   kubectl get pods -l job-name=santa-ruler-4k-fwe-starve100 -o wide -w
echo.
echo Once running, follow logs with:
echo   kubectl logs -f job/santa-ruler-4k-fwe-starve100
endlocal
