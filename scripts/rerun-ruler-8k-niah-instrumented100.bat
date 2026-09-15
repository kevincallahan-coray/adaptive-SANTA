@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo === Re-running only 8K NIAH-multivalue instrumented job ===
kubectl delete job santa-ruler-8k-niah-multivalue-instrumented100 --ignore-not-found
if errorlevel 1 exit /b 1
kubectl apply -f k8s\instrumented8k\job-ruler-8k-niah-multivalue-100.yaml
if errorlevel 1 exit /b 1

echo.
echo Submitted. FWE and all earlier jobs are untouched.
echo Watch with:
echo   kubectl get pods -l job-name=santa-ruler-8k-niah-multivalue-instrumented100 -o wide -w
echo   kubectl logs -f job/santa-ruler-8k-niah-multivalue-instrumented100
endlocal
