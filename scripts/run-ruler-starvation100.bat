@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

echo === Launching 4K/100 starvation add-on jobs ===
echo Reusing the existing prepared datasets on kevin-ruler-shared.
echo The original RULER jobs are NOT deleted or modified.
echo.

for %%J in (
  santa-ruler-4k-fwe-starve100
  santa-ruler-4k-niah-starve100
  santa-ruler-4k-qa1-starve100
  santa-ruler-4k-qa2-starve100
) do kubectl delete job %%J --ignore-not-found >nul 2>&1

kubectl apply -f k8s\starvation\job-ruler-4k-fwe-starve100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\starvation\job-ruler-4k-niah_multivalue-starve100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\starvation\job-ruler-4k-qa1-starve100.yaml
if errorlevel 1 exit /b 1
kubectl apply -f k8s\starvation\job-ruler-4k-qa2-starve100.yaml
if errorlevel 1 exit /b 1

echo.
echo Submitted four add-on jobs.
echo Methods: fixed1 fixed2 fixed4 adaptive0.5 adaptive1 adaptive2 adaptive4
echo Adaptive candidates: 1,2,4,8,16,32,64,128,256

echo.
echo Check status:
echo   kubectl get jobs,pods -o wide
echo.
echo Example logs:
echo   kubectl logs -f job/santa-ruler-4k-qa1-starve100
endlocal
