@echo off
setlocal EnableExtensions
cd /d "%~dp0.."

REM ===================================================================
REM  Midpoint-offset comparison: randomized systematic offset vs a
REM  constant 0.5, on RULER FWE and NIAH-multivalue.
REM
REM  Usage:  scripts\run-midpoint.bat [8k^|4k] [seed2]
REM
REM    scripts\run-midpoint.bat            8K, seed 1690
REM    scripts\run-midpoint.bat 4k         4K, seed 1690   <- widest node pool
REM    scripts\run-midpoint.bat 8k seed2   8K, seed 991
REM    scripts\run-midpoint.bat 4k seed2   4K, seed 991
REM
REM  Each variant is its own plain YAML file under k8s\instrumented8k\,
REM  so this script only ever does kubectl delete + kubectl apply. No
REM  text substitution, nothing to go wrong on a different shell.
REM
REM  One GPU per run: both tasks execute sequentially inside the pod.
REM  The seed reaches only the random arm; the midpoint arm is
REM  deterministic. Run both seeds to get the noise floor.
REM ===================================================================

set "CTX=%~1"
if "%CTX%"=="" set "CTX=8k"
set "SEEDARG=%~2"

set "SUFFIX="
if /I "%SEEDARG%"=="seed2" set "SUFFIX=-s991"

if /I "%CTX%"=="8k" goto :ok
if /I "%CTX%"=="4k" goto :ok
echo ERROR: first argument must be 8k or 4k.
exit /b 1

:ok
set "MANIFEST=k8s\instrumented8k\job-ruler-midpoint100-%CTX%%SUFFIX%.yaml"
set "JOB=santa-ruler-midpoint100-%CTX%%SUFFIX%"

if not exist "%MANIFEST%" (
    echo ERROR: %MANIFEST% not found.
    exit /b 1
)

echo === %JOB% ===
echo manifest: %MANIFEST%
echo.

kubectl get pvc kevin-ruler-shared
if errorlevel 1 (
    echo ERROR: kevin-ruler-shared PVC not found.
    exit /b 1
)

kubectl delete job %JOB% --ignore-not-found >nul 2>&1
kubectl apply -f "%MANIFEST%"
if errorlevel 1 exit /b 1

echo.
echo Submitted. Watch scheduling:
echo   kubectl get pods -l job-name=%JOB% -o wide -w
echo.
echo Follow logs:
echo   kubectl logs -f job/%JOB%
echo.
echo If it sits Pending, the cluster has no free card matching the job.
echo Try the 4K variant, which uses the same node pool as your existing
echo k8s\parallel 4K jobs:
echo   scripts\run-midpoint.bat 4k
echo.
echo When it finishes, pull the results:
echo   scripts\download-midpoint.bat
endlocal
