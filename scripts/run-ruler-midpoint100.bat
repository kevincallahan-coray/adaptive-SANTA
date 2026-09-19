@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

REM ===================================================================
REM  Midpoint-offset comparison, single GPU, both tasks, widened pool.
REM
REM  Usage:  scripts\run-ruler-midpoint100.bat [8k^|4k] [seed]
REM
REM  8k  keeps the original context; needs an Ampere+ card in practice.
REM  4k  runs on the same pool the k8s\parallel 4K jobs already use, which
REM      is considerably larger, and reuses the 4096_100 data you already
REM      have staged. The offset question is not context-specific, so 4k
REM      is a real answer, not a consolation prize -- it just does not
REM      also tell you about long context.
REM
REM  This asks for ONE GPU and runs fwe then niah inside it. Each task's
REM  results are complete before the next begins, so preemption costs at
REM  most one task.
REM ===================================================================

set "CTX=%~1"
if "%CTX%"=="" set "CTX=8k"
set "SEED=%~2"
if "%SEED%"=="" set "SEED=1690"

if /I "%CTX%"=="8k" ( set "TOKENS=8192" ) else ( if /I "%CTX%"=="4k" ( set "TOKENS=4096" ) else (
    echo ERROR: first argument must be 8k or 4k.
    exit /b 1
))
set "JOB=santa-ruler-midpoint100-%CTX%-s%SEED%"

echo === Midpoint offset comparison: ctx=%TOKENS% seed=%SEED% job=%JOB% ===

kubectl get pvc kevin-ruler-shared
if errorlevel 1 exit /b 1

kubectl delete job %JOB% --ignore-not-found >nul 2>&1

set "DST=%TEMP%\%JOB%.yaml"
powershell -NoProfile -Command "(Get-Content -Raw 'k8s\instrumented8k\job-ruler-midpoint100.yaml').Replace('name: santa-ruler-midpoint100', 'name: %JOB%').Replace('value: \"8192\"', 'value: \"%TOKENS%\"').Replace('value: \"1690\"', 'value: \"%SEED%\"') | Set-Content -NoNewline '%DST%'"
if errorlevel 1 exit /b 1

kubectl apply -f "%DST%"
if errorlevel 1 exit /b 1

echo.
echo Submitted %JOB%. Watch scheduling with:
echo   kubectl get pods -l job-name=%JOB% -o wide -w
echo.
echo Follow logs with:
echo   kubectl logs -f job/%JOB%
echo.
echo Results land under /shared/ruler/results/%TOKENS%_100_midpoint/
echo Pull them with: scripts\download-ruler-8k-midpoint.bat
endlocal
