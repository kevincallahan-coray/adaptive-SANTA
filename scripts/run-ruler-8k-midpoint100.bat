@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0.."

REM ===================================================================
REM  Randomized systematic offset vs a constant 0.5 ("midpoint"),
REM  on 8K RULER FWE and NIAH-multivalue.
REM
REM  Usage:  scripts\run-ruler-8k-midpoint100.bat [seed]
REM
REM  Both offset arms run inside each GPU job, so this neither reads nor
REM  overwrites /shared/ruler/results/8192_100_instrumented.
REM
REM  The seed reaches only the random arm -- the midpoint arm consumes no
REM  rng and returns the identical index set at every seed. Run this twice
REM  with different seeds to measure the noise floor a claimed gap must
REM  clear.  See MIDPOINT_OFFSET.md.
REM ===================================================================

set "SEED=%~1"
if "%SEED%"=="" set "SEED=1690"
set "FWEJOB=santa-ruler-8k-fwe-midpoint100"
set "NIAHJOB=santa-ruler-8k-niah-multivalue-midpoint100"
if not "%SEED%"=="1690" set "FWEJOB=%FWEJOB%-s%SEED%"
if not "%SEED%"=="1690" set "NIAHJOB=%NIAHJOB%-s%SEED%"

echo === Offset comparison at seed %SEED% ===
echo     fwe  job: %FWEJOB%
echo     niah job: %NIAHJOB%

REM ---------------------------------------------------------------- PVC
echo.
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

REM ------------------------------------------------- 8K dataset present?
REM Reuse the short-lived export pod rather than blindly re-running the
REM 90-minute prepare job. The prepare job rm -rf's and regenerates its
REM task directories, so running it when the data is already good is not
REM merely slow, it throws away a working dataset.
echo.
echo === Checking whether the 8K RULER data is already staged ===
kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\export-ruler-results.yaml >nul
if errorlevel 1 exit /b 1
kubectl wait --for=condition=Ready pod/kevin-ruler-export --timeout=5m
if errorlevel 1 (
    echo ERROR: could not start the check pod.
    kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
    exit /b 1
)
set "DATA_OK=1"
kubectl exec kevin-ruler-export -- sh -c "test -s /shared/ruler/data/8192_100/fwe/validation.jsonl && test -s /shared/ruler/data/8192_100/niah_multivalue/validation.jsonl"
if errorlevel 1 set "DATA_OK=0"
kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1

if "%DATA_OK%"=="1" (
    echo 8K FWE and NIAH-multivalue validation.jsonl are both present. Skipping preparation.
) else (
    echo 8K data missing. Running preparation ^(this takes up to ~90 minutes^)...
    kubectl delete job ruler-prepare-8k-100 --ignore-not-found >nul 2>&1
    kubectl apply -f k8s\instrumented8k\job-ruler-prepare-8k-100.yaml
    if errorlevel 1 exit /b 1
    kubectl wait --for=condition=complete job/ruler-prepare-8k-100 --timeout=90m
    if errorlevel 1 (
        echo ERROR: 8K RULER data preparation failed. Logs:
        kubectl logs job/ruler-prepare-8k-100
        exit /b 1
    )
    kubectl logs job/ruler-prepare-8k-100
)

REM ------------------------------------------------------- launch the jobs
echo.
echo === Launching both GPU jobs in parallel ===
kubectl delete job %FWEJOB% %NIAHJOB% --ignore-not-found >nul 2>&1

if "%SEED%"=="1690" (
    kubectl apply -f k8s\instrumented8k\job-ruler-8k-fwe-midpoint100.yaml
    if errorlevel 1 exit /b 1
    kubectl apply -f k8s\instrumented8k\job-ruler-8k-niah-multivalue-midpoint100.yaml
    if errorlevel 1 exit /b 1
) else (
    REM Non-default seed: rename the job and rewrite SANTA_SEED into a temp
    REM copy, so a second seed can coexist with the first instead of
    REM clobbering it. A job's pod template is immutable once created, so
    REM this has to happen before apply, not via kubectl set env.
    REM
    REM powershell does the substitution because a cmd for/f rewrite mangles
    REM blank lines and the '!' characters inside the embedded shell script.
    REM It ships with Windows 10 and later, as certutil and tar already do.
    for %%T in (fwe niah-multivalue) do (
        set "SRC=k8s\instrumented8k\job-ruler-8k-%%T-midpoint100.yaml"
        set "DST=%TEMP%\job-ruler-8k-%%T-midpoint100-s%SEED%.yaml"
        powershell -NoProfile -Command "(Get-Content -Raw '!SRC!').Replace('value: \"1690\"', 'value: \"%SEED%\"').Replace('midpoint100', 'midpoint100-s%SEED%') | Set-Content -NoNewline '!DST!'"
        if errorlevel 1 exit /b 1
        kubectl apply -f "!DST!"
        if errorlevel 1 exit /b 1
    )
)

echo.
echo Submitted. Results will land under:
echo   /shared/ruler/results/8192_100_midpoint/fwe_seed%SEED%
echo   /shared/ruler/results/8192_100_midpoint/niah_multivalue_seed%SEED%
echo.
echo Follow logs in another window with:
echo   kubectl logs -f job/%FWEJOB%
echo   kubectl logs -f job/%NIAHJOB%

REM ------------------------------------------------------------ wait
echo.
echo === Waiting for both jobs ^(13 methods x 100 examples each; hours^) ===
echo Ctrl+C here is safe: it stops the wait, not the jobs.
kubectl wait --for=condition=complete job/%FWEJOB% --timeout=86400s
if errorlevel 1 (
    echo ERROR: %FWEJOB% did not complete. Last log lines:
    kubectl logs --tail=60 job/%FWEJOB%
    exit /b 1
)
kubectl wait --for=condition=complete job/%NIAHJOB% --timeout=86400s
if errorlevel 1 (
    echo ERROR: %NIAHJOB% did not complete. Last log lines:
    kubectl logs --tail=60 job/%NIAHJOB%
    exit /b 1
)

echo.
echo === Paired offset tables from the pods ===
kubectl logs job/%FWEJOB% --tail=40
kubectl logs job/%NIAHJOB% --tail=40

REM ------------------------------------------------------------ download
echo.
echo === Downloading results ===
call scripts\download-ruler-8k-midpoint.bat
if errorlevel 1 (
    echo WARNING: download failed, but the results are safe on the PVC.
    echo Retry with: scripts\download-ruler-8k-midpoint.bat
)

echo.
echo Done. To pair across seeds after a second run:
echo   python compare_offsets.py --results "8192_100_midpoint\*" --out-dir summary
endlocal
