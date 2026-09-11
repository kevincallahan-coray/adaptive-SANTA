@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0.."

echo Creating or confirming PVC kevin-workspace...
kubectl apply -f k8s\pvc.yaml
if errorlevel 1 exit /b 1

echo Waiting for kevin-workspace to become Bound...
for /L %%N in (1,1,150) do (
    set "phase="
    for /F "delims=" %%P in ('kubectl get pvc kevin-workspace -o jsonpath^="{.status.phase}" 2^>nul') do set "phase=%%P"
    if "!phase!"=="Bound" goto :pvc_ready
    timeout /t 2 /nobreak >nul
)

echo ERROR: Timed out waiting for kevin-workspace PVC to bind.
exit /b 1

:pvc_ready
kubectl get pvc kevin-workspace santa-work
if errorlevel 1 exit /b 1

echo.
echo Running read-only migration from santa-work to kevin-workspace\santa-adaptive-z...
kubectl delete job santa-migrate-workspace --ignore-not-found >nul 2>&1
kubectl apply -f k8s\workspace\job-migrate-santa-work.yaml
if errorlevel 1 exit /b 1

kubectl wait --for=condition=complete job/santa-migrate-workspace --timeout=15m
if errorlevel 1 (
    echo Migration job did not complete successfully. Showing logs:
    kubectl logs job/santa-migrate-workspace
    exit /b 1
)

kubectl logs job/santa-migrate-workspace

echo.
set /P "answer=Migration completed. Type DELETE to permanently delete the old santa-work PVC: "
if /I "%answer%"=="DELETE" (
    kubectl delete pvc santa-work
    if errorlevel 1 exit /b 1
    echo Old PVC santa-work deleted.
) else (
    echo Old PVC retained. Delete it later with: kubectl delete pvc santa-work
)

echo.
echo Current PVCs:
kubectl get pvc
endlocal
