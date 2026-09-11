@echo off
setlocal
cd /d "%~dp0.."

kubectl delete pod kevin-workspace-shell --ignore-not-found >nul 2>&1
kubectl apply -f k8s\workspace\workspace-shell.yaml
if errorlevel 1 exit /b 1

kubectl wait --for=condition=Ready pod/kevin-workspace-shell --timeout=2m
if errorlevel 1 exit /b 1

kubectl exec -it kevin-workspace-shell -- bash
endlocal
