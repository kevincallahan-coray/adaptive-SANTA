@echo off
setlocal
cd /d "%~dp0.."

kubectl delete pod kevin-ruler-shell --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\workspace-shell-ruler-shared.yaml
if errorlevel 1 exit /b 1
kubectl wait --for=condition=Ready pod/kevin-ruler-shell --timeout=2m
if errorlevel 1 exit /b 1

echo Entering /shared/ruler. Type exit when finished.
kubectl exec -it kevin-ruler-shell -- bash -lc "cd /shared/ruler && exec bash"

echo Removing inspection pod...
kubectl delete pod kevin-ruler-shell --ignore-not-found
endlocal
