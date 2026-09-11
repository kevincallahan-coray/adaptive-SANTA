@echo off
setlocal
cd /d "%~dp0.."

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
kubectl apply -f k8s\parallel\export-ruler-results.yaml
if errorlevel 1 exit /b 1
kubectl wait --for=condition=Ready pod/kevin-ruler-export --timeout=2m
if errorlevel 1 exit /b 1

kubectl exec kevin-ruler-export -- sh -c "test -d /shared/ruler/results && tar -czf /tmp/ruler-results.tar.gz -C /shared/ruler results"
if errorlevel 1 exit /b 1

kubectl cp kevin-ruler-export:/tmp/ruler-results.tar.gz .\ruler-results.tar.gz
if errorlevel 1 exit /b 1

kubectl delete pod kevin-ruler-export --ignore-not-found >nul 2>&1
echo Downloaded .\ruler-results.tar.gz
endlocal
