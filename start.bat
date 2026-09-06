@echo off
title Land Acquisition Delay Predictive System - Full Stack Launcher
color 0B

echo =====================================================================
echo   Predictive Analytics System for Early Detection of Delays
echo   Starting Python FastAPI (Port 8000) ^& Node.js Express (Port 3000)
echo =====================================================================
echo.

cd /d "%~dp0"

:: Detect Python executable
set "PYTHON_CMD=python"
if exist "W:\Project Work\ML~ env\venv\Scripts\python.exe" (
    set "PYTHON_CMD=W:\Project Work\ML~ env\venv\Scripts\python.exe"
    echo [OK] Using virtual environment: W:\Project Work\ML~ env\venv
) else (
    echo [OK] Using system Python
)

:: 1. Launch Python FastAPI Server in a separate window
echo [*] Launching Python FastAPI Server (Port 8000)...
start "Python FastAPI ML Server [Port 8000]" cmd /k "cd /d "%~dp0" && "%PYTHON_CMD%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

:: 2. Wait for FastAPI to load model
echo [*] Initializing ML Model Pipeline...
timeout /t 3 /nobreak >nul

:: 3. Launch Node.js Express Server in a separate window
echo [*] Launching Node.js Express Web Server (Port 3000)...
start "Node.js Express Web Server [Port 3000]" cmd /k "cd /d "%~dp0" && node app.js"

:: 4. Open default browser
timeout /t 2 /nobreak >nul
echo.
echo =====================================================================
echo   [SUCCESS] Both servers are online!
echo   - Web Application UI: http://localhost:3000
echo   - ML API Swagger Docs: http://localhost:8000/docs
echo =====================================================================
echo Opening web application in your default browser...
start http://localhost:3000

echo.
echo Leave the two server command windows open while testing.
echo To stop the servers, simply close their command windows.
pause
