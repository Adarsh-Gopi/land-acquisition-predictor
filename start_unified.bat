@echo off
title Land Acquisition Delay System - Unified Terminal (FastAPI + Express)
color 0A

echo =====================================================================
echo   Starting Both Servers in a Single Unified Terminal Console
echo   Press Ctrl+C to stop both servers at once
echo =====================================================================
echo.

cd /d "%~dp0"
timeout /t 2 /nobreak >nul
start http://localhost:3000
npm run dev
