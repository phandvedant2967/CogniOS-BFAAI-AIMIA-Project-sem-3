@echo off
rem CogniOS - start the local server on Windows. Keep this window open while you use the app.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo .venv not found. Run setup_windows.bat first.
  pause
  exit /b 1
)

echo Starting CogniOS. When you see "Running on http://127.0.0.1:5000", open that address in your browser.
echo Keep this window open while you use the app. Press Ctrl+C here to stop the server.
echo.
".venv\Scripts\python.exe" run.py
pause
