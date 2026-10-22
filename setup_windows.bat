@echo off
rem CogniOS - one-time setup on Windows: creates .venv and installs the pinned dependencies.
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo The Python launcher "py" was not found.
  echo Install Python 3.12 ^(64-bit^) from https://www.python.org/downloads/ and run this file again.
  goto fail
)
py -3.12 -c "import sys" >nul 2>nul
if errorlevel 1 (
  echo Python 3.12 was not found. Install Python 3.12 ^(64-bit^) from https://www.python.org/downloads/
  echo Other versions are not supported: the saved models need xgboost 3.4.1 ^(Python 3.12+^) and scikit-learn 1.9.1.
  goto fail
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment in .venv ...
  py -3.12 -m venv .venv
  if errorlevel 1 goto fail
)

echo Installing dependencies ^(torch is about 200 MB; this can take several minutes^) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo.
echo Setup complete. Start the app with start_windows.bat
pause
exit /b 0

:fail
echo.
echo Setup did not finish. See the message above and the Windows section of README.md.
pause
exit /b 1
