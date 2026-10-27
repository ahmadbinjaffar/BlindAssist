@echo off
setlocal
rem ==========================================================================
rem  BlindAssist - first-time setup
rem
rem  Creates the virtual environment, links it into the project as .venv,
rem  installs the pinned dependencies and verifies the result.
rem
rem  Safe to re-run: it only does what is missing.
rem
rem  NOTE FOR MAINTAINERS: cmd.exe parses a parenthesized IF block as a whole,
rem  even when the branch is not taken. Never put an unescaped "(" or ")" in
rem  echo text inside such a block - it aborts with "was unexpected at this
rem  time." Keep all messages parenthesis-free.
rem ==========================================================================

set "PROJ=%~dp0"
rem Keep the venv path SHORT. torch ships deep header paths and Windows'
rem default MAX_PATH is 260 characters - a long project path breaks the install.
set "VENV_DIR=%USERPROFILE%\ba_venv"

echo.
echo === BlindAssist setup ==============================================
echo Project : %PROJ%
echo venv    : %VENV_DIR%
echo.

cd /d "%PROJ%"
if errorlevel 1 (
  echo [FAIL] Cannot cd into the project folder.
  exit /b 1
)

rem --- 1. Python ----------------------------------------------------------
where python >nul 2>&1
if errorlevel 1 (
  echo [FAIL] No python on PATH.
  echo        Install 64-bit Python 3.13 from python.org/downloads
  echo        and tick "Add python.exe to PATH" in the installer.
  exit /b 1
)
for /f "delims=" %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [1/5] Found %PYVER%

rem --- 2. Create the virtual environment ----------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" goto :make_venv
echo [2/5] venv already exists, skipping creation.
goto :venv_ok
:make_venv
echo [2/5] Creating venv at %VENV_DIR% ...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
  echo [FAIL] Could not create the virtual environment.
  exit /b 1
)
echo        created.
:venv_ok

rem --- 3. Link it into the project as .venv -------------------------------
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [FAIL] venv is missing after creation.
  exit /b 1
)
if exist "%PROJ%.venv\Scripts\python.exe" (
  echo [3/5] .venv link already in place, skipping.
  goto :link_ok
)
echo [3/5] Creating junction .venv pointing at %VENV_DIR%
if exist "%PROJ%.venv" rmdir "%PROJ%.venv"
mklink /J "%PROJ%.venv" "%VENV_DIR%" >nul
if errorlevel 1 (
  echo [FAIL] mklink /J failed.
  echo        Run this script from an elevated prompt, or create the link
  echo        by hand with:
  echo        cmd /c mklink /J "%PROJ%.venv" "%VENV_DIR%"
  exit /b 1
)
echo        linked.
:link_ok

set "VPY=%VENV_DIR%\Scripts\python.exe"
if not exist "%VPY%" (
  echo [FAIL] %VPY% not found.
  exit /b 1
)

rem --- 4. Install pinned dependencies -------------------------------------
echo [4/5] Installing dependencies - first run downloads about 1.2 GB.
"%VPY%" -m pip install --upgrade pip
if errorlevel 1 (
  echo [FAIL] pip upgrade failed - check your internet connection.
  exit /b 1
)
"%VPY%" -m pip install -r "%PROJ%requirements.txt"
if errorlevel 1 (
  echo [FAIL] pip install failed.
  echo        Common causes: no internet, a proxy blocking PyPI,
  echo        or a project path that is too long for torch.
  exit /b 1
)

rem --- 5. Verify ----------------------------------------------------------
echo [5/5] Verifying imports ...
"%VPY%" -c "import cv2, numpy, torch, ultralytics, pyttsx3; print('        opencv      ', cv2.__version__); print('        torch       ', torch.__version__); print('        ultralytics ', ultralytics.__version__); print('        torch device', 'CUDA' if torch.cuda.is_available() else 'CPU-only')"
if errorlevel 1 (
  echo [FAIL] A required package did not import.
  exit /b 1
)

if exist "%PROJ%assets\yolov8n.pt" goto :model_ok
echo         model      : NOT downloaded yet - the first launch fetches
echo                     yolov8n.pt, about 6.25 MB, then stays offline.
goto :model_done
:model_ok
echo         model      : assets\yolov8n.pt present - offline ready
:model_done

echo.
echo === Setup complete ===================================================
echo   Start the app : run.bat
echo   Run the tests : .venv\Scripts\python.exe -m pytest -q
echo   Docs          : README.md  docs\ARCHITECTURE.md
echo.
exit /b 0
