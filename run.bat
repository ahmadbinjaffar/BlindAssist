@echo off
setlocal
rem ==========================================================================
rem  BlindAssist - start the app
rem
rem  Any arguments are passed straight through, e.g.
rem      run.bat --no-speech
rem      run.bat --video assets\samples\chair.jpg --headless-test --tag chair
rem ==========================================================================

set "PROJ=%~dp0"

if not exist "%PROJ%.venv\Scripts\python.exe" (
  echo [setup] No virtual environment found.
  echo         Run setup.bat first, then try again.
  exit /b 1
)

cd /d "%PROJ%"

rem The app must run as a module (relative imports), from the project folder.
".venv\Scripts\python.exe" -m blindassist.app %*

set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%
