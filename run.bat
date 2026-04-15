@echo off
title MidiMaker
echo ============================================================
echo  MidiMaker - Local Symbolic Music Workstation
echo ============================================================
echo.

REM ---------------------------------------------------------------------------
REM Check virtual environment exists
REM ---------------------------------------------------------------------------
if not exist venv\Scripts\activate.bat (
    echo ERROR: Virtual environment not found.
    echo.
    echo Please run setup.bat first to install MidiMaker.
    echo.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Activate virtual environment
REM ---------------------------------------------------------------------------
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment.
    echo Try deleting the venv\ folder and running setup.bat again.
    pause
    exit /b 1
)

REM ---------------------------------------------------------------------------
REM Launch the application
REM ---------------------------------------------------------------------------
echo Starting MidiMaker...
echo.
python app\main.py %*
set EXIT_CODE=%errorlevel%

if %EXIT_CODE% neq 0 (
    echo.
    echo ============================================================
    echo  MidiMaker exited with an error (code %EXIT_CODE%).
    echo ============================================================
    echo.
    echo Possible causes:
    echo  - A Python error occurred: check the log files for details
    echo    Logs: %%USERPROFILE%%\.midimaker\logs\
    echo  - GPU driver issue: the app will fall back to CPU automatically
    echo  - If dependencies are missing, re-run setup.bat to reinstall
    echo.
    echo For support, include the log file from the logs folder.
    pause
)
