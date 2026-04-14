@echo off
title MidiMaker Setup
echo ============================================================
echo  MidiMaker - Local Symbolic Music Workstation
echo  Setup Script
echo ============================================================
echo.

REM ---------------------------------------------------------------------------
REM 1. Check Python 3.10+
REM ---------------------------------------------------------------------------
echo [1/7] Checking Python installation...
python --version 2>NUL
if errorlevel 1 (
    echo.
    echo ERROR: Python not found in PATH.
    echo Please install Python 3.10 or newer from:
    echo   https://www.python.org/downloads/
    echo Make sure to tick "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Check version is at least 3.10
python -c "import sys; exit(0 if sys.version_info >= (3,10) else 1)" 2>NUL
if errorlevel 1 (
    echo.
    echo ERROR: Python 3.10 or newer is required.
    echo Your current version is too old.
    echo Please download the latest Python from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo   OK - Python version is compatible.
echo.

REM ---------------------------------------------------------------------------
REM 2. Create virtual environment
REM ---------------------------------------------------------------------------
echo [2/7] Creating virtual environment at venv\...
if exist venv\ (
    echo   Virtual environment already exists, skipping creation.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create virtual environment.
        echo Try running: python -m pip install virtualenv
        pause
        exit /b 1
    )
    echo   Virtual environment created successfully.
)
echo.

REM ---------------------------------------------------------------------------
REM 3. Activate venv and install requirements
REM ---------------------------------------------------------------------------
echo [3/7] Activating virtual environment and installing dependencies...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo.
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

python -m pip install --upgrade pip --quiet
echo   pip upgraded.

echo   Installing requirements (this may take several minutes)...
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo   Dependencies installed successfully.
echo.

REM ---------------------------------------------------------------------------
REM 4. Download / validate model assets
REM ---------------------------------------------------------------------------
echo [4/7] Downloading model assets...
python scripts\download_assets.py
if errorlevel 1 (
    echo.
    echo WARNING: Asset download reported issues.
    echo The application can still run in stub mode without model weights.
    echo See above for instructions on manual download.
    echo.
)
echo.

REM ---------------------------------------------------------------------------
REM 5. Validate CUDA availability
REM ---------------------------------------------------------------------------
echo [5/7] Checking GPU / CUDA availability...
python -c "from core.utils import get_device; d=get_device(); print('  Device:', d)"
echo.

REM ---------------------------------------------------------------------------
REM 6. Create application data directories
REM ---------------------------------------------------------------------------
echo [6/7] Creating application data directories...
if not exist projects\ mkdir projects\
if not exist models\  mkdir models\
if not exist logs\    mkdir logs\
echo   projects\, models\, logs\ directories ready.
echo.

REM ---------------------------------------------------------------------------
REM 7. Run smoke test
REM ---------------------------------------------------------------------------
echo [7/7] Running smoke test...
python scripts\smoke_test.py
if errorlevel 1 (
    echo.
    echo WARNING: Some smoke tests did not pass.
    echo The application may still be usable - see output above for details.
    echo.
) else (
    echo   All smoke tests passed.
)
echo.

REM ---------------------------------------------------------------------------
REM Done
REM ---------------------------------------------------------------------------
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo  To start MidiMaker, run:
echo    run.bat
echo.
pause
