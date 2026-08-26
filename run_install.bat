@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ============================================
echo  MSST GUI - Install
echo ============================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo        Download: https://www.python.org/downloads/
    echo        Check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM --- Create virtual environment ---
if exist ".venv\" (
    echo [INFO] Virtual environment (.venv) already exists.
) else (
    echo [STEP] Creating virtual environment...
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

echo [STEP] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [STEP] Upgrading pip...
python -m pip install --upgrade pip >nul

echo [STEP] Installing dependencies...
pip install -r requirements_gui.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo [STEP] Installing PyTorch with CUDA support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ============================================
echo  Install complete!
echo  Run:  run_gui.bat
echo ============================================
pause
