@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM --- Isolate pip from machine-wide pip.ini. NVIDIA PyIndex tooling adds
REM     pypi.ngc.nvidia.com as a global extra index; when that host is
REM     unreachable every package lookup stalls with 5 DNS retries.
set "PIP_CONFIG_FILE=nul"
set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
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
    echo [INFO] Virtual environment .venv already exists.
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

REM --- PyTorch FIRST, from the CUDA index. If requirements_gui.txt runs
REM     before this, pip resolves torch from PyPI as the CPU-only wheel and
REM     the later CUDA install is skipped because the requirement is already
REM     satisfied. torchvision is included so segmentation-models-pytorch /
REM     timm don't pull a PyPI torchvision that drags torch back to CPU.
echo [STEP] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install PyTorch.
    pause
    exit /b 1
)

echo [STEP] Installing dependencies...
pip install -r requirements_gui.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

REM --- Guard: a dependency may still have replaced torch with a CPU build.
echo [STEP] Verifying CUDA build...
python -c "import torch, sys; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else 1)"
if %errorlevel% neq 0 (
    echo [WARN] torch reports no CUDA. Reinstalling CUDA wheels...
    pip install --force-reinstall --no-deps torch torchvision torchaudio --index-url %TORCH_INDEX%
    python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
)

echo.
echo ============================================
echo  Install complete!
echo  Run:  run_gui.bat
echo ============================================
pause