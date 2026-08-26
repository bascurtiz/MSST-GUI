@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\" (
    echo [ERROR] Virtual environment not found.
    echo        Run run_install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python main.py
pause
