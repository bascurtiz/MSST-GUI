@echo off
REM ============================================================
REM  MSST Modern GUI - full build: exe + Inno Setup installer
REM  Outputs: dist\MSST-GUI\           (portable app)
REM           dist\MSST-GUI-Setup-*.exe (installer)
REM ============================================================
cd /d "%~dp0.."

echo [1/3] Preparing bundled runtime (skipped if already present)...
python tools\prepare_runtime.py
if errorlevel 1 exit /b 1

echo [2/3] Building app bundle...
python -m PyInstaller MSST-GUI.spec --noconfirm
if errorlevel 1 exit /b 1

echo [3/3] Building installer...
"%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" MSST-GUI.iss
if errorlevel 1 exit /b 1

echo.
echo Done. See dist\MSST-GUI-Setup-*.exe
