@echo off
REM deploy.bat - Regenerate site and deploy to server
REM Run this manually or via Task Scheduler

setlocal

set DOC_ID=17fjNvJzj8ZGSer7c7OFe_CNfUKbAxEh_OBv94ZdRG5c
set SITE_DIR=%~dp0site
set REMOTE_USER=your_username
set REMOTE_HOST=msst-bible.x10.mx
set REMOTE_DIR=/public_html

echo [%date% %time%] Starting deployment...

REM Step 1: Regenerate the site
echo Regenerating site...
python "%~dp0gdoc_site.py" --doc %DOC_ID% --out "%SITE_DIR%"
if errorlevel 1 (
    echo ERROR: Site generation failed!
    exit /b 1
)

REM Step 2: Deploy via SCP (uncomment and configure one method)

REM Option A: SCP (if you have SSH access)
REM scp -r "%SITE_DIR%\*" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/

REM Option B: rsync (if available)
REM rsync -avz "%SITE_DIR%/" %REMOTE_USER%@%REMOTE_HOST%:%REMOTE_DIR%/

REM Option C: FTP using lftp (install lftp first)
REM lftp -u %REMOTE_USER%,your_password %REMOTE_HOST% -e "mirror -R %SITE_DIR% %REMOTE_DIR%; quit"

echo [%date% %time%] Deployment complete!
echo.
echo NOTE: Uncomment your preferred deployment method in this script.
echo       Edit REMOTE_USER, REMOTE_HOST, and REMOTE_DIR as needed.

endlocal
