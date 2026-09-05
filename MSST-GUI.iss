; MSST GUI — Inno Setup script
; Build the app bundle first (pyinstaller MSST-GUI.spec --noconfirm),
; then compile this script with ISCC. Output: dist\MSST-GUI-Setup-<ver>.exe

#define AppName "MSST GUI"
#define AppExe "MSST-GUI.exe"
#define AppVersion "1.0.4"
#define AppMutex "MSST-GUI-Mutex"

[Setup]
AppId={{7E1A2C64-5B9D-4A0E-9C3F-2A8B1D6E5F01}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=MSST GUI contributors
AppMutex={#AppMutex}
DefaultDirName={autopf}\MSST-GUI
DefaultGroupName={#AppName}
; Per-user install: no admin prompt, installs to %LocalAppData%\Programs
PrivilegesRequired=lowest
SetupIconFile=resources\app_icon.ico
WizardImageFile=build\setup\wizard.png
WizardSmallImageFile=build\setup\wizardsmall.png
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
WizardStyle=modern
OutputDir=dist
OutputBaseFilename=MSST-GUI-Setup-{#AppVersion}
CloseApplications=yes
RestartApplications=no
DisableProgramGroupPage=yes
SetupLogging=yes
ArchitecturesInstallIn64BitMode=x64compatible

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\MSST-GUI\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[InstallDelete]
; TESTING MODE: only wipe the bundled app code (_internal), so stale
; artifacts from a previous install (e.g. a broken bundled zstd/backports
; stub) can never survive an upgrade — but keep the installed GPU runtime
; (runtime\, ~2.5-3 GB / ~5 min to reinstall), downloaded models\ and
; configs\, and msst_settings.json intact while iterating on builds.
; For release, consider restoring the full wipe:
;   Type: filesandordirs; Name: "{app}\*"
Type: filesandordirs; Name: "{app}\_internal\*"

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; \
    Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Nothing: user data (msst_settings.json, models\, runtime\, configs\
; downloads) is intentionally preserved across uninstalls.
