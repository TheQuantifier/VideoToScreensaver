#ifndef AppVersion
#define AppVersion "0.0.0"
#endif

[Setup]
AppId={{9B3E3D47-6C94-4E5A-9C2A-6F5646A55C58}
AppName=VideoToScreensaver
AppVersion={#AppVersion}
AppVerName=VideoToScreensaver v{#AppVersion}
AppPublisher=ManusWebworks
DefaultDirName={autopf}\VideoToScreensaver
DefaultGroupName=VideoToScreensaver
UninstallDisplayIcon={app}\VideoToScreensaver.exe
Compression=lzma
SolidCompression=yes
WizardStyle=modern
OutputDir=..\release
OutputBaseFilename=VideoToScreensaver-Setup

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "..\release\VideoToScreensaver\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\VideoToScreensaver"; Filename: "{app}\VideoToScreensaver.exe"
Name: "{autodesktop}\VideoToScreensaver"; Filename: "{app}\VideoToScreensaver.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\VideoToScreensaver.exe"; Description: "Launch VideoToScreensaver"; Flags: nowait postinstall skipifsilent
