# VideoToScreensaver Windows

This project is for **Windows only**.

VideoToScreensaver lets you turn your own video files into a real Windows screensaver with a simple UI. It supports one-click install, per-screen video fit modes (`contain`, `cover`, `stretch`) including multi-monitor playback, quick preview, and built-in management for removing old app-created screensavers.

Use the standalone installer to install the app quickly:

1. Download `VideoToScreensaver-Setup.exe` from the latest GitHub Release.
2. Run the installer.
3. Launch `VideoToScreensaver` from Start Menu.

The app lets you pick a video file, install it as a screensaver, and then open Screen Saver settings to enable or verify it.

## What you get
- A GUI executable (`VideoToScreensaver.exe`).
- In-app video picker.
- One-click install as active screensaver.
- Built-in steps for what to do next in Windows settings.
- Fullscreen video playback in screensaver mode (`/s`).

## 1) Setup in VS Code (for building from source)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## 2) Build the Windows app executable

```powershell
python build_app.py
```

Output:
- `release\VideoToScreensaver\VideoToScreensaver.exe`

## 2b) Build a Windows installer (easy distribution)

Install Inno Setup 6, then run:

```powershell
.\build_installer.ps1
```

Output:
- `release\VideoToScreensaver-Setup.exe`

## 3) Use the app
1. Run `release\VideoToScreensaver\VideoToScreensaver.exe`.
2. Click `Browse` and select your video.
3. Optionally adjust:
   - `Mouse move threshold`
   - `Timeout seconds`
   - `Video fit` (`contain`, `cover`, `stretch`)
4. Click `Install as Screensaver`.
5. Click `Open Screen Saver Settings`.
6. Confirm `"<yourfilename>_vts.scr"` is selected, set idle wait, click `Apply`.

The app also includes a `Managed screensavers` section where you can:
- list all `.scr` files created by this app
- delete selected old screensaver files (and auto-delete copied video when no other screensaver uses it)

If you used the installer, users can launch from Start Menu shortcut: `VideoToScreensaver`.

## Notes
- The app stores files in `%LOCALAPPDATA%\VideoToScreensaver`.
- During install, the app copies itself to `%LOCALAPPDATA%\VideoToScreensaver\VideoToScreensaver.scr`.
- Build now uses `PyInstaller --onedir` for faster startup compared to onefile extraction.
- H.264 `.mp4` is the most compatible format.
- If it exits too fast, increase `Mouse move threshold`.
- `contain` keeps full frame with black bars, `cover` fills screen by cropping edges, `stretch` fills screen by distortion.
- Multi-monitor playback is supported. The same video is shown on each monitor, and fit/stretch is applied per monitor resolution.
