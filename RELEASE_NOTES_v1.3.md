# VideoToScreensaver v1.3

## Windows Installer (If you don't have the app)
- Download `VideoToScreensaver-Setup.exe` from this release.
- Run the installer and launch `VideoToScreensaver` from Start Menu.

## Windows Updater (If you already have the app)
- Open the app
- Select `Update App`
- Wait for new update to be installed

## Updates
- Reworked the app UI to use themed `ttk` controls and grouped sections for cleaner layout.
- Improved control alignment so related labels and inputs stay visually paired.
- Replaced raw timeout-seconds entry with a `Wait` (minutes) input in the main settings row.
- Added a standard footer action row with `OK`, `Cancel`, and `Apply`.
- Updater now supports elevation flow (UAC prompt) when install folder write access requires admin rights.
- Added clearer status messaging when launching elevated updater mode.
- App version is now sourced from `src/version.py` (single source of truth).
- Installer version now comes from the app version at build time (`build_installer.ps1` passes `AppVersion` into Inno Setup).
- Inno Setup script now uses version preprocessor variables, reducing version mismatch risk.

## Highlights
- Turn local videos into a real Windows screensaver from a simple UI.
- One-click install as active screensaver.
- Video fit modes: `contain`, `cover`, `stretch`.
- Multi-monitor playback support (same video on all monitors, per-monitor fit).
- Preview mode from inside the app.
- Managed screensaver list with delete options.
- Auto-cleanup of copied video files when their screensaver is removed and no longer referenced.

## Notes
- Windows only.
- Unsigned build: Windows SmartScreen may show a warning.
