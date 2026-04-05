# VideoToScreensaver v1.6

## Windows Installer (If you don't have the app)
- Download `VideoToScreensaver-Setup.exe` from this release.
- Run the installer and launch `VideoToScreensaver` from Start Menu.

## Windows Updater (If you already have the app)
- Open the app
- Select `Update App`
- Wait for new update to be installed

## Updates
- Reworked installed screensavers so they are copied into the Windows screensaver directory instead of relying on the app install folder.
- Switched the packaged app to a standalone `--onefile` build so installed `.scr` files can run independently.
- Added per-screensaver storage under `%ProgramData%` for managed video/config data used by installed screensavers.
- Split the main flow into `Install as Screensaver` for default install and `Save Settings` for updating the active managed screensaver.
- Added `Set selected active` in the managed screensaver list and expanded list support to include legacy app-created screensavers.
- Updated preview behavior so selected-source previews use the current behavior settings and exit on `Esc` or timeout.
- Refined the dark UI with section reordering, button-state improvements, and clearer settings actions.
- Bumped app version to `v1.6`.

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
