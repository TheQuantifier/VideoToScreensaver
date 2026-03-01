# VideoToScreensaver v1.3

## Windows Installer
- **Option 1: Install (new users)**
  - Download `VideoToScreensaver-Setup.exe` from this release.
  - Run the installer and launch `VideoToScreensaver` from Start Menu.
- **Option 2: Update (existing users)**
  - Open `VideoToScreensaver`.
  - Click `Update App` and follow the prompts to update in place.

## Updates
- Added an in-app `Update App` button so users can update in place without reinstalling.
- App now checks the latest GitHub release, downloads the zip asset, applies the update, and restarts automatically.
- Improved runtime asset sync to handle file/folder path collisions more safely during updates.

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
