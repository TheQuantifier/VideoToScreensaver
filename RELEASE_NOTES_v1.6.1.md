# VideoToScreensaver v1.6.1

## Windows Installer (If you don't have the app)
- Download `VideoToScreensaver-Setup.exe` from this release.
- Run the installer and launch `VideoToScreensaver` from Start Menu.

## Windows Updater (If you already have the app)
- Open the app
- Select `Update App`
- Wait for new update to be installed

## Updates
- Fixed in-app uninstall detection so installed builds correctly recognize the Windows uninstall entry even when the registry display name includes a version suffix.
- Fixed `Uninstall App` launching so the app starts the registered Windows uninstall command reliably.
- Rebuilt release artifacts for `v1.6.1`.
- Bumped app version to `v1.6.1`.

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
