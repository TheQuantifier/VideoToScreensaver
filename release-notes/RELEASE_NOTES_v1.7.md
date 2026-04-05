# VideoToScreensaver v1.7

## Windows Installer (If you don't have the app)
- Download `VideoToScreensaver-Setup.exe` from this release.
- Run the installer and launch `VideoToScreensaver` from Start Menu.

## Windows Updater (If you already have the app)
- Open the app
- Select `Update App`
- Wait for the new update to install

## Updates
- Fixed the in-app uninstall flow so the button closes the app and launches the installed `unins000.exe` uninstaller reliably.
- Fixed `Install as Screensaver` so the elevated install path does not reopen the app UI as a second window.
- Fixed managed screensaver install refresh so successful installs are validated and immediately appear in the managed list.
- Moved per-screensaver config into `%LOCALAPPDATA%\VideoToScreensaver\Screensavers\...` so `Save Settings` no longer needs repeated admin elevation.
- Updated uninstall cleanup so the app removes its own `%LOCALAPPDATA%\VideoToScreensaver` files before launching the uninstaller.
- Fixed mouse-wheel scrolling for the main page while keeping the managed-screensaver list scroll behavior intact.
- Restored balanced left/right page spacing in the main app layout.
- Strengthened Windows dark title-bar requests so supported systems are more likely to show a dark caption bar.
- Bumped app version to `v1.7`.

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
