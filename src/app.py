import json
import os
import re
import shutil
import subprocess
import sys
import time
import ctypes
import urllib.error
import urllib.request
import webbrowser
import winreg
from ctypes import wintypes
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

import cv2
import numpy as np

try:
    from version import APP_VERSION
except ModuleNotFoundError:
    from src.version import APP_VERSION


APP_NAME = "VideoToScreensaver"
APP_DISPLAY_NAME = f"{APP_NAME} v{APP_VERSION}"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME
PROGRAM_DATA_DIR = Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / APP_NAME
MANAGED_ROOT_DIR = PROGRAM_DATA_DIR / "Screensavers"
SCR_NAME = f"{APP_NAME}.scr"
CONFIG_NAME = "config.json"
SCR_TAG = "_vts"
UPDATE_API_URL = "https://api.github.com/repos/TheQuantifier/VideoToScreensaver/releases/latest"
RELEASES_PAGE_URL = "https://github.com/TheQuantifier/VideoToScreensaver/releases/latest"
DEFAULT_THRESHOLD = 15
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_FIT_MODE = "contain"
VALID_FIT_MODES = {"contain", "cover", "stretch"}
DEFAULT_STARTUP_TRANSITION = "none"
VALID_STARTUP_TRANSITIONS = {"none", "fade", "wipe_left", "wipe_down", "zoom_in"}
UNINSTALL_REGISTRY_PATH = r"Software\Microsoft\Windows\CurrentVersion\Uninstall"


class MONITORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
    ]


MONITORINFOF_PRIMARY = 0x00000001


def show_message(title: str, message: str) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, title, 0)


def get_mouse_pos() -> tuple[int, int]:
    point = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def get_monitor_rects() -> list[tuple[int, int, int, int]]:
    user32 = ctypes.windll.user32
    monitors: list[tuple[int, int, int, int, int]] = []

    monitor_enum_proc = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HANDLE,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    def callback(h_monitor, _hdc, _rect, _lparam):
        info = MONITORINFO()
        info.cbSize = ctypes.sizeof(MONITORINFO)
        if user32.GetMonitorInfoW(h_monitor, ctypes.byref(info)):
            left = int(info.rcMonitor.left)
            top = int(info.rcMonitor.top)
            right = int(info.rcMonitor.right)
            bottom = int(info.rcMonitor.bottom)
            width = max(right - left, 1)
            height = max(bottom - top, 1)
            is_primary = 1 if (info.dwFlags & MONITORINFOF_PRIMARY) else 0
            monitors.append((left, top, width, height, is_primary))
        return True

    user32.EnumDisplayMonitors(0, 0, monitor_enum_proc(callback), 0)
    monitors.sort(key=lambda m: (-m[4], m[0], m[1]))
    return [(m[0], m[1], m[2], m[3]) for m in monitors]


def get_wallpaper_frame() -> np.ndarray | None:
    SPI_GETDESKWALLPAPER = 0x0073
    buf = ctypes.create_unicode_buffer(1024)
    wallpaper_path: Path | None = None

    if ctypes.windll.user32.SystemParametersInfoW(SPI_GETDESKWALLPAPER, len(buf), buf, 0):
        candidate = Path(buf.value.strip())
        if candidate.is_file():
            wallpaper_path = candidate

    if wallpaper_path is None:
        appdata = Path(os.environ.get("APPDATA", str(Path.home())))
        transcoded = appdata / "Microsoft" / "Windows" / "Themes" / "TranscodedWallpaper"
        if transcoded.is_file():
            wallpaper_path = transcoded

    if wallpaper_path is None:
        return None

    wallpaper = cv2.imread(str(wallpaper_path), cv2.IMREAD_COLOR)
    if wallpaper is None or wallpaper.size == 0:
        return None
    return wallpaper


def get_resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def get_system_screensaver_dir() -> Path:
    buffer = ctypes.create_unicode_buffer(260)
    length = ctypes.windll.kernel32.GetSystemDirectoryW(buffer, len(buffer))
    if length:
        return Path(buffer.value)
    return Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32"


def get_runtime_scr_path() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    candidate = Path(sys.executable)
    if candidate.suffix.lower() == ".scr":
        return candidate
    return None


def get_storage_dir_for_scr(scr_path: Path) -> Path:
    return MANAGED_ROOT_DIR / scr_path.stem


def get_config_path_for_scr(scr_path: Path) -> Path:
    return get_storage_dir_for_scr(scr_path) / CONFIG_NAME


def get_legacy_config_path() -> Path:
    return APP_DIR / CONFIG_NAME


def read_config(config_path: Path | None = None) -> dict:
    if config_path is None:
        runtime_scr = get_runtime_scr_path()
        if runtime_scr is None:
            return {}
        config_path = get_config_path_for_scr(runtime_scr)
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict, config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _norm_path_str(value: str) -> str:
    return os.path.normcase(str(Path(value)))


def write_config(
    source_video_path: Path,
    video_path: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    startup_transition: str,
    scr_path: Path,
) -> None:
    config = {
        "source_video_path": str(source_video_path),
        "video_path": str(video_path),
        "mouse_move_threshold": mouse_threshold,
        "timeout_seconds": timeout_seconds,
        "fit_mode": fit_mode,
        "startup_transition": startup_transition,
        "scr_path": str(scr_path),
    }
    save_config(config, get_config_path_for_scr(scr_path))


def get_installed_scr_path() -> Path:
    return get_system_screensaver_dir() / SCR_NAME


def sanitize_file_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in name).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned[:80] if cleaned else APP_NAME


def is_managed_screensaver_path(scr_path: Path | None) -> bool:
    if scr_path is None:
        return False
    try:
        return (
            scr_path.suffix.lower() == ".scr"
            and scr_path.stem.endswith(SCR_TAG)
            and _norm_path_str(str(scr_path.parent)) == _norm_path_str(str(get_system_screensaver_dir()))
        )
    except OSError:
        return False


def is_legacy_managed_screensaver_path(scr_path: Path | None) -> bool:
    if scr_path is None:
        return False
    try:
        return (
            scr_path.suffix.lower() == ".scr"
            and scr_path.stem.endswith(SCR_TAG)
            and _norm_path_str(str(scr_path.parent)) == _norm_path_str(str(APP_DIR))
        )
    except OSError:
        return False


def is_app_created_screensaver_path(scr_path: Path | None) -> bool:
    return is_managed_screensaver_path(scr_path) or is_legacy_managed_screensaver_path(scr_path)


def get_active_scr_path(config: dict) -> Path:
    configured = str(config.get("scr_path", "")).strip()
    if configured:
        return Path(configured)
    return get_installed_scr_path()


def get_installed_video_path(config: dict) -> Path | None:
    value = str(config.get("video_path", "")).strip()
    if value:
        return Path(value)
    return None


def get_source_video_path(config: dict) -> str:
    source = str(config.get("source_video_path", "")).strip()
    if source:
        return source
    installed_video = get_installed_video_path(config)
    return str(installed_video) if installed_video else ""


def read_config_for_scr(scr_path: Path) -> dict:
    if is_managed_screensaver_path(scr_path):
        return read_config(get_config_path_for_scr(scr_path))
    if is_legacy_managed_screensaver_path(scr_path):
        return read_config(get_legacy_config_path())
    return {}


def normalize_screensaver_args(argv: list[str]) -> str:
    if not argv:
        return "gui"
    first = argv[0].lower()
    if first.startswith("/s"):
        return "screensaver"
    if first.startswith("/c"):
        return "configure"
    if first.startswith("/p"):
        return "preview"
    return "gui"


def parse_version_tuple(value: str) -> tuple[int, ...]:
    nums = [int(part) for part in re.findall(r"\d+", value)]
    return tuple(nums) if nums else (0,)


def is_newer_version(latest: str, current: str) -> bool:
    latest_parts = list(parse_version_tuple(latest))
    current_parts = list(parse_version_tuple(current))
    max_len = max(len(latest_parts), len(current_parts))
    latest_parts.extend([0] * (max_len - len(latest_parts)))
    current_parts.extend([0] * (max_len - len(current_parts)))
    return tuple(latest_parts) > tuple(current_parts)


def fetch_latest_release_info() -> dict[str, str]:
    req = urllib.request.Request(
        UPDATE_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"{APP_NAME}/{APP_VERSION}",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    tag_name = str(payload.get("tag_name", "")).strip()
    latest_version = tag_name.lstrip("vV")
    html_url = str(payload.get("html_url", "")).strip() or RELEASES_PAGE_URL

    zip_url = ""
    for asset in payload.get("assets", []):
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", "")).lower()
        if name == f"{APP_NAME.lower()}.zip":
            zip_url = str(asset.get("browser_download_url", "")).strip()
            break
    if not zip_url:
        for asset in payload.get("assets", []):
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "")).lower()
            if name.endswith(".zip"):
                zip_url = str(asset.get("browser_download_url", "")).strip()
                break

    return {
        "version": latest_version,
        "zip_url": zip_url,
        "html_url": html_url,
    }


def download_file(url: str, destination: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": f"{APP_NAME}/{APP_VERSION}"})
    with urllib.request.urlopen(req, timeout=60) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def launch_self_update(zip_path: Path, install_dir: Path, current_pid: int, require_admin: bool = False) -> None:
    def ps_quote(text: str) -> str:
        return text.replace("'", "''")

    script_path = APP_DIR / "run_update.ps1"
    script = (
        "$ErrorActionPreference = 'Stop'\n"
        f"$pidToWait = {current_pid}\n"
        f"$zipPath = '{ps_quote(str(zip_path))}'\n"
        f"$installDir = '{ps_quote(str(install_dir))}'\n"
        "$exePath = Join-Path $installDir 'VideoToScreensaver.exe'\n"
        "$tempDir = Join-Path ([System.IO.Path]::GetTempPath()) ('VideoToScreensaverUpdate_' + [Guid]::NewGuid())\n"
        "for ($i = 0; $i -lt 240; $i++) {\n"
        "  if (-not (Get-Process -Id $pidToWait -ErrorAction SilentlyContinue)) { break }\n"
        "  Start-Sleep -Milliseconds 500\n"
        "}\n"
        "New-Item -Path $tempDir -ItemType Directory -Force | Out-Null\n"
        "Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force\n"
        "Copy-Item -Path (Join-Path $tempDir '*') -Destination $installDir -Recurse -Force\n"
        "Remove-Item -Path $zipPath -Force -ErrorAction SilentlyContinue\n"
        "Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue\n"
        "Start-Process -FilePath $exePath\n"
    )
    APP_DIR.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script, encoding="utf-8")
    if require_admin:
        args = subprocess.list2cmdline(
            [
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-WindowStyle",
                "Hidden",
                "-File",
                str(script_path),
            ]
        )
        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            "powershell.exe",
            args,
            None,
            0,
        )
        if result <= 32:
            raise OSError(f"Could not start elevated updater (ShellExecuteW code {result}).")
        return

    subprocess.Popen(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-File",
            str(script_path),
        ],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def render_frame(frame: np.ndarray, screen_w: int, screen_h: int, fit_mode: str) -> np.ndarray:
    fh, fw = frame.shape[:2]
    if fit_mode == "stretch":
        return cv2.resize(frame, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)

    if fit_mode == "cover":
        scale = max(screen_w / fw, screen_h / fh)
        new_w = int(fw * scale)
        new_h = int(fh * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        x0 = max((new_w - screen_w) // 2, 0)
        y0 = max((new_h - screen_h) // 2, 0)
        return resized[y0 : y0 + screen_h, x0 : x0 + screen_w]

    scale = min(screen_w / fw, screen_h / fh)
    new_w = int(fw * scale)
    new_h = int(fh * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x0 = (screen_w - new_w) // 2
    y0 = (screen_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def render_transition_frame(
    backdrop: np.ndarray,
    target: np.ndarray,
    transition_mode: str,
    alpha: float,
) -> np.ndarray:
    if alpha >= 1.0 or transition_mode == "none":
        return target

    if transition_mode == "fade":
        return cv2.addWeighted(backdrop, 1.0 - alpha, target, alpha, 0.0)

    h, w = target.shape[:2]
    out = backdrop.copy()

    if transition_mode == "wipe_left":
        x = max(1, min(w, int(w * alpha)))
        out[:, :x] = target[:, :x]
        return out

    if transition_mode == "wipe_down":
        y = max(1, min(h, int(h * alpha)))
        out[:y, :] = target[:y, :]
        return out

    if transition_mode == "zoom_in":
        vis_w = max(1, min(w, int(w * alpha)))
        vis_h = max(1, min(h, int(h * alpha)))
        resized = cv2.resize(target, (vis_w, vis_h), interpolation=cv2.INTER_LINEAR)
        x0 = (w - vis_w) // 2
        y0 = (h - vis_h) // 2
        out[y0 : y0 + vis_h, x0 : x0 + vis_w] = resized
        return out

    return target


def run_fullscreen_video(
    video_path: Path,
    mouse_threshold: int,
    fit_mode: str,
    startup_transition: str,
    *,
    exit_on_mouse_move: bool = True,
    max_runtime_seconds: int | None = None,
) -> None:
    if not video_path.is_file():
        show_message("VideoToScreensaver", f"Video not found:\n{video_path}")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        show_message("VideoToScreensaver", f"Could not open video:\n{video_path}")
        return

    ok, first_frame = cap.read()
    if not ok:
        cap.release()
        show_message("VideoToScreensaver", f"Could not decode video:\n{video_path}")
        return

    user32 = ctypes.windll.user32
    monitor_rects = get_monitor_rects()
    if not monitor_rects:
        monitor_rects = [(0, 0, user32.GetSystemMetrics(0), user32.GetSystemMetrics(1))]

    windows: list[tuple[str, int, int]] = []
    for index, (left, top, width, height) in enumerate(monitor_rects):
        window_name = f"VideoToScreensaverPlayer_{index}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, left, top)
        cv2.resizeWindow(window_name, width, height)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        windows.append((window_name, width, height))

    first_cache: dict[tuple[int, int], np.ndarray] = {}
    backdrop_cache: dict[tuple[int, int], np.ndarray] = {}
    wallpaper = get_wallpaper_frame()
    for window_name, screen_w, screen_h in windows:
        key_dims = (screen_w, screen_h)
        if key_dims not in first_cache:
            first_cache[key_dims] = render_frame(first_frame, screen_w, screen_h, fit_mode)
        if key_dims not in backdrop_cache:
            if wallpaper is None:
                backdrop_cache[key_dims] = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            else:
                backdrop_cache[key_dims] = render_frame(wallpaper, screen_w, screen_h, "cover")
        cv2.imshow(window_name, backdrop_cache[key_dims])
    cv2.waitKey(1)

    transition_steps = [1.0]
    if startup_transition != "none":
        transition_steps = [0.18, 0.36, 0.54, 0.72, 0.86, 1.0]

    for alpha in transition_steps:
        blended_cache: dict[tuple[int, int], np.ndarray] = {}
        for window_name, screen_w, screen_h in windows:
            key_dims = (screen_w, screen_h)
            if key_dims not in blended_cache:
                blended_cache[key_dims] = render_transition_frame(
                    backdrop_cache[key_dims],
                    first_cache[key_dims],
                    startup_transition,
                    alpha,
                )
            frame_to_show = blended_cache[key_dims]
            cv2.imshow(window_name, frame_to_show)
        cv2.waitKey(1)
        if alpha < 1.0:
            time.sleep(0.03)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 240:
        fps = 30.0
    frame_delay = 1.0 / fps

    start_mouse = get_mouse_pos()
    start_time = time.time()
    last_frame_time = time.time()

    while True:
        if exit_on_mouse_move:
            x, y = get_mouse_pos()
            if abs(x - start_mouse[0]) > mouse_threshold or abs(y - start_mouse[1]) > mouse_threshold:
                break

        if max_runtime_seconds is not None and (time.time() - start_time) >= max_runtime_seconds:
            break

        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        cache: dict[tuple[int, int], np.ndarray] = {}
        for window_name, screen_w, screen_h in windows:
            key_dims = (screen_w, screen_h)
            if key_dims not in cache:
                cache[key_dims] = render_frame(frame, screen_w, screen_h, fit_mode)
            cv2.imshow(window_name, cache[key_dims])

        key = cv2.waitKey(1)
        if key in (27, ord("q"), ord("Q")):
            break
        if key != -1 and max_runtime_seconds is None:
            break

        now = time.time()
        sleep_for = frame_delay - (now - last_frame_time)
        if sleep_for > 0:
            time.sleep(sleep_for)
        last_frame_time = time.time()

        should_exit = False
        for window_name, _, _ in windows:
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                should_exit = True
                break
        if should_exit:
            break

    cap.release()
    cv2.destroyAllWindows()


def apply_registry_settings(scr_path: Path, timeout_seconds: int) -> None:
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Control Panel\Desktop",
        0,
        winreg.KEY_SET_VALUE,
    ) as key:
        winreg.SetValueEx(key, "SCRNSAVE.EXE", 0, winreg.REG_SZ, str(scr_path))
        winreg.SetValueEx(key, "ScreenSaveActive", 0, winreg.REG_SZ, "1")
        winreg.SetValueEx(key, "ScreenSaveTimeOut", 0, winreg.REG_SZ, str(timeout_seconds))

    subprocess.run(
        ["rundll32.exe", "user32.dll,UpdatePerUserSystemParameters"],
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def get_registry_scr_path() -> Path | None:
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Control Panel\Desktop",
            0,
            winreg.KEY_READ,
        ) as key:
            value, _ = winreg.QueryValueEx(key, "SCRNSAVE.EXE")
            text = str(value).strip()
            return Path(text) if text else None
    except OSError:
        return None


def clear_registry_screensaver() -> None:
    with winreg.OpenKey(
        winreg.HKEY_CURRENT_USER,
        r"Control Panel\Desktop",
        0,
        winreg.KEY_SET_VALUE,
    ) as key:
        winreg.SetValueEx(key, "SCRNSAVE.EXE", 0, winreg.REG_SZ, "")
        winreg.SetValueEx(key, "ScreenSaveActive", 0, winreg.REG_SZ, "0")

    subprocess.run(
        ["rundll32.exe", "user32.dll,UpdatePerUserSystemParameters"],
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


def find_uninstall_command() -> str | None:
    registry_views = [0]
    if hasattr(winreg, "KEY_WOW64_64KEY"):
        registry_views.append(winreg.KEY_WOW64_64KEY)
    if hasattr(winreg, "KEY_WOW64_32KEY"):
        registry_views.append(winreg.KEY_WOW64_32KEY)

    seen: set[tuple[int, int]] = set()
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view_flag in registry_views:
            access = winreg.KEY_READ | view_flag
            key_id = (int(hive), access)
            if key_id in seen:
                continue
            seen.add(key_id)
            try:
                with winreg.OpenKey(hive, UNINSTALL_REGISTRY_PATH, 0, access) as uninstall_root:
                    subkey_count, _, _ = winreg.QueryInfoKey(uninstall_root)
                    for index in range(subkey_count):
                        subkey_name = winreg.EnumKey(uninstall_root, index)
                        try:
                            with winreg.OpenKey(uninstall_root, subkey_name) as app_key:
                                display_name, _ = winreg.QueryValueEx(app_key, "DisplayName")
                                uninstall_cmd, _ = winreg.QueryValueEx(app_key, "UninstallString")
                        except OSError:
                            continue

                        if str(display_name).strip() != APP_NAME:
                            continue
                        text = str(uninstall_cmd).strip()
                        if text:
                            return text
            except OSError:
                continue
    return None


def list_managed_screensavers() -> list[Path]:
    items: dict[str, Path] = {}

    screensaver_dir = get_system_screensaver_dir()
    if screensaver_dir.is_dir():
        for path in screensaver_dir.glob(f"*{SCR_TAG}.scr"):
            items[_norm_path_str(str(path))] = path

    if APP_DIR.is_dir():
        for path in APP_DIR.glob(f"*{SCR_TAG}.scr"):
            items[_norm_path_str(str(path))] = path

    return sorted(items.values(), key=lambda p: (p.name.lower(), str(p.parent).lower()))


def copy_with_lock_fallback(source: Path, preferred_target: Path, max_attempts: int = 20) -> Path:
    if source.resolve() == preferred_target.resolve():
        return preferred_target

    try:
        shutil.copy2(source, preferred_target)
        return preferred_target
    except PermissionError:
        pass

    timestamp = int(time.time())
    for i in range(1, max_attempts + 1):
        if preferred_target.suffix.lower() == ".scr" and preferred_target.stem.endswith(SCR_TAG):
            base_stem = preferred_target.stem[: -len(SCR_TAG)]
            candidate_name = f"{base_stem}_{timestamp}_{i}{SCR_TAG}{preferred_target.suffix}"
        else:
            candidate_name = f"{preferred_target.stem}_{timestamp}_{i}{preferred_target.suffix}"
        candidate = preferred_target.with_name(candidate_name)
        try:
            shutil.copy2(source, candidate)
            return candidate
        except PermissionError:
            continue

    raise PermissionError(
        f"Could not copy file because destination is in use:\n{preferred_target}"
    )


def install_from_gui(
    source_video: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    startup_transition: str,
) -> tuple[Path, Path]:
    if not getattr(sys, "frozen", False):
        raise RuntimeError("Build the app executable first, then run the .exe to install a screensaver.")

    current_exe = Path(sys.executable)
    result = run_self_elevated(
        [
            "--admin-install",
            str(current_exe),
            str(source_video),
            str(mouse_threshold),
            str(timeout_seconds),
            fit_mode,
            startup_transition,
        ]
    )
    if result != 0:
        raise RuntimeError("Install was canceled or failed before screensaver files were written.")

    actual_scr_path = get_registry_scr_path() or (get_system_screensaver_dir() / f"{sanitize_file_stem(source_video.stem)}{SCR_TAG}.scr")
    config = read_config(get_config_path_for_scr(actual_scr_path))
    actual_video = get_installed_video_path(config) if config else get_storage_dir_for_scr(actual_scr_path) / source_video.name
    return actual_scr_path, actual_video


def install_managed_screensaver(
    source_exe: Path,
    source_video: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    startup_transition: str,
) -> tuple[Path, Path]:
    if not source_exe.is_file():
        raise FileNotFoundError(f"App executable not found:\n{source_exe}")
    if not source_video.is_file():
        raise FileNotFoundError(f"Video file not found:\n{source_video}")
    if fit_mode not in VALID_FIT_MODES:
        raise ValueError(f"Invalid fit mode: {fit_mode}")
    if startup_transition not in VALID_STARTUP_TRANSITIONS:
        raise ValueError(f"Invalid startup transition: {startup_transition}")

    MANAGED_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    preferred_scr = get_system_screensaver_dir() / f"{sanitize_file_stem(source_video.stem)}{SCR_TAG}.scr"
    scr_path = copy_with_lock_fallback(source_exe, preferred_scr)

    storage_dir = get_storage_dir_for_scr(scr_path)
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    video_destination = storage_dir / source_video.name
    shutil.copy2(source_video, video_destination)
    write_config(source_video, video_destination, mouse_threshold, timeout_seconds, fit_mode, startup_transition, scr_path)
    apply_registry_settings(scr_path, timeout_seconds)
    return scr_path, video_destination


def save_managed_screensaver_settings(
    scr_path: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    startup_transition: str,
) -> dict:
    if not is_managed_screensaver_path(scr_path):
        raise ValueError(f"Not a managed screensaver path:\n{scr_path}")

    config_path = get_config_path_for_scr(scr_path)
    config = read_config(config_path)
    video_path = get_installed_video_path(config)
    if not video_path or not video_path.is_file():
        raise FileNotFoundError(f"Installed video not found for screensaver:\n{scr_path}")

    source_video_text = get_source_video_path(config)
    source_video = Path(source_video_text) if source_video_text else video_path
    write_config(source_video, video_path, mouse_threshold, timeout_seconds, fit_mode, startup_transition, scr_path)

    active_scr = get_registry_scr_path()
    if active_scr and _norm_path_str(str(active_scr)) == _norm_path_str(str(scr_path)):
        apply_registry_settings(scr_path, timeout_seconds)

    return read_config(config_path)


def save_legacy_screensaver_settings(
    scr_path: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    startup_transition: str,
) -> dict:
    if not is_legacy_managed_screensaver_path(scr_path):
        raise ValueError(f"Not a legacy managed screensaver path:\n{scr_path}")

    config_path = get_legacy_config_path()
    config = read_config(config_path)
    video_path = str(config.get("video_path", "")).strip()
    if not video_path:
        raise FileNotFoundError(f"Installed video not found for legacy screensaver:\n{scr_path}")

    source_video_path = str(config.get("source_video_path", "")).strip() or video_path
    updated = {
        "source_video_path": source_video_path,
        "video_path": video_path,
        "mouse_move_threshold": mouse_threshold,
        "timeout_seconds": timeout_seconds,
        "fit_mode": fit_mode,
        "startup_transition": startup_transition,
        "scr_path": str(scr_path),
    }
    save_config(updated, config_path)

    active_scr = get_registry_scr_path()
    if active_scr and _norm_path_str(str(active_scr)) == _norm_path_str(str(scr_path)):
        apply_registry_settings(scr_path, timeout_seconds)

    return read_config(config_path)


def delete_managed_screensaver(scr_path: Path) -> None:
    if not is_managed_screensaver_path(scr_path):
        raise ValueError(f"Not a managed screensaver path:\n{scr_path}")

    active_scr = get_registry_scr_path()
    if active_scr and _norm_path_str(str(active_scr)) == _norm_path_str(str(scr_path)):
        clear_registry_screensaver()

    try:
        scr_path.unlink(missing_ok=True)
    except TypeError:
        if scr_path.exists():
            scr_path.unlink()

    storage_dir = get_storage_dir_for_scr(scr_path)
    if storage_dir.exists():
        shutil.rmtree(storage_dir)


def load_seed_config() -> dict:
    candidates: list[Path] = []
    runtime_scr = get_runtime_scr_path()
    if is_managed_screensaver_path(runtime_scr):
        candidates.append(runtime_scr)

    active_scr = get_registry_scr_path()
    if is_managed_screensaver_path(active_scr) and active_scr not in candidates:
        candidates.append(active_scr)

    for scr_path in list_managed_screensavers():
        if scr_path not in candidates:
            candidates.append(scr_path)

    for scr_path in candidates:
        config = read_config(get_config_path_for_scr(scr_path))
        if config:
            return config
    return {}


def run_self_elevated(arguments: list[str]) -> int:
    if not getattr(sys, "frozen", False):
        raise RuntimeError("Install and delete actions require the built app executable.")

    def ps_quote(text: str) -> str:
        return text.replace("'", "''")

    exe_path = str(Path(sys.executable))
    arg_list = ", ".join(f"'{ps_quote(arg)}'" for arg in arguments)
    command = (
        f"$p = Start-Process -FilePath '{ps_quote(exe_path)}' -ArgumentList @({arg_list}) "
        "-Verb RunAs -Wait -PassThru; "
        "exit $p.ExitCode"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    return int(result.returncode)


def open_screen_saver_settings() -> None:
    subprocess.Popen(["control.exe", "desk.cpl,,1"])


def launch_gui() -> None:
    root = tk.Tk()
    root.title(APP_DISPLAY_NAME)
    root.geometry("980x760")
    root.minsize(940, 720)
    root.resizable(False, False)
    icon_path = get_resource_path("assets/vts_icon.ico")
    if icon_path.is_file():
        try:
            root.iconbitmap(str(icon_path))
        except Exception:
            pass

    palette = {
        "bg": "#0b0d10",
        "panel": "#12161b",
        "panel_alt": "#181d24",
        "panel_soft": "#1d232b",
        "border": "#2d3641",
        "text": "#eef2f6",
        "muted": "#8f9bab",
        "accent": "#f4a949",
        "accent_active": "#ffbe67",
        "accent_text": "#16100a",
        "danger": "#cf6679",
        "danger_active": "#e07f91",
        "select": "#243040",
        "success": "#8fd694",
    }

    root.configure(bg=palette["bg"])

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(
        ".",
        font=("Bahnschrift", 10),
        background=palette["bg"],
        foreground=palette["text"],
        fieldbackground=palette["panel_soft"],
    )
    style.configure("TFrame", background=palette["bg"])
    style.configure("Shell.TFrame", background=palette["bg"])
    style.configure("Card.TFrame", background=palette["panel"], relief="flat")
    style.configure("Inset.TFrame", background=palette["panel_alt"])
    style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
    style.configure("Card.TLabel", background=palette["panel"], foreground=palette["text"])
    style.configure("Muted.TLabel", background=palette["panel"], foreground=palette["muted"])
    style.configure(
        "Hero.TLabel",
        background=palette["panel"],
        foreground=palette["text"],
        font=("Bahnschrift SemiBold", 22),
    )
    style.configure(
        "Eyebrow.TLabel",
        background=palette["panel"],
        foreground=palette["accent"],
        font=("Bahnschrift SemiBold", 10),
    )
    style.configure(
        "Status.TLabel",
        background=palette["panel_alt"],
        foreground=palette["success"],
        font=("Bahnschrift SemiBold", 10),
    )
    style.configure(
        "Section.TLabelframe",
        background=palette["panel"],
        borderwidth=1,
        relief="solid",
        bordercolor=palette["border"],
        padding=0,
    )
    style.configure(
        "Section.TLabelframe.Label",
        background=palette["panel"],
        foreground=palette["text"],
        font=("Bahnschrift SemiBold", 11),
    )
    style.configure(
        "TEntry",
        fieldbackground=palette["panel_soft"],
        foreground=palette["text"],
        bordercolor=palette["border"],
        lightcolor=palette["border"],
        darkcolor=palette["border"],
        insertcolor=palette["text"],
        padding=8,
    )
    style.configure(
        "TSpinbox",
        fieldbackground=palette["panel_soft"],
        foreground=palette["text"],
        bordercolor=palette["border"],
        lightcolor=palette["border"],
        darkcolor=palette["border"],
        arrowsize=13,
        padding=6,
    )
    style.configure(
        "TCombobox",
        fieldbackground=palette["panel_soft"],
        foreground=palette["text"],
        bordercolor=palette["border"],
        lightcolor=palette["border"],
        darkcolor=palette["border"],
        arrowcolor=palette["accent"],
        padding=6,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette["panel_soft"])],
        selectbackground=[("readonly", palette["panel_soft"])],
        selectforeground=[("readonly", palette["text"])],
    )
    style.configure(
        "TButton",
        background=palette["panel_soft"],
        foreground=palette["text"],
        borderwidth=0,
        focusthickness=0,
        padding=(12, 9),
        font=("Bahnschrift SemiBold", 10),
    )
    style.map(
        "TButton",
        background=[("active", palette["border"]), ("pressed", palette["select"])],
        foreground=[("disabled", palette["muted"])],
    )
    style.configure("Primary.TButton", background=palette["accent"], foreground=palette["accent_text"])
    style.map(
        "Primary.TButton",
        background=[
            ("disabled", palette["panel_soft"]),
            ("active", palette["accent_active"]),
            ("pressed", palette["accent"]),
        ],
        foreground=[
            ("disabled", palette["muted"]),
            ("active", palette["accent_text"]),
            ("pressed", palette["accent_text"]),
        ],
    )
    style.configure("Danger.TButton", background=palette["danger"], foreground=palette["text"])
    style.map(
        "Danger.TButton",
        background=[("active", palette["danger_active"]), ("pressed", palette["danger"])],
    )
    style.configure("Outline.TButton", background=palette["panel"], foreground=palette["text"])
    style.map(
        "Outline.TButton",
        background=[("active", palette["panel_alt"]), ("pressed", palette["panel_soft"])],
    )
    style.configure("TSeparator", background=palette["border"])
    style.configure(
        "Vertical.TScrollbar",
        background=palette["panel_soft"],
        troughcolor=palette["bg"],
        bordercolor=palette["bg"],
        arrowcolor=palette["accent"],
        darkcolor=palette["panel_soft"],
        lightcolor=palette["panel_soft"],
    )

    selected_video = tk.StringVar(value="")
    threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
    timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT_SECONDS))
    timeout_minutes_var = tk.StringVar(value="5")
    fit_mode_var = tk.StringVar(value=DEFAULT_FIT_MODE)
    startup_transition_var = tk.StringVar(value=DEFAULT_STARTUP_TRANSITION)
    status_var = tk.StringVar(value="Select a video, then click 'Install as Screensaver'.")
    screensaver_items: list[Path] = []
    settings_baseline = {
        "mouse_move_threshold": DEFAULT_THRESHOLD,
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "fit_mode": DEFAULT_FIT_MODE,
        "startup_transition": DEFAULT_STARTUP_TRANSITION,
    }

    def refresh_install_button_state(*_args) -> None:
        if Path(selected_video.get().strip()).is_file():
            install_button.state(["!disabled"])
        else:
            install_button.state(["disabled"])

    def capture_settings_baseline(config: dict) -> None:
        try:
            settings_baseline["mouse_move_threshold"] = int(config.get("mouse_move_threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            settings_baseline["mouse_move_threshold"] = DEFAULT_THRESHOLD
        try:
            settings_baseline["timeout_seconds"] = int(config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            settings_baseline["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS

        fit_mode = str(config.get("fit_mode", DEFAULT_FIT_MODE)).lower()
        settings_baseline["fit_mode"] = fit_mode if fit_mode in VALID_FIT_MODES else DEFAULT_FIT_MODE
        transition = str(config.get("startup_transition", DEFAULT_STARTUP_TRANSITION)).lower()
        settings_baseline["startup_transition"] = (
            transition if transition in VALID_STARTUP_TRANSITIONS else DEFAULT_STARTUP_TRANSITION
        )

    def load_settings_into_form(config: dict) -> None:
        capture_settings_baseline(config)
        threshold_var.set(str(settings_baseline["mouse_move_threshold"]))
        timeout_var.set(str(settings_baseline["timeout_seconds"]))
        timeout_minutes_var.set(str(max(int(settings_baseline["timeout_seconds"] / 60), 1)))
        fit_mode_var.set(str(settings_baseline["fit_mode"]))
        startup_transition_var.set(str(settings_baseline["startup_transition"]))

    def refresh_save_button_state(*_args) -> None:
        active_scr = get_registry_scr_path()
        if not active_scr or not is_app_created_screensaver_path(active_scr):
            save_button.state(["disabled"])
            return

        try:
            threshold = int(threshold_var.get().strip())
            timeout_minutes = int(timeout_minutes_var.get().strip())
            timeout_seconds = timeout_minutes * 60
        except ValueError:
            save_button.state(["disabled"])
            return

        fit_mode = fit_mode_var.get().strip().lower()
        startup_transition = startup_transition_var.get().strip().lower()
        if fit_mode not in VALID_FIT_MODES or startup_transition not in VALID_STARTUP_TRANSITIONS:
            save_button.state(["disabled"])
            return

        has_changes = (
            threshold != settings_baseline["mouse_move_threshold"]
            or timeout_seconds != settings_baseline["timeout_seconds"]
            or fit_mode != settings_baseline["fit_mode"]
            or startup_transition != settings_baseline["startup_transition"]
        )
        if has_changes:
            save_button.state(["!disabled"])
        else:
            save_button.state(["disabled"])

    existing = load_seed_config()
    selected_video.set(get_source_video_path(existing))
    selected_video.trace_add("write", refresh_install_button_state)
    load_settings_into_form(existing)
    threshold_var.trace_add("write", refresh_save_button_state)
    timeout_minutes_var.trace_add("write", refresh_save_button_state)
    fit_mode_var.trace_add("write", refresh_save_button_state)
    startup_transition_var.trace_add("write", refresh_save_button_state)

    def browse_video() -> None:
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            selected_video.set(file_path)
            status_var.set("Video selected. Install to set it as your active screensaver.")

    def install_screensaver() -> bool:
        video_path = Path(selected_video.get().strip())
        if not video_path.is_file():
            messagebox.showerror("VideoToScreensaver", "Pick a valid video file first.")
            return False

        try:
            scr_path, copied_video = install_from_gui(
                video_path,
                DEFAULT_THRESHOLD,
                DEFAULT_TIMEOUT_SECONDS,
                DEFAULT_FIT_MODE,
                DEFAULT_STARTUP_TRANSITION,
            )
        except Exception as exc:
            messagebox.showerror("VideoToScreensaver", str(exc))
            return False

        load_settings_into_form(
            {
                "mouse_move_threshold": DEFAULT_THRESHOLD,
                "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
                "fit_mode": DEFAULT_FIT_MODE,
                "startup_transition": DEFAULT_STARTUP_TRANSITION,
            }
        )
        refresh_save_button_state()
        status_var.set(
            "Installed with default settings. Use Save Settings to update playback behavior for the active screensaver."
        )
        messagebox.showinfo(
            "Installed",
            "Screensaver installed.\n\n"
            f"SCR file: {scr_path}\n"
            f"Video copy: {copied_video}\n\n"
            "Default settings were applied.\n"
            "Use the controls below and click 'Save Settings' to change playback behavior.",
        )
        refresh_screensaver_list()
        return True

    def parse_settings_input() -> tuple[int, int, str, str] | None:
        try:
            threshold = int(threshold_var.get().strip())
            timeout_minutes = int(timeout_minutes_var.get().strip())
            timeout_seconds = timeout_minutes * 60
            timeout_var.set(str(timeout_seconds))
            if threshold < 1 or timeout_seconds < 30:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "VideoToScreensaver",
                "Mouse threshold must be >= 1 and wait minutes must be >= 1.",
            )
            return None

        fit_mode = fit_mode_var.get().strip().lower()
        if fit_mode not in VALID_FIT_MODES:
            messagebox.showerror("VideoToScreensaver", "Invalid fit mode. Choose Contain, Cover, or Stretch.")
            return None
        startup_transition = startup_transition_var.get().strip().lower()
        if startup_transition not in VALID_STARTUP_TRANSITIONS:
            messagebox.showerror(
                "VideoToScreensaver",
                "Invalid startup transition. Choose none, fade, wipe_left, wipe_down, or zoom_in.",
            )
            return None

        return threshold, timeout_seconds, fit_mode, startup_transition

    def save_settings() -> bool:
        active_scr = get_registry_scr_path()
        if not active_scr:
            messagebox.showerror("VideoToScreensaver", "No active screensaver is currently configured.")
            return False
        if not is_app_created_screensaver_path(active_scr):
            messagebox.showerror(
                "VideoToScreensaver",
                f"Active screensaver is not managed by this app:\n{active_scr}",
            )
            return False

        parsed = parse_settings_input()
        if not parsed:
            return False
        threshold, timeout_seconds, fit_mode, startup_transition = parsed

        try:
            if is_managed_screensaver_path(active_scr):
                saved_config = save_managed_screensaver_settings(
                    active_scr, threshold, timeout_seconds, fit_mode, startup_transition
                )
            else:
                saved_config = save_legacy_screensaver_settings(
                    active_scr, threshold, timeout_seconds, fit_mode, startup_transition
                )
        except Exception as exc:
            messagebox.showerror("VideoToScreensaver", f"Could not save settings:\n{exc}")
            return False

        load_settings_into_form(saved_config)
        status_var.set("Saved settings for the active screensaver.")
        messagebox.showinfo("Settings saved", f"Updated settings for:\n{active_scr.name}")
        refresh_screensaver_list()
        return True

    def preview_now() -> None:
        selected_path = Path(selected_video.get().strip()) if selected_video.get().strip() else None
        if selected_path and selected_path.is_file():
            parsed = parse_settings_input()
            if not parsed:
                return
            threshold, _timeout_seconds, fit_mode, startup_transition = parsed
            run_fullscreen_video(
                selected_path,
                threshold,
                fit_mode,
                startup_transition,
                exit_on_mouse_move=False,
                max_runtime_seconds=10,
            )
            return

        scr_path = get_registry_scr_path()
        if not scr_path:
            messagebox.showerror("VideoToScreensaver", "Select a source file or install a screensaver first.")
            return
        if not scr_path.is_file():
            messagebox.showerror("VideoToScreensaver", f"Screensaver file missing:\n{scr_path}")
            return
        subprocess.Popen([str(scr_path), "/s"])

    def set_selected_active_screensaver() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("VideoToScreensaver", "Select a screensaver from the list first.")
            return

        target = screensaver_items[selection[0]]
        if not is_app_created_screensaver_path(target):
            messagebox.showerror("VideoToScreensaver", f"Not an app-created screensaver:\n{target}")
            return

        config = read_config_for_scr(target)
        try:
            timeout_seconds = int(config.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
        except (TypeError, ValueError):
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS
        if timeout_seconds < 30:
            timeout_seconds = DEFAULT_TIMEOUT_SECONDS

        try:
            apply_registry_settings(target, timeout_seconds)
        except OSError as exc:
            messagebox.showerror("VideoToScreensaver", f"Could not set active screensaver:\n{exc}")
            return

        load_settings_into_form(config)
        selected_video.set(get_source_video_path(config))
        refresh_save_button_state()
        status_var.set(f"Set active screensaver to {target.name}.")
        refresh_screensaver_list()

    def open_settings() -> None:
        open_screen_saver_settings()

    uninstall_command = find_uninstall_command()

    def update_app() -> None:
        if not getattr(sys, "frozen", False):
            messagebox.showinfo(
                "VideoToScreensaver",
                "Updater is only available in the built .exe.\nRun from release build or installer package.",
            )
            return

        install_dir = Path(sys.executable).parent
        test_file = install_dir / ".vts_update_check.tmp"
        require_admin = False
        try:
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink(missing_ok=True)
        except OSError:
            require_admin = True

        status_var.set("Checking for updates...")
        root.update_idletasks()
        try:
            release = fetch_latest_release_info()
        except urllib.error.URLError as exc:
            status_var.set("Update check failed.")
            messagebox.showerror("VideoToScreensaver", f"Could not check updates:\n{exc}")
            return
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            status_var.set("Update check failed.")
            messagebox.showerror("VideoToScreensaver", f"Invalid update response:\n{exc}")
            return

        latest_version = release.get("version", "").strip()
        zip_url = release.get("zip_url", "").strip()
        html_url = release.get("html_url", RELEASES_PAGE_URL).strip() or RELEASES_PAGE_URL

        if not latest_version:
            status_var.set("Update check failed.")
            messagebox.showerror("VideoToScreensaver", "Could not determine latest version from GitHub.")
            return

        if not is_newer_version(latest_version, APP_VERSION):
            status_var.set("App is up to date.")
            messagebox.showinfo(
                "VideoToScreensaver",
                f"You're up to date.\nCurrent: {APP_VERSION}\nLatest: {latest_version}",
            )
            return

        if not zip_url:
            status_var.set("Update available.")
            messagebox.showinfo(
                "VideoToScreensaver",
                f"Update {latest_version} is available.\nNo zip asset found, opening release page.",
            )
            webbrowser.open(html_url)
            return

        if not messagebox.askyesno(
            "Update available",
            f"Update from {APP_VERSION} to {latest_version} now?\n\n"
            "The app will close, update files, then restart.",
        ):
            status_var.set("Update canceled.")
            return

        update_zip = APP_DIR / f"{APP_NAME}-{latest_version}.zip"
        try:
            status_var.set(f"Downloading update {latest_version}...")
            root.update_idletasks()
            download_file(zip_url, update_zip)
            if require_admin:
                status_var.set("Starting elevated updater (UAC prompt)...")
                root.update_idletasks()
            launch_self_update(update_zip, install_dir, os.getpid(), require_admin=require_admin)
        except Exception as exc:
            status_var.set("Update failed.")
            messagebox.showerror("VideoToScreensaver", f"Update failed:\n{exc}")
            return

        status_var.set("Updater started. Closing app...")
        messagebox.showinfo(
            "VideoToScreensaver",
            "Updater started.\nThe app will close now and reopen after updating.",
        )
        root.destroy()

    def uninstall_app() -> None:
        if not uninstall_command:
            messagebox.showinfo(
                "VideoToScreensaver",
                "Uninstall is only available from the installed app.\nUse Windows Installed Apps if this is a local source run.",
            )
            return

        if not messagebox.askyesno(
            "Uninstall app",
            "Uninstall VideoToScreensaver now?\n\nInstalled screensavers in Windows will keep working, but app update and uninstall controls will be removed.",
        ):
            return

        try:
            subprocess.Popen(uninstall_command)
        except OSError as exc:
            messagebox.showerror("VideoToScreensaver", f"Could not start uninstall:\n{exc}")
            return

        status_var.set("Uninstaller started. Closing app...")
        root.destroy()

    def refresh_screensaver_list() -> None:
        nonlocal screensaver_items
        screensaver_items = list_managed_screensavers()
        listbox.delete(0, tk.END)
        active_scr = get_registry_scr_path()
        active_resolved = active_scr.resolve() if active_scr else None
        for path in screensaver_items:
            marker = ""
            try:
                if active_resolved and path.resolve() == active_resolved:
                    marker = " (active)"
            except OSError:
                pass
            listbox.insert(tk.END, f"{path.name}{marker}")

    def delete_selected_screensaver() -> None:
        selection = listbox.curselection()
        if not selection:
            messagebox.showerror("VideoToScreensaver", "Select a screensaver from the list first.")
            return

        target = screensaver_items[selection[0]]
        if not messagebox.askyesno(
            "Delete screensaver",
            f"Delete this screensaver and its stored video?\n{target}\n\nThis cannot be undone.",
        ):
            return

        try:
            if is_managed_screensaver_path(target):
                result = run_self_elevated(["--admin-delete", str(target)])
                if result != 0:
                    raise RuntimeError("Delete was canceled or failed before files were removed.")
            elif is_legacy_managed_screensaver_path(target):
                active_scr = get_registry_scr_path()
                if active_scr and _norm_path_str(str(active_scr)) == _norm_path_str(str(target)):
                    clear_registry_screensaver()
                target.unlink(missing_ok=False)
            else:
                raise RuntimeError(f"Not an app-created screensaver:\n{target}")
        except PermissionError:
            messagebox.showerror(
                "VideoToScreensaver",
                "Could not delete screensaver files because they are currently in use.\nClose preview/screensaver and try again.",
            )
            return
        except (OSError, RuntimeError) as exc:
            messagebox.showerror("VideoToScreensaver", f"Delete failed:\n{exc}")
            return

        if is_legacy_managed_screensaver_path(target):
            status_var.set("Deleted selected legacy screensaver file.")
        else:
            status_var.set("Deleted selected screensaver and its stored video.")
        refresh_screensaver_list()

    shell = ttk.Frame(root, style="Shell.TFrame")
    shell.pack(fill="both", expand=True)

    scroll_canvas = tk.Canvas(
        shell,
        bg=palette["bg"],
        bd=0,
        highlightthickness=0,
        relief="flat",
    )
    scroll_canvas.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(shell, orient="vertical", command=scroll_canvas.yview, style="Vertical.TScrollbar")
    scrollbar.pack(side="right", fill="y")
    scroll_canvas.configure(yscrollcommand=scrollbar.set)

    frame = ttk.Frame(scroll_canvas, style="Shell.TFrame", padding=(20, 18, 20, 18))
    canvas_window = scroll_canvas.create_window((0, 0), window=frame, anchor="nw")

    def sync_scroll_region(_event=None) -> None:
        scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

    def sync_canvas_width(event) -> None:
        scroll_canvas.itemconfigure(canvas_window, width=event.width)

    def on_mousewheel(event) -> None:
        scroll_canvas.yview_scroll(int(-event.delta / 120), "units")

    frame.bind("<Configure>", sync_scroll_region)
    scroll_canvas.bind("<Configure>", sync_canvas_width)
    scroll_canvas.bind_all("<MouseWheel>", on_mousewheel)

    header = ttk.Frame(frame, style="Card.TFrame", padding=(18, 16, 18, 16))
    header.pack(fill="x", pady=(0, 14))

    accent_bar = tk.Canvas(
        header,
        height=6,
        highlightthickness=0,
        bd=0,
        bg=palette["panel"],
    )
    accent_bar.pack(fill="x", pady=(0, 14))
    accent_bar.create_rectangle(0, 0, 700, 6, fill=palette["accent"], outline=palette["accent"])
    accent_bar.create_rectangle(700, 0, 920, 6, fill=palette["panel_soft"], outline=palette["panel_soft"])

    ttk.Label(header, text="VIDEO TO SCREENSAVER", style="Eyebrow.TLabel").pack(anchor="w")
    ttk.Label(header, text="Turn any video into your Windows screensaver.", style="Hero.TLabel").pack(
        anchor="w", pady=(4, 6)
    )
    ttk.Label(
        header,
        text="Choose a video to install with default playback, then adjust the active screensaver settings below whenever you need to.",
        style="Muted.TLabel",
        wraplength=880,
        justify="left",
    ).pack(anchor="w")

    content = ttk.Frame(frame, style="Shell.TFrame")
    content.pack(fill="both", expand=True, pady=(0, 8))

    info_group = ttk.LabelFrame(content, text="Status + Flow", style="Section.TLabelframe", padding=(16, 14, 16, 16))
    info_group.pack(fill="x", pady=(0, 8))
    status_panel = ttk.Frame(info_group, style="Inset.TFrame", padding=(12, 10, 12, 10))
    status_panel.pack(fill="x")
    ttk.Label(status_panel, textvariable=status_var, style="Status.TLabel", wraplength=840, justify="left").pack(
        anchor="w"
    )

    steps_text = (
        "Workflow:\n"
        "1) Choose a video and click Install as Screensaver.\n"
        "2) Adjust fit, transition, wait time, and mouse threshold.\n"
        "3) Click Save Settings to update the active managed screensaver.\n"
        "4) Use Open Settings if you want to verify it in Windows."
    )
    ttk.Label(info_group, text=steps_text, style="Muted.TLabel", justify="left").pack(anchor="w", pady=(10, 0))

    source_group = ttk.LabelFrame(content, text="Source File", style="Section.TLabelframe", padding=(16, 14, 16, 16))
    source_group.pack(fill="x", pady=(0, 8))

    path_row = ttk.Frame(source_group, style="Card.TFrame")
    path_row.pack(fill="x")
    ttk.Entry(path_row, textvariable=selected_video).pack(side="left", fill="x", expand=True)
    ttk.Button(path_row, text="Browse", command=browse_video, style="Outline.TButton", width=12).pack(
        side="left", padx=(10, 0)
    )

    source_hint = ttk.Label(
        source_group,
        text="Supported: .mp4 .mov .avi .mkv .wmv",
        style="Muted.TLabel",
    )
    source_hint.pack(anchor="w", pady=(10, 0))
    install_button = ttk.Button(
        source_group,
        text="Install as Screensaver",
        command=install_screensaver,
        style="Primary.TButton",
        width=22,
    )
    install_button.pack(anchor="w", pady=(12, 0))
    refresh_install_button_state()

    screen_saver_group = ttk.LabelFrame(content, text="Behavior", style="Section.TLabelframe", padding=(16, 14, 16, 16))
    screen_saver_group.pack(fill="x", pady=(0, 8))

    screen_saver_group.columnconfigure(0, weight=1)

    controls_row = ttk.Frame(screen_saver_group, style="Card.TFrame")
    controls_row.grid(row=0, column=0, sticky="ew")
    controls_row.columnconfigure(0, weight=1)
    controls_row.columnconfigure(1, weight=1)
    controls_row.columnconfigure(2, weight=0)

    fit_row = ttk.Frame(controls_row, style="Card.TFrame")
    fit_row.grid(row=0, column=0, sticky="w")
    ttk.Label(fit_row, text="Video fit", style="Card.TLabel").pack(side="left")
    fit_mode_menu = ttk.Combobox(
        fit_row,
        textvariable=fit_mode_var,
        values=("contain", "cover", "stretch"),
        width=14,
        state="readonly",
    )
    fit_mode_menu.pack(side="left", padx=(6, 0))

    transition_row = ttk.Frame(controls_row, style="Card.TFrame")
    transition_row.grid(row=0, column=1, sticky="w", padx=(12, 0))
    ttk.Label(transition_row, text="Startup transition", style="Card.TLabel").pack(side="left")
    startup_transition_menu = ttk.Combobox(
        transition_row,
        textvariable=startup_transition_var,
        values=("none", "fade", "wipe_left", "wipe_down", "zoom_in"),
        width=14,
        state="readonly",
    )
    startup_transition_menu.pack(side="left", padx=(6, 0))

    action_row = ttk.Frame(controls_row, style="Card.TFrame")
    action_row.grid(row=0, column=2, sticky="e", padx=(12, 0))
    save_button = ttk.Button(action_row, text="Save Settings", command=save_settings, style="Primary.TButton", width=18)
    save_button.pack(side="left")
    ttk.Button(action_row, text="Open Settings", command=open_settings, style="Outline.TButton", width=14).pack(
        side="left", padx=(8, 0)
    )
    ttk.Button(action_row, text="Preview", command=preview_now, style="Outline.TButton", width=10).pack(
        side="left", padx=(8, 0)
    )

    timing_row = ttk.Frame(screen_saver_group, style="Card.TFrame")
    timing_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    timing_row.columnconfigure(0, weight=1)
    timing_row.columnconfigure(1, weight=1)

    wait_row = ttk.Frame(timing_row, style="Card.TFrame")
    wait_row.grid(row=0, column=0, sticky="w")
    ttk.Label(wait_row, text="Wait", style="Card.TLabel").pack(side="left")
    ttk.Spinbox(wait_row, from_=1, to=1440, textvariable=timeout_minutes_var, width=6).pack(side="left", padx=(6, 0))
    ttk.Label(wait_row, text="minutes", style="Card.TLabel").pack(side="left", padx=(6, 0))

    threshold_row = ttk.Frame(timing_row, style="Card.TFrame")
    threshold_row.grid(row=0, column=1, sticky="e")
    ttk.Label(threshold_row, text="Mouse threshold", style="Card.TLabel").pack(side="left")
    ttk.Entry(threshold_row, textvariable=threshold_var, width=7).pack(side="left", padx=(6, 0))

    managed_group = ttk.LabelFrame(
        content, text="Managed Screensavers", style="Section.TLabelframe", padding=(16, 14, 16, 16)
    )
    managed_group.pack(fill="both", expand=True, pady=(0, 8))

    managed_buttons = ttk.Frame(managed_group, style="Card.TFrame")
    managed_buttons.pack(fill="x", pady=(0, 8))
    ttk.Button(managed_buttons, text="Refresh", command=refresh_screensaver_list, style="Outline.TButton", width=11).pack(
        side="left"
    )
    ttk.Button(
        managed_buttons,
        text="Set selected active",
        command=set_selected_active_screensaver,
        style="Primary.TButton",
        width=17,
    ).pack(side="left", padx=(8, 0))
    ttk.Button(
        managed_buttons,
        text="Delete selected",
        command=delete_selected_screensaver,
        style="Danger.TButton",
        width=16,
    ).pack(side="left", padx=(8, 0))

    listbox = tk.Listbox(
        managed_group,
        height=9,
        font=("Bahnschrift", 10),
        bg=palette["panel_alt"],
        fg=palette["text"],
        relief="flat",
        borderwidth=0,
        highlightthickness=1,
        highlightbackground=palette["border"],
        highlightcolor=palette["accent"],
        selectbackground=palette["accent"],
        selectforeground=palette["accent_text"],
        selectborderwidth=0,
        activestyle="none",
    )
    listbox.pack(fill="both", expand=True)

    app_settings_group = ttk.LabelFrame(
        content, text="App Settings", style="Section.TLabelframe", padding=(16, 14, 16, 16)
    )
    app_settings_group.pack(fill="x")

    app_settings_row = ttk.Frame(app_settings_group, style="Card.TFrame")
    app_settings_row.pack(fill="x")
    app_settings_text = (
        "Manage application-level actions separately from installed screensavers."
        if uninstall_command
        else "Update and uninstall are available only from the installed app build."
    )
    ttk.Label(
        app_settings_row,
        text=app_settings_text,
        style="Muted.TLabel",
        justify="left",
    ).pack(side="left")
    app_action_row = ttk.Frame(app_settings_row, style="Card.TFrame")
    app_action_row.pack(side="right")

    uninstall_button = ttk.Button(
        app_action_row,
        text="Uninstall App",
        command=uninstall_app,
        style="Danger.TButton",
        width=12,
    )
    uninstall_button.pack(side="right")
    if not uninstall_command:
        uninstall_button.state(["disabled"])

    update_button = ttk.Button(
        app_action_row,
        text="Update App",
        command=update_app,
        style="Outline.TButton",
        width=12,
    )
    update_button.pack(side="right", padx=(0, 8))
    if not getattr(sys, "frozen", False):
        update_button.state(["disabled"])

    refresh_save_button_state()

    refresh_screensaver_list()

    root.mainloop()


def main() -> None:
    args = sys.argv[1:]
    if args:
        command = args[0].lower()
        if command == "--admin-install" and len(args) == 7:
            source_exe = Path(args[1])
            source_video = Path(args[2])
            mouse_threshold = int(args[3])
            timeout_seconds = int(args[4])
            fit_mode = args[5].strip().lower()
            startup_transition = args[6].strip().lower()
            install_managed_screensaver(
                source_exe,
                source_video,
                mouse_threshold,
                timeout_seconds,
                fit_mode,
                startup_transition,
            )
            return

        if command == "--admin-delete" and len(args) == 2:
            delete_managed_screensaver(Path(args[1]))
            return

    mode = normalize_screensaver_args(sys.argv[1:])

    if mode == "screensaver":
        config = read_config()
        video_path = Path(config.get("video_path", ""))
        try:
            mouse_threshold = int(config.get("mouse_move_threshold", DEFAULT_THRESHOLD))
        except (TypeError, ValueError):
            mouse_threshold = DEFAULT_THRESHOLD
        if mouse_threshold < 1:
            mouse_threshold = DEFAULT_THRESHOLD
        fit_mode = str(config.get("fit_mode", DEFAULT_FIT_MODE)).lower()
        if fit_mode not in VALID_FIT_MODES:
            fit_mode = DEFAULT_FIT_MODE
        startup_transition = str(config.get("startup_transition", DEFAULT_STARTUP_TRANSITION)).lower()
        if startup_transition not in VALID_STARTUP_TRANSITIONS:
            startup_transition = DEFAULT_STARTUP_TRANSITION
        run_fullscreen_video(video_path, mouse_threshold, fit_mode, startup_transition)
        return

    if mode == "preview":
        return

    if mode == "configure":
        launch_gui()
        return

    launch_gui()


if __name__ == "__main__":
    main()

