import json
import os
import shutil
import subprocess
import sys
import time
import ctypes
import winreg
from ctypes import wintypes
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np


APP_NAME = "VideoToScreensaver"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / APP_NAME
SCR_NAME = f"{APP_NAME}.scr"
CONFIG_NAME = "config.json"
SCR_TAG = "_vts"
DEFAULT_THRESHOLD = 15
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_FIT_MODE = "contain"
VALID_FIT_MODES = {"contain", "cover", "stretch"}


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


def get_resource_path(relative_path: str) -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(getattr(sys, "_MEIPASS")) / relative_path
    return Path(__file__).resolve().parent.parent / relative_path


def read_config() -> dict:
    config_path = APP_DIR / CONFIG_NAME
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    (APP_DIR / CONFIG_NAME).write_text(json.dumps(config, indent=2), encoding="utf-8")


def _norm_path_str(value: str) -> str:
    return os.path.normcase(str(Path(value)))


def get_managed_entries(config: dict) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    raw = config.get("managed_entries")
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            scr = str(item.get("scr_path", "")).strip()
            video = str(item.get("video_path", "")).strip()
            if scr and video:
                entries.append({"scr_path": scr, "video_path": video})

    legacy_scr = str(config.get("scr_path", "")).strip()
    legacy_video = str(config.get("video_path", "")).strip()
    if legacy_scr and legacy_video:
        legacy_key = (_norm_path_str(legacy_scr), _norm_path_str(legacy_video))
        has_legacy = any(
            (_norm_path_str(entry["scr_path"]), _norm_path_str(entry["video_path"])) == legacy_key for entry in entries
        )
        if not has_legacy:
            entries.append({"scr_path": legacy_scr, "video_path": legacy_video})

    return entries


def upsert_managed_entry(config: dict, scr_path: Path, video_path: Path) -> None:
    entries = get_managed_entries(config)
    scr_norm = _norm_path_str(str(scr_path))
    replaced = False
    for entry in entries:
        if _norm_path_str(entry["scr_path"]) == scr_norm:
            entry["video_path"] = str(video_path)
            replaced = True
            break
    if not replaced:
        entries.append({"scr_path": str(scr_path), "video_path": str(video_path)})
    config["managed_entries"] = entries


def remove_entry_and_cleanup_video(scr_path: Path) -> str | None:
    config = read_config()
    entries = get_managed_entries(config)
    target_norm = _norm_path_str(str(scr_path))

    removed_video: str | None = None
    remaining: list[dict[str, str]] = []
    for entry in entries:
        if _norm_path_str(entry["scr_path"]) == target_norm and removed_video is None:
            removed_video = str(entry["video_path"])
            continue
        remaining.append(entry)

    config["managed_entries"] = remaining

    if _norm_path_str(str(config.get("scr_path", ""))) == target_norm:
        config["scr_path"] = ""

    if removed_video and _norm_path_str(str(config.get("video_path", ""))) == _norm_path_str(removed_video):
        config["video_path"] = remaining[-1]["video_path"] if remaining else ""

    video_delete_warning: str | None = None
    if removed_video:
        removed_video_norm = _norm_path_str(removed_video)
        still_used = any(_norm_path_str(entry["video_path"]) == removed_video_norm for entry in remaining)
        if not still_used:
            video_path = Path(removed_video)
            try:
                if video_path.is_file() and video_path.parent.resolve() == APP_DIR.resolve():
                    video_path.unlink()
            except PermissionError:
                video_delete_warning = f"Deleted screensaver, but video copy is locked:\n{video_path}"
            except OSError as exc:
                video_delete_warning = f"Deleted screensaver, but video cleanup failed:\n{exc}"

    save_config(config)
    return video_delete_warning


def write_config(
    video_path: Path,
    mouse_threshold: int,
    timeout_seconds: int,
    fit_mode: str,
    scr_path: Path,
) -> None:
    config = read_config()
    config["video_path"] = str(video_path)
    config["mouse_move_threshold"] = mouse_threshold
    config["timeout_seconds"] = timeout_seconds
    config["fit_mode"] = fit_mode
    config["scr_path"] = str(scr_path)
    upsert_managed_entry(config, scr_path, video_path)
    save_config(config)


def get_installed_scr_path() -> Path:
    return APP_DIR / SCR_NAME


def sanitize_file_stem(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in (" ", "-", "_") else "_" for ch in name).strip()
    cleaned = cleaned.replace(" ", "_")
    return cleaned[:80] if cleaned else APP_NAME


def get_active_scr_path(config: dict) -> Path:
    configured = str(config.get("scr_path", "")).strip()
    if configured:
        return Path(configured)
    return get_installed_scr_path()


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


def run_fullscreen_video(video_path: Path, mouse_threshold: int, fit_mode: str) -> None:
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
    for window_name, screen_w, screen_h in windows:
        key_dims = (screen_w, screen_h)
        if key_dims not in first_cache:
            first_cache[key_dims] = render_frame(first_frame, screen_w, screen_h, fit_mode)
        cv2.imshow(window_name, first_cache[key_dims])
    cv2.waitKey(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1 or fps > 240:
        fps = 30.0
    frame_delay = 1.0 / fps

    start_mouse = get_mouse_pos()
    last_frame_time = time.time()

    while True:
        x, y = get_mouse_pos()
        if abs(x - start_mouse[0]) > mouse_threshold or abs(y - start_mouse[1]) > mouse_threshold:
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
        if key != -1:
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


def list_managed_screensavers() -> list[Path]:
    if not APP_DIR.is_dir():
        return []
    return sorted(APP_DIR.glob(f"*{SCR_TAG}.scr"), key=lambda p: p.name.lower())


def is_onedir_runtime(exe_path: Path) -> bool:
    return (exe_path.parent / "_internal").is_dir()


def sync_onedir_runtime_assets(source_exe: Path) -> None:
    source_dir = source_exe.parent
    for item in source_dir.iterdir():
        if item.name.lower().endswith(".scr"):
            continue
        target = APP_DIR / item.name
        try:
            if item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        except PermissionError:
            # Keep existing runtime assets when files are locked by an active screensaver process.
            continue


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
) -> tuple[Path, Path]:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    video_destination = APP_DIR / source_video.name
    video_destination = copy_with_lock_fallback(source_video, video_destination)

    if not getattr(sys, "frozen", False):
        raise RuntimeError("Build the app executable first, then run the .exe to install a screensaver.")

    current_exe = Path(sys.executable)
    if is_onedir_runtime(current_exe):
        sync_onedir_runtime_assets(current_exe)
        current_exe = APP_DIR / current_exe.name

    scr_name = f"{sanitize_file_stem(source_video.stem)}{SCR_TAG}.scr"
    scr_path = APP_DIR / scr_name
    scr_path = copy_with_lock_fallback(current_exe, scr_path)

    write_config(video_destination, mouse_threshold, timeout_seconds, fit_mode, scr_path)
    apply_registry_settings(scr_path, timeout_seconds)
    return scr_path, video_destination


def open_screen_saver_settings() -> None:
    subprocess.Popen(["control.exe", "desk.cpl,,1"])


def launch_gui() -> None:
    root = tk.Tk()
    root.title("VideoToScreensaver")
    root.geometry("760x620")
    root.resizable(False, False)
    icon_path = get_resource_path("assets/vts_icon.ico")
    if icon_path.is_file():
        try:
            root.iconbitmap(str(icon_path))
        except Exception:
            pass

    selected_video = tk.StringVar(value="")
    threshold_var = tk.StringVar(value=str(DEFAULT_THRESHOLD))
    timeout_var = tk.StringVar(value=str(DEFAULT_TIMEOUT_SECONDS))
    fit_mode_var = tk.StringVar(value=DEFAULT_FIT_MODE)
    status_var = tk.StringVar(value="Select a video, then click 'Install as Screensaver'.")
    screensaver_items: list[Path] = []

    existing = read_config()
    existing_video = existing.get("video_path")
    if isinstance(existing_video, str):
        selected_video.set(existing_video)
    threshold_var.set(str(existing.get("mouse_move_threshold", DEFAULT_THRESHOLD)))
    timeout_var.set(str(existing.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)))
    existing_fit_mode = str(existing.get("fit_mode", DEFAULT_FIT_MODE)).lower()
    fit_mode_var.set(existing_fit_mode if existing_fit_mode in VALID_FIT_MODES else DEFAULT_FIT_MODE)

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

    def install_screensaver() -> None:
        video_path = Path(selected_video.get().strip())
        if not video_path.is_file():
            messagebox.showerror("VideoToScreensaver", "Pick a valid video file first.")
            return

        try:
            threshold = int(threshold_var.get().strip())
            timeout_seconds = int(timeout_var.get().strip())
            if threshold < 1 or timeout_seconds < 30:
                raise ValueError
        except ValueError:
            messagebox.showerror(
                "VideoToScreensaver",
                "Mouse threshold must be >= 1 and timeout seconds must be >= 30.",
            )
            return

        fit_mode = fit_mode_var.get().strip().lower()
        if fit_mode not in VALID_FIT_MODES:
            messagebox.showerror("VideoToScreensaver", "Invalid fit mode. Choose Contain, Cover, or Stretch.")
            return

        try:
            scr_path, copied_video = install_from_gui(video_path, threshold, timeout_seconds, fit_mode)
        except Exception as exc:
            messagebox.showerror("VideoToScreensaver", str(exc))
            return

        status_var.set(
            "Installed successfully. Next: click 'Open Screen Saver Settings', verify selection, then Apply."
        )
        messagebox.showinfo(
            "Installed",
            "Screensaver installed.\n\n"
            f"SCR file: {scr_path}\n"
            f"Video copy: {copied_video}\n\n"
            "Next steps:\n"
            "1) Click 'Open Screen Saver Settings'\n"
            "2) Ensure '<yourfilename>_vts.scr' is selected\n"
            "3) Set wait time and click Apply",
        )
        refresh_screensaver_list()

    def preview_now() -> None:
        config = read_config()
        video = config.get("video_path")
        if not video:
            messagebox.showerror("VideoToScreensaver", "Install first so the app has a saved video.")
            return
        scr_path = get_active_scr_path(config)
        if not scr_path.is_file():
            messagebox.showerror("VideoToScreensaver", f"Screensaver file missing:\n{scr_path}")
            return
        subprocess.Popen([str(scr_path), "/s"])

    def open_settings() -> None:
        open_screen_saver_settings()

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
            f"Delete this file?\n{target}\n\nThis cannot be undone.",
        ):
            return

        active_scr = get_registry_scr_path()
        try:
            if active_scr and active_scr.resolve() == target.resolve():
                clear_registry_screensaver()
            target.unlink(missing_ok=False)
        except PermissionError:
            messagebox.showerror(
                "VideoToScreensaver",
                "Could not delete file because it is currently in use.\nClose preview/screensaver and try again.",
            )
            return
        except OSError as exc:
            messagebox.showerror("VideoToScreensaver", f"Delete failed:\n{exc}")
            return

        warning = remove_entry_and_cleanup_video(target)
        if warning:
            messagebox.showwarning("VideoToScreensaver", warning)
        status_var.set("Deleted selected screensaver file and cleaned up video if unused.")
        refresh_screensaver_list()

    def delete_active_screensaver() -> None:
        active_scr = get_registry_scr_path()
        if not active_scr:
            messagebox.showerror("VideoToScreensaver", "No active screensaver is currently configured.")
            return
        if not active_scr.is_file():
            clear_registry_screensaver()
            status_var.set("Active screensaver reference was invalid and has been cleared.")
            refresh_screensaver_list()
            return

        if active_scr.parent.resolve() != APP_DIR.resolve():
            messagebox.showerror(
                "VideoToScreensaver",
                f"Active screensaver is not managed by this app:\n{active_scr}",
            )
            return

        if not messagebox.askyesno(
            "Delete active screensaver",
            f"Delete active screensaver?\n{active_scr}\n\nThis cannot be undone.",
        ):
            return

        try:
            clear_registry_screensaver()
            active_scr.unlink(missing_ok=False)
        except PermissionError:
            messagebox.showerror(
                "VideoToScreensaver",
                "Could not delete active file because it is currently in use.\nClose preview/screensaver and try again.",
            )
            return
        except OSError as exc:
            messagebox.showerror("VideoToScreensaver", f"Delete failed:\n{exc}")
            return

        warning = remove_entry_and_cleanup_video(active_scr)
        if warning:
            messagebox.showwarning("VideoToScreensaver", warning)
        status_var.set("Deleted active screensaver file and cleaned up video if unused.")
        refresh_screensaver_list()

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="Video To Screensaver", font=("Segoe UI", 18, "bold")).pack(anchor="w")
    tk.Label(
        frame,
        text="Choose a video and install it as your active Windows screensaver.",
        font=("Segoe UI", 10),
    ).pack(anchor="w", pady=(6, 14))

    path_row = tk.Frame(frame)
    path_row.pack(fill="x", pady=(0, 10))
    tk.Entry(path_row, textvariable=selected_video, font=("Segoe UI", 10)).pack(side="left", fill="x", expand=True)
    tk.Button(path_row, text="Browse", command=browse_video, width=10).pack(side="left", padx=(8, 0))

    options_row = tk.Frame(frame)
    options_row.pack(fill="x", pady=(0, 12))
    tk.Label(options_row, text="Mouse move threshold:", width=22, anchor="w").grid(row=0, column=0, sticky="w")
    tk.Entry(options_row, textvariable=threshold_var, width=8).grid(row=0, column=1, sticky="w")
    tk.Label(options_row, text="Timeout seconds:", width=16, anchor="w").grid(row=0, column=2, padx=(20, 0), sticky="w")
    tk.Entry(options_row, textvariable=timeout_var, width=8).grid(row=0, column=3, sticky="w")
    tk.Label(options_row, text="Video fit:", width=10, anchor="w").grid(row=1, column=0, sticky="w", pady=(10, 0))
    fit_mode_menu = tk.OptionMenu(options_row, fit_mode_var, "contain", "cover", "stretch")
    fit_mode_menu.config(width=10)
    fit_mode_menu.grid(row=1, column=1, sticky="w", pady=(10, 0))

    buttons = tk.Frame(frame)
    buttons.pack(fill="x", pady=(4, 12))
    tk.Button(buttons, text="Install as Screensaver", command=install_screensaver, width=24).pack(side="left")
    tk.Button(buttons, text="Preview Now", command=preview_now, width=16).pack(side="left", padx=(8, 0))
    tk.Button(buttons, text="Open Screen Saver Settings", command=open_settings, width=24).pack(side="left", padx=(8, 0))

    tk.Label(frame, textvariable=status_var, fg="#0a4d24", wraplength=620, justify="left").pack(anchor="w", pady=(0, 8))

    steps_text = (
        "After installing:\n"
        "1) Open Screen Saver Settings.\n"
        "2) Confirm '<yourfilename>_vts.scr' is selected.\n"
        "3) Set 'Wait' to your preferred idle time.\n"
        "4) Click Apply, then OK."
    )
    tk.Label(frame, text=steps_text, justify="left", font=("Segoe UI", 10)).pack(anchor="w")

    managed_frame = tk.LabelFrame(frame, text="Managed screensavers", padx=10, pady=10)
    managed_frame.pack(fill="both", expand=True, pady=(12, 0))

    managed_buttons = tk.Frame(managed_frame)
    managed_buttons.pack(fill="x", pady=(0, 8))
    tk.Button(managed_buttons, text="Refresh", command=refresh_screensaver_list, width=12).pack(side="left")
    tk.Button(managed_buttons, text="Delete Active", command=delete_active_screensaver, width=14).pack(
        side="left", padx=(8, 0)
    )
    tk.Button(managed_buttons, text="Delete selected", command=delete_selected_screensaver, width=16).pack(
        side="left", padx=(8, 0)
    )

    listbox = tk.Listbox(managed_frame, height=9)
    listbox.pack(fill="both", expand=True)

    refresh_screensaver_list()

    root.mainloop()


def main() -> None:
    mode = normalize_screensaver_args(sys.argv[1:])

    if mode == "screensaver":
        config = read_config()
        video_path = Path(config.get("video_path", ""))
        mouse_threshold = int(config.get("mouse_move_threshold", DEFAULT_THRESHOLD))
        fit_mode = str(config.get("fit_mode", DEFAULT_FIT_MODE)).lower()
        if fit_mode not in VALID_FIT_MODES:
            fit_mode = DEFAULT_FIT_MODE
        run_fullscreen_video(video_path, mouse_threshold, fit_mode)
        return

    if mode == "preview":
        return

    if mode == "configure":
        launch_gui()
        return

    launch_gui()


if __name__ == "__main__":
    main()
