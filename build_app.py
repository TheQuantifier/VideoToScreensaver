import shutil
import subprocess
import sys
from pathlib import Path


APP_NAME = "VideoToScreensaver"


def run(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    root = Path(__file__).resolve().parent
    source = root / "src" / "app.py"
    build_root = root / "build_artifacts"
    dist_dir = build_root / "dist"
    work_dir = build_root / "work"
    spec_dir = build_root / "spec"
    release_dir = root / "release"
    icon_path = root / "assets" / "vts_icon.ico"

    if not source.is_file():
        raise SystemExit(f"Missing source file: {source}")

    if build_root.exists():
        shutil.rmtree(build_root)
    build_root.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--icon",
        str(icon_path),
        "--add-data",
        f"{icon_path};assets",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        str(source),
    ]
    run(command, cwd=root)

    built_bundle = dist_dir / APP_NAME
    built_exe = built_bundle / f"{APP_NAME}.exe"
    if not built_exe.is_file():
        raise SystemExit(f"Expected output not found: {built_exe}")

    release_bundle = release_dir / APP_NAME
    try:
        if release_bundle.exists():
            shutil.rmtree(release_bundle)
        shutil.copytree(built_bundle, release_bundle)
    except PermissionError:
        raise SystemExit(
            f"Cannot overwrite {release_bundle} because it is in use. "
            "Close VideoToScreensaver.exe and run build_app.py again."
        )

    legacy_onefile = release_dir / f"{APP_NAME}.exe"
    if legacy_onefile.exists():
        try:
            legacy_onefile.unlink()
        except OSError:
            pass

    print(f"Build complete: {release_bundle / (APP_NAME + '.exe')}")
    print("Run this executable to choose a video and install it as screensaver.")


if __name__ == "__main__":
    main()
