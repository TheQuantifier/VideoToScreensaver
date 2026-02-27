import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(command, cwd=cwd, text=True)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Windows .scr from the current VideoToScreensaver app source."
    )
    parser.add_argument(
        "--name",
        default="VideoToScreensaver",
        help="Output screensaver name without extension.",
    )
    parser.add_argument(
        "--output-dir",
        default="release",
        help="Folder to place the generated .scr file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    source_script = root / "src" / "app.py"
    icon_path = root / "assets" / "vts_icon.ico"

    if not source_script.is_file():
        raise SystemExit(f"Missing source script: {source_script}")
    if not icon_path.is_file():
        raise SystemExit(f"Missing icon file: {icon_path}")

    build_dir = root / "build_artifacts"
    dist_dir = build_dir / "dist"
    work_dir = build_dir / "work"
    spec_dir = build_dir / "spec"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    dist_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--windowed",
        "--name",
        args.name,
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
        str(source_script),
    ]
    run_command(pyinstaller_cmd, cwd=root)

    exe_path = dist_dir / f"{args.name}.exe"
    if not exe_path.is_file():
        raise SystemExit(f"Expected build output not found: {exe_path}")

    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scr_path = output_dir / f"{args.name}.scr"
    shutil.copy2(exe_path, scr_path)

    print("Build complete:")
    print(f"- Screensaver: {scr_path}")
    print("")
    print("Next:")
    print("1) Run VideoToScreensaver.exe and click 'Install as Screensaver' for normal setup, or")
    print(f'2) Right-click "{scr_path.name}" and choose Install')


if __name__ == "__main__":
    main()
