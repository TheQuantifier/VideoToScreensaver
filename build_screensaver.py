import argparse
import json
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
        description="Build a Windows .scr screensaver from a video file."
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the input video (mp4 recommended).",
    )
    parser.add_argument(
        "--name",
        default="VideoScreensaver",
        help="Output screensaver name without extension.",
    )
    parser.add_argument(
        "--output-dir",
        default="release",
        help="Folder to place .scr + config + video.",
    )
    parser.add_argument(
        "--mouse-threshold",
        type=int,
        default=15,
        help="Pixels of mouse movement before exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    video_path = Path(args.video).resolve()
    source_script = root / "src" / "video_screensaver.py"

    if not video_path.is_file():
        raise SystemExit(f"Video file not found: {video_path}")
    if not source_script.is_file():
        raise SystemExit(f"Missing source script: {source_script}")

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
        "--noconsole",
        "--name",
        args.name,
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

    output_video = output_dir / video_path.name
    shutil.copy2(video_path, output_video)

    config_path = output_dir / f"{args.name}.json"
    config = {
        "video_path": output_video.name,
        "mouse_move_threshold": args.mouse_threshold,
    }
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Build complete:")
    print(f"- Screensaver: {scr_path}")
    print(f"- Config:      {config_path}")
    print(f"- Video copy:  {output_video}")
    print("")
    print("Install:")
    print(f'1) Right-click "{scr_path.name}" and choose Install')
    print("2) Select it in Windows Screen Saver Settings")


if __name__ == "__main__":
    main()
