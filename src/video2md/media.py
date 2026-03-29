from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def run_cmd(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MediaError(f"Command failed ({' '.join(cmd)}): {proc.stderr.strip()}")
    return proc.stdout.strip()


def _ffmpeg_bin() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise MediaError(
            "ffmpeg not found in PATH, and imageio-ffmpeg is not installed."
        ) from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _ffprobe_bin() -> str | None:
    return shutil.which("ffprobe")


def ensure_ffmpeg() -> None:
    ffmpeg = _ffmpeg_bin()
    run_cmd([ffmpeg, "-version"])


def probe_duration(video_path: Path) -> float:
    ffprobe = _ffprobe_bin()
    if ffprobe:
        out = run_cmd(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ]
        )
        try:
            return float(out)
        except ValueError as exc:
            raise MediaError(f"Unable to parse duration from ffprobe output: {out}") from exc

    try:
        import cv2
    except ImportError as exc:
        raise MediaError(
            "ffprobe is not available, and OpenCV fallback could not be imported."
        ) from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise MediaError(f"Unable to open video for duration probe: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise MediaError(f"Unable to determine fps for video: {video_path}")
    return float(frames / fps)


def extract_audio(video_path: Path, audio_path: Path) -> None:
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_bin()
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_path),
        ]
    )


def extract_frame(video_path: Path, second: float, image_path: Path) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_bin()
    run_cmd(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{second:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(image_path),
        ]
    )


def detect_scenes_pyscenedetect(video_path: Path, threshold: float) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, detect

    scenes = detect(str(video_path), ContentDetector(threshold=threshold))
    return [(start.get_seconds(), end.get_seconds()) for start, end in scenes]


def uniform_windows(duration: float, step: float) -> list[tuple[float, float]]:
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration:
        end = min(start + step, duration)
        if end > start:
            windows.append((start, end))
        start = end
    if not windows:
        windows.append((0.0, max(duration, 1.0)))
    return windows


def dump_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
