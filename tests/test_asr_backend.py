from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from video2md.backends import asr as asr_backend
from video2md.config import PipelineConfig


class _Segment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


def test_transcribe_with_info_fallback_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[tuple[str, str, str]] = []

    class _WhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:
            attempts.append((model, device, compute_type))
            if device == "cuda":
                raise RuntimeError("cuda unavailable")
            if device == "cpu" and compute_type == "float16":
                raise RuntimeError("float16 unsupported on cpu")

        def transcribe(
            self, audio_path: str, language: str | None = None, vad_filter: bool = True
        ) -> tuple[list[_Segment], object]:  # noqa: ARG002
            return ([_Segment(0.0, 1.0, "测试")], object())

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_WhisperModel)
    )
    asr_backend._MODEL_CACHE.clear()  # noqa: SLF001

    cfg = PipelineConfig(
        input_video=Path("demo.mp4"),
        asr_backend="faster-whisper",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_fallback_to_cpu=True,
    )
    segments, runtime = asr_backend.transcribe_with_info(Path("/tmp/demo.wav"), cfg)

    assert [s.text for s in segments] == ["测试"]
    assert runtime is not None
    assert runtime.used_fallback is True
    assert runtime.actual_device == "cpu"
    assert runtime.actual_compute_type == "int8"
    assert attempts == [
        ("small", "cuda", "float16"),
        ("small", "cpu", "float16"),
        ("small", "cpu", "int8"),
    ]


def test_transcribe_with_info_no_fallback_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _WhisperModel:
        def __init__(self, model: str, device: str, compute_type: str) -> None:  # noqa: ARG002
            raise RuntimeError("init failed")

    monkeypatch.setitem(
        sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=_WhisperModel)
    )
    asr_backend._MODEL_CACHE.clear()  # noqa: SLF001

    cfg = PipelineConfig(
        input_video=Path("demo.mp4"),
        asr_backend="faster-whisper",
        asr_device="cuda",
        asr_compute_type="float16",
        asr_fallback_to_cpu=False,
    )
    with pytest.raises(asr_backend.ASRError, match="fallback_to_cpu=False"):
        asr_backend.transcribe_with_info(Path("/tmp/demo.wav"), cfg)
