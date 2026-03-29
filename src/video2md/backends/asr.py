from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from video2md.config import PipelineConfig
from video2md.models import TranscriptSegment


class ASRError(RuntimeError):
    pass


@dataclass(frozen=True)
class ASRRuntimeInfo:
    requested_device: str
    requested_compute_type: str
    actual_device: str
    actual_compute_type: str
    used_fallback: bool


_MODEL_CACHE: dict[tuple[str, str, str, bool], tuple[object, ASRRuntimeInfo]] = {}


def _device_compute_candidates(config: PipelineConfig) -> list[tuple[str, str]]:
    requested = (config.asr_device, config.asr_compute_type)
    candidates: list[tuple[str, str]] = [requested]

    if config.asr_fallback_to_cpu and config.asr_device != "cpu":
        candidates.append(("cpu", config.asr_compute_type))
        if config.asr_compute_type != "int8":
            candidates.append(("cpu", "int8"))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _get_model(config: PipelineConfig) -> tuple[object, ASRRuntimeInfo]:
    key = (
        config.asr_model,
        config.asr_device,
        config.asr_compute_type,
        config.asr_fallback_to_cpu,
    )
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ASRError(
            "faster-whisper is not installed. Run `uv sync --extra asr` first."
        ) from exc

    errors: list[str] = []
    for device, compute_type in _device_compute_candidates(config):
        try:
            model = WhisperModel(
                config.asr_model,
                device=device,
                compute_type=compute_type,
            )
        except Exception as exc:  # pragma: no cover - backend runtime exceptions
            errors.append(f"{device}/{compute_type}: {exc}")
            continue

        runtime_info = ASRRuntimeInfo(
            requested_device=config.asr_device,
            requested_compute_type=config.asr_compute_type,
            actual_device=device,
            actual_compute_type=compute_type,
            used_fallback=(device, compute_type)
            != (config.asr_device, config.asr_compute_type),
        )
        _MODEL_CACHE[key] = (model, runtime_info)
        return model, runtime_info

    detail = "; ".join(errors) if errors else "unknown reason"
    raise ASRError(
        "ASR model initialization failed. "
        f"requested={config.asr_device}/{config.asr_compute_type}, "
        f"fallback_to_cpu={config.asr_fallback_to_cpu}, details={detail}"
    )


def transcribe_with_info(
    audio_path: Path, config: PipelineConfig
) -> tuple[list[TranscriptSegment], ASRRuntimeInfo | None]:
    if config.asr_backend == "none":
        return [], None

    if config.asr_backend != "faster-whisper":
        raise ASRError(f"Unsupported ASR backend: {config.asr_backend}")

    model, runtime_info = _get_model(config)
    segments, _info = model.transcribe(
        str(audio_path),
        language=config.language,
        vad_filter=True,
    )

    results: list[TranscriptSegment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        results.append(TranscriptSegment(start=float(seg.start), end=float(seg.end), text=text))
    return results, runtime_info


def transcribe(audio_path: Path, config: PipelineConfig) -> list[TranscriptSegment]:
    segments, _runtime = transcribe_with_info(audio_path, config)
    return segments
