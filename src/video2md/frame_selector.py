from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import cv2

from video2md.config import PipelineConfig
from video2md.models import TranscriptSegment


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _contains_cue(text: str, keywords: list[str]) -> bool:
    lowered = text.casefold()
    return any(keyword.casefold() in lowered for keyword in keywords)


def select_candidate_times(
    scene_start: float,
    scene_end: float,
    transcript: list[TranscriptSegment],
    config: PipelineConfig,
    pass_index: int,
) -> list[tuple[float, str]]:
    duration = scene_end - scene_start
    if duration <= 0:
        return []

    points: list[tuple[float, str]] = []

    # Base triplet: beginning, midpoint, ending.
    points.append((_clamp(scene_start + 0.15 * duration, scene_start, scene_end), "base_start"))
    points.append((_clamp(scene_start + 0.50 * duration, scene_start, scene_end), "base_mid"))
    points.append((_clamp(scene_start + 0.85 * duration, scene_start, scene_end), "base_end"))

    # Dense sampling for long static scenes.
    if duration >= config.dense_scene_threshold_sec:
        t = scene_start + config.dense_frame_interval_sec
        while t < scene_end:
            points.append((t, "dense"))
            t += config.dense_frame_interval_sec

    # ASR-triggered frames when language hints point to visual references.
    cue_hit = False
    for seg in transcript:
        if not _contains_cue(seg.text, config.cue_keywords):
            continue
        cue_hit = True
        points.append((_clamp(seg.start, scene_start, scene_end), "cue_start"))
        points.append((_clamp(seg.end, scene_start, scene_end), "cue_end"))
        points.append(
            (_clamp(seg.start + config.cue_window_sec, scene_start, scene_end), "cue_after")
        )

    # Extra probes in later mining passes for weak scenes.
    if pass_index > 1:
        points.append((_clamp(scene_start + 0.33 * duration, scene_start, scene_end), "pass_probe"))
        points.append((_clamp(scene_start + 0.66 * duration, scene_start, scene_end), "pass_probe"))

    # De-duplicate by time proximity.
    points = sorted(points, key=lambda item: item[0])
    deduped: list[tuple[float, str]] = []
    for ts, reason in points:
        if not deduped:
            deduped.append((ts, reason))
            continue

        last_ts, last_reason = deduped[-1]
        if ts - last_ts >= 0.8:
            deduped.append((ts, reason))
            continue

        # If timestamps are close, keep the more informative reason.
        if reason_score(reason) > reason_score(last_reason):
            deduped[-1] = (ts, reason)

    # Keep sparse scenes from exploding when no cues are present.
    if not cue_hit and len(deduped) > 12:
        deduped = deduped[:12]

    return deduped


def reason_score(reason: str) -> float:
    if reason.startswith("cue"):
        return 3.0
    if reason == "pass_probe":
        return 2.2
    if reason == "dense":
        return 1.4
    return 1.0


def image_hash_bits(image_path: Path) -> int:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    thumb = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
    mean_value = float(thumb.mean())
    bits = 0
    for idx, value in enumerate(thumb.flatten()):
        if float(value) >= mean_value:
            bits |= 1 << idx
    return bits


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def dedupe_by_hash(
    existing_hashes: Iterable[int],
    new_hash: int,
    threshold: int,
) -> bool:
    for known in existing_hashes:
        if hamming_distance(known, new_hash) <= threshold:
            return True
    return False
