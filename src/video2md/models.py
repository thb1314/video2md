from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True)
class SceneSegment:
    index: int
    start: float
    end: float
    keyframe_path: Path
    transcript: list[TranscriptSegment]
    source_video: str | None = None
    ocr_text: str | None = None
    visual_insights: list[str] = field(default_factory=list)
    frame_ids: list[str] = field(default_factory=list)

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(slots=True)
class FrameSample:
    frame_id: str
    scene_index: int
    source_video: str
    global_ts: float
    local_ts: float
    image_path: Path
    reason: str
    score: float
    ocr_text: str | None = None
    ocr_char_count: int = 0
    hash_bits: int = 0
    vlm_summary: str | None = None
    vlm_facts: list[str] = field(default_factory=list)


@dataclass(slots=True)
class PipelineResult:
    video_path: Path
    markdown_path: Path
    lecture_path: Path | None
    scene_count: int
    transcript_segment_count: int
    duration: float
    summary: str
    evidence_record_count: int = 0
