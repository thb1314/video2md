from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from video2md.config import PipelineConfig
from video2md.models import FrameSample, SceneSegment, TranscriptSegment


@dataclass(slots=True)
class EvidenceRecord:
    record_id: str
    kind: str
    scene_index: int | None
    source_video: str | None
    global_ts: float | None
    time_start: float | None
    time_end: float | None
    frame_id: str | None
    text: str
    confidence: float | None
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": self.kind,
            "scene_index": self.scene_index,
            "source_video": self.source_video,
            "global_ts": self.global_ts,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "frame_id": self.frame_id,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def _scene_index_for_segment(scenes: list[SceneSegment], seg: TranscriptSegment) -> int | None:
    mid = (seg.start + seg.end) / 2.0
    for scene in scenes:
        if scene.start <= mid <= scene.end:
            return scene.index
    return None


def build_evidence_records(
    scenes: list[SceneSegment],
    transcript_segments: list[TranscriptSegment],
    frame_bank: dict[str, FrameSample],
) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []

    for idx, seg in enumerate(transcript_segments, start=1):
        scene_index = _scene_index_for_segment(scenes, seg)
        records.append(
            EvidenceRecord(
                record_id=f"transcript_{idx:06d}",
                kind="transcript",
                scene_index=scene_index,
                source_video=None,
                global_ts=seg.start,
                time_start=seg.start,
                time_end=seg.end,
                frame_id=None,
                text=seg.text,
                confidence=None,
                metadata={"speaker": seg.speaker},
            )
        )

    for frame in sorted(frame_bank.values(), key=lambda item: item.global_ts):
        if frame.ocr_text:
            records.append(
                EvidenceRecord(
                    record_id=f"ocr_{frame.frame_id}",
                    kind="ocr",
                    scene_index=frame.scene_index,
                    source_video=frame.source_video,
                    global_ts=frame.global_ts,
                    time_start=None,
                    time_end=None,
                    frame_id=frame.frame_id,
                    text=frame.ocr_text,
                    confidence=None,
                    metadata={
                        "reason": frame.reason,
                        "score": frame.score,
                        "path": str(frame.image_path),
                    },
                )
            )

        if frame.vlm_summary:
            records.append(
                EvidenceRecord(
                    record_id=f"vlm_summary_{frame.frame_id}",
                    kind="vlm_summary",
                    scene_index=frame.scene_index,
                    source_video=frame.source_video,
                    global_ts=frame.global_ts,
                    time_start=None,
                    time_end=None,
                    frame_id=frame.frame_id,
                    text=frame.vlm_summary,
                    confidence=None,
                    metadata={
                        "reason": frame.reason,
                        "score": frame.score,
                        "path": str(frame.image_path),
                    },
                )
            )

        for fact_idx, fact in enumerate(frame.vlm_facts, start=1):
            records.append(
                EvidenceRecord(
                    record_id=f"vlm_fact_{frame.frame_id}_{fact_idx:02d}",
                    kind="vlm_fact",
                    scene_index=frame.scene_index,
                    source_video=frame.source_video,
                    global_ts=frame.global_ts,
                    time_start=None,
                    time_end=None,
                    frame_id=frame.frame_id,
                    text=fact,
                    confidence=None,
                    metadata={
                        "reason": frame.reason,
                        "score": frame.score,
                        "path": str(frame.image_path),
                    },
                )
            )

    for scene in scenes:
        if scene.ocr_text:
            records.append(
                EvidenceRecord(
                    record_id=f"scene_ocr_{scene.index:04d}",
                    kind="scene_ocr",
                    scene_index=scene.index,
                    source_video=scene.source_video,
                    global_ts=scene.midpoint,
                    time_start=scene.start,
                    time_end=scene.end,
                    frame_id=None,
                    text=scene.ocr_text,
                    confidence=None,
                    metadata={"frame_ids": scene.frame_ids},
                )
            )

        for insight_idx, insight in enumerate(scene.visual_insights, start=1):
            records.append(
                EvidenceRecord(
                    record_id=f"scene_insight_{scene.index:04d}_{insight_idx:02d}",
                    kind="scene_insight",
                    scene_index=scene.index,
                    source_video=scene.source_video,
                    global_ts=scene.midpoint,
                    time_start=scene.start,
                    time_end=scene.end,
                    frame_id=None,
                    text=insight,
                    confidence=None,
                    metadata={"frame_ids": scene.frame_ids},
                )
            )

    return records


def _write_jsonl(path: Path, records: list[EvidenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def _write_sqlite(path: Path, records: list[EvidenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE evidence (
                record_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                scene_index INTEGER,
                source_video TEXT,
                global_ts REAL,
                time_start REAL,
                time_end REAL,
                frame_id TEXT,
                text TEXT NOT NULL,
                confidence REAL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        conn.execute("CREATE INDEX idx_evidence_kind ON evidence(kind);")
        conn.execute("CREATE INDEX idx_evidence_scene ON evidence(scene_index);")
        conn.execute("CREATE INDEX idx_evidence_source ON evidence(source_video);")
        conn.execute("CREATE INDEX idx_evidence_ts ON evidence(global_ts);")

        conn.executemany(
            """
            INSERT INTO evidence (
                record_id, kind, scene_index, source_video, global_ts,
                time_start, time_end, frame_id, text, confidence, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            [
                (
                    r.record_id,
                    r.kind,
                    r.scene_index,
                    r.source_video,
                    r.global_ts,
                    r.time_start,
                    r.time_end,
                    r.frame_id,
                    r.text,
                    r.confidence,
                    json.dumps(r.metadata, ensure_ascii=False),
                )
                for r in records
            ],
        )

        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE evidence_fts
                USING fts5(record_id, text, content='evidence', content_rowid='rowid');
                """
            )
            conn.execute(
                """
                INSERT INTO evidence_fts(rowid, record_id, text)
                SELECT rowid, record_id, text FROM evidence;
                """
            )
        except sqlite3.DatabaseError:
            # FTS5 might not be enabled in some sqlite builds.
            pass

        conn.commit()
    finally:
        conn.close()


def write_evidence_outputs(
    config: PipelineConfig,
    scenes: list[SceneSegment],
    transcript_segments: list[TranscriptSegment],
    frame_bank: dict[str, FrameSample],
) -> tuple[int, Path, Path]:
    records = build_evidence_records(scenes, transcript_segments, frame_bank)
    jsonl_path = config.artifacts_dir() / config.evidence_jsonl_name
    db_path = config.artifacts_dir() / config.evidence_db_name

    _write_jsonl(jsonl_path, records)
    _write_sqlite(db_path, records)
    return len(records), jsonl_path, db_path


def search_evidence(db_path: Path, query: str, limit: int = 20) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        try:
            rows = conn.execute(
                """
                SELECT e.*
                FROM evidence_fts f
                JOIN evidence e ON e.rowid = f.rowid
                WHERE f.text MATCH ?
                ORDER BY e.global_ts ASC
                LIMIT ?;
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.DatabaseError:
            rows = []

        if not rows:
            rows = conn.execute(
                """
                SELECT * FROM evidence
                WHERE text LIKE ?
                ORDER BY global_ts ASC
                LIMIT ?;
                """,
                (f"%{query}%", limit),
            ).fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()
