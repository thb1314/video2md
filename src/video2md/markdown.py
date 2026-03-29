from __future__ import annotations

import re
from collections import OrderedDict
from pathlib import Path

from video2md.config import PipelineConfig
from video2md.models import PipelineResult, SceneSegment
from video2md.utils import format_ts


def render_markdown(config: PipelineConfig, scenes: list[SceneSegment], duration: float) -> str:
    title = config.title or config.input_video.stem
    lines: list[str] = [
        f"# {title}",
        "",
        "## Metadata",
        f"- Source: `{config.input_video}`",
        f"- Duration: `{format_ts(duration)}`",
        f"- ASR backend: `{config.asr_backend}`",
        f"- OCR backend: `{config.ocr_backend}`",
        f"- Keyframe backend: `{config.keyframe_backend}`",
        "",
        "## Scene Notes",
        "",
    ]

    if not scenes:
        lines.append("No scene detected.")
        return "\n".join(lines)

    for scene in scenes:
        start = format_ts(scene.start)
        end = format_ts(scene.end)
        rel_img = scene.keyframe_path.relative_to(config.output_dir)

        lines.append(f"### [{start} - {end}] Scene {scene.index:04d}")
        lines.append(f"![scene_{scene.index:04d}]({rel_img.as_posix()})")
        if scene.source_video:
            lines.append(f"- Source clip: `{scene.source_video}`")
        lines.append(f"- Evidence frames: `{len(scene.frame_ids)}`")

        if scene.ocr_text:
            lines.append(f"- OCR: {scene.ocr_text}")

        if scene.visual_insights:
            lines.append("- Visual Insights:")
            for insight in scene.visual_insights[:5]:
                lines.append(f"  - {insight}")

        if scene.transcript:
            lines.append("- Transcript:")
            for seg in scene.transcript:
                lines.append(f"  - [{format_ts(seg.start)}] {seg.text}")
        else:
            lines.append("- Transcript: (none)")

        lines.append("")

    return "\n".join(lines)


def write_markdown(config: PipelineConfig, content: str) -> Path:
    output_path = config.output_dir / config.markdown_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def write_lecture_markdown(config: PipelineConfig, content: str) -> Path:
    output_path = config.output_dir / config.lecture_markdown_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def _normalize_text(text: str) -> str:
    normalized = " ".join(text.strip().split())
    normalized = normalized.replace(" ,", ",").replace(" .", ".")
    return normalized


def _split_sentences(text: str) -> list[str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    pieces = re.split(r"(?<=[。！？!?；;])", cleaned)
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 6]


def _chapter_title(raw_source_name: str | None, chapter_index: int) -> str:
    if not raw_source_name:
        return f"章节 {chapter_index}"

    stem = Path(raw_source_name).stem
    stem = re.sub(r"^\d+\s*[-_—]*\s*", "", stem)
    stem = stem.replace("_", " ").replace("【", "").replace("】", "")
    stem = re.sub(r"\s+", " ", stem).strip(" -_")
    return stem or f"章节 {chapter_index}"


def _group_scenes_by_source(scenes: list[SceneSegment]) -> list[tuple[str, list[SceneSegment]]]:
    grouped: OrderedDict[str, list[SceneSegment]] = OrderedDict()
    for scene in scenes:
        key = scene.source_video or "continuous"
        grouped.setdefault(key, []).append(scene)
    return list(grouped.items())


def _collect_chapter_transcript(scenes: list[SceneSegment]) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for scene in scenes:
        for seg in scene.transcript:
            text = _normalize_text(seg.text)
            if len(text) < 6 or text in seen:
                continue
            seen.add(text)
            lines.append(text)
    return lines


def _collect_chapter_points(scenes: list[SceneSegment], limit: int = 8) -> list[str]:
    points: list[str] = []
    seen: set[str] = set()

    for scene in scenes:
        candidates: list[str] = []
        candidates.extend(scene.visual_insights[:4])
        if scene.ocr_text:
            candidates.extend(piece.strip() for piece in scene.ocr_text.split("|"))
        for seg in scene.transcript[:2]:
            candidates.extend(_split_sentences(seg.text)[:1])

        for item in candidates:
            text = _normalize_text(item)
            if len(text) < 8 or text in seen:
                continue
            seen.add(text)
            points.append(text)
            if len(points) >= limit:
                return points

    return points


def _lines_to_paragraphs(
    lines: list[str], paragraph_chars: int, max_paragraphs: int
) -> list[str]:
    if not lines:
        return []

    paragraphs: list[str] = []
    buffer: list[str] = []
    buffer_chars = 0

    for line in lines:
        sentence = line
        if sentence[-1] not in "。！？!?；;":
            sentence += "。"

        if buffer and buffer_chars + len(sentence) > paragraph_chars:
            paragraphs.append("".join(buffer))
            if len(paragraphs) >= max_paragraphs:
                return paragraphs
            buffer = []
            buffer_chars = 0

        buffer.append(sentence)
        buffer_chars += len(sentence)

    if buffer and len(paragraphs) < max_paragraphs:
        paragraphs.append("".join(buffer))

    return paragraphs


def render_lecture_markdown(
    config: PipelineConfig, scenes: list[SceneSegment], duration: float
) -> str:
    title = config.title or config.input_video.stem
    chapter_groups = _group_scenes_by_source(scenes)
    source_count = len({scene.source_video for scene in scenes if scene.source_video})
    transcript_count = sum(len(scene.transcript) for scene in scenes)

    lines: list[str] = [
        f"# {title}：视频讲义",
        "",
        "## 内容概览",
        f"- 视频来源：`{config.input_video}`",
        f"- 总时长：`{format_ts(duration)}`",
        f"- 连续视频数：`{source_count}`",
        f"- 章节数：`{len(chapter_groups)}`",
        f"- 转写片段数：`{transcript_count}`",
        "",
    ]

    if not scenes:
        lines.append("未检测到可用场景。")
        return "\n".join(lines)

    if transcript_count == 0:
        lines.append("> 当前未获得语音转写，正文将以 OCR/VLM 视觉线索为主。")
        lines.append("")

    for chapter_index, (source_name, chapter_scenes) in enumerate(chapter_groups, start=1):
        chapter_title = _chapter_title(source_name, chapter_index)
        chapter_start = min(scene.start for scene in chapter_scenes)
        chapter_end = max(scene.end for scene in chapter_scenes)
        transcript_lines = _collect_chapter_transcript(chapter_scenes)
        key_points = _collect_chapter_points(chapter_scenes)

        lines.append(f"## 第 {chapter_index} 章：{chapter_title}")
        lines.append(
            f"时间范围：`{format_ts(chapter_start)} - {format_ts(chapter_end)}` | "
            f"场景数：`{len(chapter_scenes)}`"
        )
        lines.append("")

        if key_points:
            lines.append("### 本章要点")
            for point in key_points:
                lines.append(f"- {point}")
            lines.append("")

        lines.append("### 讲义正文")
        if transcript_lines:
            paragraphs = _lines_to_paragraphs(
                transcript_lines,
                paragraph_chars=config.lecture_paragraph_chars,
                max_paragraphs=config.lecture_max_paragraphs_per_chapter,
            )
            for paragraph in paragraphs:
                lines.append(paragraph)
                lines.append("")
        elif key_points:
            lines.append("本章暂无可用转写，依据画面识别信息整理如下：")
            lines.append("")
            for point in key_points:
                lines.append(point if point[-1] in "。！？!?；;" else f"{point}。")
            lines.append("")
        else:
            lines.append("本章暂无可用内容。")
            lines.append("")

    return "\n".join(lines)


def render_index_markdown(root_output: Path, results: list[PipelineResult]) -> str:
    lines = [
        "# Video Batch Index",
        "",
        f"- Total files: {len(results)}",
        "",
    ]
    for item in results:
        rel_notes = item.markdown_path.relative_to(root_output)
        lines.append(f"- [{item.video_path.name}]({rel_notes.as_posix()})")
        lines.append(
            "  - "
            f"Duration: `{format_ts(item.duration)}` | "
            f"scenes: `{item.scene_count}` | "
            f"ASR segments: `{item.transcript_segment_count}`"
        )
        if item.lecture_path:
            rel_lecture = item.lecture_path.relative_to(root_output)
            lines.append(f"  - Lecture: [{rel_lecture.name}]({rel_lecture.as_posix()})")
        lines.append(f"  - Summary: {item.summary}")
    lines.append("")
    return "\n".join(lines)
