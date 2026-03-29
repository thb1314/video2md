from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from rich.console import Console

from video2md.backends.asr import ASRError, transcribe_with_info
from video2md.backends.ocr import OCRError, ocr_image
from video2md.backends.vlm import (
    VLMError,
    enrich_frame,
    infer_term_overrides_from_evidence,
    normalize_lecture_markdown_strong,
    refine_lecture_markdown,
)
from video2md.config import PipelineConfig
from video2md.evidence import write_evidence_outputs
from video2md.frame_selector import (
    dedupe_by_hash,
    image_hash_bits,
    reason_score,
    select_candidate_times,
)
from video2md.markdown import (
    render_lecture_markdown,
    render_markdown,
    write_lecture_markdown,
    write_markdown,
)
from video2md.media import (
    MediaError,
    detect_scenes_pyscenedetect,
    dump_json,
    ensure_ffmpeg,
    extract_audio,
    extract_frame,
    probe_duration,
    uniform_windows,
)
from video2md.models import FrameSample, PipelineResult, SceneSegment, TranscriptSegment
from video2md.utils import overlap

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
_OCR_PRIORITY_PATTERNS = (
    r"([\u4e00-\u9fff]{2,4})教你开公司",
    r"([\u4e00-\u9fff]{2,4})陪你开公司",
    r"我是([\u4e00-\u9fff]{2,4})",
)
_NON_NAME_SUBSTRINGS = (
    "公司",
    "股东",
    "老板",
    "课程",
    "注册",
    "资本",
    "企业",
    "经理",
    "监事",
    "董事",
)


def collect_videos(path: Path) -> list[Path]:
    resolved = path.resolve()
    if resolved.is_file():
        return [resolved]

    if not resolved.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = [p for p in sorted(resolved.rglob("*")) if p.suffix.lower() in VIDEO_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No video files found in directory: {path}")
    return files


def _safe_name(video_path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in video_path.stem)


def _align_transcript_to_scenes(
    scenes: list[SceneSegment], transcript_segments: list[TranscriptSegment]
) -> None:
    for scene in scenes:
        scene.transcript = [
            seg
            for seg in transcript_segments
            if overlap(scene.start, scene.end, seg.start, seg.end)
        ]


def _build_summary(transcript_segments: list[TranscriptSegment], max_chars: int = 220) -> str:
    if not transcript_segments:
        return "No transcript available."

    text = " ".join(seg.text.strip() for seg in transcript_segments if seg.text.strip())
    if not text:
        return "No transcript available."
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _get_scene_windows(
    config: PipelineConfig, duration: float, video_path: Path
) -> list[tuple[float, float]]:
    if config.keyframe_backend == "uniform":
        return uniform_windows(duration, config.frame_interval_sec)

    try:
        windows = detect_scenes_pyscenedetect(video_path, config.scene_threshold)
    except Exception:
        windows = []

    if not windows:
        windows = uniform_windows(duration, config.frame_interval_sec)

    return windows


def _scene_excerpt(scene: SceneSegment, max_chars: int = 500) -> str:
    transcript = " ".join(seg.text for seg in scene.transcript)
    if len(transcript) <= max_chars:
        return transcript
    return transcript[:max_chars].rstrip() + "..."


def _build_term_evidence_blocks(
    scenes: list[SceneSegment],
    max_scenes: int = 80,
    max_block_chars: int = 360,
) -> list[str]:
    blocks: list[str] = []
    for scene in scenes[:max_scenes]:
        transcript = " ".join(seg.text.strip() for seg in scene.transcript[:6] if seg.text.strip())
        if not transcript:
            continue

        parts = [
            f"scene={scene.index}",
            f"ASR: {transcript[:max_block_chars]}",
        ]
        if scene.ocr_text:
            parts.append(f"OCR: {scene.ocr_text[:max_block_chars]}")
        if scene.visual_insights:
            merged_visual = " | ".join(
                item.strip() for item in scene.visual_insights[:2] if item.strip()
            )
            if merged_visual:
                parts.append(f"VIS: {merged_visual[:max_block_chars]}")
        blocks.append("\n".join(parts))
    return blocks


def _needs_more_mining(scene: SceneSegment) -> bool:
    text_chars = sum(len(seg.text) for seg in scene.transcript)
    weak_visual = not scene.visual_insights
    weak_ocr = not scene.ocr_text
    return text_chars >= 80 and (weak_visual or weak_ocr)


def _scene_ocr_aggregate(scene: SceneSegment, frame_bank: dict[str, FrameSample]) -> str | None:
    pieces: list[str] = []
    seen: set[str] = set()
    for frame_id in scene.frame_ids:
        value = frame_bank[frame_id].ocr_text
        if not value:
            continue
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        pieces.append(normalized)
        if len(pieces) >= 6:
            break

    if not pieces:
        return None
    return " | ".join(pieces)


def _scene_visual_aggregate(scene: SceneSegment, frame_bank: dict[str, FrameSample]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()

    for frame_id in scene.frame_ids:
        frame = frame_bank[frame_id]
        if frame.vlm_summary:
            line = frame.vlm_summary.strip()
            if line and line not in seen:
                seen.add(line)
                merged.append(line)

        for fact in frame.vlm_facts:
            line = fact.strip()
            if line and line not in seen:
                seen.add(line)
                merged.append(line)

        if len(merged) >= 8:
            break

    return merged


def _choose_primary_frame(scene: SceneSegment, frame_bank: dict[str, FrameSample]) -> Path:
    if not scene.frame_ids:
        return scene.keyframe_path

    best = max(
        (frame_bank[frame_id] for frame_id in scene.frame_ids),
        key=lambda frame: frame.score + frame.ocr_char_count / 80.0,
    )
    return best.image_path


def _vlm_budget(config: PipelineConfig, total_duration: float) -> int:
    if config.max_vlm_frames_per_minute <= 0:
        return 0
    return max(1, int((total_duration / 60.0) * config.max_vlm_frames_per_minute))


def _extract_context_terms(text: str, pattern: str) -> list[str]:
    if not text:
        return []
    return [
        token.strip()
        for token in re.findall(pattern, text)
        if 2 <= len(token.strip()) <= 8
    ]


def _is_name_like_term(token: str) -> bool:
    value = token.strip()
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", value):
        return False
    return not any(bad in value for bad in _NON_NAME_SUBSTRINGS)


def _infer_ocr_term_overrides(scenes: list[SceneSegment]) -> dict[str, str]:
    """Infer typo overrides from OCR/visual hints vs transcript context."""
    votes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for scene in scenes:
        ocr_blob = " ".join(
            part.strip()
            for part in [scene.ocr_text or "", *scene.visual_insights]
            if part and part.strip()
        )
        transcript_blob = " ".join(seg.text.strip() for seg in scene.transcript if seg.text.strip())
        if not ocr_blob or not transcript_blob:
            continue

        for pattern in _OCR_PRIORITY_PATTERNS:
            right_terms = _extract_context_terms(ocr_blob, pattern)
            wrong_terms = _extract_context_terms(transcript_blob, pattern)
            if not right_terms or not wrong_terms:
                continue
            right = max(set(right_terms), key=right_terms.count)
            if not _is_name_like_term(right):
                continue
            for wrong in wrong_terms:
                if (
                    wrong == right
                    or len(wrong) != len(right)
                    or not _is_name_like_term(wrong)
                ):
                    continue
                votes[wrong][right] += 1

        # If OCR gives a clear account/brand name, use it to fix self-intro names in ASR.
        account_terms = _extract_context_terms(ocr_blob, _OCR_PRIORITY_PATTERNS[0])
        account_terms.extend(_extract_context_terms(ocr_blob, _OCR_PRIORITY_PATTERNS[1]))
        intro_terms = _extract_context_terms(transcript_blob, _OCR_PRIORITY_PATTERNS[2])
        if account_terms and intro_terms:
            right = max(set(account_terms), key=account_terms.count)
            if not _is_name_like_term(right):
                continue
            for wrong in intro_terms:
                if (
                    wrong == right
                    or len(wrong) != len(right)
                    or not _is_name_like_term(wrong)
                ):
                    continue
                votes[wrong][right] += 1

    overrides: dict[str, str] = {}
    for wrong, right_votes in votes.items():
        right = max(right_votes.items(), key=lambda item: item[1])[0]
        overrides[wrong] = right
    return overrides


def _extract_forced_terms(scenes: list[SceneSegment], max_terms: int = 80) -> list[str]:
    weighted_score: dict[str, int] = defaultdict(int)
    has_visual_evidence: dict[str, bool] = defaultdict(bool)
    for scene in scenes:
        if scene.ocr_text:
            for term in re.findall(r"[A-Za-z0-9\+\-]{2,24}|[\u4e00-\u9fff]{2,8}", scene.ocr_text):
                cleaned = term.strip()
                if len(cleaned) < 2:
                    continue
                weighted_score[cleaned] += 3
                has_visual_evidence[cleaned] = True

        for insight in scene.visual_insights[:3]:
            for term in re.findall(r"[A-Za-z0-9\+\-]{2,24}|[\u4e00-\u9fff]{2,8}", insight):
                cleaned = term.strip()
                if len(cleaned) < 2:
                    continue
                weighted_score[cleaned] += 2
                has_visual_evidence[cleaned] = True

        for seg in scene.transcript[:2]:
            blob = seg.text
            for term in re.findall(r"[A-Za-z0-9\+\-]{2,24}|[\u4e00-\u9fff]{2,8}", blob):
                cleaned = term.strip()
                if len(cleaned) < 2:
                    continue
                weighted_score[cleaned] += 1

    ranked = sorted(weighted_score.items(), key=lambda item: item[1], reverse=True)
    return [
        term
        for term, score in ranked
        if score >= 2 or has_visual_evidence.get(term, False)
    ][:max_terms]


def _apply_term_overrides(text: str, overrides: dict[str, str]) -> str:
    if not overrides:
        return text
    result = text
    # Longest key first to avoid partial shadowing.
    for wrong, right in sorted(overrides.items(), key=lambda item: len(item[0]), reverse=True):
        wrong_norm = str(wrong).strip()
        right_norm = str(right).strip()
        if not wrong_norm or not right_norm:
            continue
        result = result.replace(wrong_norm, right_norm)
    return result


def _is_vlm_triggered_frame(frame: FrameSample, config: PipelineConfig) -> bool:
    return frame.ocr_char_count >= config.ocr_trigger_min_chars or frame.reason.startswith("cue")


def _select_vlm_candidates(
    config: PipelineConfig,
    total_duration: float,
    new_frame_ids: list[str],
    frame_bank: dict[str, FrameSample],
) -> list[FrameSample]:
    base_budget = _vlm_budget(config, total_duration)
    if base_budget <= 0 or not new_frame_ids:
        return []

    fresh_frames = [
        frame_bank[frame_id]
        for frame_id in new_frame_ids
        if frame_bank[frame_id].vlm_summary is None
    ]
    if not fresh_frames:
        return []

    triggered_frames = [
        frame for frame in fresh_frames if _is_vlm_triggered_frame(frame, config)
    ]
    if not triggered_frames:
        return sorted(fresh_frames, key=lambda frame: frame.score, reverse=True)[:base_budget]

    triggered_sorted = sorted(triggered_frames, key=lambda frame: frame.score, reverse=True)
    selected: list[FrameSample] = []
    selected_ids: set[str] = set()
    selected_scene_indexes: set[int] = set()

    # Guarantee at least one triggered frame per scene.
    for frame in triggered_sorted:
        if frame.scene_index in selected_scene_indexes:
            continue
        selected.append(frame)
        selected_ids.add(frame.frame_id)
        selected_scene_indexes.add(frame.scene_index)

    budget = min(len(fresh_frames), max(base_budget, len(selected)))

    for frame in triggered_sorted:
        if len(selected) >= budget:
            break
        if frame.frame_id in selected_ids:
            continue
        selected.append(frame)
        selected_ids.add(frame.frame_id)

    if len(selected) < budget:
        for frame in sorted(fresh_frames, key=lambda frame: frame.score, reverse=True):
            if len(selected) >= budget:
                break
            if frame.frame_id in selected_ids:
                continue
            selected.append(frame)
            selected_ids.add(frame.frame_id)

    return selected


def run_continuous(
    config: PipelineConfig,
    input_path: Path,
    console: Console | None = None,
) -> PipelineResult:
    console = console or Console()
    ensure_ffmpeg()

    videos = collect_videos(input_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.assets_dir().mkdir(parents=True, exist_ok=True)
    config.artifacts_dir().mkdir(parents=True, exist_ok=True)

    all_scenes: list[SceneSegment] = []
    all_transcript_segments: list[TranscriptSegment] = []
    frame_bank: dict[str, FrameSample] = {}
    scene_video_meta: dict[int, dict[str, object]] = {}

    global_offset = 0.0
    global_scene_index = 1

    for video in videos:
        duration = probe_duration(video)
        windows = _get_scene_windows(config, duration, video)
        video_tag = _safe_name(video)
        asset_video_dir = config.assets_dir() / video_tag
        asset_video_dir.mkdir(parents=True, exist_ok=True)

        console.rule(f"Processing {video.name}")
        console.log(
            f"duration={duration:.1f}s scenes={len(windows)} offset={global_offset:.1f}s"
        )

        for _local_idx, (start, end) in enumerate(windows, start=1):
            midpoint = (global_offset + start + global_offset + end) / 2
            placeholder = asset_video_dir / f"scene_{global_scene_index:04d}_main.jpg"
            scene = SceneSegment(
                index=global_scene_index,
                start=global_offset + start,
                end=global_offset + end,
                keyframe_path=placeholder,
                transcript=[],
                source_video=video.name,
                ocr_text=None,
            )
            all_scenes.append(scene)
            scene_video_meta[scene.index] = {
                "video_path": video,
                "video_name": video.name,
                "video_tag": video_tag,
                "offset": global_offset,
                "asset_dir": asset_video_dir,
                "midpoint": midpoint,
            }
            global_scene_index += 1

        audio_path = config.artifacts_dir() / f"{video_tag}.wav"
        extract_audio(video, audio_path)

        try:
            local_transcript, asr_runtime = transcribe_with_info(audio_path, config)
            if asr_runtime and asr_runtime.used_fallback:
                console.log(
                    "ASR fallback applied: "
                    f"{asr_runtime.requested_device}/{asr_runtime.requested_compute_type} -> "
                    f"{asr_runtime.actual_device}/{asr_runtime.actual_compute_type}"
                )
            for seg in local_transcript:
                all_transcript_segments.append(
                    TranscriptSegment(
                        start=global_offset + seg.start,
                        end=global_offset + seg.end,
                        text=seg.text,
                        speaker=seg.speaker,
                    )
                )
            console.log(f"asr_segments={len(local_transcript)}")
        except ASRError as exc:
            console.log(f"ASR skipped: {exc}")

        global_offset += duration

    _align_transcript_to_scenes(all_scenes, all_transcript_segments)

    per_scene_counter: defaultdict[int, int] = defaultdict(int)
    ocr_failed = False
    vlm_failed = False

    for pass_idx in range(1, config.mining_passes + 1):
        console.rule(f"Mining pass {pass_idx}/{config.mining_passes}")

        new_frame_ids: list[str] = []
        for scene in all_scenes:
            if pass_idx > 1 and not _needs_more_mining(scene):
                continue

            meta = scene_video_meta[scene.index]
            video_path = meta["video_path"]
            offset = float(meta["offset"])
            asset_dir = meta["asset_dir"]

            candidates = select_candidate_times(
                scene.start,
                scene.end,
                scene.transcript,
                config,
                pass_idx,
            )

            existing_hashes = [frame_bank[frame_id].hash_bits for frame_id in scene.frame_ids]
            for ts, reason in candidates:
                per_scene_counter[scene.index] += 1
                local_ts = max(ts - offset, 0.0)
                frame_id = f"s{scene.index:04d}_p{pass_idx}_f{per_scene_counter[scene.index]:03d}"
                image_path = Path(asset_dir) / f"{frame_id}.jpg"

                extract_frame(Path(video_path), local_ts, image_path)
                hash_bits = image_hash_bits(image_path)
                if dedupe_by_hash(
                    existing_hashes,
                    hash_bits,
                    config.frame_dedupe_hamming_threshold,
                ):
                    continue

                existing_hashes.append(hash_bits)
                sample = FrameSample(
                    frame_id=frame_id,
                    scene_index=scene.index,
                    source_video=str(meta["video_name"]),
                    global_ts=ts,
                    local_ts=local_ts,
                    image_path=image_path,
                    reason=reason,
                    score=reason_score(reason),
                    hash_bits=hash_bits,
                )
                frame_bank[frame_id] = sample
                scene.frame_ids.append(frame_id)
                new_frame_ids.append(frame_id)

        # Ensure every scene has at least one keyframe.
        for scene in all_scenes:
            if scene.frame_ids:
                continue
            meta = scene_video_meta[scene.index]
            per_scene_counter[scene.index] += 1
            frame_id = f"s{scene.index:04d}_fallback"
            image_path = Path(meta["asset_dir"]) / f"{frame_id}.jpg"
            local_ts = max(scene.midpoint - float(meta["offset"]), 0.0)
            extract_frame(Path(meta["video_path"]), local_ts, image_path)
            sample = FrameSample(
                frame_id=frame_id,
                scene_index=scene.index,
                source_video=str(meta["video_name"]),
                global_ts=scene.midpoint,
                local_ts=local_ts,
                image_path=image_path,
                reason="fallback",
                score=0.5,
                hash_bits=image_hash_bits(image_path),
            )
            frame_bank[frame_id] = sample
            scene.frame_ids.append(frame_id)
            new_frame_ids.append(frame_id)

        if not ocr_failed:
            ocr_errors = 0
            for frame_id in new_frame_ids:
                sample = frame_bank[frame_id]
                try:
                    sample.ocr_text = ocr_image(sample.image_path, config)
                except OCRError as exc:
                    ocr_errors += 1
                    if ocr_errors <= 3:
                        console.log(f"OCR frame skipped ({sample.frame_id}): {exc}")
                    continue

                sample.ocr_char_count = len(sample.ocr_text or "")
                sample.score += min(sample.ocr_char_count / 60.0, 3.0)

            if new_frame_ids and ocr_errors == len(new_frame_ids):
                ocr_failed = True
                console.log("OCR skipped for following passes: all frame OCR attempts failed.")

        if config.vlm_backend != "none" and not vlm_failed:
            candidates = _select_vlm_candidates(
                config=config,
                total_duration=global_offset,
                new_frame_ids=new_frame_ids,
                frame_bank=frame_bank,
            )
            vlm_errors = 0
            for sample in candidates:
                scene = all_scenes[sample.scene_index - 1]
                excerpt = _scene_excerpt(scene)
                try:
                    payload = enrich_frame(
                        sample.image_path,
                        transcript_excerpt=excerpt,
                        ocr_text=sample.ocr_text,
                        config=config,
                    )
                except VLMError as exc:
                    vlm_errors += 1
                    if vlm_errors <= 3:
                        console.log(f"VLM frame skipped ({sample.frame_id}): {exc}")
                    continue

                if not payload:
                    continue
                sample.vlm_summary = str(payload.get("summary", "")).strip() or None
                sample.vlm_facts = [
                    str(item).strip() for item in payload.get("facts", []) if str(item).strip()
                ]
                sample.score += 1.0 * float(payload.get("confidence", 0.0))

            if candidates and vlm_errors == len(candidates):
                vlm_failed = True
                console.log("VLM skipped for following passes: all frame requests failed.")

        for scene in all_scenes:
            scene.ocr_text = _scene_ocr_aggregate(scene, frame_bank)
            scene.visual_insights = _scene_visual_aggregate(scene, frame_bank)
            scene.keyframe_path = _choose_primary_frame(scene, frame_bank)

    markdown = render_markdown(config, all_scenes, global_offset)
    markdown_path = write_markdown(config, markdown)

    lecture_markdown = render_lecture_markdown(config, all_scenes, global_offset)

    auto_term_overrides: dict[str, str] = {}
    if config.vlm_backend != "none":
        evidence_blocks = _build_term_evidence_blocks(all_scenes)
        try:
            auto_term_overrides = infer_term_overrides_from_evidence(
                lecture_markdown,
                evidence_blocks,
                config,
            )
            if auto_term_overrides:
                preview = ", ".join(
                    f"{wrong}->{right}" for wrong, right in list(auto_term_overrides.items())[:6]
                )
                console.log(f"VLM term overrides: {preview}")
        except VLMError as exc:
            console.log(f"VLM term override inference skipped: {exc}")

    if not auto_term_overrides:
        auto_term_overrides = _infer_ocr_term_overrides(all_scenes)
        if auto_term_overrides:
            preview = ", ".join(
                f"{wrong}->{right}" for wrong, right in list(auto_term_overrides.items())[:6]
            )
            console.log(f"OCR fallback term overrides: {preview}")

    effective_term_overrides = dict(auto_term_overrides)
    effective_term_overrides.update(config.lecture_term_overrides)
    lecture_markdown = _apply_term_overrides(lecture_markdown, effective_term_overrides)
    forced_terms = _extract_forced_terms(all_scenes)
    forced_terms.extend(effective_term_overrides.values())
    # de-duplicate while preserving order
    dedup_forced_terms: list[str] = []
    seen_terms: set[str] = set()
    for term in forced_terms:
        normalized = term.strip()
        if not normalized or normalized in seen_terms:
            continue
        seen_terms.add(normalized)
        dedup_forced_terms.append(normalized)

    if config.lecture_refine_with_vlm and config.vlm_backend != "none":
        ocr_hints: list[str] = []
        visual_hints: list[str] = []
        for scene in all_scenes:
            if scene.ocr_text:
                ocr_hints.append(scene.ocr_text)
            visual_hints.extend(scene.visual_insights[:2])
        try:
            refined = refine_lecture_markdown(
                lecture_markdown,
                ocr_hints,
                visual_hints,
                config,
                forced_terms=dedup_forced_terms[:120],
            )
            if refined:
                lecture_markdown = _apply_term_overrides(refined, effective_term_overrides)
                console.log("Lecture refined with VLM (OCR + visual hints).")
        except VLMError as exc:
            console.log(f"Lecture refinement skipped: {exc}")

    if config.lecture_strong_normalize_with_vlm and config.vlm_backend != "none":
        try:
            normalized = normalize_lecture_markdown_strong(
                lecture_markdown,
                config,
                forced_terms=dedup_forced_terms[:120],
            )
            if normalized:
                lecture_markdown = _apply_term_overrides(normalized, effective_term_overrides)
                console.log("Lecture strongly normalized with VLM.")
        except VLMError as exc:
            console.log(f"Lecture strong normalization skipped: {exc}")

    lecture_path = write_lecture_markdown(config, lecture_markdown)

    dump_json(
        {
            "source": str(input_path.resolve()),
            "videos": [str(v) for v in videos],
            "duration": global_offset,
            "mining_passes": config.mining_passes,
            "frames": [
                {
                    "frame_id": frame.frame_id,
                    "scene_index": frame.scene_index,
                    "source_video": frame.source_video,
                    "global_ts": frame.global_ts,
                    "local_ts": frame.local_ts,
                    "path": str(frame.image_path),
                    "reason": frame.reason,
                    "score": frame.score,
                    "ocr_text": frame.ocr_text,
                    "vlm_summary": frame.vlm_summary,
                    "vlm_facts": frame.vlm_facts,
                }
                for frame in frame_bank.values()
            ],
            "scenes": [
                {
                    "index": scene.index,
                    "source_video": scene.source_video,
                    "start": scene.start,
                    "end": scene.end,
                    "keyframe": str(scene.keyframe_path),
                    "frame_ids": scene.frame_ids,
                    "ocr": scene.ocr_text,
                    "visual_insights": scene.visual_insights,
                    "transcript": [
                        {
                            "start": t.start,
                            "end": t.end,
                            "text": t.text,
                            "speaker": t.speaker,
                        }
                        for t in scene.transcript
                    ],
                }
                for scene in all_scenes
            ],
        },
        config.artifacts_dir() / "scene_data.json",
    )

    evidence_count, _jsonl_path, _db_path = write_evidence_outputs(
        config,
        all_scenes,
        all_transcript_segments,
        frame_bank,
    )

    return PipelineResult(
        video_path=input_path.resolve(),
        markdown_path=markdown_path,
        lecture_path=lecture_path,
        scene_count=len(all_scenes),
        transcript_segment_count=len(all_transcript_segments),
        duration=global_offset,
        summary=_build_summary(all_transcript_segments),
        evidence_record_count=evidence_count,
    )


def run_batch(
    config: PipelineConfig,
    input_path: Path,
    console: Console | None = None,
) -> list[PipelineResult]:
    console = console or Console()
    videos = collect_videos(input_path)
    results: list[PipelineResult] = []
    output_root = config.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for video in videos:
        item_out = output_root / _safe_name(video)
        item_cfg = config.model_copy(
            update={
                "input_video": video,
                "output_dir": item_out,
                "title": video.stem,
            }
        )
        try:
            result = run_continuous(item_cfg, video, console=console)
            results.append(result)
        except (MediaError, FileNotFoundError, RuntimeError) as exc:
            console.log(f"Failed to process {video.name}: {exc}")

    return results
