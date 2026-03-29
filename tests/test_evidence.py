from pathlib import Path

from video2md.config import PipelineConfig
from video2md.evidence import search_evidence, write_evidence_outputs
from video2md.models import FrameSample, SceneSegment, TranscriptSegment


def test_evidence_outputs_and_search(tmp_path: Path) -> None:
    config = PipelineConfig(input_video=Path("demo.mp4"), output_dir=tmp_path)

    scene = SceneSegment(
        index=1,
        start=0.0,
        end=10.0,
        keyframe_path=tmp_path / "assets" / "f.jpg",
        transcript=[TranscriptSegment(start=0.0, end=3.0, text="公司注册流程")],
        source_video="part1.mp4",
        ocr_text="注册资本 100 万",
        visual_insights=["画面展示公司注册流程图"],
        frame_ids=["s0001_p1_f001"],
    )

    frame_bank = {
        "s0001_p1_f001": FrameSample(
            frame_id="s0001_p1_f001",
            scene_index=1,
            source_video="part1.mp4",
            global_ts=2.0,
            local_ts=2.0,
            image_path=tmp_path / "assets" / "f.jpg",
            reason="cue_start",
            score=3.5,
            ocr_text="注册资本",
            ocr_char_count=4,
            hash_bits=123,
            vlm_summary="这是一张公司注册流程图",
            vlm_facts=["包含注册资本信息"],
        )
    }

    transcripts = [TranscriptSegment(start=0.0, end=3.0, text="公司注册流程")]

    count, jsonl_path, db_path = write_evidence_outputs(config, [scene], transcripts, frame_bank)
    assert count >= 5
    assert jsonl_path.exists()
    assert db_path.exists()

    rows = search_evidence(db_path, "注册")
    assert rows
    assert any("注册" in str(row.get("text", "")) for row in rows)
