from pathlib import Path

from video2md.config import PipelineConfig
from video2md.models import FrameSample
from video2md.pipeline import _select_vlm_candidates


def _frame(
    frame_id: str,
    scene_index: int,
    score: float,
    reason: str = "base_mid",
    ocr_chars: int = 0,
) -> FrameSample:
    return FrameSample(
        frame_id=frame_id,
        scene_index=scene_index,
        source_video="demo.mp4",
        global_ts=0.0,
        local_ts=0.0,
        image_path=Path(f"/tmp/{frame_id}.jpg"),
        reason=reason,
        score=score,
        ocr_char_count=ocr_chars,
    )


def test_select_vlm_candidates_keeps_triggered_scenes_even_if_budget_small() -> None:
    config = PipelineConfig(
        input_video=Path("demo.mp4"),
        max_vlm_frames_per_minute=1,
        ocr_trigger_min_chars=12,
    )
    frame_bank = {
        "f1": _frame("f1", scene_index=1, score=1.0, ocr_chars=20),
        "f2": _frame("f2", scene_index=2, score=0.8, reason="cue_start"),
        "f3": _frame("f3", scene_index=2, score=5.0),
    }
    picked = _select_vlm_candidates(
        config=config,
        total_duration=60.0,
        new_frame_ids=["f1", "f2", "f3"],
        frame_bank=frame_bank,
    )
    picked_ids = {item.frame_id for item in picked}
    assert "f1" in picked_ids
    assert "f2" in picked_ids


def test_select_vlm_candidates_falls_back_to_score_when_no_trigger() -> None:
    config = PipelineConfig(
        input_video=Path("demo.mp4"),
        max_vlm_frames_per_minute=1,
        ocr_trigger_min_chars=50,
    )
    frame_bank = {
        "a": _frame("a", scene_index=1, score=0.5),
        "b": _frame("b", scene_index=1, score=2.0),
    }
    picked = _select_vlm_candidates(
        config=config,
        total_duration=60.0,
        new_frame_ids=["a", "b"],
        frame_bank=frame_bank,
    )
    assert len(picked) == 1
    assert picked[0].frame_id == "b"
