from pathlib import Path

from video2md.config import PipelineConfig
from video2md.frame_selector import dedupe_by_hash, reason_score, select_candidate_times
from video2md.models import TranscriptSegment


def test_select_candidate_times_with_cues() -> None:
    config = PipelineConfig(input_video=Path("demo.mp4"))
    transcript = [
        TranscriptSegment(start=10.0, end=13.0, text="请看这里这个图表"),
        TranscriptSegment(start=18.0, end=20.0, text="普通解说"),
    ]

    points = select_candidate_times(0.0, 30.0, transcript, config, pass_index=1)
    reasons = {reason for _, reason in points}

    assert "cue_start" in reasons
    assert "cue_after" in reasons
    assert len(points) >= 4


def test_reason_score_and_hash_dedupe() -> None:
    assert reason_score("cue_start") > reason_score("dense")
    assert dedupe_by_hash([0b1111], 0b1110, threshold=1)
    assert not dedupe_by_hash([0b1111], 0b0000, threshold=1)
