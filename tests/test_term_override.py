from pathlib import Path

from video2md.cli import _parse_term_overrides
from video2md.models import SceneSegment, TranscriptSegment
from video2md.pipeline import _apply_term_overrides, _infer_ocr_term_overrides


def test_apply_term_overrides_basic() -> None:
    text = "我是小惠，账号叫小惠教你开公司。"
    fixed = _apply_term_overrides(text, {"小惠": "小辉"})
    assert fixed == "我是小辉，账号叫小辉教你开公司。"


def test_parse_term_overrides() -> None:
    parsed = _parse_term_overrides(["小惠=小辉", "李子奇=李子柒"])
    assert parsed["小惠"] == "小辉"
    assert parsed["李子奇"] == "李子柒"


def test_infer_ocr_term_overrides_from_account_name_context() -> None:
    scene = SceneSegment(
        index=1,
        start=0.0,
        end=8.0,
        keyframe_path=Path("scene_0001.jpg"),
        transcript=[
            TranscriptSegment(start=0.5, end=2.0, text="先简单做个自我介绍，我是小惠"),
            TranscriptSegment(
                start=2.0,
                end=4.0,
                text="我在抖音上运营一个账号叫小惠教你开公司",
            ),
        ],
        ocr_text="宁小辉#小辉教你开公司 企业服务专家",
    )

    inferred = _infer_ocr_term_overrides([scene])
    assert inferred["小惠"] == "小辉"


def test_infer_ocr_term_overrides_skips_non_name_tokens() -> None:
    scene = SceneSegment(
        index=2,
        start=0.0,
        end=8.0,
        keyframe_path=Path("scene_0002.jpg"),
        transcript=[TranscriptSegment(start=0.0, end=2.0, text="我是公司的股")],
        ocr_text="我是公司股东",
    )

    inferred = _infer_ocr_term_overrides([scene])
    assert inferred == {}
