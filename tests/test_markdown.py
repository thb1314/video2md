from pathlib import Path

from video2md.config import PipelineConfig
from video2md.markdown import render_lecture_markdown, render_markdown
from video2md.models import SceneSegment, TranscriptSegment


def test_render_markdown_basic() -> None:
    cfg = PipelineConfig(input_video=Path("demo.mp4"), output_dir=Path("output"), title="Demo")
    scene = SceneSegment(
        index=1,
        start=0,
        end=8,
        keyframe_path=Path("output/assets/demo/scene_0001.jpg"),
        transcript=[TranscriptSegment(start=1, end=3, text="hello")],
        source_video="part1.mp4",
    )
    md = render_markdown(cfg, [scene], duration=8)
    assert "# Demo" in md
    assert "Source clip" in md
    assert "hello" in md


def test_render_lecture_markdown_basic() -> None:
    cfg = PipelineConfig(input_video=Path("demo.mp4"), output_dir=Path("output"), title="Demo")
    scene = SceneSegment(
        index=1,
        start=0,
        end=8,
        keyframe_path=Path("output/assets/demo/scene_0001.jpg"),
        transcript=[
            TranscriptSegment(start=1, end=3, text="开公司第一步是明确企业类型"),
            TranscriptSegment(start=3, end=6, text="接着准备注册登记所需材料"),
        ],
        source_video="1--课程介绍.mp4",
        ocr_text="注册登记",
        visual_insights=["画面展示公司注册流程图"],
    )
    md = render_lecture_markdown(cfg, [scene], duration=8)
    assert "# Demo：视频讲义" in md
    assert "## 第 1 章：" in md
    assert "### 讲义正文" in md
    assert "开公司第一步是明确企业类型" in md
