from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class PipelineConfig(BaseModel):
    input_video: Path
    output_dir: Path = Path("output")
    title: str | None = None
    language: str | None = "zh"

    asr_backend: str = Field(default="faster-whisper", pattern="^(faster-whisper|none)$")
    asr_model: str = "small"
    asr_device: str = "auto"
    asr_compute_type: str = "int8"
    asr_fallback_to_cpu: bool = True

    keyframe_backend: str = Field(default="pyscenedetect", pattern="^(pyscenedetect|uniform)$")
    frame_interval_sec: float = Field(default=30.0, gt=0)
    scene_threshold: float = Field(default=27.0, gt=0)

    ocr_backend: str = Field(default="none", pattern="^(paddleocr|rapidocr|none)$")
    ocr_lang: str = "ch"
    rapidocr_use_cuda: bool = False

    mining_passes: int = Field(default=1, ge=1, le=5)
    dense_scene_threshold_sec: float = Field(default=30.0, gt=0)
    dense_frame_interval_sec: float = Field(default=10.0, gt=0)
    cue_window_sec: float = Field(default=2.0, gt=0)
    frame_dedupe_hamming_threshold: int = Field(default=4, ge=0, le=64)
    max_vlm_frames_per_minute: int = Field(default=4, ge=0, le=20)
    ocr_trigger_min_chars: int = Field(default=12, ge=0)
    cue_keywords: list[str] = Field(
        default_factory=lambda: [
            "看这里",
            "如下图",
            "这个图",
            "这个表",
            "这张图",
            "注意这里",
            "如图",
            "look at",
            "as shown",
            "this chart",
            "this table",
            "this image",
        ]
    )

    vlm_backend: str = Field(default="none", pattern="^(none|openai|siliconflow)$")
    vlm_model: str = "Qwen/Qwen3-VL-32B-Instruct"
    vlm_base_url: str | None = None
    vlm_api_key_env: str = "OPENAI_API_KEY"
    vlm_timeout_sec: float = Field(default=45.0, gt=0)
    vlm_max_tokens: int = Field(default=512, ge=64, le=4096)
    vlm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    vlm_top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    vlm_use_env_proxy: bool = False

    siliconflow_api_url: str = "https://api.siliconflow.cn/v1/chat/completions"
    siliconflow_api_key_file: Path | None = Path("apptoken.txt")

    markdown_name: str = "video_notes.md"
    lecture_markdown_name: str = "video_lecture.md"
    lecture_paragraph_chars: int = Field(default=320, ge=120, le=1200)
    lecture_max_paragraphs_per_chapter: int = Field(default=24, ge=4, le=200)
    lecture_refine_with_vlm: bool = True
    lecture_refine_input_chars: int = Field(default=24000, ge=4000, le=120000)
    lecture_refine_max_tokens: int = Field(default=2200, ge=256, le=8192)
    lecture_strong_normalize_with_vlm: bool = True
    lecture_strong_chunk_chars: int = Field(default=2200, ge=600, le=6000)
    lecture_term_overrides: dict[str, str] = Field(default_factory=dict)
    evidence_jsonl_name: str = "evidence.jsonl"
    evidence_db_name: str = "evidence.db"

    def assets_dir(self) -> Path:
        return self.output_dir / "assets"

    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"
