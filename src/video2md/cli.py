from __future__ import annotations

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from video2md.config import PipelineConfig
from video2md.evidence import search_evidence
from video2md.markdown import render_index_markdown
from video2md.pipeline import run_batch, run_continuous

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("Config file must be a YAML object")
    return raw


def _parse_term_overrides(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    result: dict[str, str] = {}
    for raw in items:
        token = raw.strip()
        if "=" not in token:
            raise typer.BadParameter("term override must be `wrong=right`")
        wrong, right = token.split("=", 1)
        wrong = wrong.strip()
        right = right.strip()
        if not wrong or not right:
            raise typer.BadParameter("term override must be `wrong=right`")
        result[wrong] = right
    return result


@app.command()
def run(
    input_path: Path = typer.Option(..., "--input", "-i", help="Video file or directory path"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory"),
    title: str | None = typer.Option(None, help="Document title (single/continuous mode)"),
    mode: str = typer.Option(
        "auto",
        help="`auto` (default), `continuous` (merge directory into one timeline), or `batch`",
    ),
    config_file: Path | None = typer.Option(None, "--config", help="YAML config path"),
    asr_backend: str = typer.Option("faster-whisper", help="faster-whisper or none"),
    asr_model: str = typer.Option("small", help="faster-whisper model size"),
    asr_device: str = typer.Option("auto", help="ASR device: auto/cpu/cuda"),
    asr_compute_type: str = typer.Option("int8", help="ASR compute type"),
    asr_fallback_to_cpu: bool = typer.Option(
        True,
        help="Fallback ASR to CPU if requested device initialization fails",
    ),
    keyframe_backend: str = typer.Option("pyscenedetect", help="pyscenedetect or uniform"),
    frame_interval_sec: float = typer.Option(30.0, help="uniform mode frame interval"),
    scene_threshold: float = typer.Option(27.0, help="pyscenedetect threshold"),
    ocr_backend: str = typer.Option("none", help="paddleocr/rapidocr or none"),
    ocr_lang: str = typer.Option("ch", help="PaddleOCR language code"),
    rapidocr_use_cuda: bool = typer.Option(
        False,
        help="Enable CUDAExecutionProvider for rapidocr (requires onnxruntime-gpu)",
    ),
    language: str = typer.Option("zh", help="ASR language, e.g. zh/en/ja"),
    mining_passes: int = typer.Option(1, help="Iterative mining passes (1-5)"),
    max_vlm_frames_per_minute: int = typer.Option(
        4, help="VLM budget per minute of video in each pass"
    ),
    ocr_trigger_min_chars: int = typer.Option(
        12, help="Run VLM preference when OCR chars >= this value"
    ),
    vlm_backend: str = typer.Option("none", help="none/openai/siliconflow"),
    vlm_model: str = typer.Option(
        "Qwen/Qwen3-VL-32B-Instruct",
        help="Vision model name for VLM backend",
    ),
    vlm_base_url: str | None = typer.Option(
        None,
        help="OpenAI-compatible base URL, e.g. http://localhost:8000/v1",
    ),
    vlm_api_key_env: str = typer.Option(
        "OPENAI_API_KEY", help="Env var name for the VLM API key"
    ),
    vlm_use_env_proxy: bool = typer.Option(
        False, help="Whether VLM HTTP client should use env proxy vars"
    ),
    siliconflow_api_url: str = typer.Option(
        "https://api.siliconflow.cn/v1/chat/completions",
        help="SiliconFlow chat completions endpoint",
    ),
    siliconflow_api_key_file: Path | None = typer.Option(
        Path("apptoken.txt"),
        help="SiliconFlow key file path (fallback if env key is absent)",
    ),
    lecture_markdown_name: str = typer.Option(
        "video_lecture.md",
        help="Output filename for lecture-style article",
    ),
    lecture_paragraph_chars: int = typer.Option(
        320,
        help="Approximate characters per lecture paragraph",
    ),
    lecture_max_paragraphs_per_chapter: int = typer.Option(
        24,
        help="Maximum paragraphs in each lecture chapter",
    ),
    lecture_refine_with_vlm: bool = typer.Option(
        True,
        help="Use VLM + OCR/visual hints to polish lecture wording",
    ),
    lecture_refine_input_chars: int = typer.Option(
        24000,
        help="Max lecture draft chars sent to the refinement model",
    ),
    lecture_refine_max_tokens: int = typer.Option(
        2200,
        help="Max output tokens for lecture refinement",
    ),
    lecture_strong_normalize_with_vlm: bool = typer.Option(
        True,
        help="Run an extra strict normalization pass on lecture markdown with VLM",
    ),
    lecture_strong_chunk_chars: int = typer.Option(
        2200,
        help="Chunk size (chars) for strict lecture normalization pass",
    ),
    term_override: list[str] | None = typer.Option(
        None,
        "--term-override",
        help="Force term correction: `wrong=right`, repeatable",
    ),
) -> None:
    file_config = _load_config(config_file)
    file_term_overrides = dict(file_config.pop("lecture_term_overrides", {}) or {})
    cli_term_overrides = _parse_term_overrides(term_override)
    merged_term_overrides = dict(file_term_overrides)
    merged_term_overrides.update(cli_term_overrides)

    config = PipelineConfig(
        input_video=input_path,
        output_dir=output_dir,
        title=title,
        language=language,
        asr_backend=asr_backend,
        asr_model=asr_model,
        asr_device=asr_device,
        asr_compute_type=asr_compute_type,
        asr_fallback_to_cpu=asr_fallback_to_cpu,
        keyframe_backend=keyframe_backend,
        frame_interval_sec=frame_interval_sec,
        scene_threshold=scene_threshold,
        ocr_backend=ocr_backend,
        ocr_lang=ocr_lang,
        rapidocr_use_cuda=rapidocr_use_cuda,
        mining_passes=mining_passes,
        max_vlm_frames_per_minute=max_vlm_frames_per_minute,
        ocr_trigger_min_chars=ocr_trigger_min_chars,
        vlm_backend=vlm_backend,
        vlm_model=vlm_model,
        vlm_base_url=vlm_base_url,
        vlm_api_key_env=vlm_api_key_env,
        vlm_use_env_proxy=vlm_use_env_proxy,
        siliconflow_api_url=siliconflow_api_url,
        siliconflow_api_key_file=siliconflow_api_key_file,
        lecture_markdown_name=lecture_markdown_name,
        lecture_paragraph_chars=lecture_paragraph_chars,
        lecture_max_paragraphs_per_chapter=lecture_max_paragraphs_per_chapter,
        lecture_refine_with_vlm=lecture_refine_with_vlm,
        lecture_refine_input_chars=lecture_refine_input_chars,
        lecture_refine_max_tokens=lecture_refine_max_tokens,
        lecture_strong_normalize_with_vlm=lecture_strong_normalize_with_vlm,
        lecture_strong_chunk_chars=lecture_strong_chunk_chars,
        lecture_term_overrides=merged_term_overrides,
        **file_config,
    )

    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "continuous"

    if resolved_mode == "continuous":
        result = run_continuous(config, input_path, console=console)
        if result.lecture_path:
            console.print(f"\n[green]Done[/green] -> {result.lecture_path}")
        console.print(f"Scene notes -> {result.markdown_path}")
        console.print(
            f"scenes={result.scene_count}, "
            f"asr_segments={result.transcript_segment_count}, "
            f"duration={result.duration:.1f}s, "
            f"evidence={result.evidence_record_count}"
        )
        return

    if resolved_mode == "batch":
        results = run_batch(config, input_path, console=console)
        index = output_dir / "index.md"
        index.write_text(render_index_markdown(output_dir.resolve(), results), encoding="utf-8")
        console.print(f"\n[green]Batch done[/green] -> {index}")
        return

    raise typer.BadParameter("mode must be one of: auto, continuous, batch")


@app.command("example-config")
def example_config(output: Path = typer.Option(Path("video2md.yaml"), "--output", "-o")) -> None:
    sample = {
        "language": "zh",
        "asr_backend": "faster-whisper",
        "asr_model": "small",
        "asr_device": "auto",
        "asr_compute_type": "int8",
        "asr_fallback_to_cpu": True,
        "keyframe_backend": "pyscenedetect",
        "frame_interval_sec": 30.0,
        "scene_threshold": 27.0,
        "ocr_backend": "none",
        "ocr_lang": "ch",
        "rapidocr_use_cuda": False,
        "mining_passes": 2,
        "max_vlm_frames_per_minute": 4,
        "ocr_trigger_min_chars": 12,
        "vlm_backend": "none",
        "vlm_model": "Qwen/Qwen3-VL-32B-Instruct",
        "vlm_base_url": None,
        "vlm_api_key_env": "OPENAI_API_KEY",
        "vlm_use_env_proxy": False,
        "siliconflow_api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "siliconflow_api_key_file": "apptoken.txt",
        "lecture_markdown_name": "video_lecture.md",
        "lecture_paragraph_chars": 320,
        "lecture_max_paragraphs_per_chapter": 24,
        "lecture_refine_with_vlm": True,
        "lecture_refine_input_chars": 24000,
        "lecture_refine_max_tokens": 2200,
        "lecture_strong_normalize_with_vlm": True,
        "lecture_strong_chunk_chars": 2200,
        "lecture_term_overrides": {"小惠": "小辉"},
    }
    output.write_text(yaml.safe_dump(sample, allow_unicode=True, sort_keys=False), encoding="utf-8")
    console.print(f"Wrote {output}")


@app.command("search-evidence")
def search_evidence_cmd(
    db: Path = typer.Option(Path("output/artifacts/evidence.db"), "--db", help="SQLite DB path"),
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    limit: int = typer.Option(20, "--limit", "-n", help="Result count limit"),
) -> None:
    rows = search_evidence(db, query, limit=limit)
    if not rows:
        console.print("No evidence found.")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("kind", style="cyan")
    table.add_column("scene")
    table.add_column("ts")
    table.add_column("text")

    for row in rows:
        ts = row.get("global_ts")
        ts_text = f"{ts:.1f}" if isinstance(ts, (int, float)) else "-"
        text = str(row.get("text", "")).replace("\n", " ")
        if len(text) > 120:
            text = text[:117] + "..."
        table.add_row(
            str(row.get("kind", "")),
            str(row.get("scene_index", "")),
            ts_text,
            text,
        )

    console.print(table)


if __name__ == "__main__":
    app()
