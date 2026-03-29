# video2md

[中文 README](./README.md)

Convert single videos or continuous multi-video courses into structured Markdown.
The pipeline outputs two core files:
- `video_lecture.md`: lecture-style article (term correction + VLM refinement + strong normalization)
- `video_notes.md`: evidence notes (keyframes + OCR + VLM + transcript)

## Highlights
- Docker-only runtime (no local `uv run ...` workflow)
- Continuous course mode: merge multiple videos into one timeline/article
- Multimodal evidence fusion: ASR + OCR + VLM
- Robust correction chain: VLM term correction with OCR fallback
- GPU-first strategy: ASR/OCR prefer GPU, ASR supports CPU fallback

## Pipeline
1. Scene detection and keyframe extraction
2. ASR transcription (optional)
3. OCR extraction (optional)
4. VLM keyframe understanding (optional)
5. Lecture draft generation
6. Term correction (VLM first, OCR fallback)
7. Lecture refinement (VLM)
8. Strong lecture normalization (VLM, chunked with guardrails)

## Requirements
- Docker / Docker Compose
- `nvidia-container-toolkit` for GPU mode
- Optional proxy for SiliconFlow: `host.docker.internal:7890`

## Quick Start

### 1) Prepare folders
```text
./input
./output
```

### 2) Optional: proxy and API key
```bash
export HTTP_PROXY=http://host.docker.internal:7890
export HTTPS_PROXY=http://host.docker.internal:7890
export OPENAI_API_KEY=<your_siliconflow_key>
```

Or pass key file via CLI:
`--siliconflow-api-key-file /data/apptoken.txt`

### 3) Default GPU service
```bash
docker compose up --build video2md
```

### 4) Recommended profile (VLM + ASR + OCR)
```bash
docker compose --profile vlm up --build video2md-qwen
```

### 5) CPU fallback profile
```bash
docker compose --profile cpu up --build video2md-cpu
```

### 6) OCR-GPU fallback profile
```bash
docker compose --profile vlm-cpuocr up --build video2md-qwen-cpuocr
```

## Typical Commands

### Single video
```bash
docker compose --profile vlm run --rm video2md-qwen \
  run \
  --input /data/input/demo.mp4 \
  --output /data/output \
  --mode auto \
  --asr-backend faster-whisper \
  --asr-device cuda \
  --asr-compute-type float16 \
  --asr-fallback-to-cpu \
  --ocr-backend rapidocr \
  --rapidocr-use-cuda \
  --vlm-backend siliconflow \
  --vlm-model Qwen/Qwen3-VL-32B-Instruct \
  --lecture-refine-with-vlm \
  --lecture-strong-normalize-with-vlm
```

### Continuous multi-video mode
```bash
docker compose --profile vlm run --rm video2md-qwen \
  run \
  --input /data/input \
  --output /data/output \
  --mode continuous \
  --asr-backend faster-whisper \
  --asr-device cuda \
  --asr-compute-type float16 \
  --asr-fallback-to-cpu \
  --ocr-backend rapidocr \
  --rapidocr-use-cuda \
  --mining-passes 2 \
  --vlm-backend siliconflow \
  --vlm-model Qwen/Qwen3-VL-32B-Instruct \
  --lecture-refine-with-vlm \
  --lecture-strong-normalize-with-vlm \
  --lecture-strong-chunk-chars 2200
```

## Important CLI Flags (Explained)
- `--input PATH`: Input video file or directory path. A directory can be processed as a continuous timeline.
- `--output PATH`: Output directory for lecture, notes, keyframes, and artifacts.
- `--mode auto|continuous|batch`: Run mode. `auto` defaults to continuous behavior, `continuous` merges into one article, `batch` outputs per video.
- `--asr-backend faster-whisper|none`: ASR backend. `none` disables speech transcription.
- `--asr-model NAME`: ASR model size (for example `tiny/small/medium`), larger is usually more accurate but slower.
- `--asr-device auto|cpu|cuda`: Device selection for ASR inference.
- `--asr-compute-type int8|float16|...`: Numeric precision for ASR, affecting speed/VRAM/accuracy tradeoffs.
- `--asr-fallback-to-cpu` / `--no-asr-fallback-to-cpu`: Whether to auto-fallback to CPU if requested device init fails (enabled by default).
- `--ocr-backend none|rapidocr`: OCR backend. `none` disables OCR.
- `--rapidocr-use-cuda`: Enable CUDAExecutionProvider for RapidOCR (`onnxruntime-gpu` required).
- `--mining-passes N`: Number of iterative evidence-mining passes (keyframe/OCR/VLM); higher is usually richer but slower.
- `--vlm-backend none|openai|siliconflow`: VLM backend.
- `--vlm-model MODEL_NAME`: VLM model name (default `Qwen/Qwen3-VL-32B-Instruct`).
- `--siliconflow-api-key-file PATH`: Path to SiliconFlow API key file when env vars are not used.
- `--lecture-refine-with-vlm` / `--no-lecture-refine-with-vlm`: Toggle VLM-based lecture polishing.
- `--lecture-strong-normalize-with-vlm` / `--no-lecture-strong-normalize-with-vlm`: Toggle strict normalization step (enabled by default).
- `--lecture-strong-chunk-chars N`: Chunk size (characters) used in strict normalization.
- `--term-override wrong=right`: Manual term replacement rule, repeatable.

Default VLM model: `Qwen/Qwen3-VL-32B-Instruct`  
Note: SiliconFlow VLM is a remote API, not local GPU inference.

## GPU / Fallback Behavior
- ASR: tries requested device first (e.g., `cuda/float16`), then can fallback to CPU (enabled by default)
- OCR: `rapidocr` with `--rapidocr-use-cuda` prefers CUDA provider, with runtime skip/guardrails on failures
- VLM: remote API; frame-level failures are skipped and pipeline continues

## Output Layout
```text
output/
  video_lecture.md
  video_notes.md
  assets/
    <video_name>/scene_0001.jpg
  artifacts/
    <video_name>.wav
    scene_data.json
    evidence.jsonl
    evidence.db
```

## Evidence Search
```bash
docker compose run --rm video2md \
  search-evidence --db /data/output/artifacts/evidence.db -q "registered capital"
```

## Troubleshooting
- `CUDA driver version is insufficient`: keep `--asr-fallback-to-cpu` enabled
- `VLM network error / timed out`: verify proxy/API key/network and increase `vlm_timeout_sec`
- Root-owned output files: run container with `--user $(id -u):$(id -g)` or `chown` afterward

## Open-Source Safety Checklist
- Do not commit `.env`, `apptoken.txt`, or real API keys
- `.gitignore` already excludes local data, outputs, secrets, and temp configs
