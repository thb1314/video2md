# video2md

[English README](./README_EN.md)

将单视频或多视频连续课程转换为结构化 Markdown，输出两份核心文档：
- `video_lecture.md`：讲义正文（带术语纠错、VLM 润色、强规范化）
- `video_notes.md`：证据笔记（关键帧、OCR、VLM、转写）

## 特性
- Docker-only：仅支持容器运行，不支持本地 `uv run ...`
- 连续课程模式：目录内多视频按时间线合并为一篇讲义
- 多模态证据融合：ASR + OCR + VLM 关键帧理解
- 可控纠错链路：VLM 术语纠错失败时自动回退 OCR 证据
- GPU 优先：ASR/OCR 优先使用 GPU；ASR 支持自动回退 CPU

## 处理流程
1. 场景切分与关键帧抽取
2. ASR 转写（可选）
3. OCR 提取（可选）
4. VLM 关键帧理解（可选）
5. 讲义草稿生成
6. 术语纠错（优先 VLM，失败回退 OCR）
7. 讲义润色（VLM）
8. 讲义强规范化（VLM，分段+保护回退）

## 运行要求
- Docker / Docker Compose
- GPU 模式需要 `nvidia-container-toolkit`
- 如果访问 SiliconFlow 需要代理，可使用 `host.docker.internal:7890`

## 快速开始

### 1) 准备目录
```text
./input    # 放待处理视频
./output   # 输出目录
```

### 2) 可选：设置代理和密钥
```bash
export HTTP_PROXY=http://host.docker.internal:7890
export HTTPS_PROXY=http://host.docker.internal:7890
export OPENAI_API_KEY=<your_siliconflow_key>
```

也可不设环境变量，在命令参数传 key 文件：
`--siliconflow-api-key-file /data/apptoken.txt`

### 3) 一键运行（默认 GPU 服务）
```bash
docker compose up --build video2md
```

### 4) 推荐：VLM + ASR + OCR
```bash
docker compose --profile vlm up --build video2md-qwen
```

### 5) CPU 备用
```bash
docker compose --profile cpu up --build video2md-cpu
```

### 6) OCR-GPU 不可用时
```bash
docker compose --profile vlm-cpuocr up --build video2md-qwen-cpuocr
```

## 典型用法

### 单视频（容器内执行）
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

### 连续多视频（目录合并成一篇）
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

## 关键参数
- `--mode auto|continuous|batch`
- `--asr-backend faster-whisper|none`
- `--asr-device auto|cpu|cuda`
- `--asr-compute-type int8|float16|...`
- `--asr-fallback-to-cpu` / `--no-asr-fallback-to-cpu`
- `--ocr-backend none|rapidocr`
- `--rapidocr-use-cuda`
- `--vlm-backend none|openai|siliconflow`
- `--vlm-model Qwen/Qwen3-VL-32B-Instruct`
- `--lecture-refine-with-vlm` / `--no-lecture-refine-with-vlm`
- `--lecture-strong-normalize-with-vlm` / `--no-lecture-strong-normalize-with-vlm`
- `--lecture-strong-chunk-chars N`
- `--term-override wrong=right`（可重复）

当前默认 VLM：`Qwen/Qwen3-VL-32B-Instruct`  
说明：`siliconflow` 是远端推理接口，VLM 本身不在本机 GPU 上执行。

## GPU 与回退行为
- ASR：优先按请求设备初始化（如 `cuda/float16`），失败后可自动回退到 CPU（默认开启）
- OCR：`rapidocr` 在 `--rapidocr-use-cuda` 下优先走 CUDA provider，失败时会有跳过/降级保护
- VLM：远端 API，不涉及本机 GPU/CPU 切换；网络异常时单帧跳过，流程不中断

## 输出结构
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

## 证据检索
```bash
docker compose run --rm video2md \
  search-evidence --db /data/output/artifacts/evidence.db -q 注册资本
```

## 常见问题
- `CUDA driver version is insufficient`：当前环境无法用 CUDA，建议保留 `--asr-fallback-to-cpu`
- `VLM network error / timed out`：检查代理、API key、网络连通性，可提高 `vlm_timeout_sec`
- 输出目录权限是 `root`：容器加 `--user $(id -u):$(id -g)` 或后处理 `chown`

## 开源前安全建议
- 不要提交任何 `.env`、`apptoken.txt`、真实 API key
- 已通过 `.gitignore` 忽略本地数据、输出目录、密钥文件与临时配置
