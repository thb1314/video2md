from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import httpx

from video2md.config import PipelineConfig


class VLMError(RuntimeError):
    pass


def _encode_image(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def _extract_json_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if not match:
        raise VLMError("VLM response did not contain a JSON payload.")

    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise VLMError("VLM response JSON parsing failed.") from exc

    if not isinstance(payload, dict):
        raise VLMError("VLM JSON payload must be an object.")
    return payload


def _to_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _extract_content_text(body: dict[str, Any]) -> str:
    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)

    return json.dumps(content, ensure_ascii=False)


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _post_chat(
    endpoint: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    config: PipelineConfig,
) -> dict[str, Any]:
    try:
        with httpx.Client(
            timeout=config.vlm_timeout_sec,
            trust_env=config.vlm_use_env_proxy,
        ) as client:
            response = client.post(endpoint, json=payload, headers=headers)
    except httpx.HTTPError as exc:
        raise VLMError(f"VLM network error: {exc}") from exc

    if response.status_code >= 400:
        raise VLMError(f"VLM request failed: {response.status_code} {response.text[:300]}")

    return response.json()


def _resolve_siliconflow_key(config: PipelineConfig) -> str:
    env_candidates: list[str] = []
    if config.vlm_api_key_env.strip():
        env_candidates.append(config.vlm_api_key_env.strip())
    for env_name in ("SILICONFLOW_API_KEY", "OPENAI_API_KEY"):
        if env_name not in env_candidates:
            env_candidates.append(env_name)

    for env_name in env_candidates:
        env_value = os.getenv(env_name)
        if env_value and env_value.strip():
            return env_value.strip()

    module_dir = Path(__file__).resolve().parent
    key_file_candidates: list[Path] = []
    if config.siliconflow_api_key_file:
        key_file_candidates.append(Path(config.siliconflow_api_key_file).expanduser())

    # Keep parity with the reference SiliconFlow helper.
    key_file_candidates.extend(
        [
            Path.cwd() / "apptoken.txt",
            module_dir / "apptoken.txt",
            module_dir.parent / "apptoken.txt",
        ]
    )

    visited: set[str] = set()
    for key_path in key_file_candidates:
        key_resolved = str(key_path.resolve())
        if key_resolved in visited:
            continue
        visited.add(key_resolved)

        if key_path.exists() and key_path.is_file():
            value = key_path.read_text(encoding="utf-8").strip()
            if value:
                return value

    raise VLMError(
        "SiliconFlow API key not found. "
        f"Checked envs {env_candidates} and key files {[str(p) for p in key_file_candidates]}."
    )


def _chat_endpoint_headers(config: PipelineConfig) -> tuple[str, dict[str, str]]:
    if config.vlm_backend == "openai":
        api_key = os.getenv(config.vlm_api_key_env)
        if not api_key:
            raise VLMError(
                "Environment variable "
                f"`{config.vlm_api_key_env}` is required for VLM backend `openai`."
            )
        base_url = (config.vlm_base_url or "https://api.openai.com/v1").rstrip("/")
        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        return endpoint, headers

    if config.vlm_backend == "siliconflow":
        endpoint = config.siliconflow_api_url
        headers = {
            "Authorization": f"Bearer {_resolve_siliconflow_key(config)}",
            "Content-Type": "application/json",
        }
        return endpoint, headers

    raise VLMError(f"Unsupported VLM backend: {config.vlm_backend}")


def _openai_payload(
    image_path: Path,
    transcript_excerpt: str,
    ocr_text: str | None,
    config: PipelineConfig,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    api_key = os.getenv(config.vlm_api_key_env)
    if not api_key:
        raise VLMError(
            f"Environment variable `{config.vlm_api_key_env}` is required for VLM backend `openai`."
        )

    base_url = (config.vlm_base_url or "https://api.openai.com/v1").rstrip("/")
    endpoint = f"{base_url}/chat/completions"

    data_uri = f"data:image/jpeg;base64,{_encode_image(image_path)}"
    user_text = (
        "You are analyzing a keyframe from a business education video. "
        "Return strict JSON with fields: "
        "summary (string), facts (array of short strings), confidence (0-1). "
        "Use OCR and transcript context when useful; avoid hallucination.\n\n"
        f"Transcript context:\n{transcript_excerpt or '(none)'}\n\n"
        f"OCR context:\n{ocr_text or '(none)'}"
    )

    payload = {
        "model": config.vlm_model,
        "temperature": config.vlm_temperature,
        "top_p": config.vlm_top_p,
        "max_tokens": config.vlm_max_tokens,
        "messages": [
            {
                "role": "system",
                "content": "Return JSON only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return endpoint, payload, headers


def _siliconflow_payload(
    image_path: Path,
    transcript_excerpt: str,
    ocr_text: str | None,
    config: PipelineConfig,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    api_key = _resolve_siliconflow_key(config)
    endpoint = config.siliconflow_api_url
    data_uri = f"data:image/jpeg;base64,{_encode_image(image_path)}"

    prompt = (
        "你正在分析一个商业教学视频关键帧。"
        "只返回JSON对象，字段为：summary(字符串), facts(字符串数组), confidence(0到1浮点数)。"
        "请结合OCR与转写上下文，不要编造。\n\n"
        f"Transcript context:\n{transcript_excerpt or '(none)'}\n\n"
        f"OCR context:\n{ocr_text or '(none)'}"
    )

    payload = {
        "model": config.vlm_model,
        "max_tokens": config.vlm_max_tokens,
        "temperature": config.vlm_temperature,
        "top_p": config.vlm_top_p,
        "messages": [
            {"role": "system", "content": "你是一个有用的助手，只输出JSON。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            },
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return endpoint, payload, headers


def enrich_frame(
    image_path: Path,
    transcript_excerpt: str,
    ocr_text: str | None,
    config: PipelineConfig,
) -> dict[str, Any] | None:
    if config.vlm_backend == "none":
        return None

    if config.vlm_backend == "openai":
        endpoint, payload, headers = _openai_payload(
            image_path, transcript_excerpt, ocr_text, config
        )
    elif config.vlm_backend == "siliconflow":
        endpoint, payload, headers = _siliconflow_payload(
            image_path, transcript_excerpt, ocr_text, config
        )
    else:
        raise VLMError(f"Unsupported VLM backend: {config.vlm_backend}")

    body = _post_chat(endpoint, payload, headers, config)
    content = _extract_content_text(body)

    parsed = _extract_json_payload(content)
    summary = str(parsed.get("summary", "")).strip()
    facts = _to_str_list(parsed.get("facts"))

    confidence_raw = parsed.get("confidence", 0.0)
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(max(confidence, 0.0), 1.0)

    return {
        "summary": summary,
        "facts": facts,
        "confidence": confidence,
    }


def infer_term_overrides_from_evidence(
    draft_markdown: str,
    scene_evidence_blocks: list[str],
    config: PipelineConfig,
    max_pairs: int = 40,
) -> dict[str, str]:
    if config.vlm_backend == "none":
        return {}

    evidence_text = "\n\n".join(
        f"[证据块 {idx + 1}]\n{block.strip()}"
        for idx, block in enumerate(scene_evidence_blocks)
        if block and block.strip()
    )
    evidence_text = evidence_text[: min(24000, config.lecture_refine_input_chars)]
    draft_text = draft_markdown[: min(16000, config.lecture_refine_input_chars)]

    user_text = (
        "请根据以下视频证据，抽取“ASR错误词 -> OCR/画面正确词”的纠错映射。\n"
        "规则：\n"
        "1) 仅输出证据充分、同指代的替换；\n"
        "2) 优先人名、品牌名、机构名、术语；\n"
        "3) right 必须来源于 OCR/画面证据，wrong 来源于草稿/ASR；\n"
        "4) 不要编造，不确定就不输出；\n"
        "5) 只输出 JSON 对象，格式："
        '{"overrides":[{"wrong":"...","right":"...","confidence":0.0-1.0,"reason":"..."}]}。\n\n'
        f"讲义草稿片段：\n{draft_text or '(none)'}\n\n"
        f"OCR/视觉/转写证据：\n{evidence_text or '(none)'}"
    )

    endpoint, headers = _chat_endpoint_headers(config)
    payload = {
        "model": config.vlm_model,
        "max_tokens": min(max(config.lecture_refine_max_tokens // 2, 512), 4096),
        "temperature": 0.0,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": "你是术语纠错器，只输出JSON。"},
            {"role": "user", "content": user_text},
        ],
    }
    body = _post_chat(endpoint, payload, headers, config)
    content = _strip_markdown_fences(_extract_content_text(body))
    parsed = _extract_json_payload(content)

    raw_items = parsed.get("overrides", [])
    if not isinstance(raw_items, list):
        raise VLMError("Term override response must contain `overrides` list.")

    overrides: dict[str, str] = {}
    for item in raw_items[:max_pairs]:
        wrong = ""
        right = ""
        if isinstance(item, dict):
            wrong = str(item.get("wrong", "")).strip()
            right = str(item.get("right", "")).strip()
        elif isinstance(item, str):
            token = item.strip()
            if "->" in token:
                wrong, right = [x.strip() for x in token.split("->", 1)]
            elif "=" in token:
                wrong, right = [x.strip() for x in token.split("=", 1)]
        if (
            not wrong
            or not right
            or wrong == right
            or len(wrong) > 16
            or len(right) > 16
        ):
            continue
        overrides[wrong] = right

    return overrides


def _split_markdown_chunks(markdown: str, max_chars: int) -> list[str]:
    pieces = [item.strip() for item in markdown.split("\n\n") if item.strip()]
    if not pieces:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        add_len = len(piece) + (2 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append("\n\n".join(current))
            current = [piece]
            current_len = len(piece)
            continue
        current.append(piece)
        current_len += add_len
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _heading_count(markdown: str) -> int:
    return len(re.findall(r"(?m)^#{1,6}\s+", markdown))


def _sanitize_markdown_chunk(
    original: str,
    candidate: str,
    min_ratio: float = 0.45,
    max_ratio: float = 1.7,
) -> str:
    cleaned = candidate.strip()
    if not cleaned:
        return original

    input_headings = _heading_count(original)
    output_headings = _heading_count(cleaned)
    if input_headings != output_headings:
        return original

    ratio = len(cleaned) / max(len(original), 1)
    if ratio < min_ratio or ratio > max_ratio:
        return original
    return cleaned


def normalize_lecture_markdown_strong(
    draft_markdown: str,
    config: PipelineConfig,
    forced_terms: list[str] | None = None,
) -> str | None:
    if config.vlm_backend == "none":
        return None

    chunks = _split_markdown_chunks(draft_markdown, config.lecture_strong_chunk_chars)
    if not chunks:
        return draft_markdown

    endpoint, headers = _chat_endpoint_headers(config)
    forced_terms_text = "\n".join(
        f"- {term.strip()}" for term in (forced_terms or []) if term and term.strip()
    )

    normalized_chunks: list[str] = []
    for chunk in chunks:
        user_text = (
            "请把下面 Markdown 讲义片段做“强规范化”编辑。\n"
            "硬性要求：\n"
            "1) 仅修正错别字、同音误字、病句、断句、标点、口语赘词与重复表达；\n"
            "2) 不新增事实，不删减关键信息，不改变数字含义；\n"
            "3) 必须保持原有 Markdown 层级与顺序，不新增标题；\n"
            "4) 专有名词优先遵循强制词表，逐字一致；\n"
            "5) 只输出修订后的 Markdown 片段，不要解释。\n\n"
            f"强制词表：\n{forced_terms_text or '(none)'}\n\n"
            f"待修订片段：\n{chunk}"
        )
        payload = {
            "model": config.vlm_model,
            "max_tokens": config.lecture_refine_max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "messages": [
                {"role": "system", "content": "你是资深中文讲义编辑，只输出修订后的Markdown。"},
                {"role": "user", "content": user_text},
            ],
        }

        try:
            body = _post_chat(endpoint, payload, headers, config)
            content = _strip_markdown_fences(_extract_content_text(body))
            normalized_chunks.append(_sanitize_markdown_chunk(chunk, content))
        except VLMError:
            normalized_chunks.append(chunk)

    return "\n\n".join(normalized_chunks).strip()


def refine_lecture_markdown(
    draft_markdown: str,
    ocr_hints: list[str],
    visual_hints: list[str],
    config: PipelineConfig,
    forced_terms: list[str] | None = None,
) -> str | None:
    if config.vlm_backend == "none":
        return None

    if len(draft_markdown) > config.lecture_refine_input_chars:
        raise VLMError(
            "Lecture refinement skipped to avoid truncation: "
            f"draft length {len(draft_markdown)} > lecture_refine_input_chars "
            f"{config.lecture_refine_input_chars}."
        )

    ocr_context = "\n".join(
        f"- {line.strip()}"
        for line in ocr_hints
        if line and line.strip()
    )
    visual_context = "\n".join(
        f"- {line.strip()}"
        for line in visual_hints
        if line and line.strip()
    )
    forced_terms_text = "\n".join(
        f"- {term.strip()}" for term in (forced_terms or []) if term and term.strip()
    )

    user_text = (
        "请把下面的视频讲义草稿修订为正式讲义。\n"
        "要求：\n"
        "1) 修正明显同音错字、错别字、口语赘词和断句问题；\n"
        "2) 优先参考 OCR 与视觉线索做术语纠错；\n"
        "2.1) 当 ASR 转写与 OCR/画面词冲突时，优先采用 OCR/画面词；\n"
        "3) 不新增草稿中没有的事实，不要编造数字；\n"
        "4) 保持原有 Markdown 标题层级和章节顺序；\n"
        "5) 专有名词、人名、品牌名优先遵循“强制词表”，必须逐字一致；\n"
        "6) 只输出修订后的 Markdown 正文，不要解释。\n\n"
        f"强制词表：\n{forced_terms_text or '(none)'}\n\n"
        f"OCR 线索：\n{ocr_context or '(none)'}\n\n"
        f"视觉线索：\n{visual_context or '(none)'}\n\n"
        f"讲义草稿：\n{draft_markdown}"
    )

    endpoint, headers = _chat_endpoint_headers(config)

    payload = {
        "model": config.vlm_model,
        "max_tokens": config.lecture_refine_max_tokens,
        "temperature": 0.1,
        "top_p": 1.0,
        "messages": [
            {"role": "system", "content": "你是课程讲义编辑，只输出修订后的Markdown。"},
            {"role": "user", "content": user_text},
        ],
    }
    body = _post_chat(endpoint, payload, headers, config)
    content = _strip_markdown_fences(_extract_content_text(body))
    if not content.strip():
        raise VLMError("Lecture refinement response is empty.")
    return content.strip()
