from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from video2md.config import PipelineConfig


class OCRError(RuntimeError):
    pass


_OCR_CACHE: dict[str, object] = {}


def _get_ocr(config: PipelineConfig):
    key = f"{config.ocr_backend}:{config.ocr_lang}:{int(config.rapidocr_use_cuda)}"
    if key in _OCR_CACHE:
        return _OCR_CACHE[key]

    if config.ocr_backend == "paddleocr":
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise OCRError("paddleocr is not installed. Run `uv sync --extra ocr` first.") from exc

        ocr = PaddleOCR(
            use_angle_cls=True,
            lang=config.ocr_lang,
            use_gpu=False,
            ir_optim=False,
            show_log=False,
        )
        _OCR_CACHE[key] = ocr
        return ocr

    if config.ocr_backend == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise OCRError(
                "rapidocr-onnxruntime is not installed. Run `uv sync --extra ocr` first."
            ) from exc

        ocr = RapidOCR(use_cuda=config.rapidocr_use_cuda)
        _OCR_CACHE[key] = ocr
        return ocr

    raise OCRError(f"Unsupported OCR backend: {config.ocr_backend}")


def _clean_join(lines: Iterable[str]) -> str | None:
    seen: set[str] = set()
    ordered: list[str] = []
    for text in lines:
        normalized = str(text).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    merged = " ".join(ordered).strip()
    return merged or None


def _parse_legacy_ocr_output(raw: object) -> list[str]:
    lines: list[str] = []
    if not isinstance(raw, list):
        return lines

    for block in raw:
        if not isinstance(block, list):
            continue
        for row in block:
            if not isinstance(row, (list, tuple)) or len(row) < 2:
                continue
            rec = row[1]
            if not isinstance(rec, (list, tuple)) or not rec:
                continue
            text = rec[0]
            if text:
                lines.append(str(text))
    return lines


def _parse_predict_output(raw: object) -> list[str]:
    lines: list[str] = []
    stack: list[object] = [raw]

    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for key in ("rec_text", "text"):
                value = node.get(key)
                if isinstance(value, str) and value.strip():
                    lines.append(value.strip())
            for key in ("rec_texts", "texts"):
                value = node.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and item.strip():
                            lines.append(item.strip())
            for key in ("res", "result", "results", "data", "layout"):
                if key in node:
                    stack.append(node[key])
            continue

        if isinstance(node, (list, tuple)):
            stack.extend(node)

    return lines


def _parse_rapidocr_output(raw: object) -> list[str]:
    lines: list[str] = []
    if not isinstance(raw, list):
        return lines
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        text = row[1]
        if text:
            lines.append(str(text))
    return lines


def ocr_image(image_path: Path, config: PipelineConfig) -> str | None:
    if config.ocr_backend == "none":
        return None

    if config.ocr_backend not in {"paddleocr", "rapidocr"}:
        raise OCRError(f"Unsupported OCR backend: {config.ocr_backend}")

    ocr = _get_ocr(config)
    raw: object
    lines: list[str] = []

    if config.ocr_backend == "rapidocr":
        try:
            raw, _elapsed = ocr(str(image_path))
        except Exception as exc:  # pragma: no cover - runtime backend errors
            raise OCRError(f"OCR inference failed: {exc}") from exc
        lines = _parse_rapidocr_output(raw)
        return _clean_join(lines)

    # PaddleOCR <=2.x uses ocr(..., cls=True), while 3.x exposes predict().
    try:
        raw = ocr.ocr(str(image_path), cls=True)
        lines = _parse_legacy_ocr_output(raw)
    except TypeError:
        raw = ocr.predict(str(image_path))
        lines = _parse_predict_output(raw)
    except Exception as exc:  # pragma: no cover - runtime backend errors
        raise OCRError(f"OCR inference failed: {exc}") from exc

    return _clean_join(lines)
