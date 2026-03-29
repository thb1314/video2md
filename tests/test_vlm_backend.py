from __future__ import annotations

from pathlib import Path

import pytest

from video2md.backends.vlm import (
    enrich_frame,
    infer_term_overrides_from_evidence,
    normalize_lecture_markdown_strong,
    refine_lecture_markdown,
)
from video2md.config import PipelineConfig


class _DummyResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = "dummy"

    def json(self) -> dict:
        return self._payload


def _make_dummy_client(captured: dict):
    class _DummyClient:
        def __init__(self, *, timeout: float, trust_env: bool):
            captured["timeout"] = timeout
            captured["trust_env"] = trust_env

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint: str, json: dict, headers: dict) -> _DummyResponse:  # noqa: A003
            captured["endpoint"] = endpoint
            captured["payload"] = json
            captured["headers"] = headers
            content = (
                "analysis text\n"
                '{"summary":"画面是课程导读页","facts":["出现课程标题"],"confidence":0.9}\n'
                "trailer"
            )
            return _DummyResponse(
                200,
                {"choices": [{"message": {"content": content}}]},
            )

    return _DummyClient


def test_siliconflow_payload_with_env_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _make_dummy_client(captured))

    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"not-a-real-jpg")

    config = PipelineConfig(
        input_video=Path("demo.mp4"),
        vlm_backend="siliconflow",
        vlm_use_env_proxy=True,
        siliconflow_api_url="https://api.siliconflow.cn/v1/chat/completions",
    )
    result = enrich_frame(
        image_path=image_path,
        transcript_excerpt="这是关于公司注册的导读。",
        ocr_text="注册登记",
        config=config,
    )

    assert result == {
        "summary": "画面是课程导读页",
        "facts": ["出现课程标题"],
        "confidence": 0.9,
    }
    assert captured["endpoint"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"
    assert captured["payload"]["model"] == "Qwen/Qwen3-VL-32B-Instruct"
    assert captured["trust_env"] is True
    assert captured["timeout"] == config.vlm_timeout_sec
    user_content = captured["payload"]["messages"][1]["content"]
    assert user_content[0]["type"] == "text"
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_siliconflow_reads_key_file_when_env_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("SILICONFLOW_API_KEY", raising=False)
    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _make_dummy_client(captured))

    key_file = tmp_path / "apptoken.txt"
    key_file.write_text("file-key\n", encoding="utf-8")
    image_path = tmp_path / "f.jpg"
    image_path.write_bytes(b"fake-jpg")

    config = PipelineConfig(
        input_video=Path("demo.mp4"),
        vlm_backend="siliconflow",
        siliconflow_api_key_file=key_file,
    )
    enrich_frame(
        image_path=image_path,
        transcript_excerpt="",
        ocr_text=None,
        config=config,
    )

    assert captured["headers"]["Authorization"] == "Bearer file-key"


def test_refine_lecture_markdown_strips_fences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    class _RefineClient:
        def __init__(self, *, timeout: float, trust_env: bool):
            captured["timeout"] = timeout
            captured["trust_env"] = trust_env

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint: str, json: dict, headers: dict) -> _DummyResponse:  # noqa: A003
            captured["endpoint"] = endpoint
            captured["payload"] = json
            captured["headers"] = headers
            return _DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": "```markdown\n# 修订讲义\n\n正文内容。\n```"
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _RefineClient)

    config = PipelineConfig(
        input_video=tmp_path / "demo.mp4",
        vlm_backend="siliconflow",
    )
    refined = refine_lecture_markdown(
        draft_markdown="# 草稿\n\n错误字。",
        ocr_hints=["营业执照"],
        visual_hints=["画面显示注册流程图"],
        config=config,
    )
    assert refined == "# 修订讲义\n\n正文内容。"
    assert captured["endpoint"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_infer_term_overrides_from_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict = {}
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    class _InferClient:
        def __init__(self, *, timeout: float, trust_env: bool):
            captured["timeout"] = timeout
            captured["trust_env"] = trust_env

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint: str, json: dict, headers: dict) -> _DummyResponse:  # noqa: A003
            captured["endpoint"] = endpoint
            captured["payload"] = json
            captured["headers"] = headers
            return _DummyResponse(
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"overrides":['
                                    '{"wrong":"德道","right":"得到","confidence":0.92},'
                                    '{"wrong":"李子奇","right":"李子柒","confidence":0.95}'
                                    "]}"
                                )
                            }
                        }
                    ]
                },
            )

    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _InferClient)

    config = PipelineConfig(
        input_video=tmp_path / "demo.mp4",
        vlm_backend="siliconflow",
    )
    overrides = infer_term_overrides_from_evidence(
        draft_markdown="我们服务过德道和李子奇。",
        scene_evidence_blocks=[
            (
                "scene=2\n"
                "ASR: 我们服务过德道、李子奇、西少爷。\n"
                "OCR: 服务过得到、李子柒、西少爷。"
            )
        ],
        config=config,
    )

    assert overrides == {"德道": "得到", "李子奇": "李子柒"}
    assert captured["endpoint"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_normalize_lecture_markdown_strong_chunked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict = {"count": 0}
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    class _NormalizeClient:
        def __init__(self, *, timeout: float, trust_env: bool):
            captured["timeout"] = timeout
            captured["trust_env"] = trust_env

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint: str, json: dict, headers: dict) -> _DummyResponse:  # noqa: A003
            captured["count"] += 1
            captured["endpoint"] = endpoint
            captured["headers"] = headers
            user_text = json["messages"][1]["content"]
            if "错字段落一" in user_text:
                content = "正字段落一。" + ("甲" * 620)
            elif "错字段落二" in user_text:
                content = "正字段落二。" + ("乙" * 620)
            elif "# 标题" in user_text:
                content = "# 标题"
            else:
                content = "无变化。"
            return _DummyResponse(200, {"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _NormalizeClient)

    para1 = "错字段落一。" + ("甲" * 620)
    para2 = "错字段落二。" + ("乙" * 620)
    config = PipelineConfig(
        input_video=tmp_path / "demo.mp4",
        vlm_backend="siliconflow",
        lecture_strong_chunk_chars=600,
    )
    normalized = normalize_lecture_markdown_strong(
        draft_markdown=f"# 标题\n\n{para1}\n\n{para2}",
        config=config,
        forced_terms=["小辉"],
    )
    assert "# 标题" in normalized
    assert "正字段落一。" in normalized
    assert "正字段落二。" in normalized
    assert captured["count"] >= 2
    assert captured["endpoint"] == "https://api.siliconflow.cn/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer env-key"


def test_normalize_lecture_markdown_strong_fallback_on_structure_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    class _NormalizeClient:
        def __init__(self, *, timeout: float, trust_env: bool):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, endpoint: str, json: dict, headers: dict) -> _DummyResponse:  # noqa: A003
            return _DummyResponse(
                200,
                {"choices": [{"message": {"content": "## 新标题\n\n这里是错误输出。"}}]},
            )

    monkeypatch.setattr("video2md.backends.vlm.httpx.Client", _NormalizeClient)

    config = PipelineConfig(
        input_video=tmp_path / "demo.mp4",
        vlm_backend="siliconflow",
        lecture_strong_chunk_chars=2000,
    )
    source = "普通段落，不应新增标题。"
    normalized = normalize_lecture_markdown_strong(
        draft_markdown=source,
        config=config,
        forced_terms=[],
    )
    assert normalized == source
