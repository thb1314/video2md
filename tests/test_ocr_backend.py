from pathlib import Path

from video2md.backends import ocr as ocr_backend
from video2md.config import PipelineConfig


def test_parse_legacy_ocr_output() -> None:
    raw = [
        [
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            ("营业执照", 0.99),
        ],
        [
            [[0, 0], [1, 0], [1, 1], [0, 1]],
            ("注册资本", 0.98),
        ],
    ]
    # simulate outer blocks of legacy format
    parsed = ocr_backend._parse_legacy_ocr_output([raw])  # noqa: SLF001
    assert "营业执照" in parsed
    assert "注册资本" in parsed


def test_ocr_image_fallbacks_to_predict(monkeypatch) -> None:
    class _FakeOCR:
        def ocr(self, image_path: str, cls: bool = True):  # noqa: ARG002
            raise TypeError("unexpected keyword argument 'cls'")

        def predict(self, image_path: str):  # noqa: ARG002
            return [
                {
                    "res": {
                        "rec_texts": ["公司名称", "注册地址"],
                    }
                }
            ]

    monkeypatch.setattr(ocr_backend, "_get_ocr", lambda config: _FakeOCR())
    cfg = PipelineConfig(input_video=Path("demo.mp4"), ocr_backend="paddleocr")

    text = ocr_backend.ocr_image(Path("/tmp/demo.jpg"), cfg)
    assert text is not None
    assert "公司名称" in text
    assert "注册地址" in text


def test_ocr_image_with_rapidocr(monkeypatch) -> None:
    class _RapidOCR:
        def __call__(self, image_path: str):  # noqa: ARG002
            return (
                [
                    [[[0, 0], [1, 0], [1, 1], [0, 1]], "营业执照", 0.99],
                    [[[0, 0], [1, 0], [1, 1], [0, 1]], "注册资本", 0.98],
                ],
                0.01,
            )

    monkeypatch.setattr(ocr_backend, "_get_ocr", lambda config: _RapidOCR())
    cfg = PipelineConfig(input_video=Path("demo.mp4"), ocr_backend="rapidocr")
    text = ocr_backend.ocr_image(Path("/tmp/demo.jpg"), cfg)
    assert text == "营业执照 注册资本"
