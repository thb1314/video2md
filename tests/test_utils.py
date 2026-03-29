from video2md.utils import format_ts, overlap


def test_format_ts() -> None:
    assert format_ts(0) == "00:00:00"
    assert format_ts(3661) == "01:01:01"


def test_overlap() -> None:
    assert overlap(0, 5, 4, 8)
    assert not overlap(0, 2, 2, 4)
