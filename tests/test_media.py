from seepat.preprocessing.media import parse_fraction


def test_parse_fraction() -> None:
    assert parse_fraction("30000/1001") == 30000 / 1001
    assert parse_fraction("25") == 25.0
    assert parse_fraction("0/0") is None
