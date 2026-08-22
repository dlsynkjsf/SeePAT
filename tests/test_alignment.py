from __future__ import annotations

from pathlib import Path

from seepat.preprocessing.alignment import normalize_arpa_phone, parse_mfa_json


def test_normalize_arpa_phone_removes_stress() -> None:
    assert normalize_arpa_phone("M") == "M"
    assert normalize_arpa_phone("b1") == "B"


def test_parse_mfa_json_keeps_only_bilabials(tmp_path: Path) -> None:
    output = tmp_path / "alignment.json"
    output.write_text(
        '{"tiers":{"words":{"entries":[[0.0,0.3,"word"]]},'
        '"phones":{"entries":[[0.0,0.1,"AH0"],[0.1,0.2,"B"]]}}}',
        encoding="utf-8",
    )

    assert parse_mfa_json(output) == [
        {
            "phoneme": "b",
            "phone_start_s": 0.1,
            "phone_end_s": 0.2,
            "speaker": "",
        }
    ]
