import json
from pathlib import Path

import pytest

from seepat.preprocessing.media import parse_fraction, probe_media


def test_parse_fraction() -> None:
    assert parse_fraction("30000/1001") == 30000 / 1001
    assert parse_fraction("25") == 25.0
    assert parse_fraction("0/0") is None


def test_probe_media_preserves_stream_timing_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_bytes(b"binary")
    monkeypatch.setattr(
        "seepat.preprocessing.media.find_media_binary",
        lambda name, explicit_path: ffprobe,
    )
    monkeypatch.setattr(
        "seepat.preprocessing.media.subprocess.run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {
                "stdout": json.dumps(
                    {
                        "streams": [
                            {
                                "codec_type": "video",
                                "codec_name": "h264",
                                "width": 640,
                                "height": 360,
                                "duration": "2.0",
                                "avg_frame_rate": "30000/1001",
                                "r_frame_rate": "30/1",
                                "time_base": "1/30000",
                                "nb_frames": "60",
                                "start_time": "0.040",
                            },
                            {
                                "codec_type": "audio",
                                "codec_name": "aac",
                                "sample_rate": "48000",
                                "start_time": "0.060",
                            },
                        ]
                    }
                )
            },
        )(),
    )

    result = probe_media(tmp_path / "video.mp4")

    assert result["average_frame_rate"] == pytest.approx(30000 / 1001)
    assert result["nominal_frame_rate"] == 30.0
    assert result["video_time_base"] == "1/30000"
    assert result["video_frame_count"] == 60
    assert result["video_start_time_s"] == 0.04
    assert result["audio_start_time_s"] == 0.06
    assert result["audio_video_start_offset_s"] == pytest.approx(0.02)
