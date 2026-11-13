"""Approved concise-report seam: presentation only, no model/media/provider jobs."""

import hashlib
import json
from pathlib import Path

import pytest


def test_plain_summary_leads_with_verdict_and_keeps_technical_details_last():
    from seepat.inference.readable_xai import human_summary

    video = {"verdict": "manipulated", "clips_scored": 3, "video_score": 0.6390966176986694,
             "threshold": 0.46726664900779724, "highlight": {
                 "kind": "bilabial", "probability": 0.6390966176986694,
                 "timestamp": {"status": "verified", "kind": "video_phoneme_interval",
                               "start_s": 1.58, "end_s": 1.64}}}

    text = human_summary(video, technical=True)

    plain, technical = text.split(" Technical: ")
    assert plain == ("SeePAT flagged this video as potentially manipulated. "
                     "It checked 3 short clips; start your review at 1.58 seconds.")
    assert "score" not in plain and "cutoff" not in plain and "/p/" not in plain
    assert "0.639097" in technical and "0.467267" in technical
    assert "not a confirmed edit" in technical
    assert video["verdict"] == "manipulated" and video["threshold"] == 0.46726664900779724


def test_authentic_wording_reports_no_flag_without_claiming_proven_authenticity():
    from seepat.inference.readable_xai import human_summary

    video = {"verdict": "authentic", "clips_scored": 1, "video_score": 0.2, "threshold": 0.5,
             "highlight": {"kind": "bilabial", "probability": 0.2,
                           "timestamp": {"status": "verified", "kind": "video_phoneme_interval",
                                         "start_s": 0.96, "end_s": 1.02}}}

    text = human_summary(video, technical=True)

    assert text.startswith("SeePAT did not flag this video as manipulated.")
    assert "1 short clip;" in text and "is below the cutoff" in text
    assert "confirmed authentic" not in text and "genuine" not in text


@pytest.mark.parametrize("stamp,expected", [
    ({"status": "verified", "kind": "sampled_context_source_frame", "start_s": 0.01,
      "last_sample_s": 0.61}, "start your review at 0.01 seconds"),
    ({"status": "unavailable", "reason": "no_bound_source_trace_clock"},
     "A review timestamp is unavailable"),
])
def test_visual_context_does_not_become_a_speech_event_or_invent_a_clock(stamp, expected):
    from seepat.inference.readable_xai import human_summary

    video = {"verdict": "manipulated", "clips_scored": 4, "video_score": 0.8, "threshold": 0.5,
             "highlight": {"kind": "context", "probability": 0.8, "timestamp": stamp}}

    text = human_summary(video, technical=True)

    assert expected in text
    assert "aligned speech event" not in text and "/p/" not in text
    if stamp["status"] == "verified":
        assert "sampled visual frame" in text


def test_unscored_video_stays_without_a_verdict_or_review_highlight():
    from seepat.inference.readable_xai import human_summary

    text = human_summary({"verdict": "not_evaluated", "clips_scored": 0, "video_score": None,
                          "threshold": 0.5, "highlight": None})

    assert text == ("SeePAT could not assess this video because no clips could be scored. "
                    "No manipulation verdict was made.")


def test_actual_saved_unscored_blank_score_is_missing_not_a_probability():
    from seepat.inference.readable_xai import human_summary

    # Frozen CSV reader deliberately preserves absent maximum_probability as "".
    video = {"verdict": "not_evaluated", "clips_scored": 0, "video_score": "",
             "threshold": 0.5, "highlight": None}
    null_text = human_summary({**video, "video_score": None})

    assert human_summary(video) == null_text


@pytest.mark.parametrize("fault", [{"video_score": 0.0}, {"clips_scored": 1}, {"highlight": {}}])
def test_unscored_missing_value_support_does_not_allow_scores_clips_or_highlights(fault):
    from seepat.inference.readable_xai import human_summary

    video = {"verdict": "not_evaluated", "clips_scored": 0, "video_score": "",
             "threshold": 0.5, "highlight": None}
    with pytest.raises(ValueError, match="Unscored"):
        human_summary({**video, **fault})


@pytest.fixture
def saved_report(tmp_path):
    video = {"video_id": "synthetic-one", "verdict": "manipulated", "clips_scored": 1,
             "video_score": 0.8, "threshold": 0.5, "highlight": {"kind": "bilabial", "probability": 0.8,
             "timestamp": {"status": "verified", "kind": "video_phoneme_interval",
                           "start_s": 0.96, "end_s": 1.02}}}
    content = {"report.json": json.dumps({"version": "local-inference-workflow-v1",
               "scope": "synthetic_engineering", "network_requests": 0,
               "rendering": "local_deterministic", "videos": [video]}),
               "report.md": "Original sealed presentation", "technical_report.md": "Frozen technical details"}
    for name, text in content.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "summary.json").write_text(json.dumps({
        "version": "local-inference-explanation-v1", "status": "complete",
        "scope": "synthetic_engineering", "network_requests": 0,
        "output_sha256": {name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in content},
    }), encoding="utf-8")
    return tmp_path


def test_cli_displays_sealed_report_and_restarts_without_writing(saved_report, capsys):
    from seepat.inference.readable_xai import main

    before = {p.name: p.read_bytes() for p in saved_report.iterdir()}
    main(["--report-dir", str(saved_report)])
    first = capsys.readouterr().out
    main(["--report-dir", str(saved_report)])

    assert first == capsys.readouterr().out
    assert "ENGINEERING CHECK" in first and "start your review at 0.96 seconds" in first
    assert {p.name: p.read_bytes() for p in saved_report.iterdir()} == before


def test_default_view_is_plain_and_optional_technical_sentence_is_last(saved_report, capsys):
    from seepat.inference.readable_xai import main

    args = ["--report-dir", str(saved_report)]
    main(args)
    plain = capsys.readouterr().out
    main(args + ["--technical"])
    detailed = capsys.readouterr().out

    assert "Technical:" not in plain and "cutoff" not in plain and "score" not in plain
    assert detailed.startswith(plain.rstrip()) and " Technical: " in detailed
    assert detailed.rstrip().endswith("not a confirmed edit.")


@pytest.mark.parametrize("changed", ["report.md", "report.json", "technical_report.md"])
def test_changed_source_stops_display_without_writing(saved_report, capsys, changed):
    from seepat.inference.readable_xai import main

    (saved_report / changed).write_text("altered source", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in saved_report.iterdir()}
    with pytest.raises(SystemExit, match="hash differs"):
        main(["--report-dir", str(saved_report)])
    assert not capsys.readouterr().out
    assert {p.name: p.read_bytes() for p in saved_report.iterdir()} == before


@pytest.mark.parametrize("fault", ["verdict", "highlight", "clock", "score"])
def test_resealed_contradictory_or_invalid_facts_are_rejected(saved_report, fault):
    from seepat.inference.readable_xai import render_saved_report

    path = saved_report / "report.json"
    report = json.loads(path.read_text())
    video = report["videos"][0]
    if fault == "verdict":
        video["verdict"] = "authentic"
    elif fault == "highlight":
        video["highlight"]["probability"] = 0.1
    elif fault == "clock":
        video["highlight"]["timestamp"]["start_s"] = -1
    else:
        video["video_score"] = float("nan")
    path.write_text(json.dumps(report), encoding="utf-8")
    summary_path = saved_report / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["output_sha256"]["report.json"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        render_saved_report(saved_report)


def test_cli_has_no_run_or_overwrite_option(saved_report):
    from seepat.inference.readable_xai import main

    before = {p.name: p.read_bytes() for p in saved_report.iterdir()}
    for flag in ("--run", "--output-dir"):
        with pytest.raises(SystemExit) as stopped:
            main(["--report-dir", str(saved_report), flag])
        assert stopped.value.code == 2
    assert {p.name: p.read_bytes() for p in saved_report.iterdir()} == before


def test_accepted_paragraph_layout_displays_saved_facts_without_rewriting(saved_report):
    from seepat.inference.readable_xai import render_saved_report

    before = {p.name: p.read_bytes() for p in saved_report.iterdir()}

    text = render_saved_report(saved_report, technical=True, paragraph_layout=True)

    assert "SeePAT classified this video as manipulated (fake)." in text
    assert "Start your review around 0.96 seconds" in text
    assert "\n\nTechnical details: 1 clip analyzed;" in text
    assert "0.800000" in text and "0.500000" in text
    assert {p.name: p.read_bytes() for p in saved_report.iterdir()} == before


def test_paragraph_display_rejects_unreviewed_formatter_before_execution(saved_report, monkeypatch):
    from seepat.inference.readable_xai import render_saved_report

    original = Path.read_bytes

    def changed(path):
        if path.name == "xai_paragraphs.py":
            return b"raise AssertionError('unreviewed source must not execute')"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", changed)

    with pytest.raises(ValueError, match="wording source changed"):
        render_saved_report(saved_report, technical=True, paragraph_layout=True)
