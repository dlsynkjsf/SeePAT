"""Approved concise-report seam: clear wording from existing structured facts."""

import copy

import pytest


def video(verdict="authentic", **changes):
    score = 0.2 if verdict == "authentic" else 0.8
    return {"verdict": verdict, "clips_scored": 6, "video_score": score, "threshold": 0.5,
            "highlight": {"kind": "bilabial", "probability": score, "timestamp": {
                "status": "verified", "kind": "video_phoneme_interval", "start_s": 3.73, "end_s": 3.79}},
            **changes}


def test_real_label_and_explanation_come_before_separate_technical_paragraph():
    from seepat.inference.xai_paragraphs import human_summary

    source = video()
    before = copy.deepcopy(source)

    plain, technical = human_summary(source).split("\n\n")

    assert plain == ("SeePAT classified this video as authentic (real). "
                     "The clips it analyzed did not provide enough evidence to flag manipulation.")
    assert "cutoff" not in plain and "score" not in plain and "3.73" not in plain
    assert technical.startswith("Technical details: 6 clips analyzed;")
    assert "0.200000" in technical and "0.500000" in technical and "3.73 seconds" in technical
    assert "No anomalous mouth movements" not in plain
    assert source == before


def test_fake_label_gives_a_review_time_without_claiming_a_word_mouth_mismatch():
    from seepat.inference.xai_paragraphs import human_summary

    source = video("manipulated", word="banana", claimed_mismatch="the word does not align")
    plain, technical = human_summary(source).split("\n\n")

    assert plain == ("SeePAT classified this video as manipulated (fake). "
                     "Start your review around 3.73 seconds, at the clip linked to its strongest warning.")
    assert "banana" not in plain and "does not align" not in plain
    assert "meets or exceeds" in technical
    assert "not a confirmed mismatch or edit" in technical


@pytest.mark.parametrize("stamp,kind,expected", [
    ({"status": "verified", "kind": "sampled_context_source_frame", "start_s": 0.01,
      "last_sample_s": 0.61}, "context", "sampled visual frame at 0.01 seconds"),
    ({"status": "unavailable", "reason": "no_bound_source_trace_clock"}, "bilabial",
     "A reliable review timestamp is unavailable"),
])
def test_context_and_missing_clocks_never_become_confirmed_speech_mismatches(stamp, kind, expected):
    from seepat.inference.xai_paragraphs import human_summary

    source = video("manipulated")
    source["highlight"].update(kind=kind, timestamp=stamp)
    text = human_summary(source)

    assert expected in text
    assert "word" not in text and "does not align" not in text
    if kind == "context":
        assert "aligned speech sound" not in text
    if stamp["status"] == "unavailable":
        assert "Start your review around" not in text


@pytest.mark.parametrize("score", [None, ""])
def test_unscored_video_is_neither_real_nor_fake(score):
    from seepat.inference.xai_paragraphs import human_summary

    source = video("not_evaluated", clips_scored=0, video_score=score, highlight=None)
    plain, technical = human_summary(source).split("\n\n")

    assert plain == ("SeePAT could not assess this video. "
                     "No real/fake classification was made because no clips could be scored.")
    assert "0 clips scored" in technical and "no manipulation score" in technical


def test_plain_only_option_omits_technical_paragraph_and_handles_one_clip():
    from seepat.inference.xai_paragraphs import human_summary

    source = video(clips_scored=1)

    assert "\n\n" not in human_summary(source, technical=False)
    assert "Technical details" not in human_summary(source, technical=False)
    assert "1 clip analyzed" in human_summary(source)


@pytest.mark.parametrize("fault", ["verdict", "score", "clock"])
def test_new_wording_keeps_existing_saved_fact_rejection(fault):
    from seepat.inference.xai_paragraphs import human_summary

    source = video()
    if fault == "verdict":
        source["verdict"] = "manipulated"  # Contradicts the saved score/cutoff.
    elif fault == "score":
        source["video_score"] = float("nan")
    else:
        source["highlight"]["timestamp"]["start_s"] = -1

    with pytest.raises(ValueError):
        human_summary(source)
