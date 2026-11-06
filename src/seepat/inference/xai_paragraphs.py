"""Two-paragraph wording candidate; existing frozen renderers stay unchanged."""

from .readable_xai import human_summary as validate_video


def human_summary(video, *, technical=True):
    """Reuse saved-fact validation; labels describe model predictions, not proof."""
    validate_video(video)
    if video["verdict"] == "not_evaluated":
        plain = ("SeePAT could not assess this video. "
                 "No real/fake classification was made because no clips could be scored.")
        detail = "Technical details: 0 clips scored; no manipulation score or verdict is available."
        return f"{plain}\n\n{detail}" if technical else plain
    plain = ("SeePAT classified this video as authentic (real). "
             "The clips it analyzed did not provide enough evidence to flag manipulation.")
    stamp = video["highlight"]["timestamp"]
    verified = stamp["status"] == "verified"
    if video["verdict"] == "manipulated":
        plain = ("SeePAT classified this video as manipulated (fake). "
                 + (f"Start your review around {stamp['start_s']:.2f} seconds, "
                    "at the clip linked to its strongest warning." if verified
                    else "A clip triggered a warning, but a reliable review timestamp is unavailable."))
    if not technical:
        return plain
    count, score, cutoff = (video[key] for key in ("clips_scored", "video_score", "threshold"))
    comparison = "is below" if video["verdict"] == "authentic" else "meets or exceeds"
    noun = "clip" if count == 1 else "clips"
    if verified:
        clock = "an aligned speech sound" if video["highlight"]["kind"] == "bilabial" else "a sampled visual frame"
        timing = (f"The highest-scoring clip is linked to {clock} at {stamp['start_s']:.2f} seconds; "
                  "this is not a confirmed mismatch or edit.")
    else:
        timing = "A reliable review timestamp is unavailable; no mismatch or edit location is established."
    detail = (f"Technical details: {count} {noun} analyzed; the highest score ({score:.6f}) "
              f"{comparison} the decision cutoff ({cutoff:.6f}), both rounded. {timing}")
    return f"{plain}\n\n{detail}"
