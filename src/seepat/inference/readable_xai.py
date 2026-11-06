"""Plain-language display of a saved report; never score, refit or rewrite evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def human_summary(video, *, technical=False):
    """Explain a saved verdict; classifier ranking is review priority, not localization."""
    count, score, cutoff = (video[key] for key in ("clips_scored", "video_score", "threshold"))
    verdict = video["verdict"]
    if verdict == "not_evaluated":
        if type(count) is not int or count != 0 or score not in (None, "") or video["highlight"] is not None:
            raise ValueError("Unscored video has a score or review highlight")
        return ("SeePAT could not assess this video because no clips could be scored. "
                "No manipulation verdict was made.")
    if verdict not in {"manipulated", "authentic"} or type(count) is not int or count < 1:
        raise ValueError("Unsupported verdict or analyzed-clip count")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in (score, cutoff)) or (score >= cutoff) != (verdict == "manipulated"):
        raise ValueError("Saved score and verdict disagree")
    highlight = video["highlight"]
    stamp = highlight["timestamp"]
    kind = highlight["kind"]
    if kind not in {"bilabial", "context"} or highlight["probability"] != score:
        raise ValueError("Invalid saved review highlight")
    noun = "clip" if count == 1 else "clips"
    if stamp["status"] == "verified":
        start = stamp["start_s"]
        end = stamp["end_s"] if kind == "bilabial" else stamp["last_sample_s"]
        expected = "video_phoneme_interval" if kind == "bilabial" else "sampled_context_source_frame"
        if (stamp["kind"] != expected or any(type(value) not in (int, float) or not math.isfinite(value)
                                            for value in (start, end))
                or start < 0 or end < start or (kind == "bilabial" and end == start)):
            raise ValueError("Invalid saved review timestamp")
        review = f"It checked {count} short {noun}; start your review at {start:.2f} seconds."
        clock = "an aligned speech event" if kind == "bilabial" else "a sampled visual frame"
        detail = f"the time marks {clock}, not a confirmed edit"
    elif stamp["status"] == "unavailable":
        review = f"It checked {count} short {noun}. A review timestamp is unavailable."
        detail = "no confirmed edit location is established"
    else:
        raise ValueError("Unsupported saved timestamp status")
    lead = ("SeePAT flagged this video as potentially manipulated." if verdict == "manipulated"
            else "SeePAT did not flag this video as manipulated.")
    comparison = "meets or exceeds" if verdict == "manipulated" else "is below"
    plain = f"{lead} {review}"
    if not technical:
        return plain
    return (f"{plain} "
            f"Technical: the highest clip score ({score:.6f}) {comparison} the cutoff "
            f"({cutoff:.6f}), both rounded; {detail}.")


def render_saved_report(report_dir, *, technical=False, paragraph_layout=False):
    """Check the small presentation seal only; do not revalidate models/media."""
    directory = Path(report_dir)
    summary_bytes = (directory / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    names = {"report.md", "report.json", "technical_report.md"}
    if (summary["version"] != "local-inference-explanation-v1" or summary["status"] != "complete"
            or summary["scope"] not in {"synthetic_engineering", "local_unlabelled_inference"}
            or summary["network_requests"] != 0 or set(summary["output_sha256"]) != names):
        raise ValueError("Unsupported saved explanation")
    files = {name: (directory / name).read_bytes() for name in names}
    if any(hashlib.sha256(data).hexdigest() != summary["output_sha256"][name] for name, data in files.items()):
        raise ValueError("Saved explanation hash differs; preserve the source")
    report = json.loads(files["report.json"])
    if (report["version"] != "local-inference-workflow-v1" or report["scope"] != summary["scope"]
            or report["network_requests"] != 0 or report["rendering"] != "local_deterministic"
            or not isinstance(report["videos"], list) or not report["videos"]):
        raise ValueError("Unsupported saved report")
    summarize = human_summary
    wording_path, wording_bytes = None, None
    if paragraph_layout:
        from .xai_paragraphs import human_summary as summarize

        wording_path = Path(__file__).with_name("xai_paragraphs.py")
        wording_bytes = wording_path.read_bytes()
        if hashlib.sha256(wording_bytes).hexdigest() != "45e18b25e621b3bc1c2c6bd409a07dab2ddccc5504791191674af3ea2c53360f":
            raise ValueError("Paragraph wording source changed; review its release binding")
    paragraphs = [summarize(video, technical=technical) for video in report["videos"]]
    if ((directory / "summary.json").read_bytes() != summary_bytes
            or any((directory / name).read_bytes() != data for name, data in files.items())):
        raise ValueError("Saved report changed during display")
    if wording_path is not None and wording_path.read_bytes() != wording_bytes:
        raise ValueError("Paragraph wording changed during display")
    prefix = "ENGINEERING CHECK: synthetic input, not thesis performance.\n\n" if report["scope"] == "synthetic_engineering" else ""
    return prefix + "\n\n".join(paragraphs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--technical", action="store_true", help="Append technical details as the last sentence")
    args = parser.parse_args(argv)
    try:
        print(render_saved_report(args.report_dir, technical=args.technical))
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"READABLE XAI FAILED | {error}") from None


if __name__ == "__main__":
    main()
