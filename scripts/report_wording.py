"""Plain-language presentation of sealed local results; never change evidence."""

import csv
import hashlib
import io
import json
import math
from pathlib import Path

from seepat.inference.xai_paragraphs import human_summary


def video_summary(video, output, bindings):
    fallback = human_summary(video, technical=False)
    if video["verdict"] == "not_evaluated":
        return fallback
    pins = {Path(row["path"]).resolve(): row["sha256"] for row in bindings}

    def bound(relative):
        path = (output / relative).resolve()
        if path not in pins:
            return None
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != pins[path]:
            raise ValueError("Saved explanation evidence changed")
        return data.decode("utf-8-sig")

    count = video["clips_scored"]
    noun = "clip" if count == 1 else "clips"
    checked = f"It checked {count} short {noun}."
    highlight = video["highlight"]
    phone, word = "", ""
    saved = bound("result/event_predictions.csv")
    if saved is not None:
        rows = list(csv.DictReader(io.StringIO(saved)))
        scores = [float(row["manipulated_probability"]) for row in rows]
        if (len(rows) != count or len({row["event_id"] for row in rows}) != count
                or any(row["video_id"] != video["video_id"] for row in rows)
                or any(not math.isfinite(score) or not 0 <= score <= 1 for score in scores)
                or max(scores) != video["video_score"]):
            raise ValueError("Saved clip predictions differ from the video report")
        selected = next((row for row in rows if row["event_id"] == highlight["event_id"]), None)
        if (selected is None or float(selected["manipulated_probability"]) != video["video_score"]
                or selected["kind"] != highlight["kind"]):
            raise ValueError("Saved review clip differs from the video report")
        phone = selected["phoneme"].lower()
        flagged = sum(score >= video["threshold"] for score in scores)
        checked = (f"It flagged {flagged} of the {count} short {noun} it checked." if flagged
                   else f"None of the {count} short {noun} it checked triggered a warning.")
        if flagged == count:
            checked = ("It flagged the short clip it checked." if count == 1 else
                       f"It flagged all {count} short clips it checked.")
        elif count == 1:
            checked = "The short clip it checked did not trigger a warning."
    if video["verdict"] == "authentic":
        return (f"SeePAT classified this video as authentic (real). {checked} "
                "This result covers the analyzed clips, not every moment of the video.")
    stamp = highlight["timestamp"]
    if stamp["status"] == "verified" and highlight["kind"] == "bilabial":
        events = bound("prepared/events_raw.csv")
        alignment = bound(f"prepared/videos/{video['video_id']}/alignment.json")
        if events is not None and alignment is not None:
            event = next((row for row in csv.DictReader(io.StringIO(events))
                          if row["event_id"] == highlight["event_id"] and row["video_id"] == video["video_id"]), None)
            if (event is not None and float(event["video_phone_start_s"]) == stamp["start_s"]
                    and float(event["video_phone_end_s"]) == stamp["end_s"]):
                start, end = float(event["phone_start_s"]), float(event["phone_end_s"])
                words = [text for a, b, text in json.loads(alignment)["tiers"]["words"]["entries"]
                         if a <= start < end <= b and text and len(text) <= 60
                         and all(char.isalnum() or char in "'-\u2019" for char in text)]
                word = words[0] if len(words) == 1 else ""
    if stamp["status"] == "verified":
        context = (f'near "{word}" in the transcript' if word else
                   f'during the "{phone}" sound' if highlight["kind"] == "bilabial" and phone in {"m", "b", "p"} else
                   "in a short mouth-movement clip" if highlight["kind"] == "context" else "in a speech clip")
        warning = f"Around {stamp['start_s']:.2f} seconds, {context}, this clip triggered SeePAT's strongest warning."
    else:
        warning = "A reliable review time is unavailable."
    return (f"SeePAT classified this video as manipulated (fake). "
            f"{checked} {warning}")
