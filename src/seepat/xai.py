"""Reasoning-LLM forensic trace (Figure 4.1 XAI output).

The trace layer is strictly downstream of the verdict:

* it consumes the frozen verdict artifacts (``evaluation.json``,
  ``video_verdicts.csv``, ``event_predictions.csv``) as read-only evidence;
* it never recomputes probabilities, thresholds or verdicts;
* the reasoning LLM only translates the supplied numerical evidence into
  human-readable text, with grounding rules that forbid inventing evidence.

The default client is a deterministic template renderer that runs fully
offline. :class:`OpenAICompatibleClient` provides the paper's GPT-4o-mini
integration over plain HTTP (no extra dependency); LangChain or any other
orchestrator can be used by implementing the same :class:`ReasoningClient`
protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from seepat.artifacts import atomic_write_json, file_sha256, read_csv_rows
from seepat.evidence import EVIDENCE_LABELS, FUSION_EVIDENCE_FIELDS

FORENSIC_TRACE_VERSION = "forensic-trace-v1"
SYSTEM_GROUNDING_RULES = (
    "You are a forensic explanation writer for the SeePAT audio-visual "
    "deepfake detector. You receive frozen numerical evidence as JSON. "
    "Rules: (1) every statement must be grounded in the supplied JSON; "
    "(2) never change, recompute or contradict probabilities, thresholds or "
    "verdicts; (3) never claim evidence that is absent; explicitly mention "
    "missing measurements; (4) explain which measurements support or "
    "contradict manipulation, in the Observation-Thought-Action style; "
    "(5) the trace is explanatory only and must not propose a different verdict."
)


class ReasoningClient(Protocol):
    """Minimal interface for a reasoning LLM used for trace generation."""

    label: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _event_evidence(record: dict[str, str]) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for field in FUSION_EVIDENCE_FIELDS:
        value = record.get(field, "").strip()
        if value:
            evidence[EVIDENCE_LABELS[field]] = value
    return evidence


def build_evidence_bundle(
    verdict_dir: Path,
    video_id: str | None = None,
    max_events: int = 5,
    max_videos: int = 25,
) -> dict[str, object]:
    """Assemble the frozen evidence bundle the reasoning LLM may cite."""
    if max_events < 1 or max_videos < 1:
        raise ValueError("max_events and max_videos must be positive")
    evaluation_path = verdict_dir / "evaluation.json"
    verdicts_path = verdict_dir / "video_verdicts.csv"
    events_path = verdict_dir / "event_predictions.csv"
    for path in (evaluation_path, verdicts_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"Verdict artifact is missing: {path}")
    evaluation = _read_json(evaluation_path)
    verdict_rows = read_csv_rows(verdicts_path)
    event_rows = read_csv_rows(events_path)
    if video_id is not None:
        verdict_rows = [row for row in verdict_rows if row.get("video_id") == video_id]
        event_rows = [row for row in event_rows if row.get("video_id") == video_id]
        if not verdict_rows:
            raise ValueError(f"No verdict row for video {video_id!r}")

    events_by_video: dict[str, list[dict[str, str]]] = {}
    for row in event_rows:
        events_by_video.setdefault(row.get("video_id", ""), []).append(row)

    ordered_videos = sorted(
        verdict_rows,
        key=lambda row: float(row.get("maximum_probability") or 0.0),
        reverse=True,
    )
    selected = ordered_videos[:max_videos]
    videos: list[dict[str, object]] = []
    for row in selected:
        identifier = row.get("video_id", "")
        events = events_by_video.get(identifier, [])
        events.sort(
            key=lambda event: float(event.get("manipulated_probability") or 0.0),
            reverse=True,
        )
        videos.append(
            {
                "video_id": identifier,
                "verdict": row.get("verdict", ""),
                "maximum_probability": row.get("maximum_probability", ""),
                "event_count": row.get("event_count", ""),
                "events": [
                    {
                        "event_id": event.get("event_id", ""),
                        "phoneme": event.get("phoneme", ""),
                        "manipulated_probability": event.get(
                            "manipulated_probability", ""
                        ),
                        "sync_gap_score": event.get("sync_gap_score", ""),
                        "evidence": _event_evidence(event),
                    }
                    for event in events[:max_events]
                ],
            }
        )
    missing_evidence_events = sum(
        1
        for event in event_rows
        for field in FUSION_EVIDENCE_FIELDS
        if not event.get(field, "").strip()
    )
    return {
        "trace_version": FORENSIC_TRACE_VERSION,
        "scope": {"video_id": video_id} if video_id else {"video_id": None},
        "sources": {
            "evaluation": {
                "path": evaluation_path.as_posix(),
                "sha256": file_sha256(evaluation_path),
            },
            "video_verdicts": {
                "path": verdicts_path.as_posix(),
                "sha256": file_sha256(verdicts_path),
            },
            "event_predictions": {
                "path": events_path.as_posix(),
                "sha256": file_sha256(events_path),
            },
        },
        "model": {
            "model_name": evaluation.get("model_name"),
            "training_version": evaluation.get("training_version"),
        },
        "split": evaluation.get("split"),
        "threshold": evaluation.get("threshold"),
        "aggregation": evaluation.get("aggregation"),
        "videos_scored": len(verdict_rows),
        "videos_reported": len(videos),
        "videos_omitted": max(0, len(verdict_rows) - len(videos)),
        "events_scored": len(event_rows),
        "events_with_missing_evidence_values": missing_evidence_events,
        "videos": videos,
    }


def render_evidence_summary(bundle: dict[str, object]) -> str:
    """Deterministic, offline rendering of the evidence bundle."""
    lines = ["# SeePAT Forensic Trace", ""]
    threshold = bundle.get("threshold")
    if isinstance(threshold, dict):
        lines.append(
            "Frozen decision threshold: "
            f"{threshold.get('value')} (source: {threshold.get('source')})"
        )
    lines.append(
        f"Videos reported: {bundle.get('videos_reported')} of "
        f"{bundle.get('videos_scored')} scored"
    )
    lines.append(
        f"Events scored: {bundle.get('events_scored')} "
        f"(missing evidence values: {bundle.get('events_with_missing_evidence_values')})"
    )
    lines.append("")
    videos = bundle.get("videos")
    if not isinstance(videos, list) or not videos:
        lines.append("No evaluated videos were available.")
        return "\n".join(lines)
    for video in videos:
        if not isinstance(video, dict):
            continue
        lines.append(
            f"## Video {video.get('video_id')}: {video.get('verdict')} "
            f"(maximum event probability {video.get('maximum_probability')}, "
            f"events {video.get('event_count')})"
        )
        events = video.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, dict):
                    continue
                evidence = event.get("evidence")
                detail = "; ".join(
                    f"{key}={value}"
                    for key, value in (
                        evidence.items() if isinstance(evidence, dict) else []
                    )
                )
                lines.append(
                    f"- Event {event.get('event_id')} (/{event.get('phoneme')}/): "
                    f"p(manipulated)={event.get('manipulated_probability')}, "
                    f"sync-gap={event.get('sync_gap_score')}"
                    + (f"; {detail}" if detail else "; no calibrated evidence available")
                )
        lines.append("")
    lines.append(
        "Note: this trace is explanatory only. It restates frozen numerical "
        "evidence and does not modify the probability, threshold or verdict."
    )
    return "\n".join(lines)


def build_prompts(bundle: dict[str, object]) -> tuple[str, str]:
    user_prompt = (
        "Frozen SeePAT verdict evidence:\n"
        + json.dumps(bundle, indent=2, sort_keys=True)
        + "\n\nWrite the forensic trace for the most relevant videos."
    )
    return SYSTEM_GROUNDING_RULES, user_prompt


class OpenAICompatibleClient:
    """GPT-4o-mini style chat completion over an OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 60.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.label = f"openai-compatible:{model}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"Missing API key: set the {self.api_key_env} environment variable"
            )
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"Reasoning LLM request failed with HTTP {error.code}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"Reasoning LLM request failed: {error.reason}") from error
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Reasoning LLM response has an unexpected shape") from error
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("Reasoning LLM returned an empty trace")
        return text


def generate_forensic_trace(
    verdict_dir: Path,
    output_path: Path,
    client: ReasoningClient | None = None,
    video_id: str | None = None,
    max_events: int = 5,
    max_videos: int = 25,
) -> dict[str, object]:
    """Write the forensic trace JSON next to the frozen verdict artifacts."""
    bundle = build_evidence_bundle(
        verdict_dir,
        video_id=video_id,
        max_events=max_events,
        max_videos=max_videos,
    )
    deterministic_summary = render_evidence_summary(bundle)
    if client is None:
        trace_text = deterministic_summary
        mode = "deterministic"
        client_label = None
    else:
        system_prompt, user_prompt = build_prompts(bundle)
        trace_text = client.generate(system_prompt, user_prompt)
        mode = "reasoning-llm"
        client_label = client.label
    artifact = {
        "trace_version": FORENSIC_TRACE_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "mode": mode,
        "client": client_label,
        "sources": bundle["sources"],
        "verdict_snapshot": {
            "threshold": bundle.get("threshold"),
            "aggregation": bundle.get("aggregation"),
            "videos_scored": bundle.get("videos_scored"),
            "events_scored": bundle.get("events_scored"),
        },
        "evidence_summary": deterministic_summary,
        "trace": trace_text,
        "disclaimer": (
            "Explanatory only: the trace cannot modify probabilities, thresholds "
            "or verdicts. Deterministic evidence is reproduced verbatim in "
            "evidence_summary; the reasoning text is advisory."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, artifact)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(prog="seepat-xai")
    parser.add_argument("--verdict-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--client", choices=("template", "openai"), default="template")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-events", type=int, default=5)
    parser.add_argument("--max-videos", type=int, default=25)
    args = parser.parse_args()

    client: ReasoningClient | None = None
    if args.client == "openai":
        client = OpenAICompatibleClient(
            model=args.model,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            timeout_seconds=args.timeout_seconds,
        )
    artifact = generate_forensic_trace(
        verdict_dir=args.verdict_dir,
        output_path=args.output,
        client=client,
        video_id=args.video_id,
        max_events=args.max_events,
        max_videos=args.max_videos,
    )
    print(
        json.dumps(
            {
                "trace_version": artifact["trace_version"],
                "mode": artifact["mode"],
                "output": args.output.as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
