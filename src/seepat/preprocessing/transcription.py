from __future__ import annotations

from pathlib import Path


class WhisperTranscriber:
    def __init__(self, model_name: str, device: str, compute_type: str) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                'Install preprocessing dependencies with: pip install -e ".[preprocess]"'
            ) from error

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: Path) -> dict[str, object]:
        segments, info = self.model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        materialized = [
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
            for segment in segments
            if segment.text.strip()
        ]
        text = " ".join(segment["text"] for segment in materialized).strip()
        return {
            "text": text,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": materialized,
        }
