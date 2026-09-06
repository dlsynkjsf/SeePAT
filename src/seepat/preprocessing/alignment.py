from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

BILABIAL_PHONES = frozenset({"P", "B", "M"})
SILENCE_PHONES = frozenset({"", "SIL", "<EPS>"})


class MfaAlignmentError(RuntimeError):
    """Raised when MFA cannot produce a usable alignment for one video."""


def ensure_docker_daemon() -> None:
    """Fail before a batch when Docker is installed but its daemon is stopped."""
    if shutil.which("docker") is None:
        raise FileNotFoundError("Docker is required for the MFA alignment container")
    completed = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode != 0:
        details = (completed.stdout + "\n" + completed.stderr).strip()
        raise RuntimeError(
            "Docker Desktop is not running. Start it before preprocessing."
            + (f"\n{details}" if details else "")
        )


def normalize_arpa_phone(label: str) -> str:
    return re.sub(r"\d+$", "", label.strip().upper())


def parse_mfa_phone_intervals(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = data.get("tiers", {})
    phone_tier = next(
        (tier for name, tier in tiers.items() if "phone" in str(name).lower()), None
    )
    if phone_tier is None:
        raise ValueError(f"MFA output contains no phone tier: {path}")

    intervals: list[dict[str, object]] = []
    for entry in phone_tier.get("entries", []):
        if len(entry) != 3:
            continue
        begin, end, label = entry
        phone = normalize_arpa_phone(str(label))
        intervals.append(
            {
                "phoneme": phone.lower(),
                "phone_start_s": float(begin),
                "phone_end_s": float(end),
                "speaker": "",
            }
        )
    return intervals


def parse_mfa_json(path: Path) -> list[dict[str, object]]:
    return [
        interval
        for interval in parse_mfa_phone_intervals(path)
        if str(interval["phoneme"]).upper() in BILABIAL_PHONES
    ]


class MfaDockerAligner:
    def __init__(
        self,
        image: str,
        cache_dir: Path,
        dictionary: str,
        acoustic_model: str,
    ) -> None:
        ensure_docker_daemon()
        self.image = image
        self.cache_dir = cache_dir.resolve()
        self.dictionary = dictionary
        self.acoustic_model = acoustic_model
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._models_ready = False

    def _run(self, arguments: list[str], data_dir: Path | None = None) -> None:
        command = ["docker", "run", "--rm", "-e", "MFA_ROOT_DIR=/mfa"]
        command.extend(
            [
                "--mount",
                f"type=bind,source={self.cache_dir},target=/mfa",
            ]
        )
        if data_dir is not None:
            command.extend(
                [
                    "--mount",
                    f"type=bind,source={data_dir.resolve()},target=/data",
                ]
            )
        command.append(self.image)
        command.extend(arguments)
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            details = (completed.stdout + "\n" + completed.stderr)[-6000:]
            raise RuntimeError(f"MFA Docker command failed:\n{details}")

    def ensure_models(self) -> None:
        if self._models_ready:
            return
        self._run(["mfa", "model", "download", "dictionary", self.dictionary])
        self._run(["mfa", "model", "download", "acoustic", self.acoustic_model])
        self._models_ready = True

    def align(
        self,
        audio_path: Path,
        transcript: str,
        output_path: Path,
        force: bool = False,
    ) -> list[dict[str, object]]:
        if output_path.is_file() and not force:
            return parse_mfa_json(output_path)
        if not transcript.strip():
            raise ValueError("Cannot force-align an empty transcript")

        self.ensure_models()
        work_dir = output_path.parent.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        resolved_audio = audio_path.resolve()
        try:
            relative_audio = resolved_audio.relative_to(work_dir)
        except ValueError:
            copied_audio = work_dir / "mfa_audio.wav"
            shutil.copy2(resolved_audio, copied_audio)
            relative_audio = copied_audio.relative_to(work_dir)
        transcript_path = work_dir / "transcript.txt"
        transcript_path.write_text(transcript.strip() + "\n", encoding="utf-8")

        try:
            self._run(
                [
                    "mfa",
                    "align_one",
                    f"/data/{relative_audio.as_posix()}",
                    "/data/transcript.txt",
                    self.dictionary,
                    self.acoustic_model,
                    f"/data/{output_path.name}",
                    "--output_format",
                    "json",
                ],
                data_dir=work_dir,
            )
            if not output_path.is_file():
                raise FileNotFoundError(
                    f"MFA did not create expected output: {output_path}"
                )
            return parse_mfa_json(output_path)
        except (RuntimeError, FileNotFoundError, ValueError) as error:
            raise MfaAlignmentError(str(error)) from error
