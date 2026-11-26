"""Command interface for SeePAT inference, training, and external assessment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CONFIG = "configs/local_inference.yaml"
# Frozen configs pin src/seepat/live_progress.py by hash, so the one-line tracker is
# swapped in when a command starts instead of editing that file.
BOOT = ("import runpy, sys; from scripts.seepat_cli import compact_progress; compact_progress(); "
        "runpy.run_module(sys.argv.pop(1), run_name='__main__', alter_sys=True)")

COMMANDS = {
    "ui": ("scripts.demo_ui", "Open the native single-video desktop interface"),
    "run": ("seepat.workflow", "Run all stages in a workflow config"),
    "watch": ("seepat.progress", "Watch a workflow without starting work"),
    "verify": ("seepat.progress", "Verify saved workflow artifacts"),
    "preprocess": ("seepat.pipeline", "Prepare videos from a pipeline config"),
    "train": ("seepat.training.cli", "Train/resume a model, or use --config for a training workflow"),
    "readiness": ("seepat.training.readiness", "Check training inputs and readiness"),
    "calibrate": (
        "seepat.training.calibration",
        "Fit Train calibration or apply frozen calibration",
    ),
    "verdict": ("seepat.verdict", "Score prepared AV++ inputs or select a Validation cutoff"),
    "xai": ("seepat.inference.readable_xai", "Display a saved XAI Forensic Trace"),
    "data": ("seepat.cli", "Dataset inventory, sampling, and extraction tools"),
    "pia": ("scripts.pia_assessment", "Run/resume PIA on the fixed 1,000 external requests"),
}


def progress_line(record):
    """One-line replacement for live_progress.format_live_progress."""
    if record.get("status") == "unavailable":
        return str(record["message"])
    done, total = int(record.get("finished", 0)), int(record.get("total", 0))
    stamp = time.strftime("%H:%M:%S", time.localtime(float(record.get("updated_at", time.time()))))
    parts = [f"{stamp} {record['status']}"]
    if int(record.get("jobs_total", 1)) > 1:
        parts.append(f"job {record.get('job_index', 0)}/{record['jobs_total']} {record.get('job', '')}")
    parts.append(str(record.get("phase") or "starting"))
    if total:
        parts.append(f"{done}/{total} ({min(1.0, done / total):.0%})")
    if record.get("error"):
        parts.append(str(record["error"]))
    return " | ".join(parts)


def compact_progress():
    from seepat import live_progress

    live_progress.format_live_progress = progress_line


def command_args(command, args):
    args = list(args)
    if any(value in {"--help", "-h"} for value in args):
        return args
    if command in {"run", "watch", "verify"}:
        flag = "--config" if command == "run" else "--workflow-config"
        args = [
            value.replace("--config", flag, 1) if value.split("=", 1)[0] == "--config" else value
            for value in args
        ]
        options = {value.split("=", 1)[0] for value in args}
        if not options.intersection({"--config", "--workflow-config"}):
            args = [flag, DEFAULT_CONFIG, *args]
        if command == "watch" and "--watch-seconds" not in options:
            args += ["--watch-seconds", "5"]
        if command == "verify" and "--verify" not in options:
            args += ["--verify"]
    if command == "pia" and (not args or args[0].startswith("--config")):
        args.insert(0, "run")
    return args


def main(argv=None):
    parser = argparse.ArgumentParser(prog="seepat", description=__doc__)
    parser.add_argument("command", choices=COMMANDS, help="Use COMMAND --help for its options")
    parser.add_argument("args", nargs=argparse.REMAINDER)
    parser.epilog = "\n".join(f"  {key:<12} {value[1]}" for key, value in COMMANDS.items())
    parser.epilog += (
        f"\n\nDefault run/watch/verify config: {DEFAULT_CONFIG} (local inference examples)."
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        parser.print_help()
        return 0
    args = parser.parse_args(values)
    project = Path(__file__).resolve().parents[1]
    python = project / (".venv/Scripts/python.exe" if os.name == "nt" else ".venv/bin/python")
    module = COMMANDS[args.command][0]
    if args.command == "train" and any(value.split("=", 1)[0] == "--config" for value in args.args):
        module = "seepat.workflow"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project / "src"), str(project), env.get("PYTHONPATH", "")]
    )
    try:
        result = subprocess.run(
            [
                str(python) if python.is_file() else sys.executable,
                "-u",
                "-c",
                BOOT,
                module,
                *command_args(args.command, args.args),
            ],
            env=env,
            cwd=project,
            check=False,
        )
        return result.returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
