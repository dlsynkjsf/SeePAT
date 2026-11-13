"""Command interface for SeePAT inference, training, and external assessment."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_CONFIG = "configs/local_inference.yaml"

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
                "-m",
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
