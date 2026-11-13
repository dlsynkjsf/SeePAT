"""Command-line options for model training; optimization lives in train.py."""

import argparse
import json
from dataclasses import fields
from pathlib import Path

from seepat.training import train


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m seepat.training.train")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--model", choices=train.SUPPORTED_MODELS, default=train.SWIN_BASE_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        help="Limit Train batches per epoch and mark the run as a preflight",
    )
    parser.add_argument(
        "--max-val-batches", dest="max_validation_batches",
        type=int,
        help="Limit Validation batches per epoch and mark the run as a preflight",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--unfreeze-final-backbone-stages",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--backbone-learning-rate", type=float)
    parser.add_argument(
        "--class-weighting",
        choices=("balanced", "balanced_global", "none"),
        default="balanced",
    )
    parser.add_argument("--positive-class-weight-ratio", type=float)
    parser.add_argument(
        "--loss-function",
        choices=("cross_entropy", "focal"),
        default="cross_entropy",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--supervision", choices=("event", train.VIDEO_MAX), default="event")
    parser.add_argument(
        "--selection-metric",
        choices=(train.VIDEO_F1_SELECTION, train.VIDEO_BALANCED_ACCURACY_SELECTION),
        default=train.VIDEO_F1_SELECTION,
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    options = train.TrainingOptions(**{f.name: getattr(args, f.name) for f in fields(train.TrainingOptions)})
    report = train.train_from_manifests(
        train_manifest=args.train_manifest,
        validation_manifest=args.val_manifest,
        output_dir=args.output_dir,
        project_root=args.project_root,
        options=options,
        device_name=args.device,
        pretrained=args.pretrained,
        model_name=args.model,
        resume_from=args.resume_from,
        initialize_from=args.initialize_from,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
