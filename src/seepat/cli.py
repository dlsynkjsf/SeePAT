from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

from seepat.data.archive import extract_manifest_videos
from seepat.data.inventory import build_inventory
from seepat.data.repository import (
    DATASET_SPLITS,
    DEFAULT_REPO_ID,
    audit_repository,
    download_metadata,
    write_repository_audit,
)
from seepat.data.sampling import sample_pilot, sample_training_canary

DEFAULT_CATEGORIES = ("real", "audio_modified", "visual_modified", "both_modified")


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="seepat-data")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-repo", help="Audit AV++ split sizes without download")
    audit.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    audit.add_argument("--output", type=Path, required=True)

    download = commands.add_parser("download-metadata", help="Download AV++ metadata only")
    download.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    download.add_argument("--output-dir", type=Path, required=True)

    inventory = commands.add_parser("build-inventory", help="Build SQLite metadata inventory")
    inventory.add_argument("--metadata", type=Path, action="append", required=True)
    inventory.add_argument("--database", type=Path, required=True)
    inventory.add_argument("--summary", type=Path, required=True)

    sample = commands.add_parser("sample-pilot", help="Create deterministic pilot manifest")
    sample.add_argument("--database", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--split", default="val")
    sample.add_argument("--category", action="append", dest="categories")
    sample.add_argument("--per-category", type=int, default=5)
    sample.add_argument("--seed", type=int, default=20260822)

    canary = commands.add_parser(
        "sample-training-canary",
        help="Create a deterministic, validation-safe AV++ Train canary manifest",
    )
    canary.add_argument("--database", type=Path, required=True)
    canary.add_argument("--output", type=Path, required=True)
    canary.add_argument("--summary", type=Path, required=True)
    canary.add_argument("--split", default="train")
    canary.add_argument("--category", action="append", dest="categories")
    canary.add_argument("--per-category", type=int, default=250)
    canary.add_argument("--seed", type=int, default=20260824)
    canary.add_argument("--exclude-split", action="append", dest="excluded_splits")

    extract = commands.add_parser(
        "extract-pilot", help="Extract only videos named by a pilot manifest"
    )
    extract.add_argument("--archive", type=Path, required=True)
    extract.add_argument("--manifest", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--report", type=Path, required=True)
    extract.add_argument("--seven-zip", type=Path)
    return parser


def main() -> None:
    args = create_parser().parse_args()

    if args.command == "audit-repo":
        inventories = audit_repository(args.repo_id, DATASET_SPLITS)
        write_repository_audit(args.output, inventories)
        print(json.dumps([inventory.to_dict() for inventory in inventories], indent=2))
        return

    if args.command == "download-metadata":
        for downloaded_path in download_metadata(args.output_dir, args.repo_id):
            print(downloaded_path)
        return

    if args.command == "build-inventory":
        summary = build_inventory(args.metadata, args.database, args.summary)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "sample-pilot":
        categories = args.categories or list(DEFAULT_CATEGORIES)
        rows = sample_pilot(
            database_path=args.database,
            output_path=args.output,
            split=args.split,
            categories=categories,
            per_category=args.per_category,
            seed=args.seed,
        )
        print(f"Wrote {len(rows)} rows to {args.output}")
        return

    if args.command == "sample-training-canary":
        categories = args.categories or list(DEFAULT_CATEGORIES)
        excluded_splits = args.excluded_splits or ["val"]
        _, summary = sample_training_canary(
            database_path=args.database,
            output_path=args.output,
            summary_path=args.summary,
            split=args.split,
            categories=categories,
            per_category=args.per_category,
            seed=args.seed,
            excluded_splits=excluded_splits,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "extract-pilot":
        report = extract_manifest_videos(
            archive_first_volume=args.archive,
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            report_path=args.report,
            seven_zip_path=args.seven_zip,
        )
        print(json.dumps(report, indent=2))
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
