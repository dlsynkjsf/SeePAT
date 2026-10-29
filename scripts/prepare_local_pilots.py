"""Select nested, category-balanced pilots from the audited local subset.

Run from the package root with ``python -m scripts.prepare_local_pilots``.
Only metadata and media presence are inspected; no model, decoding or training runs.
Existing identical outputs are reusable; changed outputs require a new directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from scripts import prepare_local_subset as source

CATEGORY_ORDER = ("real", "audio_modified", "visual_modified", "both_modified")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(row: dict[str, str], seed: int) -> tuple[bytes, str]:
    identity = [seed, row["split"], row["modify_type"], row["source_group"], row["file"]]
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest(), row[
        "file"
    ]


def _select(rows: list[dict[str, str]], size: int, seed: int) -> list[dict[str, str]]:
    pools = {
        category: sorted(
            (row for row in rows if row["modify_type"] == category),
            key=lambda row: _rank(row, seed),
        )
        for category in CATEGORY_ORDER
    }
    per_category = size // len(CATEGORY_ORDER)
    if any(len(pool) < per_category for pool in pools.values()):
        raise ValueError(f"Not enough unique sources for {size} category-balanced videos")
    # Interleave categories; smaller pilots are exact prefixes of larger pilots.
    return [pools[category][index] for index in range(per_category) for category in CATEGORY_ORDER]


def prepare_local_pilots(
    cleaned_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    *,
    seed: int = 20261030,
    train_sizes: tuple[int, ...] = (500, 1000),
    validation_size: int = 200,
) -> dict[str, object]:
    """Preserve complete source metadata and select before preprocessing/eligibility.

    All cleaned Train sources must be disjoint from the full cleaned Validation
    population, not merely the 200 selected development videos. No selection uses
    model scores, processing success, event counts or eligibility outcomes.
    """
    sizes = (*train_sizes, validation_size)
    if (
        not train_sizes
        or any(type(size) is not int or size < 4 or size % 4 for size in sizes)
        or tuple(sorted(set(train_sizes))) != train_sizes
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError(
            "Use ascending unique Train sizes divisible by four and a nonnegative seed"
        )
    cleaned, root, destination = cleaned_dir.resolve(), dataset_root.resolve(), output_dir.resolve()
    if destination.is_relative_to(root) or destination.is_relative_to(cleaned):
        raise ValueError("Pilot output must be outside uploaded data and cleaned input directories")
    audit_path = cleaned / "audit.json"
    upstream = json.loads(audit_path.read_text(encoding="utf-8"))
    if upstream.get("version") != "local-subset-v1":
        raise ValueError("Expected the audited local-subset-v1 manifests")
    headers, rows = {}, {}
    groups: dict[str, set[str]] = {}
    input_hashes = {"audit.json": _sha(audit_path)}
    for split in ("train", "val"):
        path = cleaned / f"{split}.csv"
        input_hashes[path.name] = _sha(path)
        if upstream.get("output_sha256", {}).get(path.name) != input_hashes[path.name]:
            raise ValueError(f"Cleaned manifest changed since its audit: {path}")
        headers[split], rows[split] = source._read_csv(path, source.METADATA_COLUMNS)
        groups[split] = set()
        files = set()
        for row in rows[split]:
            source._validate_metadata(row, split)
            group = source._group(row)
            if row["source_group"] != group or group in groups[split] or row["file"] in files:
                raise ValueError(
                    f"Noncanonical or repeated source/file in cleaned {split}: {row['file']}"
                )
            groups[split].add(group)
            files.add(row["file"])
    if groups["train"] & groups["val"]:
        raise ValueError("Canonical source overlap with the full Validation population")
    selections = {
        **{f"train{size}.csv": _select(rows["train"], size, seed) for size in train_sizes},
        f"val{validation_size}.csv": _select(rows["val"], validation_size, seed),
    }
    media_bytes = {}
    checked = set()
    for selected in selections.values():
        for row in selected:
            key = (row["split"], row["file"])
            if key not in checked:
                media_bytes[key] = source._media_path(root, *key).stat().st_size
                checked.add(key)
    outputs = {
        name: source._csv_bytes(headers[selected[0]["split"]], selected)
        for name, selected in selections.items()
    }
    audit = {
        "version": "local-pilots-v1",
        "seed": seed,
        "cleaned_dir": str(cleaned),
        "dataset_root": str(root),
        "input_sha256": input_hashes,
        "ranking": "SHA-256 of JSON [seed,split,modify_type,source_group,file]; file breaks ties",
        "order": "round-robin real/audio_modified/visual_modified/both_modified",
        "selection": "category-stratified metadata only, before processing/eligibility; no outcomes",
        "nested_train_sizes": list(train_sizes),
        "full_validation_source_overlap": 0,
        "validation_is_internal_development": True,
        "subject_identity_separation_asserted": False,
        "media_content_decoded_or_hashed": False,
        "outputs": {
            name: {
                "rows": len(selected),
                "source_groups": len({row["source_group"] for row in selected}),
                "category_counts": dict(
                    sorted(Counter(row["modify_type"] for row in selected).items())
                ),
                "media_bytes": sum(media_bytes[(row["split"], row["file"])] for row in selected),
                "sha256": hashlib.sha256(outputs[name]).hexdigest(),
            }
            for name, selected in selections.items()
        },
    }
    outputs["audit.json"] = (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    # Check every destination before writing anything; never silently replace a run.
    for name, data in outputs.items():
        path = destination / name
        if path.exists() and (not path.is_file() or path.read_bytes() != data):
            raise FileExistsError(
                f"Changed pilot output exists; choose a new output directory: {path}"
            )
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = destination / name
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(data)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-dir", type=Path, default=Path("data/manifests/local_subset_v1"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../Videos_Dataset/SeePAT_dataset_5000_train_1000_val"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests/local_pilots_v1"))
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    audit = prepare_local_pilots(
        args.cleaned_dir, args.dataset_root, args.output_dir, seed=args.seed
    )
    print(
        json.dumps(
            {name: item["category_counts"] for name, item in audit["outputs"].items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
