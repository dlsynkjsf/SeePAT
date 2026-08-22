from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DEFAULT_REPO_ID = "ControlNet/AV-Deepfake1M-PlusPlus"
DATASET_SPLITS = ("train", "val", "testA", "testB")
METADATA_FILES = ("train_metadata.json", "val_metadata.json")


@dataclass(frozen=True)
class SplitInventory:
    split: str
    file_count: int
    total_bytes: int

    @property
    def gib(self) -> float:
        return self.total_bytes / (1024**3)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["gib"] = round(self.gib, 2)
        return result


def audit_repository(
    repo_id: str = DEFAULT_REPO_ID,
    splits: Iterable[str] = DATASET_SPLITS,
    token: bool | str | None = True,
) -> list[SplitInventory]:
    api = HfApi(token=token)
    inventories: list[SplitInventory] = []

    for split in splits:
        files = [
            item
            for item in api.list_repo_tree(
                repo_id=repo_id,
                path_in_repo=split,
                repo_type="dataset",
                recursive=True,
                expand=True,
            )
            if getattr(item, "size", None) is not None
        ]
        inventories.append(
            SplitInventory(
                split=split,
                file_count=len(files),
                total_bytes=sum(int(item.size) for item in files),
            )
        )

    return inventories


def write_repository_audit(path: Path, inventories: Iterable[SplitInventory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [inventory.to_dict() for inventory in inventories]
    required = sum(row["total_bytes"] for row in rows if row["split"] in {"train", "val"})
    document = {
        "splits": rows,
        "required_train_val_bytes": required,
        "required_train_val_gib": round(required / (1024**3), 2),
        "note": "Multipart ZIP volumes are not independent example shards.",
    }
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def download_metadata(
    output_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    token: bool | str | None = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for filename in METADATA_FILES:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=output_dir,
            token=token,
        )
        downloaded.append(Path(local_path))
    return downloaded
