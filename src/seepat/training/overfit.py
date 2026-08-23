from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from seepat.artifacts import atomic_write_json
from seepat.models.swin_baseline import SwinBaseEventClassifier, parameter_counts
from seepat.training.dataset import MouthEventDataset


def select_balanced_event_indices(
    rows: list[dict[str, str]],
    events_per_class: int,
) -> list[int]:
    if events_per_class < 1:
        raise ValueError("events_per_class must be positive")

    selected: list[int] = []
    used_groups: set[str] = set()
    for class_id in ("1", "0"):
        class_indices: list[int] = []
        for index, row in enumerate(rows):
            if row["class_id"] != class_id or row["source_group"] in used_groups:
                continue
            class_indices.append(index)
            used_groups.add(row["source_group"])
            if len(class_indices) == events_per_class:
                break
        if len(class_indices) != events_per_class:
            raise ValueError(
                f"Could not select {events_per_class} distinct-source events for class {class_id}"
            )
        selected.extend(class_indices)
    return selected


def _metrics(
    classifier: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float]:
    with torch.no_grad():
        logits = classifier(features)
        loss = F.cross_entropy(logits, labels).item()
        accuracy = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, accuracy


def run_head_overfit(
    manifest_path: Path,
    report_path: Path,
    project_root: Path = Path("."),
    device_name: str = "cuda",
    events_per_class: int = 4,
    sequence_length: int = 16,
    image_size: int = 224,
    maximum_steps: int = 300,
    learning_rate: float = 0.05,
) -> dict[str, object]:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if maximum_steps < 1:
        raise ValueError("maximum_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(20260823)
    device = torch.device(device_name)
    torch_cache = project_root / ".cache" / "torch"
    torch.hub.set_dir(str(torch_cache))
    dataset = MouthEventDataset(
        manifest_path=manifest_path,
        project_root=project_root,
        sequence_length=sequence_length,
        image_size=image_size,
    )
    indices = select_balanced_event_indices(dataset.rows, events_per_class)
    model = SwinBaseEventClassifier(pretrained=True, freeze_backbone=True).to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    feature_started = perf_counter()
    feature_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    selected_events: list[dict[str, object]] = []
    for index in indices:
        event = dataset[index]
        video = event["video"].unsqueeze(0).to(device)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            feature_rows.append(model.extract_features(video).float().squeeze(0))
        label_rows.append(event["label"].to(device))
        selected_events.append(
            {
                "event_id": event["event_id"],
                "source_group": event["source_group"],
                "label": int(event["label"].item()),
            }
        )
    feature_seconds = perf_counter() - feature_started

    features = torch.stack(feature_rows)
    labels = torch.stack(label_rows)
    classifier = model.classifier
    initial_loss, initial_accuracy = _metrics(classifier, features, labels)
    classifier.train()
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
    )

    training_started = perf_counter()
    completed_steps = 0
    for step in range(1, maximum_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(classifier(features), labels)
        loss.backward()
        optimizer.step()
        completed_steps = step
        final_loss, final_accuracy = _metrics(classifier, features, labels)
        if final_accuracy == 1.0 and final_loss <= 0.01:
            break
    training_seconds = perf_counter() - training_started

    report: dict[str, object] = {
        "purpose": "wiring_check_only_not_a_reported_experiment",
        "model": "torchvision.swin3d_b",
        "pretrained_weights": "Swin3D_B_Weights.KINETICS400_V1",
        "pretrained_backbone": True,
        "backbone_frozen": True,
        "dataset_split": sorted({dataset.rows[index]["dataset_split"] for index in indices}),
        "events": len(indices),
        "events_per_class": events_per_class,
        "distinct_source_groups": len({row["source_group"] for row in selected_events}),
        "selected_events": selected_events,
        "input_shape": [3, sequence_length, image_size, image_size],
        "parameters": parameter_counts(model),
        "initial_loss": initial_loss,
        "initial_accuracy": initial_accuracy,
        "final_loss": final_loss,
        "final_accuracy": final_accuracy,
        "steps": completed_steps,
        "feature_extraction_seconds": round(feature_seconds, 3),
        "head_training_seconds": round(training_seconds, 3),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "peak_gpu_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
        "torch_cache": torch_cache.as_posix(),
        "passed": final_accuracy == 1.0 and final_loss <= 0.01,
    }
    atomic_write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m seepat.training.overfit")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/smoke/swin_head_overfit.json"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--events-per-class", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--maximum-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    args = parser.parse_args()

    report = run_head_overfit(
        manifest_path=args.manifest,
        report_path=args.report,
        project_root=args.project_root,
        device_name=args.device,
        events_per_class=args.events_per_class,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        maximum_steps=args.maximum_steps,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("The tiny classifier head did not overfit")


if __name__ == "__main__":
    main()
