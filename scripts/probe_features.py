"""Fixed, frozen-feature diagnostic; this tool never trains the SeePAT encoders."""

from __future__ import annotations

import argparse
import hashlib
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from seepat.artifacts import atomic_write_csv, atomic_write_json, file_sha256, read_csv_rows
from seepat.data.sampling import canonical_source_group
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.training import experiment as runtime
from seepat.training.dataset import video_class_id
from seepat.training.matched_inputs import inside
from seepat.training.metrics import binary_classification_metrics
from seepat.training.video_objective import video_bags

BRANCHES = ("swin", "temporal", "evidence")
VIEWS = (*BRANCHES, "concatenated", "fused")
SEED = 20261028
PROBE = {"C": 1.0, "solver": "lbfgs", "class_weight": "balanced",
         "max_iter": 2000, "tol": 1e-6, "random_state": SEED}
ROLES = {"train": "train", "inner": "val"}
SCOPE = "Reused development diagnostic; no H1, causal attribution or model promotion"
VERSION = "frozen-feature-probe-v1"
OUTPUT = "outputs/engineering/frozen_feature_probe_v1"
SELECTION = "outputs/engineering/smooth_loss_pilot_v1/selection"
SAVED_INPUTS = "outputs/engineering/sequence_evidence_pilot_v2/inputs"
DESIGN = {
    "version": VERSION, "output": OUTPUT, "selection": SELECTION,
    "saved_inputs": SAVED_INPUTS,
    "checkpoint": {
        "path": "outputs/training/fusion_phase5_v1/experiments/candidate/checkpoint_best.pt",
        "sha256": "46fb132230fb2292ecadab63ac84aca34882174d6dbd4680ade5adab82dc4b07",
        "run": "outputs/training/fusion_phase5_v1/experiments/candidate/run.json",
    },
    "roles": {r: {"events": f"outputs/numerical_calibration/guarded_per_video_v3/events_{s}_calibrated.csv",
                  "export": "outputs/phase5_context_inputs_v1"} for r, s in ROLES.items()},
    "views": list(VIEWS), "pooling": "mean_then_max_per_branch", "probe": PROBE,
    "folds": 3, "cutoff": 0.5, "gate": "stop_and_review_signal", "scope": SCOPE,
}
METADATA = {
    *(r["events"] for r in DESIGN["roles"].values()),
    "outputs/numerical_calibration/guarded_per_video_v3/calibration.json",
    "outputs/numerical_calibration/guarded_per_video_v3/summary.json",
    DESIGN["checkpoint"]["run"],
    *(f"outputs/context_budget_inputs_v1/{name}" for name in (
        "inventory_train.csv", "inventory_inner.csv", "expanded_train.csv", "expanded_inner.csv",
        "records_train.json", "records_inner.json", "summary.json", "seal.json")),
    *(f"{SELECTION}/{name}" for name in (
        "inventory_train.csv", "inventory_inner.csv", "index_train.csv", "index_inner.csv", "seal.json")),
    *(f"{SAVED_INPUTS}/{name}" for name in ("job.json", "summary.json", "train.json", "inner.json", "seal.json")),
}


def extract_clip(model, sample, device):
    """Use the model's existing token interface and unchanged classifier once."""
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Feature extraction requires an eval-only, frozen model")
    video, mask = sample["video"], sample["frame_mask"]
    if (video.ndim != 4 or mask.dtype != torch.bool or mask.shape != (video.shape[1],)
            or not mask.any() or not torch.isfinite(video).all()):
        raise ValueError("Clip must retain finite RGB and a valid saved frame mask")
    with torch.inference_mode():
        tokens = model.extract_modality_tokens(
            video[None].to(device), mask[None].to(device),
            sample["evidence_features"][None].to(device),
            sample["evidence_feature_mask"][None].to(device),
        )
        if tuple(tokens) != BRANCHES:
            raise ValueError("Expected the three original projected modality tokens")
        fused = model.fusion(torch.stack(list(tokens.values()), dim=1))
        features = {k: v[0].cpu().numpy().copy() for k, v in {**tokens, "fused": fused}.items()}
        if any(v.ndim != 1 or not np.isfinite(v).all() for v in features.values()):
            raise ValueError("Frozen embeddings must be finite vectors")
        probability = float(model.classifier(fused).softmax(-1)[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Native classifier probability must be finite and bounded")
    return {"features": features, "probability": probability,
            "frame_mask": mask.cpu().numpy().copy(),
            "evidence_mask": sample["evidence_feature_mask"].cpu().numpy().copy()}


def pool_features(clips):
    """One fixed mean+max summary per video, with equal video weight in fitting."""
    if (set(clips) != {*BRANCHES, "fused"}
            or len({len(v) for v in clips.values()}) != 1
            or any(v.ndim != 2 or not len(v) or not np.isfinite(v).all()
                   for v in clips.values())):
        raise ValueError("Clip features must be aligned, nonempty finite matrices")
    pooled = {k: np.concatenate([v.mean(axis=0), v.max(axis=0)]) for k, v in clips.items()}
    pooled["concatenated"] = np.concatenate([pooled[k] for k in BRANCHES])
    return pooled


def fit_probe(features, labels):
    """Fit just a fixed L2 logistic head and scaler on its provided Train rows."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
    if x.ndim != 2 or not np.isfinite(x).all() or y.shape != (len(x),) or set(y) != {0, 1}:
        raise ValueError("Probe fitting needs finite Train rows and both video classes")
    scaler = StandardScaler().fit(x)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        head = LogisticRegression(**PROBE).fit(scaler.transform(x), y)
    return {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coefficient": head.coef_[0].tolist(), "intercept": float(head.intercept_[0]),
            "iterations": int(head.n_iter_[0]), "classes": head.classes_.tolist()}


def predict_probe(fitted, features):
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(fitted["mean"]) or not np.isfinite(x).all():
        raise ValueError("Probe prediction needs finite rows of the frozen feature width")
    return expit(((x - fitted["mean"]) / fitted["scale"]) @ fitted["coefficient"]
                 + fitted["intercept"])


def classification_summary(labels, probabilities):
    metrics = binary_classification_metrics(labels, probabilities, 0.5)
    if set(labels) != {0, 1}:
        raise ValueError("Video-level comparison requires both classes")
    return {**metrics, "balanced_accuracy": (metrics["recall"] + metrics["specificity"]) / 2,
            "auc": float(roc_auc_score(labels, probabilities))}


def validate_inputs(datasets, inventories):
    """Keep unavailable requests and source separation before any model work."""
    if set(datasets) != set(ROLES) or set(inventories) != set(ROLES):
        raise ValueError("Only fixed Train and inner-Validation roles are allowed")
    seen_ids, seen_groups, grouped = set(), set(), {}
    for role, dataset in datasets.items():
        bags = {bag.video_id: bag for bag in video_bags(dataset.rows, True)}
        expected = set()
        for row in inventories[role]:
            identifier = row["video_id"]
            group = canonical_source_group(row["source_group"], None)
            if (not identifier or identifier in seen_ids or group in seen_groups
                    or row["dataset_split"] != ROLES[role]
                    or row["status"] not in {"paired", "unevaluated"}):
                raise ValueError("Source separation/identity/split changed, including unavailable requests")
            seen_ids.add(identifier)
            seen_groups.add(group)
            if row["status"] == "paired":
                expected.add(identifier)
                bag = bags.get(identifier)
                if (bag is None or bag.label != video_class_id(row["category"])
                        or len(bag.indices) != int(row["expanded_clips"])):
                    raise ValueError("Complete video membership/label/count differs")
                for index in bag.indices:
                    clip = dataset.rows[index]
                    if (clip["dataset_split"] != ROLES[role]
                            or canonical_source_group(clip["source_group"], None) != group):
                        raise ValueError("Clip source/split differs from its video")
            elif int(row["expanded_clips"]) != 0 or not row["reason"]:
                raise ValueError("Unavailable requests must retain zero clips and their reason")
        if expected != set(bags):
            raise ValueError("Clip bags add/drop original video requests")
        grouped[role] = bags
    return grouped


def export_features(project, output, datasets, inventories, model_loader, binding, *,
                    device="cpu", progress=None, resume=False):
    """Seal videos independently; reuse complete records, stop at partial records."""
    output = Path(output)
    grouped = validate_inputs(datasets, inventories)
    job = {"stage": "features", "binding": binding, "device": device,
           "torch": str(torch.__version__), "numpy": np.__version__, "encoder_optimizer_updates": 0}
    if output.exists():
        if (output / "seal.json").exists():
            return runtime.verify_output(output, job)
        if (output / "worker.lock").exists():
            raise ValueError("Feature worker active/interrupted; inspect before recovery")
        if not resume or runtime.read_json(output / "job.json") != job:
            raise ValueError("Feature stage partial or binding changed; preserve; explicit --resume required")
        allowed = {"job.json", "summary.json", "model.json", *ROLES,
                   *(f"inventory_{role}.csv" for role in ROLES)}
        if {p.name for p in output.iterdir()} - allowed:
            raise ValueError("Unexpected partial-stage files; preserve before recovery")

    records, reused, model = {}, 0, None
    identity_path = output / "model.json"
    identity = runtime.read_json(identity_path) if identity_path.exists() else None
    with runtime.output_stage(output, job, progress, resume=resume) as complete:
        for role_number, (role, dataset) in enumerate(datasets.items(), 1):
            if progress:
                progress.start_job(role_number, f"frozen-features-{role}")
            atomic_write_csv(output / f"inventory_{role}.csv", inventories[role])
            role_dir = output / role
            expected = set(grouped[role])
            if role_dir.exists() and {p.name for p in role_dir.iterdir()} - expected:
                raise ValueError("Unexpected feature-video records; preserve before recovery")
            clips_total = 0
            for number, bag in enumerate(grouped[role].values(), 1):
                if progress:
                    progress("extract frozen branches", number - 1, len(expected), bag.video_id)
                rows = [dataset.rows[i] for i in bag.indices]
                original = next(r for r in inventories[role] if r["video_id"] == bag.video_id)
                lineage = [{k: r[k] for k in ("event_id", "kind", "mouth_clip_path",
                                             "class_id", "phoneme")} for r in rows]
                video_job = {"stage": "feature-video", "parent": job, "video": original,
                             "clips": lineage}
                directory = inside(role_dir, bag.video_id)
                if directory.exists():
                    if not (directory / "seal.json").is_file():
                        raise ValueError(f"Partial video record: {directory}; preserve before reviewed recovery")
                    saved = runtime.verify_output(directory, video_job)
                    if identity is None or saved["model_state_sha256"] != identity["state_sha256"]:
                        raise ValueError("Completed records do not share the same model weights")
                    for item in saved["inputs"]:
                        if file_sha256(inside(project, item["path"])) != item["sha256"]:
                            raise ValueError("Completed feature input bytes changed")
                    reused += 1
                else:
                    if model is None:
                        if progress:
                            progress("load fixed checkpoint; no encoder fitting", 0, 0, role)
                        model = model_loader().to(device).eval().requires_grad_(False)
                        actual = {"state_sha256": runtime.state_digest(model)}
                        if identity is not None and actual != identity:
                            raise ValueError("Resumed model weights differ from completed feature records")
                        identity = actual
                        atomic_write_json(identity_path, identity)
                    with runtime.output_stage(directory, video_job) as video_complete:
                        values, inputs = [], []
                        for index, row in zip(bag.indices, rows, strict=True):
                            path = inside(project, row["mouth_clip_path"])
                            before = file_sha256(path)
                            item = dataset[index]
                            if (item["event_id"] != row["event_id"] or item["video_id"] != bag.video_id
                                    or item["source_group"] != row["source_group"]
                                    or int(item["label"]) != int(row["class_id"])
                                    or int(item["video_label"]) != bag.label):
                                raise ValueError("Decoded clip identity or stored targets changed")
                            values.append(extract_clip(model, item, device))
                            if file_sha256(path) != before:
                                raise ValueError("Input clip bytes changed during extraction")
                            inputs.append({"path": row["mouth_clip_path"], "sha256": before})
                        arrays = {k: np.stack([v["features"][k] for v in values])
                                  for k in (*BRANCHES, "fused")}
                        pool_features(arrays)
                        np.savez(directory / "features.npz", **arrays,
                                 frame_mask=np.stack([v["frame_mask"] for v in values]),
                                 evidence_mask=np.stack([v["evidence_mask"] for v in values]),
                                 probabilities=np.array([v["probability"] for v in values]))
                        video_complete({"status": "complete", "label": bag.label, "inputs": inputs,
                                        "clips": len(rows), "clip_ids": list(bag.event_ids),
                                        "model_state_sha256": identity["state_sha256"]})
                clips_total += len(rows)
                if progress:
                    progress("extract frozen branches", number, len(expected), bag.video_id)
            records[role] = {"requested": len(inventories[role]), "evaluated": len(expected),
                             "unavailable": len(inventories[role]) - len(expected), "clips": clips_total}
        return complete({"status": "complete", "roles": records, "reused_videos": reused,
                         "model_state_sha256": identity["state_sha256"],
                         "encoder_optimizer_updates": 0, "gate": "ready_for_fixed_probes", "scope": SCOPE})


def load_features(directory, binding):
    """Read sealed feature arrays only; never reload a model or decode media."""
    summary = runtime.verify_output(directory)
    parent = runtime.read_json(directory / "job.json")
    if parent["binding"] != binding or parent["stage"] != "features":
        raise ValueError("Feature binding/stage differs")
    cache, seen, groups = {}, set(), set()
    for role, split in ROLES.items():
        inventory = read_csv_rows(directory / f"inventory_{role}.csv")
        available, pooled, probabilities, clips = [], [], [], 0
        for row in inventory:
            group = canonical_source_group(row["source_group"], None)
            if (row["video_id"] in seen or group in groups or row["dataset_split"] != split
                    or row["status"] not in {"paired", "unevaluated"}):
                raise ValueError("Cached source separation/coverage differs")
            seen.add(row["video_id"])
            groups.add(group)
            if row["status"] == "unevaluated":
                if int(row["expanded_clips"]) != 0 or not row["reason"]:
                    raise ValueError("Cached unavailable request differs")
                continue
            video_dir = inside(directory / role, row["video_id"])
            saved = runtime.verify_output(video_dir)
            job = runtime.read_json(video_dir / "job.json")
            count = int(row["expanded_clips"])
            if (job["parent"] != parent or job["video"] != row
                    or saved["model_state_sha256"] != summary["model_state_sha256"]
                    or saved["clips"] != count or len(saved["inputs"]) != count
                    or saved["clip_ids"] != [r["event_id"] for r in job["clips"]]
                    or saved["label"] != video_class_id(row["category"])):
                raise ValueError("Cached video/checkpoint/clip lineage differs")
            with np.load(video_dir / "features.npz", allow_pickle=False) as arrays:
                if (any(len(arrays[k]) != count for k in arrays.files)
                        or arrays["frame_mask"].ndim != 2 or arrays["frame_mask"].dtype != bool
                        or not arrays["frame_mask"].any(axis=1).all()
                        or arrays["evidence_mask"].ndim != 2 or arrays["evidence_mask"].dtype != bool):
                    raise ValueError("Cached frame/evidence masks differ")
                pooled.append(pool_features({k: arrays[k] for k in (*BRANCHES, "fused")}))
                p = arrays["probabilities"]
                if p.shape != (count,) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
                    raise ValueError("Cached probabilities differ")
                probabilities.append(float(p.max()))
            available.append(row)
            clips += count
        counts = {"requested": len(inventory), "evaluated": len(available),
                  "unavailable": len(inventory) - len(available), "clips": clips}
        if counts != summary["roles"][role] or not available:
            raise ValueError("Cached role counts differ")
        cache[role] = {"inventory": inventory, "rows": available, "counts": counts,
                       "labels": np.array([int(r["category"] != "real") for r in available]),
                       "groups": [canonical_source_group(r["source_group"], None) for r in available],
                       "features": {k: np.stack([v[k] for v in pooled]).astype(np.float64) for k in VIEWS},
                       "probabilities": np.array(probabilities)}
    return cache


def compare_probes(features_directory, output, binding, *, progress=None):
    """Fixed Train-only fitting; exactly one reused-inner comparison, no selection."""
    import sklearn

    features_directory, output = Path(features_directory), Path(output)
    job = {"stage": "probes", "binding": binding,
           "feature_seal_sha256": file_sha256(features_directory / "seal.json"),
           "probe": PROBE, "pooling": "mean_then_max_per_branch", "folds": 3,
           "cutoff": 0.5, "sklearn": sklearn.__version__, "numpy": np.__version__}
    cache = load_features(features_directory, binding)
    if output.exists():
        return runtime.verify_output(output, job)
    train, inner = cache["train"], cache["inner"]
    y, groups = train["labels"], np.asarray(train["groups"])
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED).split(
        np.zeros((len(y), 1)), y, groups))
    for fitted, held in folds:
        if (set(groups[fitted]) & set(groups[held]) or set(y[fitted]) != {0, 1}
                or set(y[held]) != {0, 1}):
            raise ValueError("Fixed Train folds need separate source groups and both classes")
    results, predictions = {}, []

    def record(view, role, original, probabilities):
        labels = original["labels"]
        score = classification_summary(labels.tolist(), probabilities.tolist())
        lookup = {r["video_id"]: (int(label), float(p)) for r, label, p in zip(
            original["rows"], labels, probabilities, strict=True)}
        score["categories"] = {}
        for category in sorted({r["category"] for r in original["inventory"]}):
            requested = [r for r in original["inventory"] if r["category"] == category]
            evaluated = [lookup[r["video_id"]] for r in requested if r["video_id"] in lookup]
            score["categories"][category] = {
                "requested": len(requested), "evaluated": len(evaluated),
                "unavailable": len(requested) - len(evaluated),
                "flagged": sum(p >= .5 for _, p in evaluated),
                "correct": sum(int(p >= .5) == label for label, p in evaluated),
            }
        for row in original["inventory"]:
            label, probability = lookup.get(row["video_id"], ("", ""))
            predictions.append({"view": view, "role": role, "video_id": row["video_id"],
                                "source_group": row["source_group"], "category": row["category"],
                                "status": "evaluated" if probability != "" else "not_evaluated",
                                "reason": row["reason"], "label": label, "probability": probability})
        return score

    with runtime.output_stage(output, job, progress) as complete:
        atomic_write_json(output / "folds.json", [
            {"fold": i, "fit": [train["rows"][j]["video_id"] for j in fitted],
             "held": [train["rows"][j]["video_id"] for j in held]}
            for i, (fitted, held) in enumerate(folds, 1)
        ])
        for number, view in enumerate(VIEWS, 1):
            if progress:
                progress.start_job(number, view)
            x = train["features"][view]
            out_of_fold, heads = np.full(len(y), np.nan), []
            for i, (fitted, held) in enumerate(folds, 1):
                if progress:
                    progress("fit Train-fold probe; encoder already exposed to Train", i - 1, 4, view)
                head = fit_probe(x[fitted], y[fitted])
                out_of_fold[held] = predict_probe(head, x[held])
                heads.append(head)
            head = fit_probe(x, y)
            atomic_write_json(output / f"{view}_probe.json", {"all_train": head, "folds": heads})
            results[view] = {
                "train_fit": record(view, "train_fit", train, predict_probe(head, x)),
                "train_probe_oof": record(view, "train_probe_oof", train, out_of_fold),
                "inner_reused": record(view, "inner", inner,
                                       predict_probe(head, inner["features"][view])),
                "feature_width": x.shape[1],
            }
            if progress:
                progress("fixed probe complete; no cutoff search", 4, 4, view)
        results["checkpoint_max"] = {
            "train_fit": record("checkpoint_max", "train_fit", train, train["probabilities"]),
            "inner_reused": record("checkpoint_max", "inner", inner, inner["probabilities"]),
        }
        atomic_write_csv(output / "predictions.csv", predictions)
        return complete({"status": "complete", "views": results, "head_fits": len(VIEWS) * 4,
                         "encoder_optimizer_updates": 0, "gate": "stop_and_review_signal",
                         "coverage": {r: c["counts"] for r, c in cache.items()}, "scope": SCOPE,
                         "fold_exposure": "Encoder already trained on Train; these folds measure probe stability only",
                         "inner_exposure": "Reused for past model/checkpoint/threshold choices; not an independent test"})


def load_protocol(project, config):
    """Check exact design and declared small metadata/source bindings; no media."""
    protocol = runtime.read_json(config)
    if (set(protocol) != {"design", "metadata_bindings", "code_bindings"}
            or protocol["design"] != DESIGN or set(protocol["metadata_bindings"]) != METADATA):
        raise ValueError("Unsupported feature diagnostic design; preserve this frozen version")
    sources = {p.relative_to(project).as_posix() for p in (project / "src/seepat").rglob("*.py")}
    if set(protocol["code_bindings"]) != sources | {"scripts/probe_features.py"}:
        raise ValueError("Frozen source membership changed")
    for relative, expected in protocol["code_bindings"].items():
        raw = inside(project, relative).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {relative}")
    for relative, expected in protocol["metadata_bindings"].items():
        if file_sha256(inside(project, relative)) != expected:
            raise ValueError(f"Frozen metadata changed: {relative}")
    return protocol, {"protocol_sha256": hashlib.sha256(config.read_bytes().replace(b"\r\n", b"\n")).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Frozen-feature diagnostic; no encoder training or promotion.")
    parser.add_argument("command", choices=("preflight", "export", "fit", "verify", "watch"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/frozen_feature_probe_v1.json"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--stage", choices=("features", "probes"), default="features")
    parser.add_argument("--resume", action="store_true", help="Reuse sealed feature videos; partial videos stop")
    args = parser.parse_args(argv)
    project = args.project.resolve()
    output = inside(project, OUTPUT)
    if args.command == "watch":
        return runtime.watch_progress(live_progress_path(output / f"{args.stage}_status.json"), "FEATURE CHECK")
    if args.resume and args.command != "export":
        parser.error("--resume applies only to an interrupted feature export")
    os.chdir(project)
    config = inside(project, str(args.config))
    protocol, binding = load_protocol(project, config)
    from seepat.training.sequence_inputs import SequenceInputDataset
    from seepat.training.sequence_pilot import base_dataset, selected_inputs

    inventories, indexes = selected_inputs(project)
    checkpoint = protocol["design"]["checkpoint"]
    if not inside(project, checkpoint["path"]).is_file():
        raise ValueError("Pinned checkpoint is missing; do not substitute another model")
    if args.command == "preflight":
        counts = {r: {"requested": len(rows), "usable": sum(v["status"] == "paired" for v in rows)}
                  for r, rows in inventories.items()}
        print(f"FEATURE INPUTS READY | {counts} | no model/media load", flush=True)
        return counts
    features, probes = output / "features", output / "probes"
    if args.command == "verify":
        load_features(features, binding)
        if args.stage == "probes":
            if not (probes / "seal.json").is_file():
                raise ValueError("Probe result missing/partial; verify never starts fitting")
            result = compare_probes(features, probes, binding)
        else:
            result = runtime.verify_output(features)
        print(f"FEATURE CHECK VERIFIED | {args.stage} | saved arrays only; no model/media load", flush=True)
        return result
    stage = "features" if args.command == "export" else "probes"
    progress = WorkflowProgress(config, output / f"{stage}_status.json", 2 if stage == "features" else len(VIEWS))
    try:
        if stage == "features":
            if args.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA requested but unavailable; preserve the frozen settings")
            progress.start_job(1, "validate fixed saved clip inputs")
            datasets = {
                r: SequenceInputDataset(base_dataset(project, protocol["design"], r, indexes[r]),
                                        inside(project, SAVED_INPUTS), r, False)
                for r in ROLES
            }
            def model_loader():
                run = runtime.read_json(inside(project, checkpoint["run"]))
                return runtime.warm_start(inside(project, checkpoint["path"]),
                                          run["resume_contract"], checkpoint["sha256"])
            result = export_features(project, features, datasets, inventories, model_loader,
                                     binding, device=args.device, progress=progress, resume=args.resume)
        else:
            progress.start_job(1, "verify frozen feature cache")
            result = compare_probes(features, probes, binding, progress=progress)
            print_comparison(result)
        progress.finish()
        print(f"FEATURE CHECK COMPLETE | {stage} | {result['gate']}", flush=True)
        return result
    except BaseException as error:
        progress.finish(error)
        raise


def print_comparison(result):
    print("\nFixed views            Train BA  Train-fold BA  Inner AUC  Inner BA   TP   FP")
    for name, scores in result["views"].items():
        inner = scores["inner_reused"]
        fold = scores.get("train_probe_oof", {}).get("balanced_accuracy")
        fold_text = f"{fold:6.2%}" if fold is not None else "     --"
        print(f"{name:<22} {scores['train_fit']['balanced_accuracy']:7.2%}  {fold_text:>13}"
              f"  {inner['auc']:9.4f}  {inner['balanced_accuracy']:8.2%}"
              f"  {inner['true_positive']:3}  {inner['false_positive']:3}")
    print("REUSED DEVELOPMENT DIAGNOSTIC | Train folds are encoder-exposed; no H1 or promotion\n")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ConvergenceWarning) as error:
        raise SystemExit(f"FEATURE CHECK STOPPED | {error}") from error
