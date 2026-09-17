"""Documentation tables derived from study results; never runs or selects a model."""
from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, stdev

from seepat.artifacts import atomic_write_csv, atomic_write_json, atomic_write_text

REPORT_VERSION = "paper-study-report-v1"
MODEL_NAMES = {
    "efficientnet_v2_s_only": "EfficientNetV2-S only",
    "tempcnn_gray16": "TempCNN only (16x16 grayscale)",
    "efficientnet_v2_s_tempcnn": "EfficientNetV2-S + TempCNN",
    "swin3d_b": "Video Swin Base (visual baseline)",
    "swin3d_b_vild_fusion": "SeePAT full fusion",
    "swin3d_b_visual_fusion": "Fusion without biological evidence",
    "geometry_static": "Static geometric threshold",
    "geometry_dynamic": "Per-video Isolation Forest",
}
METRICS = ("accuracy", "precision", "recall", "f1", "false_positive_rate", "specificity")
COUNTS = ("samples", "true_positive", "true_negative", "false_positive", "false_negative")
H3_MODELS = ("efficientnet_v2_s_only", "tempcnn_gray16", "efficientnet_v2_s_tempcnn")


def report_paths(prefix: Path) -> list[Path]:
    return [prefix.with_suffix(ext) for ext in (".csv", ".json", ".md")] + [
        prefix.with_name(prefix.name + suffix + ".csv")
        for suffix in ("_summary", "_statistics", "_coverage")
    ]


def _comparison_specs():
    for control in ("efficientnet_v2_s_only", "tempcnn_gray16"):
        for metric in ("f1", "accuracy"):
            yield "H3", "efficientnet_v2_s_tempcnn", control, metric
    for metric in ("false_positive_rate", "recall"):
        yield "H2", "geometry_dynamic", "geometry_static", metric
    for control in ("swin3d_b", "efficientnet_v2_s_tempcnn"):
        for metric in ("accuracy", "f1"):
            yield "H1 internal comparison", "swin3d_b_vild_fusion", control, metric
    yield "Optional evidence comparison", "swin3d_b_vild_fusion", "swin3d_b_visual_fusion", "f1"


def build_report(results: list[dict], expected_folds: int, expected_models: tuple[str, ...]) -> dict:
    from seepat.training.study_evaluation import paired_statistics

    if expected_folds < 2:
        raise ValueError("Study reporting requires at least two expected folds")
    fold_ids = set(range(1, expected_folds + 1))
    by_model = {}
    results = sorted(results, key=lambda row: (row["model"], row["fold"]))
    study_hashes = {r.get("provenance", {}).get("study_sha256") for r in results}
    if len(study_hashes) > 1:
        raise ValueError("Reports must not combine results from different study provenance")
    for result in results:
        model, fold = result["model"], result["fold"]
        if type(fold) is not int or fold not in fold_ids:
            raise ValueError("Result fold is outside the configured study")
        if fold in by_model.setdefault(model, {}):
            raise ValueError("Duplicate model/fold result")
        by_model[model][fold] = result
        for level in ("video", "event"):
            for metric, value in result.get(f"{level}_metrics", {}).items():
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"Non-finite or invalid {level} metric: {metric}")
                if metric in (*METRICS, "threshold") and not 0 <= value <= 1:
                    raise ValueError(f"{metric} must be between zero and one")

    inventory, summaries, coverage = [], [], []
    for model in sorted(set(expected_models) | set(by_model)):
        observed = by_model.get(model, {})
        inventory.append({"model": model, "model_name": MODEL_NAMES.get(model, model),
                          "folds_present": sorted(observed), "folds_expected": expected_folds,
                          "missing_folds": sorted(fold_ids - set(observed)),
                          "status": "complete" if set(observed) == fold_ids else "incomplete"})
        for level in ("video", "event"):
            records = [r[f"{level}_metrics"] for r in observed.values() if f"{level}_metrics" in r]
            if not records:
                continue
            summary = {"model": model, "level": level, "folds_present": len(records),
                       "folds_expected": expected_folds,
                       "status": "complete" if len(records) == expected_folds else "incomplete"}
            for metric in METRICS:
                values = [r[metric] for r in records if metric in r]
                summary[f"{metric}_folds"] = len(values)
                complete = len(values) == len(records)
                summary[f"{metric}_mean"] = mean(values) if complete and values else None
                summary[f"{metric}_sd"] = stdev(values) if complete and len(values) > 1 else None
            for count in COUNTS:
                summary[f"{count}_sum"] = (
                    sum(r[count] for r in records) if all(count in r for r in records) else None)
            summaries.append(summary)
    for result in results:
        for split, counts in result.get("coverage", {}).items():
            row = {"model": result["model"], "fold": result["fold"], "split": split,
                   **{key: value for key, value in counts.items() if not isinstance(value, dict)}}
            for key, values in counts.items():
                if isinstance(values, dict):
                    row.update({f"{key}_{name}": value for name, value in values.items()})
            coverage.append(row)

    comparisons = []
    for comparison_id, (hypothesis, first, second, metric) in enumerate(_comparison_specs(), 1):
        one, two = by_model.get(first, {}), by_model.get(second, {})
        entry = {"comparison_id": f"C{comparison_id:02}",
                 "hypothesis": hypothesis, "first": first, "second": second, "metric": metric,
                 "direction": "lower" if metric == "false_positive_rate" else "higher",
                 "folds": sorted(one.keys() & two.keys()), "alpha": .05,
                 "confirmatory_decision": "not_assessed", "status": "missing_results"}
        if one and two:
            if set(one) != set(two):
                raise ValueError("Paired comparisons require identical fold inventories")
            values = [r["video_metrics"].get(metric) for group in (one, two) for _, r in sorted(group.items())]
            if any(value is None for value in values):
                entry["status"] = "missing_metric"
            elif len(one) < 2:
                entry["status"] = "insufficient_pairs"
            else:
                diagnostics = paired_statistics(values[:len(one)], values[len(one):])
                entry.update(diagnostics, status="diagnostic_only" if set(one) == fold_ids else "incomplete_folds")
                entry["mean_difference_percentage_points"] = 100 * diagnostics["mean_difference"]
                shapiro = diagnostics["shapiro_p"]
                selected = ("paired_t" if shapiro > .05 else "exact_two_sided_signed_rank") if shapiro is not None else None
                p_value = diagnostics.get(f"{selected}_p") if selected else None
                entry.update(paper_rule_test=selected, paper_rule_p=p_value,
                             paper_rule_p_below_alpha=p_value < .05 if p_value is not None else None,
                             observed_direction=("no_difference" if diagnostics["mean_difference"] == 0 else
                                                 "higher" if diagnostics["mean_difference"] > 0 else "lower"))
        comparisons.append(entry)
    return {"report_version": REPORT_VERSION, "expected_folds": expected_folds,
            "expected_models": list(expected_models), "model_inventory": inventory,
            "fold_results": results, "summary": summaries, "coverage": coverage,
            "paper_comparisons": comparisons,
            "paired_comparisons": [r for r in comparisons if "pairs" in r],
            "multiple_comparisons": "Unadjusted diagnostic p-values; no confirmatory significance decision"}


def _table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(value):
        return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")

    return ["| " + " | ".join(map(cell, headers)) + " |",
            "| " + " | ".join("---" for _ in headers) + " |", *[
                "| " + " | ".join(map(cell, row)) + " |" for row in rows], ""]


def _number(value, digits=6):
    return "N/A" if value is None else f"{value:.{digits}g}"


def render_markdown(report: dict, source_name: str) -> str:
    def name(model):
        return MODEL_NAMES.get(model, model)

    inventory = {row["model"]: row for row in report["model_inventory"]}

    def available(models):
        return all(inventory.get(model, {}).get("status") == "complete" for model in models)

    def metric_cell(row, metric):
        average, spread = row[f"{metric}_mean"], row[f"{metric}_sd"]
        if average is None:
            return "N/A"
        return f"{average * 100:.2f} +/- {spread * 100:.2f}" if spread is not None else f"{average * 100:.2f} (SD N/A)"

    lines = ["# SeePAT study results", "", "**Scope: held-out AV++ Train source groups; not external-test results.**",
             "Official Validation selects checkpoints and thresholds. Videos use maximum-event aggregation.",
             f"Configured folds: {report['expected_folds']}. The paper specifies five folds.",
             f"Source: [{source_name}]({source_name}). Missing results are not zeros or successful experiments.",
             "", "## Paper experiment status", ""]
    lines += _table(["Requirement", "Status in this report"], [
        ["H1: SeePAT versus external multimodal baselines", "Incomplete: exact external baselines are not configured; internal comparisons do not establish SOTA superiority"],
        ["H2: static versus dynamic calibration", "Fold results available" if available(("geometry_static", "geometry_dynamic")) else "Not complete in this report"],
        ["H3: three visual classifiers", "Fold results available" if available(H3_MODELS) else "Not complete in this report"],
        ["Confirmatory statistical conclusions", "Not assessed; statistical review required"],
        ["Locked external evaluation / explanation quality", "Not evaluated by this study report"],
    ])
    lines += _table(["Model", "Folds present / expected", "Missing folds"], [
        [row["model_name"], f"{len(row['folds_present'])} / {row['folds_expected']}",
         ", ".join(map(str, row["missing_folds"])) or "None"] for row in inventory.values()])
    for level, title in (("video", "Video results (primary)"), ("event", "Event results (secondary)")):
        lines += [f"## {title}", "", "Unweighted fold mean +/- sample SD, in percent. SD describes fold spread, not a confidence interval.",
                  "Incomplete rows summarize only the available folds; see the fold inventory above.", ""]
        lines += _table(["Model", "Folds", "Accuracy", "Precision", "Recall", "F1", "FPR", "Specificity"], [
            [name(row["model"]), row["folds_present"], *[metric_cell(row, metric) for metric in METRICS]]
            for row in report["summary"] if row["level"] == level])
    lines += ["## Video confusion counts and thresholds by fold", "",
              "Positive = manipulated. Thresholds are selected on Validation; these counts are from held-out folds.", ""]
    lines += _table(["Model", "Fold", "Videos", "TP", "TN", "FP", "FN", "Threshold"], [
        [name(row["model"]), row["fold"], *[row["video_metrics"].get(key, "N/A") for key in COUNTS],
         _number(row["video_metrics"].get("threshold"), 17)] for row in report["fold_results"]])
    lines += ["## Paired video comparisons", "",
              "Difference = first model minus second, in percentage points. Lower FPR is favorable; other listed metrics favor higher values.",
              "The paper-rule column follows Shapiro p > 0.05 -> paired t; otherwise exact two-sided signed-rank.",
              "Undefined/constant differences leave that choice N/A. A normality non-rejection does not establish normality.",
              "Reference alpha is 0.05. All p-values are unadjusted diagnostics, including the paper-rule value; no hypothesis is declared supported or rejected.", ""]
    lines += _table(["ID", "Paper", "First / second", "Metric", "Pairs", "Difference (pp)", "Status"], [
        [row["comparison_id"], row["hypothesis"], f"{name(row['first'])} / {name(row['second'])}",
         {"f1": "F1", "false_positive_rate": "FPR"}.get(row["metric"], row["metric"].capitalize()),
         row.get("pairs", "N/A"), _number(row.get("mean_difference_percentage_points")),
         row["status"].replace("_", " ").capitalize()]
        for row in report["paper_comparisons"]])
    lines += ["### Statistical diagnostics", ""]
    lines += _table(["ID", "Shapiro p", "Paired-t p", "Signed-rank p", "Paper-rule test", "Paper-rule p"], [
        [row["comparison_id"], _number(row.get("shapiro_p")), _number(row.get("paired_t_p")),
         _number(row.get("exact_two_sided_signed_rank_p")),
         {"paired_t": "Paired t-test", "exact_two_sided_signed_rank": "Exact signed-rank"}.get(row.get("paper_rule_test"), "N/A"),
         _number(row.get("paper_rule_p"))]
        for row in report["paper_comparisons"]])
    lines += ["## H2 evidence coverage", "",
              "Both methods use the same available events, then aggregate per video. Excluded evidence is counted, not classified as genuine.",
              "Validation coverage repeats across folds and must not be summed as unique videos.", ""]
    lines += _table(["Model", "Fold", "Split", "Events used / input", "Videos used / input", "Missing static", "Missing forest"], [
        [name(row["model"]), row["fold"], row["split"], f"{row.get('paired_events', 'N/A')} / {row.get('input_events', 'N/A')}",
         f"{row.get('paired_videos', 'N/A')} / {row.get('input_videos', 'N/A')}",
         row.get("missing_events_static", "N/A"), row.get("missing_events_dynamic", "N/A")]
        for row in report["coverage"]])
    if not report["coverage"]:
        lines += ["H2 coverage is not available in this phase's results. See the calibration report after that experiment completes.", ""]
    lines += ["## Provenance and interpretation", "",
              "- JSON retains every fold's source hashes, threshold provenance and prediction-artifact references supplied by evaluation.",
              "- CSV rates and thresholds retain numeric precision on the 0-1 scale. Only Markdown display values are rounded.",
              "- Mean F1 is the mean of fold F1 scores, not F1 recomputed from summed confusion counts.",
              "- H2 measures standalone geometry thresholding, not Isolation Forest's isolated contribution inside fusion.",
              "- TempCNN-only uses fixed 16x16 grayscale frames without a pretrained spatial encoder. EfficientNet variants use pretrained features; this difference must be disclosed.",
              "- Overlapping training folds are dependent. Five nonzero pairs have minimum exact two-sided signed-rank p = 0.0625. Multiple comparisons also require review.",
              "- These tables cannot supply missing external baselines, prove superiority, or validate explanation quality.", ""]
    return "\n".join(lines)


def write_report(prefix: Path, results: list[dict], *, expected_folds: int = 5,
                 expected_models: tuple[str, ...] = H3_MODELS) -> list[Path]:
    report = build_report(results, expected_folds, expected_models)
    from seepat.training.study_evaluation import EVALUATION_VERSION

    report["version"] = EVALUATION_VERSION
    paths = report_paths(prefix)
    atomic_write_csv(paths[0], [{"model": row["model"], "fold": row["fold"], **row["video_metrics"]}
                               for row in report["fold_results"]])
    atomic_write_json(paths[1], report)
    atomic_write_text(paths[2], render_markdown(report, paths[1].name))
    atomic_write_csv(paths[3], report["summary"])
    atomic_write_csv(paths[4], [{key: value for key, value in row.items() if not isinstance(value, list)}
                               for row in report["paper_comparisons"]])
    atomic_write_csv(paths[5], report["coverage"])
    return paths
