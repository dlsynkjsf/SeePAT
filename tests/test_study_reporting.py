import copy
import json
from statistics import mean, stdev

import pytest

from seepat.artifacts import read_csv_rows
from seepat.training.metrics import binary_classification_metrics
from seepat.training.study_evaluation import calibration_comparison, write_comparison
from seepat.training.study_reporting import H3_MODELS, build_report, report_paths


def model_results():
    rows = []
    for model in H3_MODELS:
        for fold in range(1, 6):
            metrics = binary_classification_metrics([0] * fold + [1, 1], [.1] * fold + [.4, .8], .5000000000000001)
            if model == "efficientnet_v2_s_tempcnn":
                metrics = binary_classification_metrics([0] * fold + [1, 1], [.6] * fold + [.9, .8], .5000000000000001)
            rows.append({"model": model, "fold": fold, "video_metrics": metrics,
                         "event_metrics": metrics.copy(),
                         "provenance": {"study_sha256": "synthetic-study", "checkpoint_sha256": f"synthetic-{model}-{fold}"}})
    return rows


def test_documentation_exports_preserve_precision_and_use_fold_means(tmp_path):
    results = model_results()
    original = copy.deepcopy(results)
    prefix = tmp_path / "evaluate_comparison"
    paths = write_comparison(prefix, results)
    assert paths == report_paths(prefix) and all(path.is_file() for path in paths)
    document = json.loads(paths[1].read_text())
    summary = next(row for row in document["summary"] if row["model"] == H3_MODELS[2] and row["level"] == "video")
    f1 = [r["video_metrics"]["f1"] for r in results if r["model"] == H3_MODELS[2]]
    assert summary["f1_mean"] == pytest.approx(mean(f1))
    assert summary["f1_sd"] == pytest.approx(stdev(f1))
    pooled_f1 = 2 * summary["true_positive_sum"] / (
        2 * summary["true_positive_sum"] + summary["false_positive_sum"] + summary["false_negative_sum"])
    assert summary["f1_mean"] != pytest.approx(pooled_f1)
    assert float(read_csv_rows(paths[0])[0]["threshold"]) == .5000000000000001
    markdown = paths[2].read_text()
    assert "0.50000000000000011" in markdown
    assert "not external-test results" in markdown and "Incomplete: exact external baselines" in markdown
    assert "not a confidence interval" in markdown and "0.0625" in markdown
    assert document["fold_results"][0]["provenance"]["checkpoint_sha256"].startswith("synthetic-")
    assert all(row["confirmatory_decision"] == "not_assessed" for row in document["paper_comparisons"])
    assert results == original
    contents = [path.read_bytes() for path in paths]
    write_comparison(prefix, results[::-1])
    assert contents == [path.read_bytes() for path in paths]


def test_missing_results_and_metrics_are_explicit_and_zero_is_a_real_value(tmp_path):
    prefix = tmp_path / "evaluate_comparison"
    write_comparison(prefix, [])
    document = json.loads(prefix.with_suffix(".json").read_text())
    assert not document["summary"]
    assert all(row["missing_folds"] == [1, 2, 3, 4, 5] for row in document["model_inventory"])
    assert all(row["status"] == "missing_results" for row in document["paper_comparisons"])
    rows = [{"model": model, "fold": 1, "video_metrics": {"f1": 0.}}
            for model in (H3_MODELS[0], H3_MODELS[2])]
    write_comparison(prefix, rows)
    document = json.loads(prefix.with_suffix(".json").read_text())
    summary = document["summary"][0]
    assert summary["f1_mean"] == 0. and summary["f1_sd"] is None
    assert summary["accuracy_mean"] is None
    assert document["paper_comparisons"][0]["status"] == "insufficient_pairs"
    assert document["paper_comparisons"][1]["status"] == "missing_metric"
    assert "0.00 (SD N/A)" in prefix.with_suffix(".md").read_text()


def test_h2_report_retains_paired_coverage_and_does_not_count_exclusions_as_real(tmp_path):
    from test_ablation_study import geometry_rows

    held = geometry_rows()
    held[-1]["isolation_forest_available"] = "false"
    results = []
    for fold in range(1, 6):
        results.extend({**row, "fold": fold} for row in calibration_comparison(geometry_rows(), held))
    prefix = tmp_path / "calibration_comparison"
    write_comparison(prefix, results, expected_models=("geometry_static", "geometry_dynamic"))
    document = json.loads(prefix.with_suffix(".json").read_text())
    coverage = [row for row in document["coverage"] if row["split"] == "heldout"]
    assert len(coverage) == 10
    assert all(row["input_events"] == 6 and row["paired_events"] == 5 for row in coverage)
    assert all(row["missing_events_dynamic"] == 1 for row in coverage)
    assert all(row["paired_event_classes_1"] == 2 for row in coverage)
    assert all(row["video_metrics"]["samples"] == 5 for row in document["fold_results"])
    comparison = next(row for row in document["paired_comparisons"] if row["metric"] == "false_positive_rate")
    assert comparison["direction"] == "lower" and comparison["observed_direction"] == "no_difference"
    assert comparison["paper_rule_test"] is None
    assert comparison["exact_two_sided_signed_rank_p"] == 1


@pytest.mark.parametrize("differences,selected", [
    ([.01, .02, .03, .04, .05], "paired_t"),
    ([.01, .01, .01, .01, .4], "exact_two_sided_signed_rank"),
    ([.01] * 5, None),
    ([.1] * 5, None),
])
def test_paper_rule_is_reproducible_diagnostic_not_significance_claim(differences, selected):
    rows = []
    for fold, difference in enumerate(differences, 1):
        rows.extend([
            {"model": H3_MODELS[0], "fold": fold, "video_metrics": {"f1": 0.}},
            {"model": H3_MODELS[2], "fold": fold, "video_metrics": {"f1": difference}},
        ])
    report = build_report(rows, 5, H3_MODELS)
    comparison = report["paper_comparisons"][0]
    assert comparison["paper_rule_test"] == selected
    assert comparison["confirmatory_decision"] == "not_assessed"
    assert comparison["mean_difference_percentage_points"] == pytest.approx(mean(differences) * 100)
    assert comparison["exact_two_sided_signed_rank_p"] == .0625
    assert comparison["observed_direction"] == "higher"


@pytest.mark.parametrize("problem,message", [
    ("duplicate", "Duplicate"), ("mismatch", "identical fold"),
    ("nan", "Non-finite"), ("outside", "outside"), ("provenance", "provenance"),
])
def test_invalid_reports_fail_before_overwriting_existing_outputs(tmp_path, problem, message):
    results = model_results()
    prefix = tmp_path / "evaluate_comparison"
    paths = write_comparison(prefix, results)
    before = [path.read_bytes() for path in paths]
    if problem == "duplicate":
        results.append(results[0])
    elif problem == "mismatch":
        results.pop()
    elif problem == "nan":
        results[0]["video_metrics"]["f1"] = float("nan")
    elif problem == "outside":
        results[0]["fold"] = 6
    else:
        results[0]["provenance"]["study_sha256"] = "another-study"
    with pytest.raises(ValueError, match=message):
        write_comparison(prefix, results)
    assert before == [path.read_bytes() for path in paths]
