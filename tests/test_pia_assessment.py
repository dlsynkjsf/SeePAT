import json

import pytest

from scripts import pia_assessment as pia
from seepat.artifacts import file_sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def inputs(tmp_path):
    rows = []
    for name in ["a", "b"]:
        path = tmp_path / f"{name}.mp4"
        path.write_bytes(b"synthetic clip")
        rows.append(
            {
                "video_id": name,
                "file": path.name,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return rows


def completed(tmp_path, probabilities=(0.5, None)):
    rows, output = inputs(tmp_path), tmp_path / "outputs/new"
    contract = {"fixed": True}
    write(output / "freeze.json", contract)
    pia.run_cases(
        output,
        rows,
        contract,
        lambda row, directory: {
            "eligible": probabilities[rows.index(row)] is not None,
            "probability": probabilities[rows.index(row)],
            "reason": "synthetic",
        },
    )
    return rows, output


def test_exact_thousand_selection_is_label_free_and_not_resampled(tmp_path):
    names = [f"video-data/{i}.mp4" for i in range(1000)]
    selection = {
        "selected_files": names,
        "pool": [{"file": n, "sha256": "a" * 64, "bytes": 1} for n in names],
    }
    rows = pia.selected_requests(selection, tmp_path)
    assert len(rows) == 1000 and [r["file"] for r in rows] == names
    assert all(set(r) == {"video_id", "file", "path", "sha256", "bytes"} for r in rows)
    assert len({r["video_id"] for r in rows}) == 1000
    for changed in [names[:-1], names[:-1] + [names[0]]]:
        with pytest.raises(ValueError, match="1,000"):
            pia.selected_requests({**selection, "selected_files": changed}, tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        pia.selected_requests({**selection, "selected_files": ["../bad"] + names[1:]}, tmp_path)


def test_resume_reuses_commits_and_retains_unavailable_requests(tmp_path):
    rows, output = completed(tmp_path)
    before = {p: p.read_bytes() for p in (output / "cases").rglob("*") if p.is_file()}
    pia.run_cases(output, rows, {"fixed": True}, lambda *a: pytest.fail("inference repeated"))
    assert pia.verify(output)["requested"] == 2
    assert pia.verify(output)["unavailable"] == 1
    assert all(p.read_bytes() == data for p, data in before.items())
    assert pia.read(output / "predictions.json")[0]["verdict"] == "authentic"
    write(output / "cases/a/summary.json", {})
    with pytest.raises(ValueError):
        pia.verify(output)


def test_metrics_wait_for_sealed_predictions_and_verify_is_read_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    metadata = tmp_path / "truth.csv"
    metadata.write_text(
        "Filename,Video Ground Truth,Audio Ground Truth\na.mp4,real,unknown\nb.mp4,fake,real\n"
    )
    config = {"metadata": {"path": "truth.csv", "sha256": file_sha256(metadata)}}
    output = tmp_path / "outputs/new"
    monkeypatch.setattr(pia, "read_csv_rows", lambda *a: pytest.fail("early truth read"))
    with pytest.raises(FileNotFoundError):
        pia.write_metrics(config, output)
    monkeypatch.undo()
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    _, output = completed(tmp_path)
    result = pia.write_metrics(config, output)
    primary = result["metrics"]["primary_either_modality"]
    visual = result["metrics"]["secondary_visual_only"]
    assert primary["unknown_target"] == primary["unavailable_prediction"] == 1
    assert primary["scored_known_target"] == 0
    assert visual["counts"]["tn"] == 1 and visual["specificity"] == 1
    saved = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    monkeypatch.setattr(pia, "read_csv_rows", lambda *a: pytest.fail("truth re-read"))
    assert pia.verify_metrics(config, output) == result
    assert pia.write_metrics(config, output) == result
    assert all(p.read_bytes() == data for p, data in saved.items())
    (output / "report.md").write_text("changed")
    with pytest.raises(ValueError, match="assessment changed"):
        pia.verify_metrics(config, output)


def test_partial_metrics_finish_without_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    _, output = completed(tmp_path)
    truth = tmp_path / "truth.csv"
    truth.write_text(
        "Filename,Video Ground Truth,Audio Ground Truth\na.mp4,real,real\nb.mp4,fake,real\n"
    )
    config = {"metadata": {"path": "truth.csv", "sha256": file_sha256(truth)}}
    expected = pia.write_metrics(config, output)
    (output / "metrics.seal.json").unlink()
    assert pia.write_metrics(config, output) == expected


def test_historical_reuse_requires_source_runtime_and_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    row = inputs(tmp_path)[0]
    config = {"image": "fixed-image", "reuse": {"output": "outputs/old", "config_sha256": "fixed"}}
    case = tmp_path / "outputs/old/cases/a"
    assert pia.reused_prediction(config, row) is None
    write(
        case / "job.json",
        {"source": row, "contract": {"image": "fixed-image", "config_sha256": "fixed"}},
    )
    write(
        case / "summary.json",
        {
            "source_sha256": row["sha256"],
            "eligible": True,
            "probability": 0.8,
            "reason": None,
            "seconds": 5,
        },
    )
    write(case / "seal.json", {"summary_sha256": file_sha256(case / "summary.json")})
    saved = pia.reused_prediction(config, row)
    assert saved["probability"] == 0.8 and saved["reused_from"].endswith("summary.json")
    with pytest.raises(ValueError, match="binding"):
        pia.reused_prediction(config, {**row, "sha256": "b" * 64})


def test_active_historical_worker_rejects_without_output_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    lock = tmp_path / "outputs/old.worker.lock"
    lock.parent.mkdir()
    lock.write_text("123")
    monkeypatch.setattr(pia, "process_alive", lambda pid: True)
    with pytest.raises(RuntimeError, match="active"):
        pia.run(
            {"reuse": {"output": "outputs/old"}},
            tmp_path / "config.json",
            tmp_path / "outputs/new",
            [],
        )
    assert not (tmp_path / "outputs/new").exists()


@pytest.mark.parametrize('version', ['pia-assessment-1000-v2', 'pia-assessment-1000-v3'])
def test_author_worker_uses_direct_runtime_mounts_and_strict_cutoff(tmp_path, monkeypatch, version):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    config = {
        "image": "fixed",
        "version": version,
        "runtime_helpers": pia.RUNTIME_HELPERS + ([pia.DECODE_RUNTIME] if version.endswith('v3') else []),
        "case_timeout_seconds": 86400,
        "phonemes": ["p"],
    }
    observed = []

    def command(values, log, tracker, phase, **kwargs):
        observed.append((values, kwargs))
        output = log.parent / "author"
        write(
            output / "results.json",
            {
                "model": "PIA",
                "video": "/input/video.mp4",
                "weights": "checkpoints/best_model.pth",
                "probs": {"real": 0.5, "fake": 0.5},
                "predicted_label": 0,
                "predicted_label_name": "Real",
            },
        )
        write(output / "testvideo/video_aligned.json", [{"phoneme": "p", "crop_path": "p.png"}])
        (output / "testvideo/p.png").write_bytes(b"synthetic")
        write(output / "consumed_identity.json", {"synthetic": True})
        log.write_text("[SAVED] /output/results.json\n")

    monkeypatch.setattr(pia.author, "_command", command)
    case = tmp_path / "case"
    case.mkdir()
    result = pia.run_author(config, tmp_path / "work", inputs(tmp_path)[0], case, lambda *a: None)
    assert result["eligible"] and result["probability"] == 0.5
    values, options = observed[0]
    assert values[-6:] == [
        "python",
        "/helpers/pia_cpu_decode_runtime.py" if version.endswith('v3') else "/helpers/pia_cpu_consumed_runtime.py",
        "--video",
        "/input/video.mp4",
        "--outdir",
        "/output",
    ]
    assert "--gpus" not in values and options["timeout"] == 86400
    assert sum("target=/helpers/" in v for v in values) == len(config['runtime_helpers'])


def test_watch_and_verify_do_not_construct_models(tmp_path, monkeypatch):
    monkeypatch.setattr(pia, "ROOT", tmp_path)
    monkeypatch.setattr(pia, "load_profile", lambda path: ({}, tmp_path, []))
    monkeypatch.setattr(pia, "run", lambda *a: pytest.fail("worker"))
    monkeypatch.setattr(pia, "watch", lambda output: 0)
    monkeypatch.setattr(pia, "verify_metrics", lambda *a: {})
    path = write(tmp_path / "config.json", {"output": "outputs/new"})
    write(tmp_path / "job.json", {"requests": [], "contract": {"config_sha256": file_sha256(path)}})
    assert pia.main(["watch", "--config", str(path)]) == 0
    assert pia.main(["verify", "--config", str(path)]) == 0


@pytest.mark.parametrize("version", ["pia-assessment-1000-v1", "pia-assessment-1000-v2", "pia-assessment-1000-v3"])
def test_profile_and_runtime_freeze_bind_selection_parity_sources_and_cache(tmp_path, monkeypatch, version):
    monkeypatch.setattr(pia, "ROOT", tmp_path)

    def bind(name, value):
        path = write(tmp_path / name, value)
        return {"path": name, "sha256": file_sha256(path)}

    names = [f"video-data/{i}.mp4" for i in range(1000)]
    config = {
        "version": version,
        "requested_videos": 1000,
        "output": "outputs/new",
        "source_root": "sources",
        "case_timeout_seconds": 86400,
        "runtime_helpers": pia.RUNTIME_HELPERS + ([pia.DECODE_RUNTIME] if version.endswith('v3') else []),
        "image": "fixed",
        "work_dir": ".tools/pia",
        "reuse": {"output": "outputs/old"},
        "selection": bind(
            "selection.json",
            {
                "selected_files": names,
                "pool": [{"file": n, "sha256": "a" * 64, "bytes": 1} for n in names],
            },
        ),
        "setup": bind(
            "setup.json",
            {
                "status": "complete",
                "image": "fixed",
                "environment": {"device": "cpu", "strict_author_checkpoint_load": True},
                "assets": [],
                "cache_files": [],
            },
        ),
        "parity_proof": bind(
            "proof/proof.json", {"parity": {"passed": True, "absolute_probability_difference": 0}}
        ),
        "checkpoint": bind(".tools/pia/checkpoint.json", {"synthetic": True}),
        "source_sha256": {},
    }
    write(tmp_path / "proof/seal.json", {"proof_sha256": config["parity_proof"]["sha256"]})
    sources = [*pia.SOURCE_FILES, pia.DECODE_RUNTIME] if version.endswith('v3') else pia.SOURCE_FILES if version.endswith("v2") else [*pia.RUNTIME_HELPERS, "scripts/pia_assessment.py"]
    for name in sources:
        config["source_sha256"][name] = bind(name, {"synthetic": True})["sha256"]
    path = write(tmp_path / "config.json", config)
    if not version.endswith("v1"):
        incomplete = {**config, "source_sha256": {
            k: v for k, v in config["source_sha256"].items() if k != "scripts/assessment_runtime.py"
        }}
        write(path, incomplete)
        with pytest.raises(ValueError, match="explicit source bindings"):
            pia.load_profile(path)
        write(path, config)
    _, output, requests = pia.load_profile(path)
    assert len(requests) == 1000 and not output.exists()
    monkeypatch.setattr(pia.author, "_image_id", lambda name: "fixed")
    work, contract = pia.freeze_runtime(config, path, output)
    assert work == tmp_path / ".tools/pia" and contract["network"] == "none"
    assert pia.freeze_runtime(config, path, output)[1] == contract
    write(path, {**config, "new_value": True})
    with pytest.raises(ValueError, match="Frozen runtime changed"):
        pia.freeze_runtime(config, path, output)
    write(tmp_path / pia.RUNTIME_HELPERS[0], {"changed": True})
    with pytest.raises(ValueError, match="Bound file changed"):
        pia.load_profile(path)
