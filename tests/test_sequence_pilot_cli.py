from hashlib import sha256
from pathlib import Path

import pytest

from seepat.artifacts import atomic_write_csv, atomic_write_json
from seepat.live_progress import live_progress_path
from seepat.training import experiment as runtime
from seepat.training import sequence_pilot as pilot


def profile():
    return runtime.read_json(Path(__file__).parents[1] / "configs/sequence_evidence_pilot_v2.json")


@pytest.mark.parametrize("change", [
    lambda p: p["training"].update(epochs=8),
    lambda p: p["roles"]["inner"].update(events="data/deepfake_eval/events.csv"),
    lambda p: p.update(gate="accept_any_improvement"),
    lambda p: p.update(code_bindings={}),
    lambda p: p.update(frame_window_policy="strict_continuous_time"),
])
def test_profile_rejects_changed_design_or_external_role(change):
    p = profile()
    change(p)
    with pytest.raises(ValueError, match="design"):
        pilot.validate_protocol(p)


def test_v1_protocol_is_preserved_and_v2_changes_only_version_bounds_output_and_source():
    old_path = Path(__file__).parents[1] / "configs/sequence_evidence_pilot.json"
    assert sha256(old_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == "25363b88e2bf625327efa2369247fd95f506ec9c15238d055bab0079165437f8"
    old, new = runtime.read_json(old_path), profile()
    with pytest.raises(ValueError, match="design"):
        pilot.validate_protocol(old)
    pilot.validate_protocol(new)
    allowed = {"version", "output", "frame_window_policy", "code_bindings"}
    assert {k: v for k, v in new.items() if k not in allowed} == {k: v for k, v in old.items() if k not in allowed}
    changed = {k for k in old["code_bindings"] if old["code_bindings"][k] != new["code_bindings"][k]}
    assert changed == {"src/seepat/models/sequence_evidence.py", "src/seepat/training/sequence_pilot.py"}
    assert new["output"] == pilot.OUTPUT and new["version"] == pilot.VERSION


def test_frozen_source_and_metadata_bindings_are_checked_without_models(tmp_path):
    p = profile()
    data = b"synthetic binding fixture"
    expected = sha256(data).hexdigest()
    p["code_bindings"] = {name: expected for name in pilot.REQUIRED_SOURCE}
    for name in (*p["code_bindings"], *p["metadata_bindings"]):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    p["metadata_bindings"] = dict.fromkeys(p["metadata_bindings"], expected)
    config = tmp_path / "config.json"
    atomic_write_json(config, p)
    assert pilot.load_protocol(tmp_path, config)[0] == p
    source = tmp_path / next(iter(p["code_bindings"]))
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        pilot.load_protocol(tmp_path, config)
    source.write_bytes(data)
    (tmp_path / next(iter(p["metadata_bindings"]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="metadata"):
        pilot.load_protocol(tmp_path, config)


def selection(project, leaking=False):
    directory = project / pilot.SELECTION
    rows, indexes = {}, {}
    for role, split in pilot.ROLES.items():
        rows[role], indexes[role] = [], []
        for category in sorted(pilot.CATEGORIES):
            for index in range(32 if role == "train" else 16):
                identifier = f"{role}-{category}-{index}"
                missing = category == "real" and index < (3 if role == "train" else 2)
                group = "train-real-0" if leaking and role == "inner" and category == "real" and index == 0 else identifier
                rows[role].append({"video_id": identifier, "source_group": group, "dataset_split": split,
                                   "category": category, "status": "unevaluated" if missing else "paired",
                                   "reason": "missing" if missing else "", "expanded_clips": "0" if missing else "2"})
                if not missing:
                    indexes[role].extend({"video_id": identifier, "source_group": group, "dataset_split": split,
                                          "clip_id": f"{identifier}-{kind}", "kind": kind}
                                         for kind in ("bilabial", "context"))
    with runtime.output_stage(directory, {"synthetic": "original selection"}) as complete:
        for role in pilot.ROLES:
            atomic_write_csv(directory / f"inventory_{role}.csv", rows[role])
            atomic_write_csv(directory / f"index_{role}.csv", indexes[role])
        complete({"status": "complete"})


def test_original_selection_keeps_unavailable_and_checks_their_source_groups(tmp_path):
    selection(tmp_path)
    inventory, indexes = pilot.selected_inputs(tmp_path)
    assert len(inventory["train"]) == 128 and len(inventory["inner"]) == 64
    assert len(indexes["train"]) == 250 and len(indexes["inner"]) == 124
    other = tmp_path / "leaking"
    selection(other, leaking=True)
    with pytest.raises(ValueError, match="separation"):
        pilot.selected_inputs(other)


def test_cli_help_and_terminal_watch_do_not_launch_a_worker(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as stopped:
        pilot.main(["--help"])
    assert stopped.value.code == 0 and not list(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)
    status = live_progress_path(tmp_path / pilot.OUTPUT / "inputs_status.json")
    atomic_write_json(status, {"status": "complete", "jobs_total": 2, "job_index": 2,
                               "phase": "complete", "finished": 2, "total": 2})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    pilot.main(["watch", "--project", str(tmp_path), "--stage", "inputs"])
    assert sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
    assert "complete" in capsys.readouterr().out.lower()
