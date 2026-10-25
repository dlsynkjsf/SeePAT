"""Filesystem-only checks for the shared experiment-stage interface."""

import pytest

from seepat.training.experiment import output_stage, read_json, verify_output


def test_stage_commits_a_verified_result_and_rejects_unrecorded_files(tmp_path):
    output = tmp_path / "stage"
    job = {"stage": "synthetic"}
    with output_stage(output, job) as complete:
        assert (output / "worker.lock").is_file()
        (output / "result.txt").write_text("synthetic result")
        result = complete({"status": "complete", "samples": 2})
    assert not (output / "worker.lock").exists()
    assert verify_output(output, job) == result
    (output / "unexpected.txt").write_text("keep this")
    with pytest.raises(ValueError, match="membership"):
        verify_output(output, job)


def test_failed_stage_keeps_partial_files_and_cannot_be_restarted(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="synthetic failure"), output_stage(output, {}):
        (output / "partial.txt").write_text("keep this")
        raise RuntimeError("synthetic failure")
    assert read_json(output / "summary.json")["status"] == "failed"
    assert (output / "partial.txt").read_text() == "keep this"
    assert not (output / "worker.lock").exists()
    with pytest.raises(FileExistsError), output_stage(output, {}):
        raise AssertionError("A failed stage must not be overwritten")


@pytest.mark.parametrize("bad_summary", [None, {"status": "failed"}])
def test_stage_cannot_report_success_without_a_complete_result(tmp_path, bad_summary):
    output = tmp_path / "incomplete"
    with pytest.raises((ValueError, RuntimeError)), output_stage(output, {}) as complete:
        if bad_summary is not None:
            complete(bad_summary)
    assert read_json(output / "summary.json")["status"] == "failed"
    assert not (output / "seal.json").exists()
