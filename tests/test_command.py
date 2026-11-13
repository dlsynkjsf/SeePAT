import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import seepat_cli as cli


@pytest.mark.parametrize(
    "command,args,expected",
    [
        ("pia", [], ["run"]),
        ("pia", ["watch"], ["watch"]),
        ("pia", ["--help"], ["--help"]),
        (
            "watch",
            ["--config", "a b.yaml"],
            ["--workflow-config", "a b.yaml", "--watch-seconds", "5"],
        ),
        (
            "watch",
            ["--config=x", "--watch-seconds=2"],
            ["--workflow-config=x", "--watch-seconds=2"],
        ),
        ("verify", ["--config=x"], ["--workflow-config=x", "--verify"]),
        ("run", ["--config", "x"], ["--config", "x"]),
        ("run", [], ["--config", cli.DEFAULT_CONFIG]),
        ("run", ["--json"], ["--config", cli.DEFAULT_CONFIG, "--json"]),
        ("run", ["--config=x"], ["--config=x"]),
        ("run", ["--help"], ["--help"]),
        ("watch", [], ["--workflow-config", cli.DEFAULT_CONFIG, "--watch-seconds", "5"]),
        ("verify", [], ["--workflow-config", cli.DEFAULT_CONFIG, "--verify"]),
        ("verify", ["--verify"], ["--workflow-config", cli.DEFAULT_CONFIG, "--verify"]),
    ],
)
def test_command_options(command, args, expected):
    assert cli.command_args(command, args) == expected


@pytest.mark.parametrize("args,config", [
    (["run", "--config", "folder with spaces/config.yaml"], "folder with spaces/config.yaml"),
    (["run"], cli.DEFAULT_CONFIG),
    (["train", "--config", "training config.yaml"], "training config.yaml"),
])
def test_dispatch_preserves_arguments_exit_code_and_environment(monkeypatch, args, config):
    seen = []

    def run(args, **kwargs):
        seen.append((args, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", run)
    assert cli.main(args) == 7
    argv, options = seen[0]
    assert argv[-3:] == ["seepat.workflow", "--config", config]
    assert options["env"]["PYTHONPATH"] and options["check"] is False
    assert options["cwd"] == Path(cli.__file__).resolve().parents[1]
    assert "shell" not in options


def test_help_starts_nothing(monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("worker started"))
    assert cli.main([]) == 0
    assert "pia" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        cli.main(["--help"])
    assert caught.value.code == 0


def test_ui_dispatch_keeps_server_options_and_never_selects_a_training_command(monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", lambda args, **kwargs:
                        seen.append(args) or SimpleNamespace(returncode=0))
    assert cli.main(["ui", "--config", "custom-ui.json"]) == 0
    assert seen[0][-4:] == ["-m", "scripts.demo_ui", "--config", "custom-ui.json"]


@pytest.mark.parametrize("script,args,expected", [
    ("seepat.py", [], ["ui"]),
    ("seepat.py", ["train", "--help"], ["train", "--help"]),
    ("pia.py", [], ["pia"]),
    ("pia.py", ["watch"], ["pia", "watch"]),
])
def test_python_launchers_share_router_without_starting_jobs(monkeypatch, script, args, expected):
    seen = []
    monkeypatch.setattr(cli, "main", lambda values: seen.append(values) or 7)
    monkeypatch.setattr(sys, "argv", [script, *args])
    with pytest.raises(SystemExit) as caught:
        runpy.run_path(str(Path(cli.__file__).resolve().parents[1] / script), run_name="__main__")
    assert caught.value.code == 7 and seen == [expected]


def test_root_launcher_preserves_real_package_import_without_loading_models():
    root = Path(cli.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", ("import seepat, sys; "
        "from seepat import artifacts; print(seepat.__file__); print('torch' in sys.modules)")],
        cwd=root, env={**os.environ, "PYTHONPATH": ""}, capture_output=True, text=True, check=True)
    assert Path(result.stdout.splitlines()[0]) == root / "src/seepat/__init__.py"
    assert result.stdout.splitlines()[1] == "False"


@pytest.mark.parametrize("script", ["seepat.py", "pia.py"])
def test_python_launcher_help_from_another_directory(script, tmp_path):
    root = Path(cli.__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "usage:" in result.stdout and "Traceback" not in result.stderr


def test_interrupt_returns_standard_exit_code(monkeypatch):
    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess, "run", interrupted)
    assert cli.main(["run", "--config", "x"]) == 130


@pytest.mark.parametrize("verify,current,exit_code", [(True, False, 1), (True, True, 0), (False, False, 0)])
def test_progress_verification_exit_code(monkeypatch, verify, current, exit_code):
    from seepat import progress

    monkeypatch.setattr("sys.argv", ["progress", "--workflow-config", "unused.yaml", *(["--verify"] if verify else [])])
    monkeypatch.setattr(progress, "read_workflow_progress", lambda _: {"all_current": current})
    monkeypatch.setattr(progress, "format_workflow_progress", lambda _: "saved status")
    if exit_code:
        with pytest.raises(SystemExit) as caught:
            progress.main()
        assert caught.value.code == exit_code
    else:
        assert progress.main() is None


@pytest.mark.parametrize("status,exit_code", [("failed", 1), ("interrupted", 1), ("complete", 0)])
def test_workflow_watcher_exit_code(monkeypatch, status, exit_code):
    from seepat import progress

    monkeypatch.setattr("sys.argv", ["progress", "--workflow-config", "unused.yaml", "--watch-seconds", "1"])
    monkeypatch.setattr(progress, "read_live_workflow_progress", lambda _: {"status": status})
    monkeypatch.setattr(progress, "format_live_progress", lambda _: status)
    monkeypatch.setattr(progress, "sleep", lambda _: pytest.fail("terminal workflow should not wait"))
    if exit_code:
        with pytest.raises(SystemExit) as caught:
            progress.main()
        assert caught.value.code == exit_code
    else:
        assert progress.main() is None
