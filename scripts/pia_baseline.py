"""Isolated author-PIA feasibility setup; external evaluation remains locked."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from seepat.artifacts import atomic_write_json, atomic_write_text, file_sha256
from seepat.live_progress import WorkflowProgress, format_live_progress, live_progress_path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/pia_baseline.v1.json"
FACE_MEMBERS = {"1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx", "w600k_r50.onnx"}


def _json(path):
    if not path.is_file():
        raise ValueError(f"Missing or partial PIA stage; preserve existing files: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected a PIA artifact object")
    return value


def _relative(base, name):
    parts = PurePosixPath(name)
    if not parts.parts or parts.is_absolute() or ".." in parts.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe relative artifact path")
    path = base / Path(*parts.parts)
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Artifact path escaped its owned directory")
    return path


def load_plan(config: Path, *, project=ROOT):
    plan = _json(config)
    if (plan.get("version") != "pia-feasibility-v1" or plan.get("scope") != "known_development_only"
            or plan.get("revision") != "91834722cb5b814da6dbb0e16cf4931d3484d06e"):
        raise ValueError("Unsupported PIA feasibility contract; H1 stays locked")
    work = _relative(project, plan["work_dir"])
    report = _relative(project, plan["report_dir"])
    if (not work.is_relative_to(project / ".tools") or work == project / ".tools"
            or not report.is_relative_to(project / "outputs") or report == project / "outputs"):
        raise ValueError("PIA work and reports must remain isolated")
    for name, digest in plan["owned_code_sha256"].items():
        if hashlib.sha256(_relative(project, name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            raise ValueError(f"PIA controller source changed: {name}")
    names = [a["destination"] for a in plan["assets"]]
    if any(type(a.get("bytes")) is not int or a["bytes"] <= 0 for a in plan["assets"]):
        raise ValueError("Asset sizes must be positive integers")
    total = sum(a["bytes"] for a in plan["assets"])
    if len(names) != len(set(names)) or total > 6_100_000_000 or plan.get("download_bytes", total) != total:
        raise ValueError("Duplicate assets or download budget exceeded")
    for name in names:
        if not name.startswith(("source/", "cache/")):
            raise ValueError("PIA assets must stay in source or cache directories")
        _relative(work, name)
    for asset in plan["assets"]:
        url = urlsplit(asset["url"])
        if (url.scheme != "https" or url.username or url.password or url.hostname not in
                {"raw.githubusercontent.com", "huggingface.co", "download.pytorch.org", "github.com"}):
            raise ValueError("Unreviewed PIA asset origin")
    if len(plan["requests"]) > 4 or not 1 <= plan.get("case_timeout_seconds", 1800) <= 7200:
        raise ValueError("PIA development budget exceeded")
    contract = {"version": plan["version"], "scope": plan["scope"],
                "plan_sha256": file_sha256(config), "project": str(project.resolve())}
    return plan, work, report, contract


def _command(command, log, tracker, phase, *, timeout=None):
    """External process boundary; append logs, heartbeat and preserve interruptions."""
    log.parent.mkdir(parents=True, exist_ok=True)
    offset = log.stat().st_size if log.exists() else 0
    with log.open("ab") as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if timeout is not None and time.monotonic() - started >= timeout:
                    raise TimeoutError(f"{phase} exceeded its reviewed time budget; preserve outputs")
                tracker(phase, 0, 0, log.name)
                time.sleep(1)
            if process.wait() != 0:
                raise RuntimeError(f"{phase} failed; preserve and inspect {log}")
        except BaseException:
            if "--name" in command:
                name = command[command.index("--name") + 1]
                subprocess.run(["docker", "stop", "--time", "10", name],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=False)
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    with log.open("rb") as stream:
        stream.seek(offset)
        return stream.read().decode("utf-8", errors="replace")


def _image_id(name):
    identifier = subprocess.check_output(
        ["docker", "image", "inspect", name, "--format", "{{.Id}}"], text=True).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", identifier):
        raise ValueError("Missing immutable Docker image identity")
    return identifier


def _mount(path, target, readonly=True):
    if "," in str(path):
        raise ValueError("Docker bind paths cannot contain commas")
    return ["--mount", f"type=bind,source={path.resolve()},target={target}" + (",readonly" if readonly else "")]


def _container(image, work):
    command = ["docker", "run", "--rm", "--name", "seepat-pia-" + uuid.uuid4().hex[:16], "--network", "none",
               *_mount(work / "source", "/pia"), *_mount(work / "cache", "/cache")]
    for setting in ("HOME=/cache/home", "HF_HOME=/cache/hf", "HF_HUB_CACHE=/cache/hf/hub",
                    "TORCH_HOME=/cache/torch", "XDG_CACHE_HOME=/tmp/xdg", "MPLCONFIGDIR=/tmp/mpl",
                    "HF_HUB_OFFLINE=1", "TRANSFORMERS_OFFLINE=1"):
        command.extend(["-e", setting])
    return command, image


def _check_environment(environment, plan):
    if (not environment.get("python", "").startswith("3.12.") or environment.get("device") != "cpu"
            or environment.get("strict_author_checkpoint_load") is not True
            or environment.get("video_inference") is not False):
        raise ValueError("PIA environment/checkpoint probe did not pass")
    for name, expected in plan["expected_versions"].items():
        if environment.get("versions", {}).get(name, "").split("+")[0] != expected:
            raise ValueError(f"PIA package version differs: {name}")


def _file_record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def initialize_cache(plan, work, *, verify_only=False):
    """Materialize pinned offline cache references and the official five-file face bundle."""
    records = []
    for reference in plan.get("references", []):
        path = _relative(work, reference["path"])
        content = reference["content"]
        if not reference["path"].startswith("cache/") or not re.fullmatch(r"[0-9a-f]{40}", content):
            raise ValueError("Invalid pinned cache reference")
        if path.exists() and path.read_text() != content:
            raise ValueError("Existing cache reference changed; preserve it")
        if not path.exists():
            if verify_only:
                raise ValueError("Pinned cache reference is missing")
            atomic_write_text(path, content)
        records.append(_file_record(path))
    if not plan.get("face_archive"):
        return records
    archive = _relative(work, plan["face_archive"])
    folder = work / "cache/home/.insightface/models/buffalo_l"
    seal = work / "cache/face-bundle.json"
    if seal.exists():
        saved = _json(seal)
        if saved["archive_sha256"] != file_sha256(archive):
            raise ValueError("Face archive changed")
        if {Path(row["path"]).name for row in saved["files"]} != FACE_MEMBERS:
            raise ValueError("Face archive membership differs")
        for row in saved["files"]:
            if Path(row["path"]).parent != folder or _file_record(folder / Path(row["path"]).name) != row:
                raise ValueError("Saved face cache changed")
        return records + saved["files"]
    if verify_only:
        raise ValueError("Face archive extraction receipt is missing")
    with zipfile.ZipFile(archive) as source:
        files = {}
        for item in source.infolist():
            name = PurePosixPath(item.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in item.filename or ":" in item.filename
                    or (item.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError("Unsafe face archive entry; preserve the archive")
            if item.is_dir():
                continue
            if name.name in files:
                raise ValueError("Duplicate face archive entry")
            files[name.name] = item
        if set(files) != FACE_MEMBERS or sum(row.file_size for row in files.values()) > 1_000_000_000:
            raise ValueError("Unexpected face archive membership or size")
        if folder.exists():
            raise ValueError("Partial face extraction; preserve it for recovery")
        folder.mkdir(parents=True)
        extracted = []
        for name, item in files.items():
            path = folder / name
            with source.open(item) as src, path.open("xb") as destination:
                while chunk := src.read(1024 * 1024):
                    destination.write(chunk)
            if path.stat().st_size != item.file_size:
                raise ValueError("Incomplete face archive member")
            extracted.append(_file_record(path))
    atomic_write_json(seal, {"archive_sha256": file_sha256(archive), "files": extracted})
    return records + extracted


def verify_setup(config: Path = CONFIG, *, project=ROOT, progress=None):
    plan, work, _, contract = load_plan(config, project=project)
    saved = _json(work / "setup.json")
    if saved.get("status") != "complete" or saved.get("contract") != contract:
        raise ValueError("PIA setup is partial or belongs to another contract")
    expected = saved["assets"]
    if len(expected) != len(plan["assets"]):
        raise ValueError("PIA asset membership differs")
    for index, (asset, old) in enumerate(zip(plan["assets"], expected, strict=True), 1):
        if progress:
            progress("verify pinned PIA assets", index - 1, len(expected), asset["destination"])
        receipt = asset_receipt(asset, _relative(work, asset["destination"]))
        if any(receipt[k] != old[k] for k in ("path", "bytes", "sha256")):
            raise ValueError("PIA saved asset changed")
    if initialize_cache(plan, work, verify_only=True) != saved["cache_files"]:
        raise ValueError("PIA offline cache differs")
    if _image_id(saved["image"]) != saved["image"]:
        raise ValueError("PIA container image differs")
    _check_environment(saved["environment"], plan)
    return saved


def _preserve_controller(plan, work, project):
    for name, digest in plan["owned_code_sha256"].items():
        target = _relative(work / "controller_source", name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_relative(project, name).read_bytes())
        if hashlib.sha256(target.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
            raise ValueError("Preserved controller differs; retain it for recovery")


def setup(config: Path = CONFIG, *, project=ROOT, allow_network=False):
    plan, work, reports, contract = load_plan(config, project=project)
    if (work / "setup.json").exists():
        return {**verify_setup(config, project=project), "action": "skipped"}
    if not allow_network:
        raise ValueError("First setup requires --allow-network; SeePAT environment stays unchanged")
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < plan.get("minimum_free_bytes", 0):
        raise ValueError("Insufficient free space for the isolated PIA setup")
    tracker = WorkflowProgress(config, reports / "setup.json", jobs_total=3)
    try:
        if (work / "job.json").exists() and _json(work / "job.json") != contract:
            raise ValueError("Existing PIA setup belongs to another source/config; preserve it")
        atomic_write_json(work / "job.json", contract)
        _preserve_controller(plan, work, project)
        tracker.start_job(1, "PIA pinned downloads")
        receipts = []
        for index, asset in enumerate(plan["assets"], 1):
            tracker("download pinned asset", 0, asset["bytes"], asset["destination"])
            receipts.append(download_asset(
                asset, _relative(work, asset["destination"]), allow_network=True,
                progress=lambda done, total, name=asset["destination"]: tracker("download pinned asset", done, total, name)))
            tracker("assets verified", index, len(plan["assets"]), asset["destination"])
        (work / "cache/home").mkdir(parents=True, exist_ok=True)
        cache_files = initialize_cache(plan, work)
        tracker.start_job(2, "PIA isolated CPU image")
        image_file = work / "image.json"
        if image_file.exists():
            image = _json(image_file)
            if image["contract"] != contract:
                raise ValueError("Existing image belongs to another setup")
            identifier = _image_id(image["image"])
        else:
            tag = "seepat-pia-feasibility:20261031"
            _command(["docker", "build", "--progress", "plain", "-t", tag,
                      str(project / "containers/pia")], work / "build.log", tracker, "build isolated image", timeout=7200)
            identifier = _image_id(tag)
            atomic_write_json(image_file, {"contract": contract, "image": identifier})
        tracker.start_job(3, "PIA import and checkpoint probe")
        command, image = _container(identifier, work)
        atomic_write_json(work / "probe-container.json", {"name": command[command.index("--name") + 1]})
        text = _command([*command, image, "python", "/opt/pia-probe.py"], work / "probe.log",
                        tracker, "imports and strict checkpoint load; no video", timeout=1800)
        payloads = [line.removeprefix("PIA_ENV_JSON:") for line in text.splitlines() if line.startswith("PIA_ENV_JSON:")]
        if len(payloads) != 1:
            raise ValueError("Container produced no unique environment receipt")
        environment = json.loads(payloads[0])
        _check_environment(environment, plan)
        if load_plan(config, project=project)[3] != contract:
            raise ValueError("PIA plan changed during setup; preserve the partial stage")
        for asset, old in zip(plan["assets"], receipts, strict=True):
            current = asset_receipt(asset, _relative(work, asset["destination"]))
            if current["sha256"] != old["sha256"]:
                raise ValueError("PIA asset changed during setup")
        saved = {"status": "complete", "contract": contract, "image": identifier,
                 "environment": environment, "assets": receipts, "cache_files": cache_files, "h1_ready": False}
        atomic_write_json(work / "setup.json", saved)
        atomic_write_json(reports / "setup.json", saved)
    except BaseException as error:
        tracker.finish(error)
        raise
    tracker.finish()
    return saved


def _sources(plan, project):
    sources = []
    if not 1 <= len(plan["requests"]) <= 4:
        raise ValueError("Development feasibility needs one to four fixed requests")
    for item in plan["requests"]:
        path = _relative(project, item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError("Development request changed")
        request = _json(path)
        if (set(request) != {"version", "dataset_source", "split", "videos"}
                or request["version"] != "raw-local-inputs-v1"
                or request["dataset_source"] != "local-unlabelled" or request["split"] != "local"
                or len(request["videos"]) != 1):
            raise ValueError("Only fixed unlabelled development inputs are supported; external test is locked")
        row = request["videos"][0]
        if (set(row) != {"source", "origin"} or row["origin"] not in {"local-demo", "personal-recording"}
                or set(row["source"]) != {"path", "sha256"}):
            raise ValueError("Input contains labels or unsupported provenance")
        source = Path(row["source"]["path"]).resolve()
        if "deepfakeeval" in re.sub("[^a-z0-9]", "", str(source).lower()) or source.suffix.lower() != ".mp4":
            raise ValueError("Unsupported or locked external video")
        if file_sha256(source) != row["source"]["sha256"]:
            raise ValueError("Raw development video differs")
        sources.append({"path": str(source), "sha256": row["source"]["sha256"], "request_sha256": item["sha256"]})
    if len({row["sha256"] for row in sources}) != len(sources):
        raise ValueError("Duplicate development video")
    return sources


def _crop_records(directory, phonemes):
    path = directory / "author/testvideo/video_aligned.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("Author alignment must be a frame list")
    counts, records = {}, {}
    for row in data:
        phone = row.get("phoneme", "").lower().strip()
        if phone not in phonemes or counts.get(phone, 0) >= 5:
            continue
        counts[phone] = counts.get(phone, 0) + 1
        if row.get("crop_path"):
            crop = _relative(directory / "author/testvideo", row["crop_path"])
            if crop.is_file():
                records[crop.relative_to(directory).as_posix()] = file_sha256(crop)
    if not records:
        raise ValueError("No usable author phoneme crops; preserve raw output")
    return records


def _read_case(directory, contract):
    saved = _json(directory / "summary.json")
    if saved.get("status") != "complete" or saved.get("contract") != contract:
        raise ValueError("Changed or partial PIA case; preserve it")
    for name, digest in saved["output_sha256"].items():
        if file_sha256(_relative(directory, name)) != digest:
            raise ValueError("Author output changed; preserve it")
    if parse_prediction(directory / "author/results.json") != saved["prediction"]:
        raise ValueError("Saved prediction contradicts author output")
    return saved


def _development_inputs(config, project, progress=None):
    plan, work, reports, contract = load_plan(config, project=project)
    saved = verify_setup(config, project=project, progress=progress)
    contract = {**contract, "setup_sha256": file_sha256(work / "setup.json"), "image": saved["image"],
                "sources": _sources(plan, project), "device": "cpu", "network": "none"}
    return plan, work, reports, contract


def verify_development(config: Path = CONFIG, *, project=ROOT, progress=None):
    _, _, reports, contract = _development_inputs(config, project, progress)
    saved = _json(reports / "development.json")
    if saved.get("status") != "complete" or saved.get("contract") != contract:
        raise ValueError("Development run is missing, partial or changed")
    cases = [_read_case(reports / f"development/case_{i:02}", {**contract, "case": i})
             for i in range(1, len(contract["sources"]) + 1)]
    if cases != saved["cases"]:
        raise ValueError("Development case records differ")
    return saved


def run_development(config: Path = CONFIG, *, project=ROOT):
    plan, work, reports, _ = load_plan(config, project=project)
    if (reports / "development.json").exists():
        return {**verify_development(config, project=project), "action": "skipped"}
    tracker = WorkflowProgress(config, reports / "development.json", jobs_total=len(plan["requests"]) + 1)
    cases = []
    try:
        tracker.start_job(1, "verify PIA setup")
        plan, work, reports, contract = _development_inputs(config, project, tracker)
        for i, source in enumerate(contract["sources"], 1):
            tracker.start_job(i + 1, f"PIA development case {i:02}")
            directory = reports / f"development/case_{i:02}"
            expected = {**contract, "case": i}
            if directory.exists():
                cases.append(_read_case(directory, expected))
                tracker("reuse verified PIA case", 1, 1, str(i))
                continue
            directory.mkdir(parents=True)
            atomic_write_json(directory / "job.json", expected)
            (directory / "author").mkdir()
            command, image = _container(contract["image"], work)
            atomic_write_json(directory / "container.json", {"name": command[command.index("--name") + 1]})
            text = _command([*command, *_mount(Path(source["path"]), "/input/video.mp4"),
                             *_mount(directory / "author", "/output", False), image,
                             "python", "main.py", "--video", "/input/video.mp4", "--outdir", "/output"],
                            directory / "author.log", tracker, "author PIA inference; CPU; offline",
                            timeout=plan["case_timeout_seconds"])
            prediction = parse_prediction(directory / "author/results.json")
            raw = _json(directory / "author/results.json")
            if raw.get("aligned_json") != "/output/testvideo/video_aligned.json":
                raise ValueError("Author alignment output escaped its case")
            outputs = _crop_records(directory, plan["phonemes"])
            for name in ("author/results.json", "author/testvideo/video_aligned.json", "author.log", "job.json", "container.json"):
                outputs[name] = file_sha256(directory / name)
            if any(contract[key] != value for key, value in load_plan(config, project=project)[3].items()):
                raise ValueError("PIA plan changed during inference; preserve outputs")
            if _sources(plan, project) != contract["sources"]:
                raise ValueError("Development input changed during inference")
            result = {"status": "complete", "contract": expected, "prediction": prediction,
                      "output_sha256": outputs, "selected_crop_files": len(outputs) - 5,
                      "alignment": "fallback" if "espeak-based aligner unavailable" in text else "preferred",
                      "scope": "development_feasibility_only", "metrics": None, "h1_ready": False}
            atomic_write_json(directory / "summary.json", result)
            cases.append(result)
            tracker("PIA case complete", 1, 1, prediction["verdict"])
        result = {"status": "complete", "contract": contract, "cases": cases,
                  "scope": "development_feasibility_only", "metrics": None, "h1_ready": False}
        atomic_write_json(reports / "development.json", result)
    except BaseException as error:
        tracker.finish(error)
        raise
    tracker.finish()
    return result


def asset_receipt(asset: dict, path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size != asset["bytes"]:
        raise ValueError(f"Asset size/path differs: {path}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {asset['bytes']}\0".encode())
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            sha.update(chunk)
            git.update(chunk)
    if ((asset.get("sha256") and sha.hexdigest() != asset["sha256"])
            or (asset.get("sha256_prefix") and not sha.hexdigest().startswith(asset["sha256_prefix"]))
            or (asset.get("git_blob_sha1") and git.hexdigest() != asset["git_blob_sha1"])):
        raise ValueError(f"Asset digest differs; preserve it: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha.hexdigest(), "mtime_ns": path.stat().st_mtime_ns}


def download_asset(asset: dict, path: Path, *, allow_network=False, progress=None) -> dict:
    """Resume only supported byte ranges; publish after the pinned digest matches."""
    release = (asset.get("release_asset_id") == 101975360 and asset.get("bytes") == 288621354
               and asset.get("url") == "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip")
    if (type(asset.get("bytes")) is not int or not 0 < asset["bytes"] <= 4_000_000_000
            or not (release or any(asset.get(key) for key in ("sha256", "git_blob_sha1", "sha256_prefix")))
            or not asset["url"].startswith("https://")):
        raise ValueError("Asset needs a bounded size and published identity")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Asset destinations cannot be symlinks")
    if path.exists():
        return asset_receipt(asset, path)
    if not allow_network:
        raise ValueError("Downloads require --allow-network")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if partial.is_symlink():
        raise ValueError("Partial destination cannot be a symlink")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > asset["bytes"]:
        raise ValueError("Oversized partial; preserve it for recovery")
    if offset < asset["bytes"]:
        headers = {"User-Agent": "SeePAT-PIA-feasibility"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(asset["url"], headers=headers), timeout=30) as response:
            if offset:
                span = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if (response.status != 206 or span is None or int(span[1]) != offset
                        or int(span[3]) != asset["bytes"] or int(span[2]) >= asset["bytes"]):
                    raise ValueError("Server rejected Range resume; preserve the partial file")
            with partial.open("ab" if partial.exists() else "xb") as stream:
                while chunk := response.read(1024 * 1024):
                    if offset + len(chunk) > asset["bytes"]:
                        raise ValueError("Response exceeds the pinned asset size")
                    stream.write(chunk)
                    offset += len(chunk)
                    if progress:
                        progress(offset, asset["bytes"])
    asset_receipt(asset, partial)
    partial.rename(path)
    return asset_receipt(asset, path)


def parse_prediction(path: Path) -> dict:
    """Validate the author entry point's video result, preserving its strict cutoff."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("probs"), dict):
        raise TypeError("Unsupported PIA probabilities object")
    probs = record.get("probs", {})
    values = [probs.get(key) for key in ("real", "fake")]
    if (record.get("model") != "PIA" or record.get("video") != "/input/video.mp4"
            or record.get("weights") != "checkpoints/best_model.pth"
            or any(type(value) not in {float, int} or not math.isfinite(value)
                   or not 0 <= value <= 1 for value in values)
            or not math.isclose(sum(values), 1, abs_tol=1e-5)):
        raise ValueError("Unsupported PIA identity or probabilities")
    fake = values[1] > .5
    if (type(record.get("predicted_label")) is not int or record["predicted_label"] != int(fake)
            or record.get("predicted_label_name") != ("Fake" if fake else "Real")):
        raise ValueError("Author verdict contradicts P(fake) > 0.5")
    return {"real_probability": values[0], "fake_probability": values[1],
            "verdict": "manipulated" if fake else "authentic", "cutoff": .5, "operator": ">"}


@contextmanager
def _worker_lock(work):
    work.mkdir(parents=True, exist_ok=True)
    path = work / "worker.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
    except FileExistsError as error:
        raise FileExistsError("PIA worker lock exists; inspect its PID/process before recovery") from error
    try:
        yield
    finally:
        if path.is_file() and path.read_text() == str(os.getpid()):
            path.unlink()


def main(argv=None, *, project=ROOT):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "run", "verify", "watch", "status"))
    parser.add_argument("--config", type=Path, default=project / "configs/pia_baseline.v1.json")
    parser.add_argument("--stage", choices=("setup", "development", "verify-setup", "verify-development"), default="setup")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--watch-seconds", type=float, default=5)
    args = parser.parse_args(argv)
    if not math.isfinite(args.watch_seconds) or args.watch_seconds <= 0:
        parser.error("--watch-seconds must be positive and finite")
    try:
        _, work, reports, _ = load_plan(args.config, project=project)
        if args.command in {"status", "watch"}:
            path = live_progress_path(reports / f"{args.stage}.json")
            while True:
                record = (_json(path) if path.exists() else
                          {"status": "unavailable", "message": "No PIA progress yet; no work started by this reader."})
                print(json.dumps(record) if args.command == "status" else format_live_progress(record), flush=True)
                if args.command == "status" or record["status"] in {"complete", "failed", "interrupted"}:
                    return 0
                try:
                    time.sleep(args.watch_seconds)
                except KeyboardInterrupt:
                    print("Tracking stopped; the PIA worker was not interrupted.")
                    return 0
        if args.allow_network and args.command != "setup":
            raise ValueError("Only setup may enable networking; development inference is offline")
        if args.command == "verify":
            if args.stage not in {"setup", "development"}:
                raise ValueError("Verification stage must be setup or development")
            tracker = WorkflowProgress(args.config, reports / f"verify-{args.stage}.json", jobs_total=1)
            tracker.start_job(1, "verify PIA " + args.stage)
            try:
                result = (verify_setup if args.stage == "setup" else verify_development)(
                    args.config, project=project, progress=tracker)
            except BaseException as error:
                tracker.finish(error)
                raise
            tracker.finish()
        else:
            with _worker_lock(work):
                result = (setup(args.config, project=project, allow_network=args.allow_network)
                          if args.command == "setup" else run_development(args.config, project=project))
        print(f"PIA {args.command.upper()} COMPLETE | {result['status']} | development only; H1 locked")
        return 0
    except KeyboardInterrupt:
        print("PIA INTERRUPTED | preserve partial files; completed cases are reusable")
        return 130
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as error:
        print(f"PIA STOPPED | {type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
