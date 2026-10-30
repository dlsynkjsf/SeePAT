"""Scoped, verified one-visit transfer for the larger auxiliary comparison."""

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from seepat.artifacts import atomic_write_json, file_sha256
from seepat.live_progress import WorkflowProgress, live_progress_path
from seepat.training import auxiliary_comparison as study
from seepat.training import experiment as runtime
from seepat.training.matched_inputs import inside

ARCHIVE = "artifacts.tar"
VERSION = "larger-auxiliary-transfer-v1"


def safe_name(name):
    path = PurePosixPath(name)
    if (not isinstance(name, str) or "\\" in name or ":" in name or path.is_absolute()
            or name != path.as_posix() or ".." in path.parts or not name.startswith(("outputs/", "data/"))):
        raise ValueError("Unsafe or out-of-scope transfer path")
    return name


def stream_digest(stream):
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def pack_files(files, bundle, *, direction, protocol_sha256, progress=None):
    if direction not in {"outbound", "return"} or not files:
        raise ValueError("Invalid transfer direction or empty package")
    names = [safe_name(n) for n in files]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Transfer has duplicate Windows paths")
    if any(p.is_symlink() or not p.is_file() for p in files.values()):
        raise ValueError("Transfer requires existing ordinary files")
    bundle.mkdir(parents=True, exist_ok=False)
    items = []
    with tarfile.open(bundle / ARCHIVE, "x") as archive:
        for number, (name, source) in enumerate(sorted(files.items()), 1):
            entry = {"path": name, "bytes": source.stat().st_size, "sha256": file_sha256(source)}
            archive.add(source, arcname=name, recursive=False)
            items.append(entry)
            if progress:
                progress("hash and pack scoped files", number, len(files), name)
    manifest = {"version": VERSION, "direction": direction, "protocol_sha256": protocol_sha256, "items": items}
    atomic_write_json(bundle / "manifest.json", manifest)
    verify_bundle(bundle, progress=progress)
    return manifest


def verify_bundle(bundle, *, progress=None):
    manifest = runtime.read_json(bundle / "manifest.json")
    if (manifest.get("version") != VERSION or manifest.get("direction") not in {"outbound", "return"}
            or not manifest.get("items")):
        raise ValueError("Unsupported transfer manifest")
    names = [safe_name(r["path"]) for r in manifest["items"]]
    if len({n.casefold() for n in names}) != len(names):
        raise ValueError("Manifest has duplicate Windows paths")
    expected = {r["path"]: r for r in manifest["items"]}
    seen = set()
    with tarfile.open(bundle / ARCHIVE, "r:") as archive:
        for number, member in enumerate(archive, 1):
            name = safe_name(member.name)
            if (not member.isfile() or member.issym() or member.islnk() or name in seen or name not in expected
                    or member.size != expected[name]["bytes"]):
                raise ValueError("Archive membership/type/size differs")
            with archive.extractfile(member) as stream:
                if stream_digest(stream) != expected[name]["sha256"]:
                    raise ValueError(f"Archive bytes changed: {name}")
            seen.add(name)
            if progress:
                progress("verify every archived byte", number, len(expected), name)
    if seen != set(expected):
        raise ValueError("Archive has missing files")
    return manifest


def install_bundle(bundle, project, protocol_sha256, *, returning, progress=None):
    manifest = verify_bundle(bundle, progress=progress)
    if manifest["protocol_sha256"] != protocol_sha256:
        raise ValueError("Transfer protocol differs from this checkout")
    if manifest["direction"] != ("return" if returning else "outbound"):
        raise ValueError("Wrong transfer direction")
    equal = set()
    # Check all collisions and resolved paths before the first target write.
    for item in manifest["items"]:
        target = inside(project, item["path"])
        if target.exists():
            if not target.is_file() or target.stat().st_size != item["bytes"] or file_sha256(target) != item["sha256"]:
                raise ValueError(f"Install conflict; preserve existing file: {item['path']}")
            equal.add(item["path"])
        for parent in target.parents:
            if parent == project:
                break
            if parent.exists() and not parent.is_dir():
                raise ValueError(f"Install parent conflict: {item['path']}")
    expected = {r["path"]: r for r in manifest["items"]}
    with tarfile.open(bundle / ARCHIVE, "r:") as archive:
        for number, member in enumerate(archive, 1):
            name = safe_name(member.name)
            if name not in expected or not member.isfile() or member.size != expected[name]["bytes"]:
                raise ValueError("Archive changed after install preflight")
            target = inside(project, name)
            if name not in equal:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
            if file_sha256(target) != expected[name]["sha256"]:
                raise ValueError(f"Installed bytes differ; preserve before recovery: {name}")
            if progress:
                progress("install and verify; existing equal files preserved", number, len(expected), name)
    return {"added": len(expected) - len(equal), "equal": len(equal)}


def collect(project, protocol, binding, *, returning=False):
    if inside(project, f"{study.OUTPUT}/operation.lock").exists():
        raise ValueError("A comparison operation is active/interrupted; preserve before packaging")
    files = {}
    if returning:
        study.verify_stage(project, "all", binding)
        roots = [f"{study.OUTPUT}/{name}" for name in ("readiness", "training", "evaluation")]
    else:
        study.verify_stage(project, "inputs", binding)
        catalog = runtime.read_json(inside(project, f"{study.OUTPUT}/inputs/media_catalog.json"))
        for name, entry in catalog.items():
            path = inside(project, safe_name(name))
            if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Prepared crop bytes changed: {name}")
            files[name] = path
        for name in protocol["metadata_bindings"]:
            files[name] = inside(project, name)
        initial = inside(project, study.INITIAL)
        if file_sha256(initial) != study.INITIAL_SHA:
            raise ValueError("Starting checkpoint bytes changed")
        files[study.INITIAL] = initial
        roots = [f"{study.OUTPUT}/{name}" for name in ("selection", "preparation", "inputs")]
    for root in roots:
        for path in inside(project, root).rglob("*"):
            if path.is_file():
                if path.name == "worker.lock":
                    raise ValueError("Active/interrupted output cannot be packaged")
                files[path.relative_to(project).as_posix()] = path
    if returning:
        for path in project.glob("larger_auxiliary_*.log"):
            if "pack" not in path.name and "transfer" not in path.name:
                files[f"{study.OUTPUT}/run_records/{path.name}"] = path
        for path in inside(project, study.OUTPUT).glob("*_status.progress.json"):
            files[f"{study.OUTPUT}/run_records/{path.name}"] = path
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "install", "pack-return", "install-return", "verify", "watch"))
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--operation", choices=("pack", "install", "pack-return", "install-return", "verify"), default="pack")
    args = parser.parse_args(argv)
    project, bundle = args.project.resolve(), args.bundle.resolve()
    operation = args.operation if args.command == "watch" else args.command
    report = bundle.parent / f"{bundle.name}.{operation}.status.json"
    if args.command == "watch":
        return runtime.watch_progress(live_progress_path(report), "LARGER AUXILIARY TRANSFER")
    protocol, binding = study.load_protocol(project, inside(project, study.CONFIG),
                                          metadata=args.command in {"pack", "pack-return"})
    progress = WorkflowProgress(inside(project, study.CONFIG), report, 1)
    progress.start_job(1, args.command)
    try:
        if args.command in {"install", "install-return"}:
            result = install_bundle(bundle, project, binding["protocol_sha256"], returning=args.command == "install-return", progress=progress)
        elif args.command == "verify":
            result = verify_bundle(bundle, progress=progress)
            if result["protocol_sha256"] != binding["protocol_sha256"]:
                raise ValueError("Bundle protocol differs")
        else:
            result = pack_files(collect(project, protocol, binding, returning=args.command == "pack-return"), bundle,
                                direction="return" if args.command == "pack-return" else "outbound",
                                protocol_sha256=binding["protocol_sha256"], progress=progress)
        progress.finish()
        print(f"LARGER AUXILIARY TRANSFER COMPLETE | {operation}", flush=True)
        return result
    except BaseException as error:
        progress.finish(error)
        raise


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, RuntimeError) as error:
        raise SystemExit(f"LARGER AUXILIARY TRANSFER STOPPED | {error}") from error
