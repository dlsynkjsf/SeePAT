import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("larger_transfer", Path(__file__).parents[1] / "scripts/transfer_auxiliary_comparison.py")
transfer = importlib.util.module_from_spec(SPEC)
if SPEC.loader and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(transfer)


def test_verified_transfer_preserves_equal_files_and_preflights_all_conflicts(tmp_path):
    one, two = tmp_path / "one.bin", tmp_path / "two.bin"
    one.write_bytes(b"synthetic video")
    two.write_bytes(b"synthetic checkpoint")
    bundle = tmp_path / "usb"
    transfer.pack_files({"outputs/crop.mp4": one, "outputs/model.pt": two}, bundle,
                         direction="outbound", protocol_sha256="a" * 64)
    assert transfer.verify_bundle(bundle)["direction"] == "outbound"
    target = tmp_path / "target"
    (target / "outputs").mkdir(parents=True)
    (target / "outputs/model.pt").write_bytes(b"existing different model")
    with pytest.raises(ValueError, match="conflict"):
        transfer.install_bundle(bundle, target, "a" * 64, returning=False)
    assert not (target / "outputs/crop.mp4").exists()
    (target / "outputs/model.pt").write_bytes(two.read_bytes())
    result = transfer.install_bundle(bundle, target, "a" * 64, returning=False)
    assert result == {"added": 1, "equal": 1}
    assert transfer.install_bundle(bundle, target, "a" * 64, returning=False) == {"added": 0, "equal": 2}
    with pytest.raises(ValueError, match="protocol"):
        transfer.install_bundle(bundle, target, "b" * 64, returning=False)
    with pytest.raises(ValueError, match="direction"):
        transfer.install_bundle(bundle, target, "a" * 64, returning=True)


def test_unsafe_names_duplicates_and_corrupt_bytes_never_install(tmp_path):
    payload = tmp_path / "source"
    payload.write_bytes(b"fixture")
    with pytest.raises(ValueError):
        transfer.pack_files({"outputs/../../secret": payload}, tmp_path / "bad", direction="outbound", protocol_sha256="a" * 64)
    assert not (tmp_path / "bad").exists()
    bundle = tmp_path / "usb"
    transfer.pack_files({"outputs/one": payload}, bundle, direction="outbound", protocol_sha256="a" * 64)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["items"].append(manifest["items"][0])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate"):
        transfer.install_bundle(bundle, tmp_path / "target", "a" * 64, returning=False)
    assert not (tmp_path / "target").exists()


def test_corrupt_archive_is_rejected_before_target_writes(tmp_path):
    source = tmp_path / "payload"
    source.write_bytes(b"unique synthetic bytes")
    bundle = tmp_path / "usb"
    transfer.pack_files({"outputs/one": source}, bundle, direction="outbound", protocol_sha256="a" * 64)
    archive = bundle / transfer.ARCHIVE
    data = archive.read_bytes()
    archive.write_bytes(data.replace(b"unique synthetic bytes", b"broken synthetic bytes", 1))
    with pytest.raises(ValueError, match="bytes"):
        transfer.install_bundle(bundle, tmp_path / "target", "a" * 64, returning=False)
    assert not (tmp_path / "target").exists()
