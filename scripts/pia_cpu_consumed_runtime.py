"""Compute raw identity vectors only for entries consumed by the unchanged PIA builder."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def consumed_indices(rows, vocabulary, keep, ignored, frames_per_phone):
    """Mirror the author's ordered first-F selection, including entries with missing crops."""
    if frames_per_phone != 5 or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Unexpected author first-five selection")
    counts = dict.fromkeys(vocabulary, 0)
    selected = []
    for index, row in enumerate(rows):
        if row["frame_idx"] != index:
            raise ValueError("Author dense row order changed")
        phone = row.get("phoneme", "").lower().strip()
        if phone in keep and phone not in ignored:
            if counts[phone] < frames_per_phone:
                selected.append(index)
            counts[phone] += 1
    return selected


def selected_embeddings(capture, embed, indices):
    """Decode sequentially as before, invoking the stateless identity model only when used."""
    values, decoded = {}, 0
    try:
        if indices != sorted(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Identity indices must be ordered unique nonnegative integers")
        selected = set(indices)
        while indices and decoded <= indices[-1] and capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            if decoded in selected:
                values[decoded] = embed(frame)
                done = len(values)
                if done % 5 == 0 or done == len(indices):
                    print(
                        f"[CPU_PROGRESS] ArcFace consumed entries {done}/{len(indices)}", flush=True
                    )
            decoded += 1
    finally:
        capture.release()
    return values, decoded


def install_consumed_hooks(utils):
    original_mouth = utils.extract_mouth_states
    original_process = utils.process_single_video
    retained = None

    def mouth(*args, **kwargs):
        nonlocal retained
        rows = original_mouth(*args, **kwargs)
        retained = len(rows)
        return rows

    def deferred_identity(_path):
        if retained is None or utils.get_device() != "cpu":
            raise ValueError("Consumed identity extraction requires the original CPU mouth stage")
        print("[STEP] Defer identity until the original phoneme alignment is available", flush=True)
        # Unconsumed identity fields are explicitly empty, not fabricated embeddings.
        return [()] * retained

    def process(*args, **kwargs):
        path, label_dir = original_process(*args, **kwargs)
        source = kwargs.get("video_path", args[0] if args else None)
        rows = json.loads(Path(path).read_text())
        indices = consumed_indices(
            rows,
            utils.default_phoneme_vocab(),
            utils.KEEP_PHONEMES,
            utils.IGNORED_PHONEMES,
            utils.IMAGES_PER_PHON,
        )
        print(
            f"[STEP] ArcFace: {len(indices)} consumed entries from {len(rows)} dense mouth rows",
            flush=True,
        )
        vectors, decoded = {}, 0
        if indices:
            app = utils.insightface.app.FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )
            app.prepare(ctx_id=-1)

            def embed(frame):
                faces = app.get(frame)
                value = (
                    faces[0].embedding
                    if faces and hasattr(faces[0], "embedding")
                    else utils.np.zeros(512)
                )
                return value.tolist()

            vectors, decoded = selected_embeddings(utils.cv2.VideoCapture(source), embed, indices)
        for index in indices:
            # The original builder pads identity with zeros beyond a short source decode.
            rows[index]["arcface"] = vectors.get(index, [0.0] * 512)
        temporary = Path(path).with_suffix(".identity.tmp")
        with temporary.open("w") as stream:
            json.dump(rows, stream, indent=2)
        temporary.replace(path)
        receipt = {
            "policy": "exact-author-first-five-identity-v1",
            "mouth_rows": len(rows),
            "selected_dense_indices": indices,
            "identity_calls": len(vectors),
            "decoded_source_frames": decoded,
            "missing_source_indices": [i for i in indices if i not in vectors],
            "unused_identity_fields": "empty; never consumed by the unchanged classifier builder",
            "mouth_tracking_alignment_and_builder_changed": False,
        }
        (Path(label_dir).parent / "consumed_identity.json").write_text(
            json.dumps(receipt, indent=2)
        )
        return path, label_dir

    utils.extract_mouth_states = mouth
    utils.extract_arcface_features = deferred_identity
    utils.process_single_video = process


def main():
    import argparse

    import onnxruntime as ort
    from insightface import app
    from insightface.model_zoo import model_zoo

    sys.path.insert(0, "/helpers")
    from pia_cpu_identity_runtime import identity_modules
    from pia_cpu_spin_runtime import disable_spinning

    disable_spinning(model_zoo, ort)
    identity_modules(app)
    sys.path.insert(0, "/pia")
    from src import utils

    # Preserve the existing decode precheck; the author reads the same audio again later.
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    utils.whisperx.load_audio(args.video)
    print("[OK] Audio decode precheck before visual inference", flush=True)
    install_consumed_hooks(utils)
    import main as author

    author.main()


if __name__ == "__main__":
    main()
