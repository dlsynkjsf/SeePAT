"""Keep PIA's raw identity inputs while skipping unused InsightFace outputs."""

from __future__ import annotations

import sys


def identity_modules(app):
    """Retain the models producing the detector keypoints and raw identity embedding."""
    original = app.FaceAnalysis

    def analysis(*args, **kwargs):
        if kwargs.get("allowed_modules") is not None:
            raise ValueError("Unexpected explicit face-task selection")
        kwargs["allowed_modules"] = ["detection", "recognition"]
        result = original(*args, **kwargs)
        if set(result.models) != {"detection", "recognition"}:
            raise ValueError("PIA requires the original detector and recognizer")
        return result

    app.FaceAnalysis = analysis


def main():
    import onnxruntime as ort
    from insightface import app
    from insightface.model_zoo import model_zoo

    sys.path.insert(0, "/helpers")
    import pia_cpu_prefix as prefix
    from pia_cpu_spin_runtime import disable_spinning, monitored_prefix

    disable_spinning(model_zoo, ort)
    identity_modules(app)
    prefix.prefix_embeddings = monitored_prefix(prefix.prefix_embeddings)
    print(
        "[CPU_POLICY] detection + recognition only; same raw embeddings, frames, FP32 and threads",
        flush=True,
    )
    prefix.main()


if __name__ == "__main__":
    main()
