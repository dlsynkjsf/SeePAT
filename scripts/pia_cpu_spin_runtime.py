"""Disable ONNX idle-worker spinning; preserve models, threads and prefix inputs."""

from __future__ import annotations

import sys


def disable_spinning(router, ort):
    original = router.PickableInferenceSession

    def session(path, **kwargs):
        if kwargs.pop("sess_options", None) is not None:
            raise ValueError("Unexpected explicit ONNX session policy")
        options = ort.SessionOptions()
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return original(path, sess_options=options, **kwargs)

    router.PickableInferenceSession = session


def monitored_prefix(original):
    def run(capture, embed, maximum):
        done = 0

        def tracked(frame):
            nonlocal done
            value = embed(frame)
            done += 1
            if done % 25 == 0 or done == maximum:
                print(f"[CPU_PROGRESS] ArcFace {done}/{maximum} frames", flush=True)
            return value

        return original(capture, tracked, maximum)

    return run


def main():
    import onnxruntime as ort
    from insightface.model_zoo import model_zoo

    disable_spinning(model_zoo, ort)
    sys.path.insert(0, "/helpers")
    import pia_cpu_prefix as prefix

    prefix.prefix_embeddings = monitored_prefix(prefix.prefix_embeddings)
    print(
        "[CPU_POLICY] ONNX idle spinning off; thread counts and model settings unchanged",
        flush=True,
    )
    prefix.main()


if __name__ == "__main__":
    main()
