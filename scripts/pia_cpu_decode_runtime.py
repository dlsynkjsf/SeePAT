"""Materialize only the identity frames consumed by the unchanged PIA builder."""

def selected_embeddings(capture, embed, indices):
    values, decoded = {}, 0
    try:
        if indices != sorted(set(indices)) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Identity indices must be ordered unique nonnegative integers")
        selected = set(indices)
        while indices and decoded <= indices[-1] and capture.isOpened():
            if not capture.grab():
                break
            if decoded in selected:
                ok, frame = capture.retrieve()
                if not ok:
                    raise RuntimeError("Selected identity frame could not be retrieved")
                values[decoded] = embed(frame)
                done = len(values)
                if done % 5 == 0 or done == len(indices):
                    print(f"[CPU_PROGRESS] ArcFace consumed entries {done}/{len(indices)}", flush=True)
            decoded += 1
    finally:
        capture.release()
    return values, decoded


if __name__ == "__main__":
    import pia_cpu_consumed_runtime as runtime

    runtime.selected_embeddings = selected_embeddings
    runtime.main()
