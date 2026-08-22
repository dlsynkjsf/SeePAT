from seepat.preprocessing.events import classify_event_label, parse_segments


def test_segment_parsing_and_event_labels() -> None:
    segments = parse_segments("[[1.0, 2.0]]")

    assert classify_event_label(0.1, 0.2, segments, 0.04) == "real"
    assert classify_event_label(1.2, 1.3, segments, 0.04) == "fake"
    assert classify_event_label(0.99, 1.05, segments, 0.04) == "ambiguous"
