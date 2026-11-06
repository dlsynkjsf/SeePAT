import csv
import json
from pathlib import Path

import pytest

from scripts.report_wording import video_summary
from seepat.artifacts import file_sha256


def evidence(tmp_path, *, verdict='manipulated', kind='bilabial', start=1.43, word='complicated', offset=0):
    scores = [.2, .8] if verdict == 'manipulated' else [.1, .2]
    video = {'video_id': 'local-demo', 'verdict': verdict, 'clips_scored': 2,
             'video_score': max(scores), 'threshold': .5,
             'highlight': {'event_id': 'clip-1', 'kind': kind, 'probability': max(scores),
                 'timestamp': {'status': 'verified', 'start_s': start, 'end_s': start + .03,
                     'last_sample_s': start + .4, 'kind': 'video_phoneme_interval' if kind == 'bilabial'
                     else 'sampled_context_source_frame'}}}
    paths = ['result/event_predictions.csv', 'prepared/events_raw.csv',
             'prepared/videos/local-demo/alignment.json']
    for name in paths:
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
    rows = [{'event_id': f'clip-{i}', 'video_id': 'local-demo', 'manipulated_probability': score,
             'kind': kind, 'phoneme': 'p'} for i, score in enumerate(scores)]
    with (tmp_path / paths[0]).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    event = {'event_id': 'clip-1', 'video_id': 'local-demo', 'video_phone_start_s': start,
             'video_phone_end_s': start + .03, 'phone_start_s': start - offset, 'phone_end_s': start - offset + .03}
    with (tmp_path / paths[1]).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=event)
        writer.writeheader()
        writer.writerow(event)
    (tmp_path / paths[2]).write_text(json.dumps({'tiers': {'words': {'entries': [
        [start - offset - .1, start - offset + .1, word]]}}}))
    return video, [{'path': str(tmp_path / name), 'sha256': file_sha256(tmp_path / name)} for name in paths]


@pytest.mark.parametrize('kind,word,offset,expected', [
    ('bilabial', 'complicated', 0, 'near "complicated" in the transcript'),
    ('bilabial', 'people', .7, 'near "people" in the transcript'),
    ('bilabial', '<eps>', 0, 'during the "p" sound'),
    ('context', 'complicated', 0, 'in a short mouth-movement clip'),
])
def test_summary_uses_video_specific_sealed_facts_without_claiming_mismatch(tmp_path, kind, word, offset, expected):
    video, bindings = evidence(tmp_path, kind=kind, word=word, offset=offset)
    before = {row['path']: file_sha256(Path(row['path'])) for row in bindings}
    text = video_summary(video, tmp_path, bindings)
    assert 'manipulated (fake)' in text and '1 of the 2 short clips' in text
    assert expected in text and '1.43 seconds' in text
    assert f"Around 1.43 seconds, {expected}, this clip triggered SeePAT's strongest warning." in text
    assert all(term not in text for term in ('cutoff', 'probability', 'phoneme', 'does not align'))
    assert text.endswith("this clip triggered SeePAT's strongest warning.")
    assert before == {row['path']: file_sha256(Path(row['path'])) for row in bindings}


def test_authentic_unavailable_and_old_reports_do_not_invent_evidence(tmp_path):
    video, bindings = evidence(tmp_path, verdict='authentic')
    text = video_summary(video, tmp_path, bindings)
    assert 'authentic (real)' in text and 'None of the 2' in text
    assert 'seconds' not in text and 'not every moment' in text
    video.update(verdict='not_evaluated', clips_scored=0, video_score=None, highlight=None)
    assert 'could not assess' in video_summary(video, tmp_path, [])
    video, _ = evidence(tmp_path)
    video['highlight']['timestamp'] = {'status': 'unavailable'}
    text = video_summary(video, tmp_path, [])
    assert '2 short clips' in text and 'unavailable' in text and 'flagged 1' not in text
    assert 'complicated' not in text and '1.43' not in text


def test_tampered_or_contradictory_clip_evidence_is_rejected(tmp_path):
    video, bindings = evidence(tmp_path)
    video['clips_scored'] = 3
    with pytest.raises(ValueError, match='predictions differ'):
        video_summary(video, tmp_path, bindings)
    video['clips_scored'] = 2
    video['highlight']['event_id'] = 'wrong-clip'
    with pytest.raises(ValueError, match='review clip differs'):
        video_summary(video, tmp_path, bindings)
    (tmp_path / 'result/event_predictions.csv').write_text('changed')
    with pytest.raises(ValueError, match='evidence changed'):
        video_summary(video, tmp_path, bindings)
