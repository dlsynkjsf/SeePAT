from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import demo_ui as ui
from seepat.artifacts import atomic_write_json, file_sha256


@pytest.fixture
def store(tmp_path):
    (tmp_path / 'freeze.json').write_text('{}')
    config = {'version': ui.VERSION, 'freeze': 'freeze.json', 'device': 'cpu',
              'output_dir': 'outputs/ui', 'max_video_mb': 1, 'examples': []}
    calls = []
    def launch(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(pid=123)
    value = ui.Analyses(tmp_path, config, launch=launch, alive=lambda pid: pid == 123)
    value.calls = calls
    return value


def upload(store, name='talk.mp4'):
    source = store.root / name
    source.write_bytes(b'0123456789')
    return store.create(source, 'personal-recording')


def complete(store, identity, verdict='manipulated'):
    folder = store.folder(identity)
    record = ui.read(folder / 'record.json')
    score = .8 if verdict == 'manipulated' else .2
    video = {'video_id': 'local-' + record['source_sha256'][:24], 'verdict': verdict,
             'threshold': .5, 'video_score': score, 'clips_scored': 2, 'reason': '',
             'highlight': {'kind': 'bilabial', 'probability': score, 'timestamp': {
                 'status': 'verified', 'kind': 'video_phoneme_interval', 'start_s': 1.5, 'end_s': 1.6}}}
    if verdict == 'not_evaluated':
        video.update(video_score=None, clips_scored=0, highlight=None, reason='no_eligible_events')
    output = folder / 'run'
    explanation = output / 'explanation'
    explanation.mkdir(parents=True, exist_ok=True)
    atomic_write_json(explanation / 'report.json', {'scope': 'local_unlabelled_inference',
        'videos': [video], 'network_requests': 0})
    for name in ('report.md', 'technical_report.md'):
        (explanation / name).write_text('Synthetic saved report')
    atomic_write_json(explanation / 'summary.json', {'status': 'complete',
        'output_sha256': {name: file_sha256(explanation / name) for name in ui.DOWNLOADS}})
    bindings = {}
    for name, directory in (('preparation', 'prepared'), ('inference', 'result'), ('explanation', 'explanation')):
        path = output / directory / 'summary.json'
        path.parent.mkdir(exist_ok=True)
        if name != 'explanation':
            atomic_write_json(path, {'status': 'complete'})
        bindings[name] = {'path': str(path), 'sha256': file_sha256(path)}
    atomic_write_json(output / 'summary.json', {'status': 'complete', 'stage_summaries': bindings})
    atomic_write_json(folder / 'worker.json', {'status': 'complete'})
    return output


def test_selection_builds_label_free_request_without_copying_the_original_video(store):
    item = upload(store)
    folder = store.folder(item['id'])
    request = ui.read(folder / 'inputs/request.json')
    assert set(request) == {'version', 'dataset_source', 'split', 'videos'}
    assert request['videos'][0]['origin'] == 'personal-recording'
    source = Path(request['videos'][0]['source']['path'])
    assert source == store.root / 'talk.mp4' and source.read_bytes() == b'0123456789'
    assert not list((folder / 'inputs').glob('*.mp4'))
    assert file_sha256(source) == request['videos'][0]['source']['sha256']
    job = ui.read(folder / 'workflow.json')['local_inference'][0]
    assert job['device'] == 'cpu' and job['batch_size'] == 1
    assert not Path(job['output_dir']).is_relative_to(Path(job['raw_inputs']).parent)
    assert store.calls == []


@pytest.mark.parametrize('name,size,origin', [('notes.txt', 10, 'local-demo'),
    ('Deepfake-Eval-2024.mp4', 10, 'local-demo'), ('test.mp4', 2 * 1024**2, 'local-demo'),
    ('test.mp4', 10, 'external-test')])
def test_invalid_files_start_no_work(store, name, size, origin):
    source = store.root / name
    source.write_bytes(b'x' * size)
    with pytest.raises(ValueError):
        store.create(source, origin)
    assert not store.state()['jobs'] and not store.calls


def test_ui_cannot_take_over_existing_scientific_outputs(store):
    folder = store.root / 'outputs/existing-study'
    folder.mkdir()
    evidence = folder / 'record.json'
    evidence.write_text('preserve scientific evidence')
    with pytest.raises(ValueError, match='Preserve'):
        ui.Analyses(store.root, {**store.config, 'output_dir': 'outputs/existing-study'})
    assert evidence.read_text() == 'preserve scientific evidence'
    assert not (folder / 'ui_workspace.json').exists()


def test_launch_is_explicit_single_worker_and_preserves_custom_configuration(store):
    one, two = upload(store), upload(store)
    item = store.start(one['id'])
    assert item['status'] == 'queued' and store.active() == one['id']
    args, options = store.calls[0]
    assert args[0][-2:] == ['--worker', one['id']] and '--config' in args[0]
    assert options['cwd'] == store.root and options['stdin'] == ui.subprocess.DEVNULL
    with pytest.raises(ValueError, match='already running'):
        store.start(two['id'])
    store.alive = lambda pid: False
    assert store.snapshot(one['id'])['status'] == 'interrupted'
    before = (store.folder(one['id']) / 'workflow.json').read_bytes()
    store.start(one['id'])
    assert (store.folder(one['id']) / 'workflow.json').read_bytes() == before


def test_parent_launcher_cannot_overwrite_a_fast_completed_worker(store):
    item = upload(store)
    def fast(*args, **kwargs):
        complete(store, item['id'])
        return SimpleNamespace(pid=123)
    store.launch = fast
    assert store.start(item['id'])['status'] == 'complete'


@pytest.mark.parametrize('verdict', ['authentic', 'manipulated', 'not_evaluated'])
def test_reports_preserve_outcome_and_missing_evidence(store, verdict):
    item = upload(store)
    output = complete(store, item['id'], verdict)
    before = file_sha256(output / 'explanation/report.json')
    value = store.snapshot(item['id'])
    assert value['report']['verdict'] == verdict
    if verdict == 'not_evaluated':
        assert value['report']['video_score'] is None and value['report']['highlight'] is None
    else:
        assert value['report']['video_score'] >= value['report']['threshold'] if verdict == 'manipulated' else value['report']['video_score'] < value['report']['threshold']
    assert file_sha256(output / 'explanation/report.json') == before
    (output / 'explanation/report.json').write_text('{}')
    assert store.snapshot(item['id'])['status'] == 'failed'


def test_partial_stages_require_preservation_and_worker_errors_are_not_verdicts(store):
    item = upload(store)
    folder = store.folder(item['id'])
    (folder / 'run/prepared').mkdir(parents=True)
    atomic_write_json(folder / 'worker.json', {'status': 'interrupted'})
    assert not store.snapshot(item['id'])['can_resume']
    with pytest.raises(ValueError, match='Preserve'):
        store.start(item['id'])
    fresh = upload(store)
    def failed(_):
        raise RuntimeError('Synthetic preprocessing failure')
    assert ui.run_worker(store, fresh['id'], failed) == 1
    assert store.snapshot(fresh['id'])['status'] == 'failed'
    assert 'report' not in store.snapshot(fresh['id'])


class FakeVideo:
    def __init__(self, parent, check_source):
        from tkinter import ttk
        self.widget = ttk.Frame(parent)
        self.check_source, self.path, self.point = check_source, None, None

    def grid(self, **options):
        self.widget.grid(**options)

    def load(self, path):
        self.path, self.point = path, None

    def close(self):
        self.path = self.point = None

    def seek(self, point):
        self.check_source()
        self.point = point


def test_native_selection_progress_explanation_toggle_and_review_keep_saved_evidence(store, monkeypatch, native_window):
    root = native_window
    try:
        app = ui.Desktop(root, store, player_factory=FakeVideo)
        source = store.root / 'speaking.mp4'
        source.write_bytes(b'0123456789')
        monkeypatch.setattr(app.dialog, 'askopenfilename', lambda **kwargs: str(source))
        app.select()
        assert app.analyze.instate(['!disabled']) and not store.calls
        assert app.player.path == source and app.demo_controls.winfo_manager() == ''
        app.launch(False)
        assert app.current['status'] == 'queued' and app.analyze.instate(['disabled'])
        identity = app.current['id']
        complete(store, identity, 'authentic')
        app.poll()
        assert app.verdict['text'] == 'Authentic (real)'
        assert 'authentic (real)' in app.trace.get('1.0', 'end').lower()
        root.geometry('940x600')
        root.deiconify()
        root.update()
        assert app.review.winfo_ismapped() and app.reports.winfo_ismapped()
        footer = app.review.master
        assert footer.winfo_y() + footer.winfo_height() <= footer.master.winfo_height()
        app.detail_view.set('Evidence details')
        app.render_text()
        assert 'highest score' in app.trace.get('1.0', 'end')
        assert 'Technical details:' not in app.trace.get('1.0', 'end')
        app.review_clip()
        assert app.player.point == 1.5
        source.write_bytes(b'changed original')
        with pytest.raises(ValueError, match='video changed'):
            app.review_clip()
        source.write_bytes(b'0123456789')
        complete(store, identity, 'not_evaluated')
        app.load_job(identity)
        app.buttons()
        assert app.verdict['text'] == 'Not evaluated' and app.review.winfo_manager() == ''
    finally:
        app.close()


def test_own_and_example_modes_clear_stale_video_results_and_ignore_background_job_updates(store, native_window):
    own = upload(store, 'my-speaking.mp4')
    example = upload(store, 'demonstration.mp4')
    example_output = complete(store, example['id'], 'manipulated')
    store.config['examples'] = [{'id':'example', 'title':'Saved example',
        'request':str(store.folder(example['id']) / 'inputs/request.json'), 'output_dir':str(example_output)}]
    store.start(own['id'])
    root = native_window
    try:
        app = ui.Desktop(root, store, player_factory=FakeVideo)
        assert app.current['id'] == own['id']
        app.mode.set('Saved examples')
        app.switch_mode()
        assert app.current is None and app.player.path is None and app.selected is None
        assert app.own_controls.winfo_manager() == ''
        assert app.detail_view.get() == 'Video summary'
        assert list(app.example_items) == ['Saved example']
        app.examples.set('Saved example')
        app.load_example()
        assert app.current.get('saved') and app.verdict['text'] == 'Manipulated (fake)'
        assert app.player.path.name == 'demonstration.mp4'
        complete(store, own['id'], 'authentic')
        app.poll()
        app.load_job(own['id'])
        assert app.current['id'] == 'example' and app.player.path.name == 'demonstration.mp4'
        app.mode.set('Your video')
        app.switch_mode()
        assert app.current is None and app.player.path is None and app.demo_controls.winfo_manager() == ''
        assert 'manipulated' not in app.trace.get('1.0', 'end').lower()
        app.recent.set(next(name for name, identity in app.job_items.items() if identity == own['id']))
        app.load_saved()
        assert app.verdict['text'] == 'Authentic (real)' and not app.current.get('saved')
        assert app.player.path.name == 'my-speaking.mp4'
    finally:
        app.close()
