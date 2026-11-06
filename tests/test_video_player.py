import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

from scripts.video_player import VideoPlayer, clock


class Media:
    def __init__(self, *args, **kwargs):
        self.paused, self.closed, self.point, self.muted = True, False, 0, False
        self.seeks = []
        self.options = kwargs

    def set_mute(self, value):
        self.muted = value

    def set_pause(self, value):
        self.paused = value

    def get_pause(self):
        return self.paused

    def get_metadata(self):
        return {'duration': 10}

    def get_pts(self):
        raise AssertionError('Do not query the native clock before a decoded frame exists')

    def get_frame(self, **options):
        assert not self.closed
        return None, 'paused' if self.paused else .02

    def seek(self, point, **options):
        self.seeks.append((point, options))
        self.point = point

    def close_player(self):
        self.closed = True


def test_controls_seek_guard_source_switch_and_close_release_player(native_window):
    root = native_window
    player = VideoPlayer(root, factory=Media)
    try:
        player.load('synthetic.mp4')
        first = player.player
        assert 'channel_layouts=stereo' in first.options['ff_opts']['af']
        assert player.play.instate(['!disabled']) and first.paused
        player.toggle()
        assert not first.paused
        player.seek(3.75)
        assert first.seeks == [(3.75, {'relative':False, 'accurate':True})]
        player.muted.set(True)
        player.mute()
        assert first.muted
        player.load('another.mp4')
        assert first.closed and player.player is not first and player.frame is None
        current = player.player
        player.position.set(7)
        current.options['callback']('audio:error', 'Synthetic audio failure')
        player.tick()
        assert current.closed and player.player is None
        assert player.position.get() == 0
        assert 'Synthetic audio failure' in player.canvas.itemcget('hint', 'text')
        player.load('another.mp4')
        current = player.player
        def changed():
            raise ValueError('The source video changed')
        player.check_source = changed
        player.toggle()
        assert current.closed and player.player is None and player.play.instate(['disabled'])
        assert 'source video changed' in player.canvas.itemcget('hint', 'text')
        player.factory = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('Unsupported video'))
        player.load('broken.mp4')
        assert 'Unsupported video' in player.canvas.itemcget('hint', 'text')
        assert clock(float('nan')) == '00:00.00' and clock(67.25) == '01:07.25'
    finally:
        player.close()
        root.destroy()


def test_generated_audio_video_renders_pauses_and_seeks_inside_tk(tmp_path):
    path = tmp_path / 'synthetic-av.mov'
    subprocess.run([shutil.which('ffmpeg'), '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'testsrc=size=160x90:rate=25', '-f', 'lavfi', '-i',
        'sine=frequency=440:sample_rate=16000', '-t', '3', '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-c:a', 'pcm_s16le', '-ac', '1', str(path)], check=True, timeout=15)
    result = subprocess.run([sys.executable, __file__, str(path)],
        env={**os.environ, 'SDL_AUDIODRIVER':'dummy', 'SDL_AUDIO_CHANNELS':'2',
             'PYTHONPATH':str(Path(__file__).resolve().parents[1])}, capture_output=True, text=True, check=True, timeout=20)
    assert 'Audio thread error' not in result.stderr and 'Channel layout change' not in result.stderr


def test_playback_controls_remain_visible_when_preview_is_short(native_window):
    root = native_window
    root.geometry('420x240')
    player = VideoPlayer(root, factory=Media)
    player.pack(fill='both', expand=True)
    try:
        root.deiconify()
        root.update()
        assert player.play.winfo_ismapped() and player.position.winfo_ismapped()
        controls = player.play.master
        assert controls.winfo_height() >= player.play.winfo_reqheight()
        assert controls.winfo_y() + controls.winfo_height() <= player.winfo_height()
        assert player.canvas.winfo_height() < 240
    finally:
        player.close()


def check_playback(path):
    root = tk.Tk()
    root.withdraw()
    widget = VideoPlayer(root)
    widget.pack()
    def until(condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            root.update()
            time.sleep(.02)
        assert condition(), widget.canvas.itemcget('hint', 'text')
    try:
        widget.load(path)
        until(lambda: widget.frame is not None)
        assert widget.player.get_pause() and widget.photo is not None
        assert widget.player.get_metadata()['duration'] >= 2.9
        widget.toggle()
        until(lambda: widget.point >= .25)
        widget.toggle()
        assert widget.player.get_pause()
        widget.seek(1.5)
        until(lambda: widget.point >= 1.5 and widget.frame is not None)
        assert widget.point < 1.8
        widget.toggle()
        assert widget.player.get_pause()
        widget.close()
        assert widget.player is None and widget.timer is None
    finally:
        widget.close()
        root.destroy()


if __name__ == '__main__':
    check_playback(Path(sys.argv[1]))
