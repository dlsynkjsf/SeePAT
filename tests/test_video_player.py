import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path

from scripts.video_player import VideoPlayer, clock, sdl_mixer


class Media:
    """Fake FFPyPlayer MediaPlayer."""

    def __init__(self, *args, **kwargs):
        self.paused, self.closed, self.point, self.volume = True, False, 0, 1.0
        self.seeks = []
        self.options = kwargs

    def set_volume(self, value):
        self.volume = value

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


def test_controls_seek_guard_source_switch_and_close_release_player(native_window, monkeypatch):
    monkeypatch.setattr('scripts.video_player.sdl_mixer', lambda: None)
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
        assert player.player is first and first.volume == 0 and len(first.seeks) == 1  # same player, no restart
        player.load('another.mp4')
        assert first.closed and player.player is not first and player.frame is None
        current = player.player
        player.tick()
        assert current.volume == 0  # a new video keeps the Mute choice
        player.muted.set(False)
        player.mute()
        assert current.volume == 1
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


def mixer_volumes():
    """Volume of each playing SDL_mixer channel; empty while the mixer is closed."""
    mixer = sdl_mixer()
    if not mixer.Mix_QuerySpec(None, None, None):
        return {}
    return {ch: mixer.Mix_Volume(ch, -1) for ch in range(mixer.Mix_AllocateChannels(-1)) if mixer.Mix_Playing(ch)}


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
        until(lambda: mixer_volumes())
        for muted, volume in ((True, 0), (False, 128)):  # Mute/unmute while playing
            point, seen = widget.point, []
            widget.muted.set(muted)
            widget.mute()
            assert set(mixer_volumes().values()) == {volume}
            until(lambda seen=seen, point=point: seen.append(widget.point) or widget.point >= point + .2)
            assert min(seen) >= point - .15  # continues, no jump back to 0:00
        widget.muted.set(True)  # Mute ticked before a new video starts
        widget.load(path)
        until(lambda: widget.frame is not None)
        widget.toggle()
        until(lambda: set(mixer_volumes().values()) == {0})
        widget.toggle()
        assert widget.player.get_pause()
        widget.seek(1.5)
        until(lambda: widget.point >= 1.5 and widget.frame is not None)
        assert widget.point < 1.8
        # After a seek, playback stays real-time with audio on the same clock (was ~0.5x, audio from 0:00).
        start, wall = widget.point, time.monotonic()
        until(lambda: time.monotonic() - wall >= .8)
        rate = (widget.point - start) / (time.monotonic() - wall)
        assert .75 <= rate <= 1.3, f'playback rate after seek {rate:.2f}x'
        assert abs(widget.player.get_pts() - widget.point) < .25, 'audio clock left the video after seek'
        widget.toggle()
        assert widget.player.get_pause()
        widget.close()
        assert widget.player is None and widget.timer is None
    finally:
        widget.close()
        root.destroy()


if __name__ == '__main__':
    check_playback(Path(sys.argv[1]))
