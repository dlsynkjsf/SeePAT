"""Video preview widget for the SeePAT desktop window (Tk + FFPyPlayer)."""

import ctypes
import functools
import hashlib
import logging
import math
import subprocess
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from seepat.preprocessing.media import find_media_binary

logger = logging.getLogger(__name__)
# Only for originals that cannot get a preview copy: mono PCM with an unknown channel layout stops
# FFPyPlayer's audio thread, but this filter also breaks audio timestamps after a seek.
STEREO_FILTER = "aresample=async=1:first_pts=0,aformat=sample_fmts=s16:channel_layouts=stereo"


def preview_copy(path):
    """All-keyframe, stereo-PCM copy of a video for the preview player only.

    FFPyPlayer's accurate seek skips only video frames before the target, so audio restarts at
    the nearest video keyframe (0 s in the AV++ clips) and the video crawls at half speed while
    waiting for it. With every frame a keyframe and a plain stereo track, both streams land on
    the target. The analysed original is never changed.
    """
    path = Path(path)
    stat = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]
    # ponytail: no cache eviction; temp folder grows ~1 MB per demo clip, ~45 MB per minute of 1080p.
    target = Path(tempfile.gettempdir()) / "seepat-preview" / f"{key}.mov"
    if not target.is_file():
        target.parent.mkdir(exist_ok=True)
        partial = target.with_name(f"{key}.partial.mov")
        # ponytail: runs on the Tk thread (~0.1 s per demo clip, ~3 s per minute of 1080p).
        subprocess.run([str(find_media_binary("ffmpeg")), "-v", "error", "-y", "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0?", "-vf", "scale=-2:'min(480,trunc(ih/2)*2)'",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "1", "-pix_fmt", "yuv420p",
            "-c:a", "pcm_s16le", "-ac", "2", str(partial)], check=True, capture_output=True, timeout=600)
        partial.replace(target)
    return target


@functools.cache
def sdl_mixer():
    """SDL_mixer library used by FFPyPlayer for audio, or None if this build has none."""
    import ffpyplayer

    dll = next((Path(p) / "SDL2_mixer.dll" for p in getattr(ffpyplayer, "dep_bins", [])
                if (Path(p) / "SDL2_mixer.dll").is_file()), None)
    return ctypes.CDLL(str(dll)) if dll else None


def seconds(value):
    """Non-negative finite seconds; anything else becomes 0."""
    value = float(value or 0)
    return value if math.isfinite(value) and value >= 0 else 0


def clock(value):
    """Seconds as mm:ss.ss."""
    value = seconds(value)
    return f"{int(value // 60):02d}:{value % 60:05.2f}"


class VideoPlayer(ttk.Frame):
    """Preview canvas with Play/Pause, seek bar, time label and Mute."""

    def __init__(self, parent, check_source=lambda: None, factory=None):
        super().__init__(parent)
        self.check_source, self.factory = check_source, factory
        self.player = self.timer = self.frame = self.photo = None
        self.point = 0
        self.error = None
        self.dragging = self.ended = False
        self.canvas = tk.Canvas(self, background="#172126", height=280, highlightthickness=0)
        self.canvas.create_text(1, 1, text="Choose a video to preview", fill="white", tags="hint")
        self.canvas.bind("<Configure>", lambda event: self.draw())
        controls = ttk.Frame(self)
        self.play = ttk.Button(controls, text="Play", command=self.toggle, state="disabled")
        self.play.pack(side="left")
        self.position = ttk.Scale(controls, from_=0, to=1, state="disabled")
        self.position.pack(side="left", fill="x", expand=True, padx=8)
        self.position.bind("<ButtonPress-1>", lambda event: setattr(self, "dragging", True))
        self.position.bind("<ButtonRelease-1>", self.scrub)
        self.position.bind("<KeyRelease>", self.scrub)
        self.time = ttk.Label(controls, text="00:00.00 / 00:00.00")
        self.time.pack(side="left")
        self.muted = tk.BooleanVar()
        ttk.Checkbutton(self, text="Mute", variable=self.muted, command=self.mute).pack(side="bottom", anchor="w", pady=4)
        controls.pack(side="bottom", fill="x", pady=(8, 0))
        self.canvas.pack(fill="both", expand=True)

    def load(self, path):
        """Open a video paused on its first frame."""
        self.close()
        self.canvas.itemconfigure("hint", text="Preparing preview...")
        self.update_idletasks()
        try:
            source, options = preview_copy(path), {}
        except (OSError, subprocess.SubprocessError) as error:
            logger.warning("Preview copy unavailable, seeking may drift out of sync: %s", error)
            source, options = path, {"af": STEREO_FILTER}
        try:
            factory = self.factory
            if factory is None:
                from ffpyplayer.player import MediaPlayer

                factory = MediaPlayer
            self.player = factory(str(source), callback=self.event,
                ff_opts={"paused": True, "out_fmt": "rgb24", **options}, loglevel="error")
            self.refresh = True
            self.tick()
        except Exception as error:
            logger.exception("Video could not be opened")
            self.fail(error)

    def toggle(self):
        """Play/Pause button; restarts from 0 after the end."""
        try:
            self.check_source()
            if self.ended:
                self.seek(0)
            else:
                self.player.set_pause(not self.player.get_pause())
        except Exception as error:
            logger.exception("Video playback could not start")
            self.fail(error)

    def seek(self, point):
        """Jump to a time in seconds and play from there."""
        try:
            self.check_source()
            self.player.set_pause(False)
            self.player.seek(seconds(point), relative=False, accurate=True)
            self.refresh, self.ended = True, False
        except Exception as error:
            logger.exception("Video seek failed")
            self.fail(error)

    def scrub(self, _event=None):
        """Seek bar released: jump to its position."""
        self.dragging = False
        if self.player:
            self.seek(self.position.get())

    def mute(self):
        """Mute checkbox: set the volume in place, so playback keeps its position."""
        if not self.player:
            return
        mixer = sdl_mixer()
        # SDL_mixer volume calls are only valid while its audio device is open.
        if mixer is None or mixer.Mix_QuerySpec(None, None, None):
            self.player.set_volume(0.0 if self.muted.get() else 1.0)

    def tick(self):
        """Playback loop: draw the next frame, update controls, reapply Mute."""
        self.timer = None
        if not self.player:
            return
        try:
            if self.error:
                raise RuntimeError(self.error)
            frame, delay = self.player.get_frame(force_refresh=self.refresh)
            duration = seconds(self.player.get_metadata().get("duration"))
            # The decoder clock is safe to read only after it has produced a frame.
            self.point = seconds(frame[1]) if frame else self.point
            self.mute()
            self.position.configure(to=duration or 1, state="normal" if duration else "disabled")
            if not self.dragging:
                self.position.set(self.point)
            self.time.configure(text=f"{clock(self.point)} / {clock(duration)}")
            if frame:
                from PIL import Image

                image, _pts = frame
                self.frame = Image.frombytes("RGB", image.get_size(), bytes(image.to_bytearray()[0]))
                self.refresh = False
                self.draw()
            if delay == "eof":
                if self.frame is None:
                    raise ValueError("No playable video stream")
                self.ended = True
                self.player.set_pause(True)
            self.play.configure(text="Play" if self.player.get_pause() else "Pause", state="normal")
            self.timer = self.after(max(10, min(100, int(delay * 1000))) if isinstance(delay, float) else 80, self.tick)
        except Exception as error:
            logger.exception("Video playback failed")
            self.fail(error)

    def draw(self):
        """Fit the current frame to the canvas."""
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.canvas.coords("hint", width / 2, height / 2)
        if self.frame:
            from PIL import Image, ImageTk

            image = self.frame.copy()
            scale = min(width / image.width, height / image.height)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.BILINEAR)
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.delete("video")
            self.canvas.create_image(width / 2, height / 2, image=self.photo, tags="video")

    def fail(self, error):
        """Close playback and show the error on the canvas."""
        self.close()
        self.canvas.itemconfigure("hint", text=f"Playback unavailable: {error}", width=400)

    def event(self, selector, value):
        """FFPyPlayer callback: remember decoder errors for the next tick."""
        if selector.endswith(":error"):
            self.error = str(value or selector)

    def close(self):
        """Release the player and reset the controls."""
        if self.timer:
            self.after_cancel(self.timer)
            self.timer = None
        player, self.player = self.player, None
        if player:
            player.close_player()
        self.frame = self.photo = None
        self.ended = False
        self.point = 0
        self.error = None
        self.canvas.delete("video")
        self.canvas.itemconfigure("hint", text="Choose a video to preview")
        self.play.configure(text="Play", state="disabled")
        self.position.configure(to=1, state="normal")
        self.position.set(0)
        self.position.configure(state="disabled")
        self.time.configure(text="00:00.00 / 00:00.00")
