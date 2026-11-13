"""Embedded Tk playback; FFPyPlayer owns the audio/video clock and decoder."""

import logging
import math
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger(__name__)

def seconds(value):
    value = float(value or 0)
    return value if math.isfinite(value) and value >= 0 else 0


def clock(value):
    value = seconds(value)
    return f"{int(value // 60):02d}:{value % 60:05.2f}"


class VideoPlayer(ttk.Frame):
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
        self.close()
        try:
            factory = self.factory
            if factory is None:
                from ffpyplayer.player import MediaPlayer

                factory = MediaPlayer
            self.player = factory(str(path), callback=self.event, ff_opts={"paused": True,
                "out_fmt": "rgb24", "af": "aresample=async=1:first_pts=0,aformat=sample_fmts=s16:channel_layouts=stereo"}, loglevel="error")
            self.player.set_mute(self.muted.get())
            self.refresh = True
            self.tick()
        except Exception as error:
            logger.exception("Video could not be opened")
            self.fail(error)

    def toggle(self):
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
        try:
            self.check_source()
            self.player.set_pause(False)
            self.player.seek(seconds(point), relative=False, accurate=True)
            self.refresh, self.ended = True, False
        except Exception as error:
            logger.exception("Video seek failed")
            self.fail(error)

    def scrub(self, _event=None):
        self.dragging = False
        if self.player:
            self.seek(self.position.get())

    def mute(self):
        if self.player:
            self.player.set_mute(self.muted.get())

    def tick(self):
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
        self.close()
        self.canvas.itemconfigure("hint", text=f"Playback unavailable: {error}", width=400)

    def event(self, selector, value):
        if selector.endswith(":error"):
            self.error = str(value or selector)

    def close(self):
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
