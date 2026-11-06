"""Local single-video interface for the unchanged frozen SeePAT workflow."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.assessment_runtime import archive_lock, process_alive
from scripts.report_wording import video_summary
from seepat.artifacts import atomic_write_json, file_sha256
from seepat.inference.xai_paragraphs import human_summary

VERSION = "seepat-local-ui-v1"
EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
DOWNLOADS = {"report.md", "report.json", "technical_report.md"}
STAGES = {"preparation": "prepared", "inference": "result", "explanation": "explanation"}
REPORT_VIEWS = {"Video summary": "plain", "Evidence details": "technical"}
INK, NAVY, ACCENT = "#203247", "#17324d", "#137d88"
OUTCOME_COLORS = {"authentic": "#22634b", "manipulated": "#98442e", "not_evaluated": "#667485"}
logger = logging.getLogger(__name__)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class Analyses:
    def __init__(self, root=ROOT, config=None, *, launch=subprocess.Popen, alive=process_alive):
        self.root = root.resolve()
        self.config = config or read(self.root / "configs/demo_ui.json")
        if self.config["version"] != VERSION or self.config["device"] not in {"cpu", "cuda"}:
            raise ValueError("Unsupported UI configuration")
        self.directory = (self.root / self.config["output_dir"]).resolve()
        if self.directory == self.root / "outputs" or not self.directory.is_relative_to(self.root / "outputs"):
            raise ValueError("UI outputs must stay under the project's outputs directory")
        self.freeze = (self.root / self.config["freeze"]).resolve()
        self.limit = int(self.config.get("max_video_mb", self.config.get("max_upload_mb", 200))) * 1024**2
        if not 1 <= self.limit <= 1024**3:
            raise ValueError("Unsupported video size limit")
        self.directory.mkdir(parents=True, exist_ok=True)
        marker = self.directory / "ui_workspace.json"
        if marker.exists():
            if read(marker) != {"version": VERSION}:
                raise ValueError("The output folder belongs to another application")
        elif any(self.directory.iterdir()):
            raise ValueError("Preserve the existing outputs; choose a dedicated UI folder")
        else:
            atomic_write_json(marker, {"version": VERSION})
        self.launch, self.alive = launch, alive
        self.lock = threading.Lock()
        self.source_hashes = {}

    def folder(self, identity):
        if not re.fullmatch(r"[a-f0-9]{32}", identity):
            raise FileNotFoundError("Unknown analysis")
        path = self.directory / identity
        if path.is_symlink() or not path.resolve().is_relative_to(self.directory):
            raise ValueError("Invalid analysis path")
        return path

    def create(self, source, origin):
        source = Path(source).resolve(strict=True)
        if (not source.is_file() or not 0 < source.stat().st_size <= self.limit
                or source.suffix.lower() not in EXTENSIONS
                or origin not in {"personal-recording", "local-demo"}
                or "deepfakeeval" in re.sub(r"[^a-z0-9]", "", source.name.lower())):
            raise ValueError("Choose a supported personal/demo video within the size limit")
        digest = file_sha256(source)
        identity = uuid.uuid4().hex
        folder = self.folder(identity)
        inputs = folder / "inputs"
        inputs.mkdir(parents=True)
        atomic_write_json(inputs / "request.json", {
            "version": "raw-local-inputs-v1", "dataset_source": "local-unlabelled", "split": "local",
            "videos": [{"source": {"path": str(source), "sha256": digest}, "origin": origin}],
        })
        atomic_write_json(folder / "workflow.json", {
            "report": str(folder / "workflow_report.json"),
            "local_inference": [{"name": "demo-" + identity, "freeze": str(self.freeze),
                "raw_inputs": str(inputs / "request.json"), "output_dir": str(folder / "run"),
                "device": self.config["device"], "batch_size": 1}],
        })
        atomic_write_json(folder / "ui_config.json", self.config)
        atomic_write_json(folder / "record.json", {
            "version": VERSION, "id": identity, "filename": source.name, "created_at": time.time(),
            "origin": origin,
            "source": str(source), "source_sha256": digest,
            "freeze_sha256": file_sha256(self.freeze),
            "workflow_sha256": file_sha256(folder / "workflow.json"),
        })
        atomic_write_json(folder / "worker.json", {"status": "ready"})
        return self.snapshot(identity)

    def worker_state(self, folder):
        state = read(folder / "worker.json")
        launched = read(folder / "launch.json") if (folder / "launch.json").exists() else {}
        pid = state.get("pid") or launched.get("pid")
        if state["status"] in {"queued", "running"} and not (pid and self.alive(pid)):
            return {**state, "status": "interrupted"}
        return state

    def active(self):
        path = self.directory / "active.json"
        if path.exists():
            identity = read(path)["id"]
            if self.worker_state(self.folder(identity))["status"] in {"queued", "running"}:
                return identity
        return None

    def start(self, identity):
        with self.lock:
            folder = self.folder(identity)
            record = read(folder / "record.json")
            if self.active():
                raise ValueError("An analysis is already running; let it finish first")
            state = self.snapshot(identity)
            if state["status"] == "complete" or not state["can_resume"]:
                raise ValueError("Preserve this analysis and select a video for a new run")
            if record["freeze_sha256"] != file_sha256(self.freeze):
                raise ValueError("The frozen configuration changed; preserve this analysis")
            if record["workflow_sha256"] != file_sha256(folder / "workflow.json"):
                raise ValueError("The saved analysis configuration changed")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(self.root / "src"), str(self.root)))}
            atomic_write_json(folder / "worker.json", {"status": "queued"})
            try:
                with (folder / "console.log").open("ab") as log:
                    process = self.launch([sys.executable, "-u", "-m", "scripts.demo_ui", "--config",
                        str(folder / "ui_config.json"), "--worker", identity],
                        cwd=self.root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                atomic_write_json(folder / "launch.json", {"pid": process.pid, "started_at": time.time()})
                atomic_write_json(self.directory / "active.json", {"id": identity})
            except OSError as error:
                atomic_write_json(folder / "worker.json", {"status": "failed", "error": str(error)})
                raise
            return self.snapshot(identity)

    def report(self, output, source=None):
        workflow = read(output / "summary.json")
        if workflow["status"] != "complete" or set(workflow["stage_summaries"]) != STAGES.keys():
            raise ValueError("The analysis does not have three completed stages")
        for name, directory in STAGES.items():
            path = output / directory / "summary.json"
            binding = workflow["stage_summaries"][name]
            if Path(binding["path"]).resolve() != path.resolve() or file_sha256(path) != binding["sha256"]:
                raise ValueError("A completed stage binding differs")
        summary_path = output / "explanation/summary.json"
        summary = read(summary_path)
        for name in DOWNLOADS:
            if file_sha256(output / "explanation" / name) != summary["output_sha256"][name]:
                raise ValueError("The saved report changed")
        report = read(output / "explanation/report.json")
        if summary["status"] != "complete" or report["scope"] != "local_unlabelled_inference" or len(report["videos"]) != 1:
            raise ValueError("Unsupported local report")
        video = report["videos"][0]
        if source and video["video_id"] != "local-" + source["sha256"][:24]:
            raise ValueError("The saved report belongs to another video")
        _, _, technical = human_summary(video).partition("\n\n")
        plain = video_summary(video, output, report.get("input_bindings", []))
        technical = technical.removeprefix("Technical details: ")
        return {**video, "video_score": None if video["video_score"] == "" else video["video_score"],
                "threshold": None if video["threshold"] == "" else video["threshold"],
                "plain": plain, "technical": technical, "network_requests": report["network_requests"]}

    def checked_source(self, path, expected):
        stamp = (path.stat().st_size, path.stat().st_mtime_ns)
        if self.source_hashes.get(path, (None, None))[0] != stamp:
            self.source_hashes[path] = (stamp, file_sha256(path))
        if self.source_hashes[path][1] != expected:
            raise ValueError("The video changed after its recorded analysis")
        return path

    def snapshot(self, identity):
        folder = self.folder(identity)
        record, worker = read(folder / "record.json"), self.worker_state(folder)
        progress_path = folder / "workflow_report.progress.json"
        progress = read(progress_path) if progress_path.exists() else {}
        stages = {name: (folder / "run" / name / "summary.json").is_file()
                  for name in STAGES.values()}
        partial = any((folder / "run" / name).exists() and not done for name, done in stages.items())
        result = {"id": identity, "filename": record["filename"], "created_at": record["created_at"],
                  "origin": record.get("origin", "personal-recording"),
                  "status": worker["status"], "saved": False, "stages": stages,
                  "progress": {k: progress.get(k) for k in ("phase", "finished", "total", "elapsed_seconds", "updated_at")},
                  "error": worker.get("error", progress.get("error", "")),
                  "can_resume": worker["status"] in {"ready", "interrupted", "failed"} and not partial}
        if worker["status"] == "complete":
            try:
                result["report"] = self.report(folder / "run", {"sha256": record["source_sha256"]})
            except (OSError, ValueError, KeyError) as error:
                result.update(status="failed", error=str(error), can_resume=False)
        return result

    def example(self, identity):
        row = next((item for item in self.config.get("examples", []) if item["id"] == identity), None)
        if row is None:
            raise FileNotFoundError("Unknown saved example")
        request = read(self.root / row["request"])["videos"][0]["source"]
        source = (self.root / request["path"]).resolve()
        self.checked_source(source, request["sha256"])
        output = (self.root / row["output_dir"]).resolve()
        return {"id": identity, "filename": row["title"], "saved": True, "status": "complete",
                "origin": "local-demo",
                "report": self.report(output, request),
                "stages": dict.fromkeys(STAGES.values(), True)}, source, output

    def state(self):
        jobs = []
        for folder in self.directory.iterdir():
            if folder.is_dir() and (folder / "record.json").is_file():
                try:
                    record = read(folder / "record.json")
                    jobs.append({k: record[k] for k in ("id", "filename", "created_at")})
                except (OSError, ValueError, KeyError):
                    continue
        return {"jobs": sorted(jobs, key=lambda r: r["created_at"], reverse=True)[:10],
                "active": self.active(), "max_file_bytes": self.limit,
                "examples": [{"id": r["id"], "title": r["title"]} for r in self.config.get("examples", [])
                             if (self.root / r["request"]).is_file()
                             and (self.root / r["output_dir"] / "summary.json").is_file()]}


def run_worker(store, identity, runner=None):
    folder = store.folder(identity)
    with archive_lock(store.directory / "analysis"):
        atomic_write_json(folder / "worker.json", {"status": "running", "pid": os.getpid(), "started_at": time.time()})
        try:
            if runner is None:
                from seepat.workflow import run_workflow

                runner = run_workflow
            runner(folder / "workflow.json")
            store.report(folder / "run", {"sha256": read(folder / "record.json")["source_sha256"]})
        except (Exception, KeyboardInterrupt) as error:
            atomic_write_json(folder / "worker.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                                                       else "failed", "error": f"{type(error).__name__}: {error}"})
            logger.exception("Local analysis worker stopped")
            return 1
        atomic_write_json(folder / "worker.json", {"status": "complete", "finished_at": time.time()})
    return 0


class Desktop:
    def __init__(self, root, store, player_factory=None):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from tkinter.scrolledtext import ScrolledText

        from scripts.video_player import VideoPlayer

        self.root, self.store = root, store
        self.dialog, self.message = filedialog, messagebox
        self.current = self.source = self.selected = self.output = None
        self.timer = None
        root.title("SeePAT - Video analysis")
        root.geometry("1120x680")
        root.minsize(940, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), foreground=INK)
        style.configure("TFrame", background="white")
        style.configure("TLabel", background="white")
        style.configure("TLabelframe", background="white", bordercolor="#d4dee7", relief="solid")
        style.configure("TLabelframe.Label", background="white", foreground=NAVY, font=("Segoe UI", 10, "bold"))
        style.configure("TButton", padding=(12, 8), background="#eef3f7", bordercolor="#ced9e3")
        style.map("TButton", background=[("active", "#e2ebf2")], foreground=[("disabled", "#84919f")])
        style.configure("Primary.TButton", background=ACCENT, foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#e3eaee"), ("active", "#096572")],
                  foreground=[("disabled", "#84919f"), ("!disabled", "white")])
        style.configure("TEntry", padding=6, fieldbackground="white", bordercolor="#ced9e3")
        style.configure("TCombobox", padding=6, bordercolor="#ced9e3", arrowsize=14)
        style.map("TCombobox", fieldbackground=[("readonly", "white")], selectbackground=[("readonly", "white")],
                  selectforeground=[("readonly", INK)])
        style.configure("TCheckbutton", background="white", padding=4)
        style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor="#e8eef3", borderwidth=0)
        style.configure("Header.TFrame", background=NAVY)
        style.configure("Brand.TLabel", background=NAVY, foreground="white", font=("Segoe UI", 21, "bold"))
        style.configure("Subtitle.TLabel", background=NAVY, foreground="#d2e2ed")
        style.configure("Status.TLabel", background="#edf2f6", foreground="#516479", padding=(20, 10))
        header = ttk.Frame(root, style="Header.TFrame", padding=(20, 12))
        header.pack(fill="x")
        ttk.Label(header, text="SeePAT", style="Brand.TLabel").pack(side="left")
        ttk.Label(header, text="Video authenticity analysis", style="Subtitle.TLabel").pack(side="left", padx=20)
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, style="Status.TLabel", wraplength=880).pack(side="bottom", fill="x")
        frame = ttk.Frame(root, padding=(20, 16))
        frame.pack(fill="both", expand=True)
        source = ttk.LabelFrame(frame, text="Video source", padding=12)
        source.pack(fill="x", pady=(0, 16))
        row = ttk.Frame(source)
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text="Source").pack(side="left", padx=(0, 8))
        self.mode = tk.StringVar(value="Your video")
        mode = ttk.Combobox(row, textvariable=self.mode, values=["Your video", "Saved examples"], state="readonly", width=18)
        mode.pack(side="left")
        mode.bind("<<ComboboxSelected>>", self.switch_mode)
        area = ttk.Frame(source)
        area.pack(fill="x")
        self.own_controls, self.demo_controls = ttk.Frame(area), ttk.Frame(area)
        self.filename = tk.StringVar(value="Choose a video to begin")
        row = ttk.Frame(self.own_controls)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.filename, state="readonly").pack(side="left", fill="x", expand=True)
        self.choose = ttk.Button(row, text="Choose video", command=self.select)
        self.choose.pack(side="left", padx=(8, 0))
        self.recent_row = ttk.Frame(self.own_controls)
        ttk.Label(self.recent_row, text="Your saved analyses").pack(side="left", padx=(0, 8))
        self.recent = ttk.Combobox(self.recent_row, state="readonly")
        self.recent.pack(fill="x", expand=True)
        self.recent.bind("<<ComboboxSelected>>", self.load_saved)
        ttk.Label(self.demo_controls, text="Saved analysis").pack(side="left", padx=(0, 8))
        self.examples = ttk.Combobox(self.demo_controls, state="readonly")
        self.examples.pack(fill="x", expand=True)
        self.examples.bind("<<ComboboxSelected>>", self.load_example)
        self.actions = ttk.Frame(self.own_controls)
        self.actions.pack(fill="x", pady=8)
        self.analyze = ttk.Button(self.actions, text="Analyze video", style="Primary.TButton", command=lambda: self.launch(False))
        self.analyze.pack(side="left")
        self.resume = ttk.Button(self.actions, text="Resume analysis", command=lambda: self.launch(True))
        self.progress = ttk.Progressbar(self.own_controls, maximum=1)
        self.progress.pack(fill="x", pady=(0, 6))
        body = ttk.Frame(frame)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1, uniform="column")
        body.columnconfigure(1, weight=1, uniform="column")
        body.rowconfigure(0, weight=1)
        preview = ttk.LabelFrame(body, text="Video preview", padding=12)
        preview.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        preview.columnconfigure(0, weight=1)
        preview.rowconfigure(0, weight=1)
        self.player = (player_factory or VideoPlayer)(preview, check_source=self.validate_source)
        self.player.grid(row=0, column=0, sticky="nsew")
        result = ttk.LabelFrame(body, text="Analysis results", padding=16)
        result.grid(row=0, column=1, sticky="nsew")
        self.verdict = ttk.Label(result, text="No result yet", font=("Segoe UI", 19, "bold"), wraplength=420)
        self.verdict.pack(anchor="w", pady=(0, 8))
        ttk.Label(result, text="XAI Forensic Trace", foreground="#65768a").pack(anchor="w", pady=(0, 12))
        self.detail_view = tk.StringVar(value="Video summary")
        self.detail_choice = ttk.Combobox(result, textvariable=self.detail_view,
            values=list(REPORT_VIEWS), state="disabled")
        self.detail_choice.pack(fill="x", pady=(0, 8))
        self.detail_choice.bind("<<ComboboxSelected>>", lambda event: self.render_text())
        self.trace = ScrolledText(result, height=10, wrap="word", font=("Segoe UI", 11),
            relief="flat", borderwidth=0, background="#f5f8fa", foreground=INK,
            padx=12, pady=12, selectbackground="#cee8ec", selectforeground=INK)
        footer = ttk.Frame(result)
        footer.pack(side="bottom", fill="x", pady=(12, 0))
        self.trace.pack(fill="both", expand=True)
        self.review = ttk.Button(footer, text="Review clip", command=self.review_clip)
        self.reports = ttk.Button(footer, text="Open report folder", command=self.open_reports)
        self.refresh_saved()
        self.switch_mode()
        self.poll()

    def text(self, value):
        self.trace.configure(state="normal")
        self.trace.delete("1.0", "end")
        self.trace.insert("end", value)
        self.trace.configure(state="disabled")

    def clear(self):
        self.player.close()
        self.current = self.source = self.selected = self.output = None
        self.filename.set("Choose a video to begin")
        self.verdict.configure(text="No result yet", foreground=INK)
        self.detail_view.set("Video summary")
        self.progress.configure(maximum=1, value=0)
        self.progress.pack_forget()
        self.render_text()

    def switch_mode(self, _event=None):
        self.clear()
        self.recent.set("")
        self.examples.set("")
        own = self.mode.get() == "Your video"
        self.own_controls.pack_forget()
        self.demo_controls.pack_forget()
        (self.own_controls if own else self.demo_controls).pack(fill="x")
        self.status.set("Choose your video, then select Analyze video." if own else "Select a saved example to inspect its existing report.")
        if not own and not self.example_items:
            self.status.set("No saved examples are available on this installation.")
        if own and self.store.active():
            self.load_job(self.store.active())
        self.buttons()

    def select(self):
        path = self.dialog.askopenfilename(title="Choose your speaking video", filetypes=[("Video", "*.mp4 *.mov *.mkv *.webm *.avi")])
        if path:
            self.clear()
            self.source = self.selected = Path(path)
            self.filename.set(self.source.name)
            self.recent.set("")
            self.player.load(self.source)
            self.status.set("Ready to analyze the selected video.")
            self.buttons()

    def launch(self, resume):
        if self.mode.get() != "Your video":
            return
        try:
            self.status.set("Checking the selected source and frozen settings...")
            self.root.update_idletasks()
            item = self.current if resume else self.store.create(self.selected, "personal-recording")
            self.store.start(item["id"])
            self.selected = None
            self.load_job(item["id"])
            self.refresh_saved()
        except (OSError, ValueError, KeyError) as error:
            self.message.showerror("SeePAT", str(error))
            self.status.set("Analysis was not started. Existing outputs are preserved.")
        self.buttons()

    def load_job(self, identity):
        if self.mode.get() != "Your video":
            return
        folder = self.store.folder(identity)
        record = read(folder / "record.json")
        self.source = self.store.checked_source(Path(record["source"]), record["source_sha256"])
        self.output = folder / "run"
        self.show(self.store.snapshot(identity))

    def refresh_saved(self):
        state = self.store.state()
        self.job_items = {f'{r["filename"]} ({r["id"][:8]})': r["id"] for r in state["jobs"]}
        self.example_items = {r["title"]: r["id"] for r in state["examples"]}
        self.recent.configure(values=list(self.job_items))
        self.examples.configure(values=list(self.example_items))
        if self.job_items:
            self.recent_row.pack(fill="x", pady=(8, 0), before=self.actions)
        else:
            self.recent_row.pack_forget()

    def load_saved(self, _event=None):
        if self.mode.get() == "Your video":
            self.load_choice(lambda: self.load_job(self.job_items[self.recent.get()]))

    def load_example(self, _event=None):
        if self.mode.get() == "Saved examples":
            def load():
                item, self.source, self.output = self.store.example(self.example_items[self.examples.get()])
                self.show(item)
            self.load_choice(load)

    def load_choice(self, load):
        try:
            self.selected = None
            load()
        except (OSError, ValueError, KeyError) as error:
            self.clear()
            self.verdict.configure(text="Report unavailable")
            self.text(str(error))
        self.buttons()

    def show(self, item):
        changed = not self.current or self.current["id"] != item["id"]
        self.current = item
        self.filename.set(item["filename"])
        status, p = item["status"], item.get("progress", {})
        done, total = p.get("finished") or 0, p.get("total") or 0
        self.progress.configure(maximum=total or 1, value=(total or 1) if status == "complete" else done)
        if status in {"queued", "running"} and not item.get("saved"):
            self.progress.pack(fill="x", pady=(0, 6))
        else:
            self.progress.pack_forget()
        self.status.set("Saved example report - no new analysis" if item.get("saved") else
                        f'{status.capitalize()} | {p.get("phase") or "Local workflow"}' + (f" | {done}/{total}" if total else ""))
        report = item.get("report")
        labels = {"authentic": "Authentic (real)", "manipulated": "Manipulated (fake)", "not_evaluated": "Not evaluated"}
        self.verdict.configure(text=labels[report["verdict"]] if report else
            "Processing stopped" if status in {"failed", "interrupted"} else "No result yet",
            foreground=OUTCOME_COLORS.get((report or {}).get("verdict"), INK))
        self.render_text()
        if changed:
            self.player.load(self.source)

    def render_text(self):
        item = self.current or {}
        report = item.get("report")
        placeholder = "Analyze your video to see its explanation." if self.mode.get() == "Your video" else "Choose a saved example to view its explanation."
        self.text(report[REPORT_VIEWS[self.detail_view.get()]] if report else item.get("error") or placeholder)

    def buttons(self):
        active, item = self.store.active(), self.current or {}
        report = item.get("report")
        stamp = ((report or {}).get("highlight") or {}).get("timestamp", {})
        self.choose.configure(state="disabled" if active else "normal")
        self.analyze.configure(state="normal" if self.selected and not active else "disabled")
        self.detail_choice.configure(state="readonly" if report else "disabled")
        for button in (self.resume, self.review, self.reports):
            button.pack_forget()
        if item.get("can_resume") and not active and self.mode.get() == "Your video":
            self.resume.pack(side="left", padx=8)
        if report:
            self.reports.pack(side="right")
            if stamp.get("status") == "verified":
                self.review.configure(text=f'Review at {stamp["start_s"]:.2f}s')
                self.review.pack(side="left", padx=(0, 8))

    def poll(self):
        if self.timer:
            self.root.after_cancel(self.timer)
        try:
            if self.mode.get() == "Your video" and self.current and self.current["status"] not in {"complete", "failed", "interrupted"}:
                self.load_job(self.current["id"])
            self.buttons()
        except (OSError, ValueError, KeyError) as error:
            self.status.set(str(error))
        self.timer = self.root.after(1000, self.poll)

    def open_reports(self):
        try:
            path = self.output / "explanation"
            if os.name == "nt":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as error:
            self.message.showerror("SeePAT", str(error))

    def review_clip(self):
        self.player.seek(self.current["report"]["highlight"]["timestamp"]["start_s"])

    def validate_source(self):
        if self.current:
            if self.current.get("saved"):
                self.store.example(self.current["id"])
            else:
                record = read(self.store.folder(self.current["id"]) / "record.json")
                self.store.checked_source(self.source, record["source_sha256"])

    def close(self):
        if self.timer:
            self.root.after_cancel(self.timer)
        self.player.close()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/demo_ui.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    store = Analyses(config=read(args.config))
    if args.worker:
        return run_worker(store, args.worker)
    import tkinter as tk

    window = tk.Tk()
    Desktop(window, store)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


