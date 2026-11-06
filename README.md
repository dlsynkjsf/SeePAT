# SeePAT

SeePAT detects audio-visual deepfakes using bilabial alignment, mouth-motion
measurements, Video Swin, EfficientNet–TempCNN, and attention fusion. Its XAI
Forensic Trace presents the saved verdict and supporting observations.

This submission contains the maintained application, command interface,
configuration templates, and focused tests. Historical experiment handoffs,
recovery controllers, research snapshots, and old phase configurations are kept
separately from the submission.

## Setup

Use Python 3.11 or 3.12. From this folder in Command Prompt:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,ui,preprocess,train,audio]"
python seepat.py --help
```

The frozen inference contract lists the required runtime versions. Use the
matching PyTorch/CUDA environment for the supplied model. FFmpeg, the configured
audio tools, and the Montreal Forced Aligner Docker image are separate runtime
requirements. PIA uses its separately installed, isolated author Docker image.

Datasets, model weights, prepared evidence, cached models, and the virtual
environment are supplied separately. Their required paths and saved hashes are
listed in `configs/required_assets.json`. Populate these paths before running
the corresponding operation. No trained weights or credentials are embedded in
this source submission.

## Run the detector

For the local video interface, run:

```bat
python seepat.py
```

A native Tkinter window opens with embedded video and audio playback, Play/Pause,
a seek bar and mute. A compact navy-and-teal layout groups Video source, Video
preview, and Analysis results, with operation status along the bottom.
In **Your video**, choose a recording and select **Analyze
video**. **Saved examples** is a separate, read-only source view. Switching views
clears the previous video and result; background jobs keep their own records.
The report selector shows either **Video summary** or **Evidence details**.
The summary states the verdict, how many analyzed clips triggered warnings,
and the transcript word near the strongest warning when that information is
available. Scores and cutoffs stay in the details view. A warning is a review
cue, not proof that a word or mouth movement was altered.
**Review at ...** plays the current video from its saved review timestamp inside
the same window. Keep the selected original file available and unchanged.

The UI uses the same frozen inference contract and existing runtime assets.
It needs no web server or browser files. The `ui` extra supplies FFPyPlayer and
Pillow for playback. On an existing installation, add it once with
`.venv\Scripts\python.exe -m pip install -e ".[ui]"`. The Python launcher
uses the project's `.venv` when present. Double-clicking `START_UI.cmd` or using
`seepat.cmd ui` remains supported. Closing the window leaves an active analysis
worker running; reopen the UI to follow it.

The command-line workflow remains:

```bat
python seepat.py run
```

The default `configs/local_inference.yaml` prepares four known example
recordings, scores them with the retained frozen fusion model and cutoff, and
exports the forensic trace. It writes a fresh result tree under
`outputs/local_inference/submission_examples_v2/`.

For other local videos, create an input inventory and workflow configuration
using the supplied examples. Each inventory records a source path, its SHA-256,
and local origin. Videos are not assigned a class label during inference.

```bat
python seepat.py run --config configs/local_inference.yaml
python seepat.py watch
python seepat.py verify
```

Run the watcher in a second terminal. It is read-only and exits when the worker
finishes or stops. Ctrl+C in the watcher stops watching. Repeat the same worker
command to resume compatible committed work; changed inputs or source bindings
are rejected.

The readable report states authentic (real), manipulated (fake), or not
evaluated. A flagged timestamp is a review cue. Technical details retain the
clip count, maximum score, cutoff, and available numerical evidence. The default
local report is generated from saved evidence without a provider API request.

## Training and individual stages

```bat
python seepat.py train --config configs/training_experiments.yaml
python seepat.py watch --config configs/training_experiments.yaml
python seepat.py readiness --config configs/training_experiments.yaml --inputs-only
```

These commands require the corresponding prepared Train and Validation
manifests. The supplied training files are explicit research configurations;
running the detector does not start training. With `--config`, `train` runs the
jobs declared by that workflow. Without it, `train` accepts explicit model,
manifest and optimization options; use `python seepat.py train --help`.

Use `python seepat.py COMMAND --help` for individual stage options:

| Command | Purpose |
| --- | --- |
| `data` | Dataset inventories, manifests, selection, and extraction |
| `preprocess` | Audio preparation, alignment, mouth clips, and VILD |
| `train` | Model learning and checkpoint resume |
| `readiness` | Training input and execution checks |
| `calibrate` | Train population calibration or frozen calibration application |
| `verdict` | Prepared-input scoring and Validation cutoff selection |
| `xai` | Display a saved XAI Forensic Trace |

## PIA assessment

Start Docker Desktop and supply the bound PIA assets before running:

```bat
python pia.py
python pia.py watch
python pia.py verify
```

The controller uses `configs/pia_assessment.optimized.json` and the unchanged fixed
1,000-video selection. It retains the released checkpoint, author inputs, and
cutoff, reuses compatible sealed predictions, and reports metrics after all
requested prediction records are sealed. Unknown targets and unavailable cases
remain recorded. Results belong to `outputs/external_assessment/pia_sparse_decode/`.
The identity stage advances through the same source-frame indices but retrieves
RGB pixels only for entries consumed by the author model. Checkpoint, phoneme
selection, raw embeddings, CPU runtime and cutoff stay fixed. The older profile
and source archives are preserved; the new runtime uses a separate output tree.

Resume by repeating the worker command. The CPU runtime depends on video length
and can be substantial; completed videos are retained, while uncommitted work
within a video can repeat.

## Source layout and checks

`src/seepat/` contains preprocessing, models, training, inference, decisions,
and verdict-linked reports. `workflow_config.py` defines jobs; `workflow.py` executes them.
`scripts/` contains the command router, native desktop interface and PIA runtime.
The root Python launchers delegate to that router. `tests/` contains focused
checks for these interfaces.

Completed five-fold experiment launchers, transfer scripts and their tests live
under `../info dump/`. They are preserved with the original ready package rather
than imported by the application. The application tests remain alongside this code.

```bat
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src scripts tests
```

In the source ZIP, `RELEASE_MANIFEST.json` records the exact submitted source and
excluded research paths. In this workspace, plotting/manuscript tools, historical
source and delivery packages live under `../info dump/`, outside the application.
The current source ZIP and its manifest are in `../info dump/delivery/`.
The supplied PIA author source is
retained at its pinned paths, with its origin recorded in
`configs/pia_source_origin.json`.
