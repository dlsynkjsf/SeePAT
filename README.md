# SeePAT

**Detecting Audio-Visual Inconsistencies in Generative AI Using Bilabial
Consonant Alignment through a Hybrid CNN-Transformer Temporal Fusion
Architecture**

SeePAT is an audio-visual deepfake detector based on the relationship between
speech and lip movement. It focuses on the bilabial consonants `/m/`, `/b/`, and
`/p/`, which require the lips to close during pronunciation. A mismatch between
the phoneme timing and the expected lip closure can provide evidence of video or
audio manipulation.

Local inference now has an opt-in `local_inference` workflow section connecting
raw-video preparation, frozen prediction and concise offline reports. Run it
through `python -m seepat.workflow`; use `python -m seepat.progress` for tracking
and verification. It requires the sibling `docs/inference/` companions and a
reviewed local freeze/input inventory. New reports and terminal summaries use a
plain-language authentic (real)/manipulated (fake) paragraph, followed by separate
technical details. Existing reports stay unchanged; old execution profiles must
not be repinned or rerun after a source update. See the
[completed four-video local check](../docs/inference/CLI_ACCEPTANCE_RESULTS.md):
12/12 stages verified. Broader human review and detection quality remain open;
this technical check is not an accuracy benchmark.
Deepfake-Eval is locked. The preparation freeze does not enable execution.

Phase-5 training and local return installation are complete; see
[development results and limitations](../PHASE5_RESULTS.md). Historical
preflight/pending descriptions below are retained for context, not current status.

The workflow preserves existing preprocessing and adds separate, verified raw
VILD and face-size measurements. Scale regression and Pearson analysis use
label-independent Train references; phoneme expectations use genuine Train
events. Each input video fits its own Isolation Forest from its eligible
non-speech frames. Insufficient references remain masked, with no pooled
fallback. Train parameters stay frozen when scoring Validation or later Test
inputs. Scaled calibration passed artifact and coverage checks. Fusion now
combines Swin, CNN-temporal, and calibrated evidence with a separate mask for
every numerical feature and independent resumable training. Closure offset is
measured timing in seconds, not a separately trained anomaly score. Unit tests
pass. Training counts completed optimizer updates and rejects zero-update epochs;
full-precision CNN/fusion preflights passed with verified updates and fusion
checkpoint resume. Validation thresholds are bound to checkpoint/calibration
provenance, and explanations consume verified frozen verdicts. The restartable
decision stages await a completed model experiment; external evaluation remains
locked. Model experiments and thesis performance validation remain pending.

Portable profiles prepare Swin, CNN-temporal, and fusion experiments with frozen
pretrained encoders. Full runs require a matching resumed readiness check with
finite saved model/optimizer state and actual updates. The new globally balanced
loss preserves minority-class weights with batch-size-one gradient accumulation.
The default workflow retains the accepted preflights until a training profile
is selected; new hardware runs have not been performed.

An opt-in ablation study adds EfficientNet-only, TempCNN over fixed grayscale
frames, and fusion without biological evidence. Source-group folds fit their
own Train calibration and use official Validation for checkpoint/threshold
selection. Static geometry and dynamic calibration are compared on matched
evidence, with coverage and paired fold diagnostics. Evaluation writes readable
Markdown tables and CSV summaries with fold averages, confusion counts, evidence
coverage and explicit missing comparisons. Neural experiments and confirmatory
statistical review remain pending.
