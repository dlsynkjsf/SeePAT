# SeePAT

**Detecting Audio-Visual Inconsistencies in Generative AI Using Bilabial
Consonant Alignment through a Hybrid CNN-Transformer Temporal Fusion
Architecture**

SeePAT is an audio-visual deepfake detector based on the relationship between
speech and lip movement. It focuses on the bilabial consonants `/m/`, `/b/`, and
`/p/`, which require the lips to close during pronunciation. A mismatch between
the phoneme timing and the expected lip closure can provide evidence of video or
audio manipulation.

## Method

The audio track is transcribed with Whisper and aligned at the phoneme level
with Montreal Forced Aligner. Around each bilabial timestamp, MediaPipe tracks
the mouth and measures the Vertical Inter-lip Distance (VILD).

EfficientNetV2-S extracts spatial mouth features, while a Temporal Convolutional
Neural Network models how the mouth changes across frames. Isolation Forest is
used for subject-specific anomaly calibration, and Video Swin Base fuses the
spatial, temporal, and alignment features for binary classification.

The final output consists of an authentic-or-manipulated verdict and the
supporting bilabial evidence. A language model converts the completed result
into a readable forensic trace but does not participate in classification.

## Datasets

| Dataset | Purpose |
|---|---|
| [AV-Deepfake1M++](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus) | Training and validation |
| [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) | External testing on unseen, in-the-wild deepfakes |

Only videos with enough usable evidence are evaluated. Eligible samples contain
clear English speech with bilabial events and a visible frontal or near-frontal
face. Evidence quality is checked independently of the real or fake label.

## Research objectives

The study evaluates whether the full architecture can outperform existing
multimodal baselines, whether subject-specific Isolation Forest calibration can
reduce false positives compared with static thresholds, and how
EfficientNetV2-S and TempCNN contribute individually and together.

Performance is measured using accuracy, precision, recall, specificity,
F1-score, false-positive rate, ablation experiments, and statistical comparison
across cross-validation folds.

## Current progress

The data inventory, deterministic pilot sampling, selective extraction,
phoneme alignment, mouth-landmark analysis, evidence filtering, caching, and QA
outputs are implemented. The training-data interface is also implemented: it
creates filtered, source-group-safe event manifests and loads fixed-length mouth
clips with VILD and timing features. Its automated checks pass, and the
100-video benchmark produced 620 usable event records.

The current pilot contains 20 AV-Deepfake1M++ validation videos and 126 aligned
bilabial events. Of these, 120 passed the automated evidence checks, and all 115
saved minimum-closure overlays passed manual review. These figures describe the
preprocessing pilot and are not model-performance results.

A subsequent 100-video local benchmark completed without pipeline failures in
30 minutes 15.7 seconds. It produced 654 bilabial events, retained 98 videos as
eligible, and generated about 115.6 MiB of derived output.

The panel-required Video Swin Base baseline is implemented. A full-size real
mouth event completed a local GPU forward pass, and a balanced eight-event
frozen-backbone classifier-head check deliberately overfit successfully. These
are implementation checks, not model-performance results.

The baseline training command now supports mixed precision, gradient
accumulation, checkpoint recovery, early stopping, and video-level validation.
Its control flow has passed automated tests with a small test model, but no real
training experiment has been completed.

A larger source-group-safe preprocessing run is now complete. It processed
4,906 of 5,000 selected Train videos and 975 of 1,000 selected Validation
videos, retaining 5,779 evidence-eligible videos and 38,727 eligible bilabial
events across both splits. These are dataset-preparation figures, not model
performance results.

Full model training, Isolation Forest calibration, feature fusion, ablation
experiments, and external evaluation have not yet been completed.

## Scope

SeePAT covers prerecorded English videos containing `/m/`, `/b/`, and `/p/`.
It does not analyze speech meaning and is not designed for live video calls.
Heavy occlusion, extreme face angles, severe noise, and insufficient bilabial
speech may prevent a video from providing enough evidence for evaluation.

## Results

### Dataset composition

### Classification performance

### Dynamic threshold comparison

### Feature ablation

### External evaluation

### Statistical analysis

### Example forensic trace

## Researchers

- Johanna Louisse V. Carigma
- Nikolas Josef P. Dalisay
- Arthur Justin Rosmar A. Evangelista
- Marian Therese J. Pineza

University of Santo Tomas, College of Information and Computing Sciences,
Department of Computer Science
