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

Whisper transcribes the audio, Montreal Forced Aligner locates each bilabial
phoneme, and MediaPipe measures the mouth around those timestamps. The proposed
model combines EfficientNetV2-S, TempCNN, VILD-based synchronization evidence,
Isolation Forest calibration, and the panel-required Video Swin Base model.

The classifier produces the authentic or manipulated result. A language model
may explain the completed result, but it does not participate in classification.

## Datasets

| Dataset | Purpose |
|---|---|
| [AV-Deepfake1M++](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M-PlusPlus) | Training and validation |
| [Deepfake-Eval-2024](https://huggingface.co/datasets/nuriachandra/Deepfake-Eval-2024) | External testing on unseen, in-the-wild deepfakes |

Only videos with enough usable evidence are evaluated. Eligible samples contain
clear English speech with bilabial events and a visible frontal or near-frontal
face. Evidence quality is checked independently of the real or fake label.

## Research objectives

The study compares SeePAT with multimodal baselines, tests whether
subject-specific calibration reduces false positives, and measures the
contribution of its spatial and temporal features.

## Current status

Implemented:

- deterministic sampling and selective archive extraction;
- restartable audio, alignment, mouth-tracking, and evidence-filtering steps;
- an optional cached Demucs and DeepFilterNet audio-enhancement path;
- filtered event manifests and a PyTorch mouth-event dataset;
- a Video Swin Base baseline with resumable, bounded training runs; and
- automated tests for the data and training controls.

The 5,000-video Train and 1,000-video Validation subsets have been preprocessed
with the original audio path. The enhanced audio path is unit-tested and awaits
one-video validation. Full model training, feature fusion, calibration,
ablation experiments, and external evaluation have not been completed.

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
