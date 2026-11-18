# SeePAT

SeePAT is an undergraduate thesis project that detects audio-visual deepfakes.
It uses a simple biological rule: a speaker's lips must close when saying the
sounds /m/, /b/ and /p/. SeePAT finds these sounds in the speech, measures the
mouth at the same moments, and combines that evidence with Video Swin and
EfficientNet–TempCNN visual features to decide whether a video is authentic or
manipulated. An XAI Forensic Trace then explains the verdict in a readable report.

## Setup

Use Python 3.11 or 3.12. From this folder:

```bat
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev,ui,preprocess,train,audio]"
```

You also need FFmpeg, Docker (for the Montreal Forced Aligner) and the model
weights, which are supplied separately. Their expected paths are listed in
`configs/required_assets.json`.

## Run

```bat
python seepat.py            :: opens the desktop app
python seepat.py run        :: analyzes the four example videos from the command line
python seepat.py --help     :: lists all commands
```

In the app, choose **Your video**, pick a recording and click **Analyze video**.
A flagged timestamp is a cue to review the video, not proof of an edit.

Run the tests with `.venv\Scripts\python.exe -m pytest -q`.

## Results

**H1 – SeePAT vs PIA on matched Deepfake-Eval-2024 videos (%)**

| Target | Model | Accuracy | Precision | Recall | Specificity | F1 | McNemar p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Audio or video (n = 269) | SeePAT | 81.04 | 95.61 | 84.17 | 0.00 | 89.53 | 0.368 |
| | PIA | 77.70 | 96.71 | 79.54 | 30.00 | 87.29 | |
| Video only (n = 570) | SeePAT | 49.47 | 45.49 | 83.81 | 23.22 | 58.97 | 0.238 |
| | PIA | 52.63 | 47.23 | 79.35 | 32.20 | 59.21 | |

**Not supported.** SeePAT was slightly more accurate for audio-or-video
manipulation and slightly less accurate for video-only manipulation, but neither
difference was statistically significant.

**H2 – Static vs per-video Isolation Forest threshold (five-fold mean, %)**

| Threshold | Accuracy | Precision | Recall | F1 | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static geometry | 53.88 | 74.89 | 58.51 | 65.66 | 60.33 |
| Isolation Forest | 68.23 | 76.43 | 83.59 | 79.82 | 79.33 |

**Not supported.** Isolation Forest improved accuracy, precision, recall and F1,
but its false-positive rate rose by 19.01 points instead of falling.

**H3 – Visual branch ablation (five-fold mean, %)**

| Branch | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| EfficientNetV2-S only | 64.77 | 76.38 | 77.20 | 76.53 |
| EfficientNet + TempCNN | 56.73 | 75.99 | 62.01 | 68.17 |
| TempCNN only | 53.52 | 75.29 | 57.20 | 62.32 |

**Not supported.** The combined branch beat TempCNN alone but scored 8.36 F1
points below EfficientNetV2-S alone, which performed best.

## Conclusions

- SeePAT works end to end: it takes a video, gives a verdict and links that
  verdict to specific clips and timestamps that a person can review.
- It detects many manipulated videos but often flags authentic ones too.
  Five-fold cross-validation gave 60.64% accuracy but only 35.42% specificity
  and 51.86% balanced accuracy.
- Future work should improve recognition of authentic videos, lip-sync
  alignment coverage and training supervision.
