# SeePAT

**Detecting Audio-Visual Inconsistencies in Generative AI Using Bilabial
Consonant Alignment through a Hybrid CNN-Transformer Temporal Fusion
Architecture**

SeePAT is an audio-visual deepfake detector based on the relationship between
speech and lip movement. It focuses on the bilabial consonants `/m/`, `/b/`, and
`/p/`, which require the lips to close during pronunciation. A mismatch between
the phoneme timing and the expected lip closure can provide evidence of video or
audio manipulation.

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
