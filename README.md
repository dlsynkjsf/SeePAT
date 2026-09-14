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
inputs. The implementation passes unit tests; real-data calibration and final
multimodal fusion still require validation.
