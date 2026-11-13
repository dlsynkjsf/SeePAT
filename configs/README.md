# Configuration

The desktop app and command-line tools use these files:

| File | Purpose |
| --- | --- |
| `demo_ui.json` | Desktop settings and saved examples |
| `model.json` | Model, calibration, cutoff, and source hashes for local analysis |
| `preprocessing.yaml` | Audio, alignment, face tracking, and clip settings |
| `local_inference.yaml` | Command-line workflow for the four example videos |
| `local_inference_case_01.json`–`local_inference_case_04.json` | Original video paths and hashes for each example |
| `required_assets.json` | Separately supplied model and runtime assets |
| `pia_assessment.optimized.json` | PIA assessment settings |
| `h1_assess_subset1000.v1.json` | Fixed 1,000-video PIA selection |
| `pia_source_origin.json` | PIA author-source references and hashes |

For training, `workflow_scaled.yaml` connects the Train and Validation subset
profiles to preprocessing and calibration. `training_readiness.yaml` checks a
short training run and checkpoint resume; `training_experiments.yaml` defines
full runs. `training_ablations.yaml` defines the source-group fold study.

Use stable filenames based on purpose. Training seeds and recorded asset paths
identify the existing experiments and must match the supplied assets.
