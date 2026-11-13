"""Niko-run container readiness: imports and trusted author checkpoint, no videos."""

import importlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch

subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
for name in ("whisperx", "mediapipe", "insightface", "onnxruntime", "phonemizer"):
    importlib.import_module(name)
if torch.cuda.is_available():
    raise RuntimeError("This reviewed baseline setup requires CPU execution")
sys.path.insert(0, "/pia")
import main as author

model = author.CNNMultistreamModel(arcface_dim=512, geo_dim=1, embed_dim=128,
                                  num_heads=4, frames_per_phon=author.IMAGES_PER_PHON)
author.load_weights(model, author.WEIGHTS_PATH, "cpu")
model.eval()
if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
    raise RuntimeError("Author model contains nonfinite state")
packages = ("torch", "torchvision", "torchaudio", "whisperx", "faster-whisper",
            "ctranslate2", "transformers", "huggingface-hub", "mediapipe",
            "insightface", "onnxruntime-gpu", "numpy")
result = {"python": platform.python_version(), "device": "cpu",
          "versions": {name: version(name) for name in packages},
          "strict_author_checkpoint_load": True, "video_inference": False,
          "parameters": sum(p.numel() for p in model.parameters()),
          "pip_reports": {name: json.loads(Path(f"/opt/{name}").read_text())
                          for name in ("torch-install.json", "pia-install.json")}}
print("PIA_ENV_JSON:" + json.dumps(result), flush=True)
