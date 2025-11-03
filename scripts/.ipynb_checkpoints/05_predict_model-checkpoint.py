import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path
import torch
from datetime import datetime

# Assign directory location
REPO_ROOT = Path(__file__).resolve().parent.parent
current_dt = datetime.now().strftime('%Y-%m-%d %H:%M')


# Load model parameters / overrides YAML
with open(REPO_ROOT / 'configurations' / 'predict_model_overrides.yaml', 'r') as f: # Modify overrides where necessary
    overrides = yaml.safe_load(f)

# Make YAML Paths Absolute
for key in ("source", "project"):
    if key in overrides:
        overrides[key] = str((REPO_ROOT / overrides[key]).resolve())

# Load Trained Model Weights
def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)

weights_path = resolve(overrides.pop("weights"))

# Model Naming
# optional: use the training run name from the weights path
train_run = weights_path.parents[1].name  # e.g., train_Yolov8s_canopy_832_...
run_name = f"predict_{train_run}_{current_dt}"

# inject name (and project if you want a different subfolder)
overrides.setdefault("project", str(REPO_ROOT / "runs"))
overrides["name"] = run_name
# overrides["exist_ok"] = True  # set True to overwrite instead of auto-appending _2

# Load Trained Model's Weights
model = YOLO(str(weights_path))  

# Predictions from the model
with torch.inference_mode():
    predictions = model.predict(**overrides)

# State output location

# Stream = True
for last in predictions:
    pass
print("Saved to:", last.save_dir)

# Stream = False
# print("Outputs saved to:", predictions[0].save_dir)
