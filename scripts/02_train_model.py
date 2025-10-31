import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path

# Directories and Paths
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CONFIG_PATH = REPO_ROOT / 'configurations' / 'model_data-seg.yaml'

# Load model parameters / overrides
with open(REPO_ROOT / 'configurations' / 'train_model_overrides.yaml', 'r') as f:
    overrides = yaml.safe_load(f)

# override the path in your dictionary before calling train
overrides['data'] = str(DATA_CONFIG_PATH)

# Load a pretrained model
model = YOLO('yolo11s-seg.pt') # yolo version 11s segementation

# Train the model
results = model.train(**overrides)

# State output location
try:
    print("Saved to:", model.trainer.save_dir)
except Exception:
    pass
