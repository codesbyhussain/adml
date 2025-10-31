# Import Packages
import ultralytics
from ultralytics import YOLO
import numpy as np
import time
import yaml
from pathlib import Path
import torch
import pandas as pd
from shutil import copy2

#-- Directories and Paths --#
REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_CONFIG_PATH = REPO_ROOT / 'configurations' / 'model_data-seg.yaml'

#-- Model --#

# Load model parameters / overrides
with open(REPO_ROOT / 'configurations' / 'test_model_overrides.yaml', 'r') as f:
    overrides = yaml.safe_load(f)

# Override path in your dictionary
overrides['data'] = str(DATA_CONFIG_PATH)

# Assign Trained Model Weights
weights = REPO_ROOT / 'runs/segment/train_Yolo11s_canopy_832__20251030-2009/weights/best.pt' # Weights from Train Model / Modify where necessary

# Load Trained Model's Weights
model = YOLO(str(weights))  

# Predictions from the model
with torch.inference_mode():
    metrics = model.val(**overrides)


#-- Results --#

#--- Results Directory ---#
# where YOLO saved this run
out_dir = Path(metrics.save_dir)
out_dir.mkdir(parents=True, exist_ok=True)

#--- Precision & Recall Calculations ---#

# names -> {id: name}; confusion matrix includes a last row/col for "background"
names = [metrics.names[i] for i in range(len(metrics.names))]
cm = metrics.confusion_matrix.matrix.astype(np.int64)   # shape = (nc+1, nc+1) # cm = confusion_matrix
nc = len(names) # nc = name classes

# diagonal (exclude background) = TP
tp = np.diag(cm)[:nc]

# row sums (include background col) minus TP = FP for that predicted class
fp = cm[:nc, :].sum(axis=1) - tp

# column sums (include background row) minus TP = FN for that true class
fn = cm[:, :nc].sum(axis=0) - tp

precision = tp / (tp + fp + 1e-16)
recall    = tp / (tp + fn + 1e-16)
f1        = 2 * precision * recall / (precision + recall + 1e-16)

df = pd.DataFrame({
    "class": names, "TP": tp, "FP": fp, "FN": fn,
    "precision": precision, "recall": recall, "f1": f1
})
print(df)
# Output P&R Results
df.to_csv(out_dir / "P&R_results.csv", header=True,index=True)   
print(f"Saved df results metrics to {out_dir/'df_results.csv'}")


#-- Export Validation Results --#
# dataframe
validation_df = metrics.to_df()

# validation_df.to_csv(csv_path, header=True, index=True)
summary_df = pd.DataFrame([metrics.results_dict])  # <-- dict -> one-row DataFrame
summary_df.to_csv(out_dir / "summary_metrics_results.csv", index=False)
print(f"Saved metrics to {out_dir/'results.csv'}")




# choose task: 'seg' for masks, 'box' for boxes
task = 'seg' # Choosing specific 'task' type of 'segmentation' selects for metric associates
seg_metrics = getattr(metrics, task)

print(f"AP@0.75 ({task}): {seg_metrics.map75:.4f}")
print(f"AP@0.50 ({task}): {seg_metrics.map50:.4f}")
print(f"Classes  mAP@0.50:0.95 ({task}): {seg_metrics.map:.4f}")

# Per-class mAP (COCO mAP@0.50:0.95) if exposed
names = metrics.names if isinstance(metrics.names, (list, tuple)) else list(metrics.names.values())
if hasattr(seg_metrics, "maps") and seg_metrics.maps is not None:
    for i, name in enumerate(names):
        print(f"{name:>16}  mAP@0.50:0.95 = {seg_metrics.maps[i]:.4f}")


# Copy test_parameters into output folder
test_model_overrides = Path(REPO_ROOT / 'configurations/test_model_overrides.yaml')
copy2(test_model_overrides, metrics.save_dir / test_model_overrides.name )
try:
    print(f'"test_model_parameters" Copied to : {metrics.save_dir}')
except Exception:
    print('Error - Parameters Not Copied To Output')

# State output location
try:
    print("Output Metrics Saved to:", metrics.save_dir)
except Exception:
    print('Error - Location Not Printed')




