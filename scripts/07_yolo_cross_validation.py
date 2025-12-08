# YOLOv11m Segmentation Training with 5-Fold Cross-Validation
# This script sets up and runs a 5-fold cross-validation training for a YOLO model.
# It handles the data preparation, training for each fold, and aggregation of results.

# ## 1. Setup
# Import libraries and define paths.
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from ultralytics import YOLO
import torch

# Training Parameters
EPOCHS = 25
BATCH_SIZE = 8 # Reduce this value if you encounter CUDA out of memory errors

# Define paths
ROOT_DIR = Path('./') # Running from the root of the project
DATA_DIR = ROOT_DIR / 'data' / 'processed'
CV_DATA_DIR = ROOT_DIR / 'cv_data_yolov8s' # New directory for this CV
MODEL_CONFIG_PATH = ROOT_DIR / 'configurations' / 'model_data-seg.yaml'
# Using the best.pt from the user's previous yolo11m training run
PRETRAINED_MODEL_PATH = ROOT_DIR / 'yolov8s-seg.pt'


# ## 2. Data Preparation for Cross-Validation
# Combine the existing training and validation sets and then split them into 5 folds.
print("--- Preparing Data for Cross-Validation ---")

# Combine all images and labels
all_images = sorted(list(DATA_DIR.glob('images/train/*.tif'))) + sorted(list(DATA_DIR.glob('images/val/*.tif')))
all_labels = sorted(list(DATA_DIR.glob('labels/train/*.txt'))) + sorted(list(DATA_DIR.glob('labels/val/*.txt')))

# Ensure images and labels correspond
assert len(all_images) == len(all_labels), "Mismatch between number of images and labels"
for img, lbl in zip(all_images, all_labels):
    assert img.stem == lbl.stem, f"Mismatch between {img.name} and {lbl.name}"

print(f"Total images for CV: {len(all_images)}")

# Create the main CV directory
if CV_DATA_DIR.exists():
    shutil.rmtree(CV_DATA_DIR)
CV_DATA_DIR.mkdir(exist_ok=True)

# Setup KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_images = np.array(all_images)

# Get class names from original data config
with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = yaml.safe_load(f)
class_names = model_config['names']

for i, (train_index, val_index) in enumerate(kf.split(all_images)):
    fold_dir = CV_DATA_DIR / f'fold_{i}'
    
    # Create directories for the fold
    (fold_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (fold_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # Copy training files
    for idx in train_index:
        img_path = all_images[idx]
        lbl_path = Path(str(img_path).replace('images', 'labels').replace('.tif', '.txt'))
        shutil.copy(img_path, fold_dir / 'images' / 'train' / img_path.name)
        shutil.copy(lbl_path, fold_dir / 'labels' / 'train' / lbl_path.name)
        
    # Copy validation files
    for idx in val_index:
        img_path = all_images[idx]
        lbl_path = Path(str(img_path).replace('images', 'labels').replace('.tif', '.txt'))
        shutil.copy(img_path, fold_dir / 'images' / 'val' / img_path.name)
        shutil.copy(lbl_path, fold_dir / 'labels' / 'val' / lbl_path.name)
        
    # Create YAML file for the fold
    fold_yaml_path = fold_dir / f'fold_{i}_data.yaml'
    fold_data_config = {
        'path': str(fold_dir.resolve()),
        'train': str((fold_dir / 'images' / 'train').resolve()),
        'val': str((fold_dir / 'images' / 'val').resolve()),
        'names': class_names
    }
    with open(fold_yaml_path, 'w') as f:
        yaml.dump(fold_data_config, f)
        
    print(f"Fold {i} created.")

print("--- Data Preparation Complete ---")

# ## 3. Cross-Validation Training
# Loop through each fold and train the model.
for i in range(5):
    print(f"\n--- Training Fold {i} ---")
    
    # Load pretrained model
    model = YOLO(PRETRAINED_MODEL_PATH)
    
    # Get data config for the fold
    fold_yaml_path = CV_DATA_DIR / f'fold_{i}' / f'fold_{i}_data.yaml'
    
    # Train the model
    model.train(
        data=str(fold_yaml_path.resolve()),
        epochs=EPOCHS, 
        batch=BATCH_SIZE, # Added batch size
        imgsz=640,
        project='YOLOv8s_CV',
        name=f'fold_{i}',
        exist_ok=True # Allows re-running the script
    )
    
    # Clear memory
    del model
    torch.cuda.empty_cache()

print("--- Cross-Validation Training Complete ---")

# ## 4. Results Aggregation
# Combine the results from all folds and calculate the mean and standard deviation of the key metrics.
print("\n--- Aggregating Results ---")
results_dir = ROOT_DIR / 'YOLOv8s_CV'
all_results = []

for i in range(5):
    fold_results_path = results_dir / f'fold_{i}' / 'results.csv'
    if fold_results_path.exists():
        df = pd.read_csv(fold_results_path)
        df['fold'] = i
        all_results.append(df)

if all_results:
    cv_results_df = pd.concat(all_results)
    
    # Get the metrics from the last epoch of each fold
    last_epoch_results = cv_results_df.groupby('fold').last().reset_index()
    
    print("\n--- Cross-Validation Results Summary (Last Epoch of Each Fold) ---")
    print(last_epoch_results[['fold', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']])
    
    print("\n--- Mean and Std Dev of Metrics ---")
    mean_metrics = last_epoch_results[['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']].mean()
    std_metrics = last_epoch_results[['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']].std()
    
    summary_df = pd.DataFrame({'mean': mean_metrics, 'std_dev': std_metrics})
    print(summary_df)
else:
    print("No results found. Please ensure the training completed successfully.")

print("\n--- Script Finished ---")
