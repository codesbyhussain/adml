import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
import segmentation_models_pytorch as smp
from shapely.geometry import Polygon
from typing import List, Tuple
from importlib import reload
import utils
reload(utils)
from utils import load_images_and_masks
from utils import mask_to_polygons
from utils import Evaluator
from utils import safe_polygon
from utils import postprocess_polygons
from utils import visualize_and_save
import albumentations as A
from albumentations.pytorch import ToTensorV2

def dice_score(pred_mask, gt_mask, class_id):
    pred_bin = (pred_mask == class_id).astype(np.uint8)
    gt_bin = (gt_mask == class_id).astype(np.uint8)
    intersection = (pred_bin & gt_bin).sum()
    union = pred_bin.sum() + gt_bin.sum()
    if union == 0:
        return 1.0  # both empty
    return 2 * intersection / union

# Paths
train_images_path = '../data/train/images/*.tif'
train_masks_path  = '../data/train/masks/train_annotations.json'

# Load the FULL training dataset
X_train_full, y_train_full = load_images_and_masks(train_images_path, train_masks_path)

# Create TRAIN/VAL SPLIT
val_ratio = 0.2
dataset_full = TensorDataset(X_train_full, y_train_full)

val_size = int(len(dataset_full) * val_ratio)
train_size = len(dataset_full) - val_size

seed = 42
g = torch.Generator().manual_seed(seed)

train_dataset, val_dataset = random_split(
    dataset_full,
    [train_size, val_size],
    generator=g
)

# class ID → class name mapping for polygon export
CLS_MAP = {
    0: "background",
    1: "individual_tree",
    2: "group_of_trees"
}

print("Running validation inference and exporting predictions...")

val_loader_for_inference = DataLoader(val_dataset, batch_size=1, shuffle=False)
val_indices = val_dataset.indices

#evaluate on validation and get a score
with open(train_masks_path) as f:
    gt_data = json.load(f)

gt_polygons_per_image = []

for item in gt_data["images"]:
    polys = []
    for ann in item.get("annotations", []):
        seg = ann["segmentation"]
        poly = safe_polygon(np.array(seg).reshape(-1,2))
        polys.append(poly)
    gt_polygons_per_image.append(polys)

# Initialize evaluator
evaluator = Evaluator()

checkpoint_path = '../checkpoints/unet_resnet50_weights.pth'

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=3,
    classes=3  # 0=background, 1=individual_tree, 2=group_of_trees
).to(device)

# Load checkpoint
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
all_scores = []

CLS_IDS = [1, 2]  # classes to compute Dice for
dice_scores_all = []  # to store per-image Dice scores

# Run inference on validation images
with torch.no_grad():
    for idx, (img_tensor, mask_gt) in enumerate(val_loader_for_inference):
        img = img_tensor.cuda()  # (1,3,H,W)
        
        # Forward pass
        logits = model(img)[0].cpu()  # (C,H,W)
        probs = torch.softmax(logits, dim=0).numpy()
        pred_mask = torch.argmax(logits, dim=0).numpy()  # (H,W)
        gt_mask = mask_gt[0].numpy()

        # Convert predictions to polygons
        annotations = mask_to_polygons(pred_mask, probs, CLS_MAP)
        annotations = postprocess_polygons(annotations, min_area=50, conf_thresh=0.3)
        pred_polygons_with_scores = []
        for ann in annotations:
            seg = np.array(ann["segmentation"]).reshape(-1,2)
            poly = safe_polygon(seg)
            pred_polygons_with_scores.append((poly, ann["confidence_score"]))

        # Get ground truth polygons for this image
        original_idx = val_indices[idx]
        gt_polys = gt_polygons_per_image[original_idx]

        # Compute mAP
        score = evaluator.compute_map(gt_polys, pred_polygons_with_scores)
        all_scores.append(score)

        # -----------------------------
        # Compute Dice scores
        # -----------------------------
        dice_scores_per_image = {}
        for cls_id in CLS_IDS:
            dice = dice_score(pred_mask, gt_mask, cls_id)
            dice_scores_per_image[cls_id] = dice
        dice_scores_all.append(dice_scores_per_image)

        print(f"Image {idx}: mAP = {score:.4f}, Dice = {dice_scores_per_image}")

        # save visualization
        pred_poly_only = [p[0] for p in pred_polygons_with_scores]

        save_path = f"../outputs/val/val_{idx:03d}.png"

        visualize_and_save(
            image=img_tensor[0],
            pred_mask=pred_mask,
            gt_mask=gt_mask,
            pred_polygons=pred_poly_only,
            gt_polygons=gt_polys,
            save_path=save_path
        )

# Overall average mAP
overall_map = np.mean(all_scores)
overall_dice = {cls_id: np.mean([d[cls_id] for d in dice_scores_all]) for cls_id in CLS_IDS}
print(f"Overall validation mAP: {overall_map:.4f}")
print(f"Overall Dice scores: {overall_dice}")