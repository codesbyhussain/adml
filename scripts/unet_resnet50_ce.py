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
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# evaluator = Evaluator()
# print(hasattr(evaluator, "compute_map"))

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

# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
# val_loader   = DataLoader(val_dataset, batch_size=4)

print(f"Train size: {train_size}, Val size: {val_size}")

# -------------------------
# Augmentations
# -------------------------
# imagenet_mean = (0.485, 0.456, 0.406)
# imagenet_std  = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    # A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2()
])

val_transform = A.Compose([
    # A.Normalize(mean=imagenet_mean, std=imagenet_std),
    ToTensorV2()])

# -------------------------
# Custom Dataset for Augmentation
# -------------------------
class TreeDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].permute(1,2,0).numpy()  # (H,W,C)
        mask = self.masks[idx].numpy()                 # (H,W)
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = torch.tensor(augmented['mask'], dtype=torch.long)
        else:
            img = torch.tensor(img).permute(2,0,1).float()
            mask = torch.tensor(mask, dtype=torch.long)
        
        return img, mask

# Wrap datasets
train_dataset_aug = TreeDataset(
    images=X_train_full[train_dataset.indices],
    masks=y_train_full[train_dataset.indices],
    transform=train_transform
)

val_dataset_aug = TreeDataset(
    images=X_train_full[val_dataset.indices],
    masks=y_train_full[val_dataset.indices],
    transform=val_transform
)

# -------------------------
# Dataloaders
# -------------------------
train_loader = DataLoader(train_dataset_aug, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset_aug, batch_size=4)

# Model setup
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=3   # 0=background, 1=individual_tree, 2=group_of_trees
).cuda()

# weights = torch.tensor([0.8, 1.0, 1.1]).to(device)
# criterion = nn.CrossEntropyLoss(weight=weights)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.cuda(), masks.cuda()
            logits = model(imgs)
            loss = criterion(logits, masks)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# Save trained weights
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.state_dict(), "../checkpoints/unet_resnet50_weights.pth")
print("Model weights saved")

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

model.eval()
all_scores = []

# Run inference on validation images
with torch.no_grad():
    for idx, (img_tensor, mask_gt) in enumerate(val_loader_for_inference):
        img = img_tensor.cuda()  # (1,3,H,W)
        
        # Forward pass
        logits = model(img)[0].cpu()  # (C,H,W)
        probs = torch.softmax(logits, dim=0).numpy()
        pred_mask = torch.argmax(logits, dim=0).numpy()  # (H,W)

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

        print(f"Image {idx}: mAP = {score:.4f}")

# Overall average mAP
overall_map = np.mean(all_scores)
print(f"Overall validation mAP: {overall_map:.4f}")