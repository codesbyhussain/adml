import os
import json
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import segmentation_models_pytorch as smp
from shapely.geometry import Polygon
from typing import List, Tuple

from utils import load_images_and_masks

# Paths
train_images_path = '../data/train/images/*.tif'
train_masks_path  = '../data/train/masks/train_annotations.json'
val_images_path   = '../data/val/images/*.tif'
val_masks_path    = '../data/val/masks/sample_answer.json'

# Load dataset
X_train, y_train = load_images_and_masks(train_images_path, train_masks_path)
X_val, y_val     = load_images_and_masks(val_images_path, val_masks_path)

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=4)

# Model setup
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=3,
    classes=3   # 0=background, 1=individual_tree, 2=group_of_trees
).cuda()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.cuda(), masks.cuda()
        optimizer.zero_grad()
        logits = model(imgs)           # (B,3,H,W)
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

