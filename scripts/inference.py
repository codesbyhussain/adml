import os
import json
import glob
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
from utils import mask_to_polygons

# Paths
eval_images_path = '../data/eval/images/*.tif'  # folder with evaluation images
checkpoint_path = '../checkpoints/unet_resnet50_weights.pth'
output_json_path = '../outputs/submission.json'

# Class ID → class name mapping for submission
CLS_MAP = {
    0: "background",
    1: "individual_tree",
    2: "group_of_trees"
}

# Load evaluation images
eval_image_files = sorted(glob.glob(eval_images_path))
eval_imgs = []
file_names = []

for img_path in eval_image_files:
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
    eval_imgs.append(torch.tensor(img).permute(2,0,1).float())  # (3,H,W)
    file_names.append(os.path.basename(img_path))

eval_imgs = torch.stack(eval_imgs)  # (N,3,H,W)

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

# Run inference and convert to polygons
submission_results = {"images": []}

with torch.no_grad():
    for idx, img_tensor in enumerate(eval_imgs):
        img_tensor = img_tensor.unsqueeze(0).to(device)  # (1,3,H,W)
        logits = model(img_tensor)[0].cpu()              # (C,H,W)
        probs = torch.softmax(logits, dim=0).numpy()
        pred_mask = torch.argmax(logits, dim=0).numpy()  # (H,W)

        # Convert predicted mask to polygons
        annotations = mask_to_polygons(pred_mask, probs, CLS_MAP)

        _, H, W = img_tensor.shape[1:]
        submission_results["images"].append({
            "file_name": file_names[idx],
            "width": W,
            "height": H,
            "cm_resolution": 10,  # adjust if necessary
            "scene_type": "unknown_scene",
            "annotations": annotations
        })

# Write submission JSON
os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(submission_results, f, indent=4)

print(f"Submission JSON saved to: {output_json_path}")
