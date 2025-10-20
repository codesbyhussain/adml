import json
import os
import cv2
import numpy as np
import torch

# Class mapping
CLASS_MAP = {
    "individual_tree": 1,
    "group_of_trees": 2
}

# Load images and generate multi-class integer masks from JSON polygons
def load_images_and_masks(img_dir_pattern, json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    imgs = []
    masks = []

    for item in data["images"]:
        file_name = item["file_name"]
        img_path = os.path.join(os.path.dirname(img_dir_pattern), file_name)

        # Load image and normalize
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.0
        h, w = img.shape[:2]

        # Initialize integer mask (H,W), 0=background
        mask = np.zeros((h, w), dtype=np.uint8)

        # Fill polygons by class in priority order: individual_tree first, group_of_trees second
        for cls_name in ["individual_tree", "group_of_trees"]:
            for ann in item.get("annotations", []):
                if ann["class"] != cls_name:
                    continue
                cls_id = CLASS_MAP[cls_name]
                poly = np.array(ann["segmentation"], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], cls_id)

        imgs.append(img)
        masks.append(mask)

    X = torch.tensor(np.stack(imgs)).permute(0,3,1,2).float()  # (N,3,H,W)
    y = torch.tensor(np.stack(masks)).long()                   # (N,H,W) int64
    return X, y