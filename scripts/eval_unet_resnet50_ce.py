import os
import segmentation_models_pytorch as smp
import torch
import glob
import cv2
import json
from shapely.geometry import Polygon
import numpy as np
from typing import List, Tuple

# Quick inference example
output_dir = '../outputs/'
os.makedirs(output_dir, exist_ok=True)

model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=3,
    classes=3
).cuda()

# Load the saved weights
model.load_state_dict(torch.load("../checkpoints/unet_resnet50_weights.pth"))

model.eval()
test_image_paths = sorted(glob.glob('../data/val/images/*.tif'))

with torch.no_grad():
    for path in test_image_paths:
        img = cv2.imread(path)[:, :, ::-1] / 255.0
        img_tensor = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().cuda()
        logits = model(img_tensor)
        pred = torch.argmax(logits, dim=1)[0].cpu().numpy().astype(np.uint8)  # (H,W)

        # Save mask (0=background, 1=individual_tree, 2=group_of_trees)
        filename = os.path.basename(path)
        cv2.imwrite(os.path.join(output_dir, filename), pred * 127)  # multiply to visualize

# -----------------------------
# Helper: Load GT polygons from JSON
# -----------------------------
def gt_polygons_from_json(json_path, class_name=None):
    """
    Returns a dict: image_file_name -> list of Shapely Polygons
    """
    with open(json_path) as f:
        data = json.load(f)
    all_polygons = {}
    for item in data['images']:
        image_id = item['file_name']
        polys = []
        for ann in item.get('annotations', []):
            if class_name is None or ann['class'] == class_name:
                poly = np.array(ann['segmentation']).reshape(-1,2)
                polys.append(Polygon(poly))
        all_polygons[image_id] = polys
    return all_polygons

# -----------------------------
# Helper: Convert predicted mask to polygons
# -----------------------------
def pred_polygons_with_confidence(mask_logits: np.ndarray, class_ids=[1,2], threshold=0.5):
    """
    Convert UNet logits to polygons with confidence scores.

    Args:
        mask_logits: np.ndarray of shape (C,H,W) — raw UNet logits for each class (C=3 here)
        class_ids: list of class indices to extract polygons for (skip background=0)
        threshold: minimum probability to include a pixel in polygon

    Returns:
        List of tuples: [(Polygon, confidence_score), ...]
    """
    # Convert logits to probabilities using softmax
    exp_logits = np.exp(mask_logits - np.max(mask_logits, axis=0, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)  # (C,H,W)

    polygons_with_scores = []

    for cls_id in class_ids:
        # binary mask for pixels with prob > threshold
        cls_mask = (probs[cls_id] > threshold).astype(np.uint8)

        # find contours
        contours, _ = cv2.findContours(cls_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) >= 3:
                poly = Polygon(cnt.squeeze())
                if poly.is_valid and poly.area > 0:
                    # Step 1: create uint8 mask for cv2.fillPoly
                    mask_inside = np.zeros_like(cls_mask, dtype=np.uint8)
                    
                    # Step 2: fill the polygon
                    cv2.fillPoly(mask_inside, [cnt], 1)
                    
                    # Step 3: convert to bool for indexing into probabilities
                    mask_inside_bool = mask_inside.astype(bool)
                    
                    confidence = probs[cls_id][mask_inside_bool].mean()
                    polygons_with_scores.append((poly, float(confidence)))
    return polygons_with_scores

def getIOU(self, poly1, poly2):
    """Compute IoU of two Shapely polygons."""
    if not poly1.is_valid or not poly2.is_valid:
        return 0.0
    inter = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0.0
    return inter / union

def compute_map(gt_polygons: List[Polygon], 
                pred_polygons_with_scores: List[Tuple[Polygon, float]], 
                iou_thresholds=None) -> float:
    """
    Compute mAP over IoU thresholds (COCO-style). Uses all-point interpolation.
    """
    if not gt_polygons:
        return 0.0
    if not pred_polygons_with_scores:
        return 0.0

    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
    elif isinstance(iou_thresholds, (float,int)):
        iou_thresholds = [float(iou_thresholds)]
    iou_thresholds = np.array(iou_thresholds)

    num_gt = len(gt_polygons)
    average_precisions = []

    # Sort predictions by confidence descending
    sorted_preds = sorted(pred_polygons_with_scores, key=lambda x: x[1], reverse=True)
    pred_polygons_sorted = [p[0] for p in sorted_preds]

    for iou_threshold in iou_thresholds:
        tp_list = []
        gt_matched_map = np.zeros(num_gt, dtype=bool)

        for pred_polygon in pred_polygons_sorted:
            best_iou = -1
            best_gt_idx = -1
            for gt_idx, gt_polygon in enumerate(gt_polygons):
                if gt_matched_map[gt_idx]:
                    continue
                iou = getIOU(None, gt_polygon, pred_polygon)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp_list.append(1)
                gt_matched_map[best_gt_idx] = True
            else:
                tp_list.append(0)

        if not tp_list:
            ap = 0.0
        else:
            tp_list = np.array(tp_list)
            fp_list = 1 - tp_list
            cum_tp = np.cumsum(tp_list)
            cum_fp = np.cumsum(fp_list)
            recalls = cum_tp / num_gt
            precisions = cum_tp / (cum_tp + cum_fp)

            # All-point interpolation
            recalls_interp = np.concatenate(([0.0], recalls, [recalls[-1]]))
            precisions_interp = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(len(precisions_interp)-2, -1, -1):
                precisions_interp[i] = max(precisions_interp[i], precisions_interp[i+1])
            idx = np.where(recalls_interp[1:] != recalls_interp[:-1])[0]
            ap = np.sum((recalls_interp[idx+1]-recalls_interp[idx]) * precisions_interp[idx+1])

        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0

# -----------------------------
# Main evaluation function
# -----------------------------
def evaluate_map(model, X_val, val_json_path, compute_map_fn, threshold=0.5):
    """
    Evaluate mAP of a trained UNet on validation data using COCO-style polygon evaluation.
    
    Args:
        model: trained UNet
        X_val: tensor of validation images (N,3,H,W)
        val_json_path: path to JSON containing ground truth
        compute_map_fn: your compute_map method
        threshold: minimum probability to include a pixel in a predicted polygon

    Returns:
        overall mAP score (float)
    """
    model.eval()
    gt_dict = gt_polygons_from_json(val_json_path)
    map_scores = []

    with torch.no_grad():
        for idx, (img_tensor, (image_file, gt_polys)) in enumerate(zip(X_val, gt_dict.items())):
            img_tensor = img_tensor.unsqueeze(0).cuda()  # add batch dim

            # Forward pass
            logits = model(img_tensor)[0].cpu().numpy()  # (C,H,W)

            # Convert logits to polygons with confidence scores
            pred_polys = pred_polygons_with_confidence(logits, class_ids=[1,2], threshold=threshold)

            # Compute mAP for this image
            map_score = compute_map_fn(gt_polys, pred_polys)
            map_scores.append(map_score)

    overall_map = np.mean(map_scores) if map_scores else 0.0
    return overall_map

# overall_map = evaluate_map(model, X_val, val_masks_path, compute_map_fn=compute_map, threshold=0.5)
# print("Initial mAP:", overall_map)