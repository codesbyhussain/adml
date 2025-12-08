import json
import os
import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import cv2
import os

# Class mapping
CLASS_MAP = {
    "individual_tree": 1,
    "group_of_trees": 2
}

CLS_MAP = {
    0: "background",
    1: "individual_tree",
    2: "group_of_trees"
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
                # The coordinates of the polygons are converted to int32
                poly = np.array(ann["segmentation"], dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(mask, [poly], cls_id)

        imgs.append(img)
        masks.append(mask)

    X = torch.tensor(np.stack(imgs)).permute(0,3,1,2).float()  # (N,3,H,W)
    y = torch.tensor(np.stack(masks)).long()                   # (N,H,W) int64
    return X, y

def mask_to_polygons(pred_mask, probs, cls_map):
    """
    pred_mask: (H,W) predicted class ID map
    probs: (C,H,W) class probabilities
    cls_map: {class_id: class_name}
    """
    annotations = []

    for cls_id, cls_name in cls_map.items():
        # Skip background
        if cls_id == 0:
            continue

        binary_mask = (pred_mask == cls_id).astype(np.uint8)

        # Connected components = individual polygons
        num_labels, labels = cv2.connectedComponents(binary_mask)

        for label in range(1, num_labels):
            component = (labels == label).astype(np.uint8)

            # Find contours → polygon extraction
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                cnt = cnt.squeeze()
                if len(cnt.shape) != 2 or len(cnt) < 3:
                    continue

                polygon_flat = cnt.flatten().tolist()
                confidence = float(probs[cls_id][component == 1].mean())

                annotations.append({
                    "class": cls_name,
                    "confidence_score": confidence,
                    "segmentation": polygon_flat
                })

    return annotations

def safe_polygon(segmentation):
    poly = Polygon(np.array(segmentation).reshape(-1,2))
    if not poly.is_valid:
        poly = poly.buffer(0)  # fixes many invalid polygons
    return poly

class Evaluator:
    def __init__(self):
        pass

    def getIOU(self, poly1: Polygon, poly2: Polygon) -> float:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        if union == 0:
            return 0.0
        return inter / union
    
    def compute_map(self, gt_polygons: List[Polygon], pred_polygons_with_scores: List[Tuple[Polygon, float]], iou_thresholds: Union[List[float], float, None] = None) -> float:
        """
        Compute mean Average Precision (mAP) over a range of IoU thresholds.
        Requires predictions with confidence scores for ranking. Uses all-point interpolation (COCO standard).

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons_with_scores (List[Tuple[Polygon, float]]): A list of tuples, where each
                tuple contains a predicted Shapely Polygon object and its associated confidence score.
            iou_thresholds (Union[List[float], float, None], optional): The IoU threshold(s) over which
                to calculate Average Precision (AP) and then average for mAP.
                - If None, defaults to COCO standard [0.5, 0.55, ..., 0.95].
                - If a float, calculates AP at that single threshold.
                - If a list of floats, calculates AP for each and averages them.
                Defaults to None.

        Returns:
            float: The mean Average Precision (mAP) score. Returns 0.0 if no ground truths or
                   no predictions are provided.
        """
        if not gt_polygons: return 0.0
        if not pred_polygons_with_scores: return 0.0

        # Default IoU thresholds: COCO standard [0.5, 0.55, ..., 0.95]
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
        elif isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [float(iou_thresholds)]
        iou_thresholds = np.array(iou_thresholds)

        num_gt = len(gt_polygons)
        average_precisions = []

        # Sort predictions by confidence score (descending)
        sorted_preds = sorted(pred_polygons_with_scores, key=lambda x: x[1], reverse=True)
        pred_polygons_sorted = [p[0] for p in sorted_preds]

        # Calculate AP for each IoU threshold
        for iou_threshold in iou_thresholds:
            tp_list = []  # Stores 1 if prediction is TP, 0 if FP
            gt_matched_map = np.zeros(num_gt, dtype=bool) # Track matched GTs for this IoU threshold

            # Match sorted predictions to GTs
            for pred_idx, pred_polygon in enumerate(pred_polygons_sorted):
                best_iou = -1.0
                best_gt_idx = -1
                for gt_idx, gt_polygon in enumerate(gt_polygons):
                    if gt_matched_map[gt_idx]: # Skip already matched GT
                        continue
                    iou = self.getIOU(gt_polygon, pred_polygon)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # Assign TP/FP status based on match quality and availability
                if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched_map[best_gt_idx]:
                    tp_list.append(1)
                    gt_matched_map[best_gt_idx] = True # Mark GT as matched
                else:
                    tp_list.append(0) # FP

            # Calculate Precision-Recall curve points
            if not tp_list:
                 ap = 0.0
            else:
                tp_list = np.array(tp_list)
                fp_list = 1 - tp_list
                cumulative_tp = np.cumsum(tp_list)
                cumulative_fp = np.cumsum(fp_list)

                recalls = cumulative_tp / num_gt
                precisions = cumulative_tp / (cumulative_tp + cumulative_fp)

                # Calculate AP using All-Point Interpolation (Area under the P-R curve)
                recalls_interp = np.concatenate(([0.0], recalls, [recalls[-1]]))
                precisions_interp = np.concatenate(([0.0], precisions, [0.0]))
                # Make precision monotonically decreasing
                for i in range(len(precisions_interp) - 2, -1, -1):
                    precisions_interp[i] = max(precisions_interp[i], precisions_interp[i+1])
                # Calculate area under curve
                recall_change_indices = np.where(recalls_interp[1:] != recalls_interp[:-1])[0]
                ap = np.sum((recalls_interp[recall_change_indices + 1] - recalls_interp[recall_change_indices]) * precisions_interp[recall_change_indices + 1])

            average_precisions.append(ap)

        # mAP is the mean of APs over the IoU thresholds
        map_score = np.mean(average_precisions) if average_precisions else 0.0
        return map_score
    

def postprocess_polygons(annotations, min_area=50, conf_thresh=0.3, epsilon_ratio=0.01):
    """
    Clean up predicted polygons.
    
    Args:
        annotations (list): List of dicts from mask_to_polygons(), each with keys:
                            'class', 'confidence_score', 'segmentation'.
        min_area (float): Minimum area to keep a polygon.
        conf_thresh (float): Minimum confidence score to keep a polygon.
        epsilon_ratio (float): Ratio for cv2.approxPolyDP simplification.
        
    Returns:
        List of cleaned annotations.
    """
    clean_annotations = []
    
    for ann in annotations:
        if ann['confidence_score'] < conf_thresh:
            continue  # discard low confidence
        
        seg = np.array(ann['segmentation']).reshape(-1, 2)
        if len(seg) < 3:
            continue  # not enough points for a polygon
        
        # Simplify polygon
        epsilon = epsilon_ratio * cv2.arcLength(seg, True)
        approx = cv2.approxPolyDP(seg, epsilon, True)
        approx = approx.reshape(-1, 2)
        
        poly = Polygon(approx)
        if not poly.is_valid:
            poly = poly.buffer(0)  # fix minor self-intersections
        
        if poly.area < min_area:
            continue  # discard tiny polygons
        
        clean_annotations.append({
            'class': ann['class'],
            'confidence_score': ann['confidence_score'],
            'segmentation': approx.flatten().tolist()
        })
    
    return clean_annotations


def visualize_and_save(
    image,
    pred_mask,
    save_path,
    gt_mask=None,
    pred_polygons=None,
    gt_polygons=None
):
    """
    Saves a 4-panel visualization figure to disk.
    """

    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert image tensor to numpy
    img = image.permute(1,2,0).cpu().numpy()

    # Scale image to 0-255 if not already
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    H, W = pred_mask.shape

    # Color map for classes
    colors = {
        0: (0, 0, 0),         # background - black
        1: (0, 255, 0),       # individual_tree - green
        2: (255, 0, 0)        # group_of_trees - blue
    }

    # Pred mask colored
    pred_color = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in colors.items():
        pred_color[pred_mask == cls] = color

    # GT mask colored
    gt_color = None
    if gt_mask is not None:
        gt_color = np.zeros((H, W, 3), dtype=np.uint8)
        for cls, color in colors.items():
            gt_color[gt_mask == cls] = color

    # Image with polygons
    img_poly = img.copy()
    if pred_polygons is not None:
        for poly in pred_polygons:
            img_poly = draw_polygon_on_image(img_poly, poly, color=(0,255,255), thickness=2)  # cyan

    if gt_polygons is not None:
        for poly in gt_polygons:
            img_poly = draw_polygon_on_image(img_poly, poly, color=(255,255,0), thickness=2)  # yellow

    # Plot 4 panels
    plt.figure(figsize=(15,8))

    plt.subplot(1,4,1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.title("Pred Mask")
    plt.imshow(pred_color)
    plt.axis("off")

    if gt_color is not None:
        plt.subplot(1,4,3)
        plt.title("GT Mask")
        plt.imshow(gt_color)
        plt.axis("off")

    plt.subplot(1,4,4)
    plt.title("Polygons")
    plt.imshow(img_poly)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()



def draw_polygon_on_image(img, poly, color, thickness=2):
    """Draws a Polygon or MultiPolygon on the image."""
    if poly.is_empty:
        return img

    if poly.geom_type == "Polygon":
        pts = np.array(list(poly.exterior.coords), dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thickness)
        for interior in poly.interiors:
            pts = np.array(list(interior.coords), dtype=np.int32)
            cv2.polylines(img, [pts], True, color, thickness)

    elif poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            img = draw_polygon_on_image(img, p, color, thickness)

    return img
