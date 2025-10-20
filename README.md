# Tree Canopy and Individual Tree Detection

This repository contains projects for instance segmentation of trees from aerial images using UNet and YOLOv8-segmentation models. It includes preprocessing, training, evaluation, and generating JSON outputs compatible with custom formats.

# Table of Contents

## Project Overview

## Folder Structure

Installation

Data Preparation

Model Training

UNet

YOLO

Inference & Predictions

Visualization

JSON Output Format

References

Project Overview

The goal of this project is to detect and segment:

Individual trees

Groups of trees

using aerial imagery. Two models are used:

UNet: Generates pixel-wise segmentation masks.

YOLOv8-seg: Generates instance segmentation with polygon outputs.

Predictions are stored in JSON files compatible with both your evaluation pipeline and competition formats.

## Folder Structure

project/
│
├─ data/
│ ├─ raw/
│ │ ├─ train_images/
│ │ ├─ test_images/
│ │ └─ train_annotations.json
│ ├─ processed_yolo/
│ │ ├─ images/
│ │ └─ labels/
│ └─ test/evaluation_images/
│
├─ notebooks/
│ ├─ unet_train_image_prediction.ipynb
│ ├─ yolo_train_image_prediction.ipynb
│ └─ visualization.ipynb
│
├─ requirements.txt
└─ README.md

Installation

Clone this repository:
git clone <repo_url>
cd project

Create a virtual environment:
python -m venv venv
source venv/bin/activate # Linux / Mac
venv\Scripts\activate # Windows

Install dependencies:
pip install -r requirements.txt

Data Preparation

Place all raw images in data/raw/train_images or data/raw/test_images.

Annotations for training are provided in JSON format (train_annotations.json).

Use the provided preprocessing scripts to convert annotations to YOLO format (labels/*.txt) if needed.

Model Training

UNet

Preprocessing: Apply augmentations: Horizontal/Vertical flips, rotations, brightness/contrast, and normalization.

Train UNet model for semantic segmentation.

Save trained model weights for inference.

YOLO

Convert raw annotations to YOLOv8-seg format.

Train using ultralytics.YOLO:
from ultralytics import YOLO
model = YOLO('yolov8s-seg.pt')
model.train(data='data/processed_yolo/dataset.yaml', epochs=100, imgsz=640, batch=8)

Trained model weights are saved in runs/train/<experiment_name>/weights/best.pt.

Inference & Predictions

UNet Predictions: Outputs JSON with polygons for individual trees and group of trees.

YOLO Predictions: Outputs instance segmentation JSON in same format.

Example structure for prediction JSON:

{
"file_name": "10cm_train_1.tif",
"width": 1024,
"height": 1024,
"cm_resolution": 10,
"scene_type": "agriculture_plantation",
"annotations_unet": [
{
"class": "individual_tree",
"confidence_score": 0.99,
"segmentation": [x1, y1, x2, y2, x3, y3, x4, y4]
}
]
}

Visualization

Overlay ground truth, UNet predictions, and YOLO predictions on the original image.

Different colors are used:

Individual trees: Green

Groups of trees: Purple

Sample code in visualization.ipynb provides plotting for single or multiple images.

JSON Output Format

Each image has:

file_name

width and height

annotations_unet or annotations_yolo

Each annotation has:

class

confidence_score

segmentation (polygon coordinates, 4 points for bounding approximation)

References

Ultralytics YOLOv8 Documentation: https://docs.ultralytics.com/

Albumentations for data augmentations: https://albumentations.ai/

Shapely for polygon operations: https://shapely.readthedocs.io/

Scikit-image for region properties and masks: https://scikit-image.org/