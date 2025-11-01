# Solafune Tree Canopy Detection Capstone

## Project Overview  
The project involved building a geospatial ML pipeline using Sentinel-2 imagery to detect tree canopies via image segmentation. Hosted by Solafune, I managed data imports of image segmentations, trained a segmentation model, and produced a competition-ready submission in the Solafune Tree Canopy Detection challenge.

This pipeline runs on both local environments and Google Colab with minimal path changes, enabling access to GPU acceleration for faster training.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge helped develop strong skils in GIS data while yielding practical impacts while building skillsets in geospatial machine learning.

This project was also my first application of YOLO-based image segmentation and geospatial data processing, providing valuable experience working with Earth Observation data formats.

### Results
My current submission model, as of **September 24th 2025**, places in the top 10 of 271 competitors on [Solafune leaderboard](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?modal=%22%22&menu=lb&tab=public), against the assesment criteria of >75% mean IoU on the prediction dataset.

### Lessons Learned
- JSON label formatting and format conversion (COCO ↔ YOLO)
- Consistent directory structure for large-scale ML projects across local and digitial directories.
- Benefits of augmentation techniques like HSV rotation and color grading
- Visual data formatting. 
- Convolutions and convolutional neural networks.


### Tools
- YOLO Machine Learning Model
    - image segmentation
- Python 3.10
    - base language for building the pipeline and running scripts
    - Tasks
        - geospatial + ML pipeline
- PyTorch 
    - deep learning framework used to train segmentation models
- Google Colab
    - Alternative notebook for accessing Google Colab's GPUs
- JSON, COCOJson
    - Retrieval and submission of labels.
- YAML
    - Organised model parameter files
- GIS Toolkits

## Installation  
```bash
git clone <https://github.com/Mitch-P-Analyst/solafune-canopy-capstone.git>
cd solafune-tree-canopy
pip install -r requirements.txt
```
## Repo Directory Structure  
```
├── configurations/                     # YAML configs (data + overrides)
│   ├── model_data-seg.yaml                 # dataset paths & class names
│   ├── train_model_overrides.yaml          # training parameters
│   ├── val_model_overrides.yaml            # validation parameters
│   └── predict_model_overrides.yaml        # prediction parameters
│
├── data/                               # Downloaded satellite imagery and mosaics
│   ├── processed/                       
│   │ ├── images/
│   │ │  ├── predict/                     # Unlabeled (no Ground Truth) data for prediction 
│   │ │  ├── train/                       # Ground Truth Data Split for model training  
│   │ │  ├── val/                         # Ground Truth Data Split for model valdiation  
│   │ │  └── test/                        # Required by YOLO structure
│   │ ├── labels/
│   │ │  ├── train/                       # Ground Truth Labels Split for model training  
│   │ │  ├── val/                         # Ground Truth Labels Split for model valdiation
│   │ │  └── test/                        # Required by YOLO structure
│   │ └── JSONs/                          # Converted JSON file
│   ├── raw/
│   │ ├── zips/                           # Raw Data ZIP files | **Restricted by NDA**
│   │ └── JSONs/
│   └── temp/
│
├── notebooks/                          
│   ├── 01_data_preparation.ipynb           # Convert JSONs, Unzip, Split Data
│   └── 04_test_model_evaluations.ipynb     # **Optional** Indepth model evaluations
│
├── scripts/                                
│   ├── 02_train_model.py                   # Train YOLO Model
│   ├── 03_val_model.py                     # Valdiate YOLO Model on GT Data
│   ├── 05_predict_model.py                 # Create predictions with trained YOLO Model on no GT Data  
│   └── 06_export_submission.py             # Convert prediction outputs into Solafune JSON format
│
├── runs/segments/                          # All model training/validation/prediction results
├── exports/                                # JSON Submission files
├── README.md                               # This file
├── README.html                             # README in HTML format for digital portfolio
└── requirements.txt                        # Package requirements
```

## Process

### Data
- Data Not Sharable by Solafune Non-Disclosure Agreement.
    - To access data, visit Solafune compeition webpage [Tree Canopy Detection](https://solafune.com/competitions/26ff758c-7422-4cd1-bfe0-daecfc40db70?menu=data&tab=&modal=%22%22) 
<!-- <https://drive.google.com/drive/folders/1sB7XVJuFYcJCqzbiHcxKC96WAWCKo3Zj?usp=drive_link> -->

### Files & Run Order

1. Data Preparation
    - 01_data_preparation.ipynb
        - JSON conversion 
        - Solafune format -> COCO format
        - COCO format -> YOLO format
        - Unpacking Raw Data
            - Extract ZIP files
                - Training
                - Prediction
        - Data Split Images & Annotations
            - Training
            - Validation 

2. Model Training
    - Two options for training model on **Local Device** or **Google Colab GPUs**
        - Local Device
            -  02_train_model.py
                - Modify train model parameters YAML file for desired training and naming
                    - Train Model Parameters
                - Select YOLO Model Version
                    - [scripts/02_train_model.py](scripts/02_train_model.py) — Line 21  
                        - Where to select pretrained model for training   
        - Google Colab    
            - [![ Open In Colab ](https://colab.research.google.com/assets/colab-badge.svg)]
                - [solafune_tree_canopy.ipynb](https://colab.research.google.com/drive/1KrtNSr8aHL5j8dGBrMzNdlHEesKB712Z?usp=drive_link)

3. Model Validation
    - 03_val_model.py
        - Select Model Weights from trained YOLO model
            - Trained Model Weights selection
        - Modify validation model parameters YAML file for fine tuning model
            - Validation Model Parameters

4. Metric Testings
    - **(Optional)** 04_test_model_evaluations.ipynb
        - Optional Unfinished Notebook file. Containing indepth measures to analyse split data validation.

5. Predictions
    - 05_predict_model.py
        - Select Model Weights from trained YOLO model
            - Trained Model Weights selection
        - Modify prediction model parameters YAML file for final model deployment
            - Prediction Model Parameters

6. Export
    - 06_export_submission.py
        - Select Predict Models Annotations for Submission Jile
            - [scripts/06_export_submission.py](scripts/06_export_submission.py) - Line 16
                - Where to select predicted model outputs to define prediction labels.txt for submission


## License
MIT License


