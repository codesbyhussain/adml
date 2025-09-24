# Solafune Tree Canopy Detection Capstone

## Project Overview  
The project involved building a geospatial ML pipeline using Sentinel-2 imagery to detect tree canopies via image segmentation. Hosted by Solafune, I managed data imports of image segmentations, trained a segmentation model, and produce a competitive-ready submission in the Solafune Tree Canopy Detection challenge.

This pipeline runs on both local environments and Google Colab with minimal path changes, enabling access to GPU acceleration for faster training.

### Motivation
Accurate tree canopy mapping supports urban planning, biodiversity conservation, and climate modeling. Participating in this challenge develops robust skills while yielding practical impacts while building skillsets in geospatial machine learning.

This project was also my first application of YOLO-based image segmentation and geospatial data processing, providing valuable experience working with Earth Observation data formats.

### Results
My current submission model, as of **September 24th 2025**, places in the top 10 of 271 competitors on Solafune leaderboard, against the assesment criteria of >75% mean IoU on the validation dataset.

### Lessons Learned
- Importance of label formatting and format conversion (COCO ↔ YOLO)
- Value of consistent directory structure for large-scale ML projects
- Benefits of augmentation techniques like HSV rotation and color grading


### Tools
- YOLO Machine Learning Model
    - image segmentation
- Python 3.10
    - base language for building the pipeline and running scripts
    - Tasks
        - geospatial + ML pipeline
- PyTorch - deep learning framework used to train segmentation models

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
│   │ │  └── test/                        # Ground Truth Data Split for model testing  
│   │ ├── labels/
│   │ │  ├── train/                       # Ground Truth Labels Split for model training  
│   │ │  ├── val/                         # Ground Truth Labels Split for model valdiation
│   │ │  └── test/                        # Ground Truth Labels Split for model testing
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
│   ├── 03_test_model.py                    # Test/Valdiate YOLO Model on GT Data
│   ├── 05_predict_model.py                 # Create predictions with trained YOLO Model on no GT Data  
│   └── 06_export_submission.py             # Convert prediction outputs into Solafune JSON format
│
├── runs/segments/                          # All model training/validation/prediction results
├── exports/                                # JSON Submission files
├── README.md                               # This file
└── requirements.txt                        # Package requirements
```

## Process

### Data
- Data Not Sharable by Solafune Non-Disclosure Agreement.
<!-- - 
<https://drive.google.com/drive/folders/1sB7XVJuFYcJCqzbiHcxKC96WAWCKo3Zj?usp=drive_link> -->

### Run Order

1. 01_data_preparation.ipynb
↓
2. 02_train_model.py
↓
3. 03_test_model.py
↓
4. (Optional) 04_test_model_evaluations.ipynb
↓
5. 05_predict_model.py
↓
6. 06_export_submission.py

### Files
- Run Notebooks & Scripts in sequence: 

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
            - Testing
            - Validation 

    - 02_train_model.py
        - Modify train model parameters YAML file for desired training and naming
            - Train Model Parameters
        - Select YOLO Model Version
            - [scripts/02_train_model.py](scripts/02_train_model.py) — Line 21  
                 - Where to select pretrained model for training
        
    - 03_test_model.py
        - Modify test model parameters YAML file for fine tuning model
            - Test Model Parameters
        - Select Models Weights for Validation
            - [scripts/03_test_model.py](scripts/03_test_model.py) — Line 23  
                 - Where to select trained model weights for evaluation

    - 04_test_model_evaluations.ipynb | **Optional**
        - Optional Unfinished Notebook file. Containing indepth measures to analyse test split data validation.

    - 05_predict_model.py
        - Modify predict model parameters YAML file for final model predictions
            - Predict Model Parameter
        - Select Models Weights for Prediction
            - [scripts/05_predict_model.py](scripts/05_predict_model.py) — Line 23  
                 - Where to select trained/validated model weights for prediction

    - 06_export_submission.py
        - Select Predict Models Annotations for Submission Jile
            - [scripts/06_export_submission.py](scripts/06_export_submission.py) - Line 16
                - Where to select predicted model outputs to define prediction labels.txt for submission


## **Optional** Google Colab 
Relative pathways and constructed Google Colab .ipynb file to utilise Google Colab GPUs.

To take advantage of Googel Colab's free or paid GPU, follow the below Google Drive Directory Structure and Google Colab Juptyer Notebook file.

### Required Google Drive Directory Structure
```
├── drive/                              # Drive Base
│   ├── MyDrive/                        # Google Drive
│   │   ├── Datasets/                   # Host of all Datasets
│   │   │  ├── solafune-tree-canopy/                   
│   │   │   │   ├── data/               # Data Folder Structure 
│   │   │   │   │   ├── processed/
│   │   │   │   │   │   ├── labels/
│   │   │   │   │   │   │   ├── train/
│   │   │   │   │   │   │   ├── test/
│   │   │   │   │   │   │   └── val/
│   │   │   │   │   │   └── images/
│   │   │   │   │   │       ├── train/
│   │   │   │   │   │       ├── test/
│   │   │   │   │   │       └── val/
│   │   │   │   │   │
│   │   │   │   │   ├── temp/
│   │   │   │   │   └──  raw/
│   │   │   │   │      ├── zips/
│   │   │   │   │      └── JSONs/
│   │   │   │   │
│   │   │   │   ├── runs/
│   │   │   │   │   ├── segments/      # All model training/validation/prediction results
│   │   │   │   │   └── notebooks/
│   │   │   │   │
│   │   │   │   └──   outputs/
│   │   │   │       └── exports/
```

### Files

- Run Notebook
    - [solafune_tree_canopy.ipynb](https://colab.research.google.com/drive/1KrtNSr8aHL5j8dGBrMzNdlHEesKB712Z?usp=drive_link)


## License
MIT License


