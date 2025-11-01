# Packages
from pathlib import Path
import pandas as pd
import numpy as np
import json
from PIL import Image
import pandas as pd
from datetime import datetime

# Create paths
REPO_ROOT = Path.cwd()

CURRENT_DATETIME = datetime.now().strftime('%Y-%m-%d %H:%M')

# Prediction Annotations
labels = REPO_ROOT / "runs/segment/pred_train_Yolo11s_canopy_832_adamW__20251101-0151_2025-10-31 20:36" / "labels" # Modify Prediction Txt annotations where necessary
IMGS = REPO_ROOT / 'data/processed/images/predict'
OUT_JSON = REPO_ROOT / 'exports'/ str(CURRENT_DATETIME) / 'submission.json'



#---Define Functions---#

# Image Metadata
def get_meta(IMGS):

    meta_data = {}
    scene_type_default = "Unknown"

    if IMGS.exists():

        for p in Path(IMGS).iterdir():
            if p.suffix.lower() not in {".tif"}: # All images .tif
                continue
            file_name = p.name
            cm_resolution = str(file_name[0:2])

            with Image.open(p) as image: # Pillow / Pill package
                width, height = image.size

            meta_data[file_name] = {
                'file_name' : file_name,
                "width" : width,
                'height' : height,
                "cm_resolution" : cm_resolution,
                "scene_type" : scene_type_default

            }
    else:
      print('Error - IMGs Not Exist')
    return meta_data

# Get Annotations
def get_annotations(LABELS):
    annotations = []

    for txt_file in sorted(Path(LABELS).iterdir()):
        
        if txt_file.is_file() and txt_file.suffix.lower() == ".txt":
            image_name = txt_file.name

            with open(txt_file, 'r') as file:
                for line in file:

                    # 1. Split the line into tokens
                    tokens = line.strip().split()

                    # 2. Convert tokens into usable pieces
                    class_id = int(tokens[0])                    # first token
                    *coords, confidence_level = map(float, tokens[1:])  
                    # all middle tokens are coords, last one is confidence
                    

                    # 3. Group coords into pairs (x,y)
                    segmentation = [
                        (coords[i], coords[i+1])
                        for i in range(0, len(coords), 2)
                    ]

                    # 4. Save everything into dict
                    annotations.append({
                        "image": image_name,
                        "class": class_id,
                        "confidence_level": confidence_level,
                        "segmentation": segmentation
                    })

    return annotations


# Denormalise segmentations to pixels

def denorm_segmentation(seg, width, height, *, flatten=True, round_int=True):
    """
    seg: list of (x,y) pairs in [0,1]
    returns: flat pixel list [x1,y1,x2,y2,...] (default) or list of pairs
    """
    # If seg is already flat [x1,y1,...], turn into pairs
    if seg and not isinstance(seg[0], (list, tuple)):
        it = iter(seg)
        seg_pairs = list(zip(it, it))
    else:
        seg_pairs = seg

    out = []
    for x, y in seg_pairs:
        X = max(0.0, min(width  - 1, x * width))
        Y = max(0.0, min(height - 1, y * height))
        if round_int:
            X = int(round(X))
            Y = int(round(Y))
        if flatten:
            out.extend([X, Y])
        else:
            out.append((X, Y))
    return out


# Append annotations to images

def append_imgs_annotations(img_list, annot_list):
    # ensure each image has an annotations list
    for img in img_list:
        if "annotations" not in img or img["annotations"] is None:
            img["annotations"] = []

    for image in img_list:
        w, h = image["width"], image["height"]
        fname = image["file_name"]

        # all annots for this image
        matched = (a for a in annot_list if a["image"] == fname)

        for a in matched:
            seg_px = denorm_segmentation(a["segmentation"], w, h, flatten=True, round_int=True) # Denormalise Segemetation to Pixels

            image["annotations"].append({
                "class": a["class"],
                "confidence_score": float(a["confidence_level"]),  # or keep as formatted string if you prefer
                "segmentation": seg_px
            })

    return img_list


# class id
ID_to_Name = {
    0 : 'individual_tree',
    1 : 'group_of_trees'
}



#--Procedure--#
print('Get Meta Data')
meta = get_meta(IMGS) # Retrieve Meta Data
meta_2items = list(meta.items())[:1]
meta_2keys = list(meta)[:1]

print(f'Meta Data Items : {meta_2items}')

print(f'Meta Data Keys : {meta_2items}')


images = [] # Produce Images list

print('Append Meta Data To Images List')
for m in meta.values(): # Append Meta Data to Images List
    images.append({
        'file_name' : m['file_name'],
        'width' : m['width'],
        'height' : m['height'],
        'cm_resolution' : m['cm_resolution'],
        'scene_type' : m['scene_type'],
        'annotations' : []
    })
    

images_sorted = sorted(images, key=lambda x: x['file_name']) # Sort Images
print(f'images_sorted ')
    
print('Get Annotations')
annotations_list = get_annotations(labels) # Retrieve Annotations & Denormalise Segmentations to Pixels

for annot in annotations_list: # Clean Annotations
    # .txt suffix -> .tif
    annot['image'] = annot['image'][:-4] + '.tif'

    # map class id -> name
    annot['class'] = ID_to_Name.get(annot['class'], 'Unknown')

    # convert confidence_level -> percentage string
    annot['confidence_level'] = f"{annot['confidence_level']:.2f}"

if annotations_list and len(annotations_list[0]) > 0:
    print('Annotations List Completed')

appended_list = append_imgs_annotations(images_sorted, annotations_list) # Append Annotations to Images

print(f'Appended_list Outputed ')

predict_answer = {}
predict_answer = {'images' : appended_list}
print(f'Predict_Answer Formulated')


# Assign Scene Type from Sample Answer to Submission

sample_answer_input = REPO_ROOT / 'exports/sample_answer.json' # Acquire Sample Answer path 

with open(sample_answer_input) as f: # Open Sample Answer Structure
    sample_answer = json.load(f)

sample_answer = sample_answer['images']


image_scenes = [] # Create list of images and associated 'scene_type'

for image in sample_answer:
    file_name = image.get('file_name') # GET each image's name
    scene_type = image.get('scene_type') # GET each image's scene_type

    image_scenes.append(  # Append to list
        {
            'filename':file_name, 
            'scene_type':scene_type
        }
    )


pd.json_normalize(image_scenes) #-- Potential redunant code --#

image_scenes_df = pd.DataFrame(image_scenes) # Transform Image_scenes list into DF

scene_map = image_scenes_df.set_index('filename')['scene_type'].to_dict() # Restructure image_scenes DF

submission = predict_answer

print('Append scene_types to predict_answer')
# Update each image in submission['images']
for image in submission['images']:
    filename = image['file_name']
    if image['scene_type'] == "Unknown" and filename in scene_map:
        image['scene_type'] = scene_map[filename]



print('Export Submission')
# Rewrite Submission JSON File
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(submission, f, indent=2)


print(f"Export Submission Complete \
      Saved As: 'Submission' \
      Saved At : {OUT_JSON}")
