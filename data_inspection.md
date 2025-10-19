Phase 0: data inspection

training images: 3MB, resolution in 10cm x37, 20cm x38, 40cm x25, 60cm x25, 80cm x25
Validation images: 3MB, resolution in 10cm x38, 20cm x37, 40cm x25, 60cm x 25, 80cm x25

Images info:
Driver: GTiff
Width: 1024
Height: 1024
Band count: 3
Data type: ('uint8', 'uint8', 'uint8')
Coordinate Reference System (CRS): None
Transform: | 1.00, 0.00, 0.00|
| 0.00, 1.00, 0.00|
| 0.00, 0.00, 1.00|
Metadata tags: {'TIFFTAG_IMAGEDESCRIPTION': '{"shape": [1024, 1024, 3]}', 'TIFFTAG_SOFTWARE': 'tifffile.py', 'TIFFTAG_XRESOLUTION': '1', 'TIFFTAG_YRESOLUTION': '1', 'TIFFTAG_RESOLUTIONUNIT': '1 (unitless)'}
Scene types: agriculture_plantation, rural_area, urban_area, open_field, industrial_area

Label info:
{
    "images": [
        {
            "file_name": "10cm_train_1.tif",
            "width": 1024,
            "height": 1024,
            "cm_resolution": 10,
            "scene_type": "agriculture_plantation",
            "annotations": [
                {
                    "class": "individual_tree",
                    "confidence_score": 1.0,
                    "segmentation": [
                        4.0,
                        146.0,
                        18.0,
                        149.0,
                        17.0,
                        160.0,
                        6.0,
                        162.0,
                        0.0,
                        152.0
                    ]
                },
                .....]}]}

The segmentation are using polygons.
classes:
individual_tree
group_of_trees

Confidence score all 1 in training but not all 1 in validation.