import matplotlib
import cv2, matplotlib.pyplot as plt
import rasterio
import json
import numpy as np

path = '../data/train/images/10cm_train_1.tif'

#look at the image
# img = cv2.imread(path)
#cv2 always convert to BGR, so need to convert back
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1)
plt.imshow(img)
plt.show()

#mata data of the image
with rasterio.open(path) as src:
    print("Driver:", src.driver)
    print("Width:", src.width)
    print("Height:", src.height)
    print("Band count:", src.count)
    print("Data type:", src.dtypes)
    print("Coordinate Reference System (CRS):", src.crs)
    print("Transform:", src.transform)
    print("Metadata tags:", src.tags())

#look at the mask
with open('../data/train/masks/train_annotations.json', 'r') as file:
    data = json.load(file)

print(data["images"][0]["file_name"])
for ann in data["images"][0]["annotations"]:
    seg = np.array(ann["segmentation"]).reshape(-1, 2).astype(np.int32) #opencv requires integers, just looking here anyways
    cv2.polylines(img, [seg], isClosed=True, color=(0, 255, 0), thickness=1)
plt.imshow(img)
plt.show()

