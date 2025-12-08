problem:
after 10 epochs, validation loss does not reduce and stays around 0.35
Epoch 39/40 - Train Loss: 0.1242 | Val Loss: 0.3299
mAP is 0
analysis: 120 images for training too small, easy to overfit

try data augmentation:
You applied on-the-fly data augmentation to your 120 training images using Albumentations: each image can be randomly flipped horizontally or vertically, rotated by 90° increments, and have brightness/contrast adjusted, while being converted to PyTorch tensors; this effectively increases the diversity of the training data each epoch without saving additional images, while the validation set remains unaugmented (only converted to tensors) to ensure consistent evaluation.

still:
Epoch 33/40 - Train Loss: 0.2350 | Val Loss: 0.3098
Epoch 34/40 - Train Loss: 0.2257 | Val Loss: 0.3003
Epoch 35/40 - Train Loss: 0.2287 | Val Loss: 0.3125
Epoch 36/40 - Train Loss: 0.2265 | Val Loss: 0.3110
Epoch 37/40 - Train Loss: 0.2322 | Val Loss: 0.3169
Epoch 38/40 - Train Loss: 0.2355 | Val Loss: 0.3043
Epoch 39/40 - Train Loss: 0.2282 | Val Loss: 0.3299
Epoch 40/40 - Train Loss: 0.2229 | Val Loss: 0.3128
mAP: 0

Smaller learning rate?
1e-4 -> 5e-5
does not help.

fix index issure, mAP finally not 0: 
Overall validation mAP: 0.0338
val_indices = val_dataset.indices  # original dataset indices for validation
original_idx = val_indices[idx]
gt_polys = gt_polygons_per_image[original_idx]

Reduce val loss:0.3
weights = torch.tensor([0.2, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
worse
try: weights = torch.tensor([0.8, 1.0, 1.1]).to(device)
not good

Normalize to imagenet weights
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std  = (0.229, 0.224, 0.225)
A.Normalize(mean=imagenet_mean, std=imagenet_std)
worse

Don't focus on reducing validation loss, we can have bad loss and good mAP.