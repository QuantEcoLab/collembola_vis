# This code prepares and visualizes dataset for collembola AI detection research
# Authors: [Jana Zovko, Domagoj K. Hackenberger]
# Based on code snippets in jana_code directory


import pandas as pd
import skimage as sk
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
import json

# Load and display the image
image_path = "data/slike/K1_Fe2O3001 (1).jpg"
image = sk.io.imread(image_path)

plt.figure(figsize=(10, 10))
plt.imshow(image)

# Load the Collembola dataset
collembola_df = pd.read_csv("data/collembolas_table.csv")

# Load only collembolas from target image (partial match)
target_collembola_df = collembola_df[collembola_df["id_collembole"].str.contains("K1_Fe2O3001", na=False)]

# Compute center coordinates for each Collembola annotation
if all(col in target_collembola_df.columns for col in ['x', 'y', 'w', 'h']):
    target_collembola_df = target_collembola_df.copy()
    target_collembola_df['center_x'] = target_collembola_df['x'] + target_collembola_df['w'] / 2
    target_collembola_df['center_y'] = target_collembola_df['y'] + target_collembola_df['h'] / 2
else:
    raise ValueError("CSV must contain 'x', 'y', 'w', 'h' columns for bounding boxes.")

# function to detect potential Collembola in the image
# this is going to be reused in the final program
def find_objects(img, visualise=False):
    # This function detects potential collembola in the image
    # It uses image processing techniques to find "circular" objects
    # and returns their properties for further analysis.
    # This part is from circles.py
    gray = rgb2gray(img)
    gray_smooth = gaussian(gray, sigma=2)
    thresh = threshold_otsu(gray_smooth)
    binary = gray_smooth > thresh
    binary = remove_small_objects(binary, min_size=20)
    binary = clear_border(binary)
    labels = label(binary)
    props = regionprops(labels)
    if visualise:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(img)
        for region in props:
            if region.area < 10:
                continue
            y, x = region.centroid
            coords = region.coords
            distances = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
            r = distances.max() + 1  
            circ = Circle(
                (x, y), r,
                edgecolor="red",
                facecolor="red",
                alpha=0.2,
                linewidth=1
            )
            ax.add_patch(circ)
        plt.show()

    return props

detected_objects = find_objects(image)

# crossreference detected objects with the target collembola dataframe
# count the number of collembola detected in the image
n=0
yellow_x = []
yellow_y = []
for obj in detected_objects:
    y, x = obj.centroid
    radius = obj.equivalent_diameter / 2
    matching_collembola = target_collembola_df[
        (target_collembola_df["center_x"] >= x - radius) & 
        (target_collembola_df["center_x"] <= x + radius) &
        (target_collembola_df["center_y"] >= y - radius) & 
        (target_collembola_df["center_y"] <= y + radius)
    ]
    
    if not matching_collembola.empty:
        print(f"Detected Collembola at ({x}, {y}) with radius {radius}:")
        n += 1
        print(matching_collembola)
        yellow_x.append(x)
        yellow_y.append(y)
    else:
        print(f"No matching Collembola found for detected object at ({x}, {y}) with radius {radius}.")
        
print(f"Total Collembola detected in the image: {n}")

# Find unmatched annotation centers
matched_indices = set()
for obj in detected_objects:
    y, x = obj.centroid
    radius = obj.equivalent_diameter / 2
    matches = target_collembola_df[
        (target_collembola_df["center_x"] >= x - radius) & 
        (target_collembola_df["center_x"] <= x + radius) &
        (target_collembola_df["center_y"] >= y - radius) & 
        (target_collembola_df["center_y"] <= y + radius)
    ]
    matched_indices.update(matches.index.tolist())

unmatched = target_collembola_df[~target_collembola_df.index.isin(matched_indices)]

# Plot yellow dots for detected Collembola, red dots for missed, and all detected circles
plt.figure(figsize=(10, 10))
plt.imshow(image)
# Draw all detected circles
for obj in detected_objects:
    y, x = obj.centroid
    r = obj.equivalent_diameter / 2
    circ = Circle((x, y), r, edgecolor="red", facecolor="none", linewidth=1, alpha=0.7)
    plt.gca().add_patch(circ)
# Draw yellow dots for matched detections
plt.scatter(yellow_x, yellow_y, c='yellow', s=40, marker='o', label='Detected Collembola')
# Draw red dots for unmatched annotation centers
plt.scatter(unmatched['center_x'], unmatched['center_y'], c='red', s=40, marker='x', label='Missed Collembola')
plt.legend()
plt.title('Detected (yellow) and Missed (red) Collembola Centroids with All Detected Circles')
plt.show()


# 522/639 81.7% of Collembola were detected in the image
# No need to be alarmed, this will be improved with deep learning


# For each suspected object (collmbola and not) create a fixed bbox with size 
# of larges collembola in the image

# crop the image to the bounding box of each detected object
def crop_to_bbox(img, bbox):
    x, y, w, h = bbox
    return img[y:y+h, x:x+w]

def crop_to_bbox_with_padding(img, center_x, center_y, crop_w, crop_h):
    H, W = img.shape[:2]
    x1 = int(center_x - crop_w // 2)
    y1 = int(center_y - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h
    # Calculate valid region in original image
    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(W, x2)
    src_y2 = min(H, y2)
    # Calculate where to place it in the crop
    dst_x1 = src_x1 - x1
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)
    crop = np.zeros((crop_h, crop_w, img.shape[2]), dtype=img.dtype)
    crop[dst_y1:dst_y2, dst_x1:dst_x2] = img[src_y1:src_y2, src_x1:src_x2]
    return crop

# Create a directory to save cropped images
output_dir = Path("data/dataset_crops")
output_dir.mkdir(parents=True, exist_ok=True)

# Find the largest bbox size in the whole collembola table
max_w = collembola_df['w'].max()
max_h = collembola_df['h'].max()

max_w = 512
max_h = 512

# Prepare CSV output
import csv
csv_rows = []

# Crop and save each detected object using the largest bbox size
for i, obj in enumerate(detected_objects):
    y, x = obj.centroid
    crop_w, crop_h = int(max_w), int(max_h)
    cropped_img = crop_to_bbox_with_padding(image, x, y, crop_w, crop_h)
    crop_path = output_dir / f"detected_object_{i}.jpg"
    sk.io.imsave(crop_path, cropped_img)
    # Calculate top-left corner of crop in original image
    H, W = image.shape[:2]
    x1 = int(x - crop_w // 2)
    y1 = int(y - crop_h // 2)
    # Calculate relative coordinates (clipped to [0,1])
    rel_x = max(0, min(1, x1 / W))
    rel_y = max(0, min(1, y1 / H))
    # Find all collembola centers inside this crop
    matches = target_collembola_df[
        (target_collembola_df["center_x"] >= x1) &
        (target_collembola_df["center_x"] <= x1 + crop_w) &
        (target_collembola_df["center_y"] >= y1) &
        (target_collembola_df["center_y"] <= y1 + crop_h)
    ]
    is_collembola = not matches.empty
    # Prepare lists of centers
    abs_centers = matches[["center_x", "center_y"]].values.tolist()
    rel_centers = [
        [
            max(0, min(1, (cx - x1) / crop_w)),
            max(0, min(1, (cy - y1) / crop_h))
        ]
        for cx, cy in abs_centers
    ]
    csv_rows.append({
        'crop_id': str(crop_path),
        'collembola': is_collembola,
        'rel_x': rel_x,
        'rel_y': rel_y,
        'collembola_centers_abs': json.dumps(abs_centers),
        'collembola_centers_rel': json.dumps(rel_centers)
    })

# Save CSV
output_dir = Path("data")
csv_path = output_dir / "crops_dataset.csv"
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=['crop_id', 'collembola', 'rel_x', 'rel_y', 'collembola_centers_abs', 'collembola_centers_rel'])
    writer.writeheader()
    writer.writerows(csv_rows)
print(f"Saved crop dataset CSV to {csv_path}")
