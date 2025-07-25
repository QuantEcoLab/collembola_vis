from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
import numpy as np

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
import time

from segmentation import detect_collembola


def find_objects(img, visualise=False):
    # This function detects potential collembola in the image
    # It uses image processing techniques to find "circular" objects
    # and returns their properties for further analysis.
    # This part is from circles.py
    print("🔍 Finding potential objects in image...")
    gray = rgb2gray(img)
    gray_smooth = gaussian(gray, sigma=2)
    thresh = threshold_otsu(gray_smooth)
    binary = gray_smooth > thresh
    binary = remove_small_objects(binary, min_size=20)
    binary = clear_border(binary)
    labels = label(binary)
    props = regionprops(labels)
    
    print(f"Found {len(props)} potential objects")
    
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

def detect_collembola_in_image(image_path, visualise_objects=False, save_results=True):
    """
    Main function to detect collembola in an image.
    
    Args:
        image_path: Path to the input image
        visualise_objects: Whether to show detected objects visualization
        save_results: Whether to save final detection results (annotated image + JSON)
    
    Returns:
        List of detected collembola with their properties
    """
    start_time = time.time()
    
    # Load and display the image
    print("📖 Loading image...")
    image = sk.io.imread(image_path)
    print(f"Image shape: {image.shape}")
    
    detected_objects = find_objects(image, visualise=visualise_objects)
    
    # Take crops of the image around detected objects and store crop metadata
    crops = []
    crop_metadata = []
    min_crop_size = 25  # Minimum crop size (50x50 pixels)
    target_crop_size = 512  # Target crop size (512x512 pixels)

    print("Creating crops from detected objects...")
    filtered_count = 0
    resized_count = 0
    
    for obj in tqdm(detected_objects, desc="Processing objects", unit="object"):
        y, x = obj.centroid
        r = int(obj.equivalent_diameter / 2)  # Use equivalent diameter for a circular crop
        
        # Calculate initial crop bounds
        y1, y2 = max(0, int(y - r)), min(image.shape[0], int(y + r))
        x1, x2 = max(0, int(x - r)), min(image.shape[1], int(x + r))
        
        # Check crop dimensions
        crop_height = y2 - y1
        crop_width = x2 - x1
        
        # Filter out crops that are too small
        if crop_height < min_crop_size or crop_width < min_crop_size:
            filtered_count += 1
            continue
        
        # If crop is larger than minimum but smaller than target, resize to target size
        if crop_height > min_crop_size or crop_width > min_crop_size:
            # Calculate half of target size for centering
            half_target = target_crop_size // 2
            
            # Recalculate bounds centered around object centroid for 512x512 crop
            y1_new = max(0, int(y - half_target))
            y2_new = min(image.shape[0], int(y + half_target))
            x1_new = max(0, int(x - half_target))
            x2_new = min(image.shape[1], int(x + half_target))
            
            # Adjust if we hit image boundaries to maintain target size when possible
            if y2_new - y1_new < target_crop_size:
                if y1_new == 0:
                    y2_new = min(image.shape[0], target_crop_size)
                elif y2_new == image.shape[0]:
                    y1_new = max(0, image.shape[0] - target_crop_size)
            
            if x2_new - x1_new < target_crop_size:
                if x1_new == 0:
                    x2_new = min(image.shape[1], target_crop_size)
                elif x2_new == image.shape[1]:
                    x1_new = max(0, image.shape[1] - target_crop_size)
            
            # Use the new bounds
            y1, y2, x1, x2 = y1_new, y2_new, x1_new, x2_new
            resized_count += 1
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        
        crops.append(crop)
        crop_metadata.append({
            'centroid': (y, x),
            'radius': r,
            'crop_bounds': (y1, y2, x1, x2),
            'original_size': (crop_height, crop_width),
            'final_size': (y2 - y1, x2 - x1)
        })

    print(f"Created {len(crops)} crops from {len(detected_objects)} detected objects")
    print(f"Filtered out {filtered_count} crops (too small: <{min_crop_size}x{min_crop_size})")
    print(f"Resized {resized_count} crops to target size ({target_crop_size}x{target_crop_size})")
    
    # Calculate memory usage of crops
    total_pixels = sum(crop.shape[0] * crop.shape[1] * crop.shape[2] for crop in crops)
    memory_mb = (total_pixels * 1) / (1024 * 1024)  # Assuming uint8 (1 byte per pixel)
    print(f"💾 Crops in memory: ~{memory_mb:.1f} MB ({len(crops)} crops)")

    # Detect collembola in the crops and collect all detections
    all_detections = []

    print("Detecting collembola in crops...")
    for i, crop in enumerate(tqdm(crops, desc="Analyzing crops", unit="crop")):
        kept_detections = detect_collembola(crop)
        if not kept_detections:
            # print(f"No detections in crop {i}")
            continue

        # Get crop bounds for coordinate conversion
        y1, y2, x1, x2 = crop_metadata[i]['crop_bounds']
        
        # Process kept detections
        for detection in kept_detections:
            box = detection['box']
            score = detection['score']
            print(f"Detection in crop {i}: {box}, Score: {score}")
            
            # Convert detections from crop coordinates to original image coordinates
            original_box = box.copy()
            original_box[0] += x1  # x1 coordinate
            original_box[1] += y1  # y1 coordinate  
            original_box[2] += x1  # x2 coordinate
            original_box[3] += y1  # y2 coordinate
            
            # Convert contours to original image coordinates
            converted_contours_data = []
            for contour_data in detection['contours_data']:
                contour = contour_data['contour'].copy()
                # Shift contour coordinates to original image space
                contour[:, :, 0] += x1  # x coordinates
                contour[:, :, 1] += y1  # y coordinates
                
                converted_contours_data.append({
                    'contour': contour,
                    'polygon_points': contour_data.get('polygon_points', []),
                    'area': contour_data['area']
                })
                
                area = contour_data['area']
                print(f"  Contour area: {area}, Points: {len(contour)}")
            
            # Store detection with original image coordinates
            converted_detection = {
                'box': original_box,
                'score': score,
                'contours_data': converted_contours_data,
                'crop_index': i
            }
            
            all_detections.append(converted_detection)
            print(f"Detection in original image: {original_box}, Score: {score}")

    print(f"\nTotal detections found: {len(all_detections)}")
    
    detection_time = time.time() - start_time
    print(f"⏱️  Detection completed in {detection_time:.2f} seconds")

    # Create annotated visualization
    if all_detections and save_results:
        print("Creating annotated visualization...")
        annotated_image = image.copy()
        
        for i, detection in enumerate(tqdm(all_detections, desc="Drawing annotations", unit="detection")):
            box = detection['box']
            score = detection['score']
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add score text
            cv2.putText(annotated_image, f"{score:.2f}", (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw contours
            for contour_data in detection['contours_data']:
                contour = contour_data['contour']
                cv2.drawContours(annotated_image, [contour], -1, (255, 0, 0), 1)
        
        # Save annotated image
        image_name = Path(image_path).stem
        output_path = f"output_collembola_{image_name}.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"Annotated results saved to {output_path}")
        
        # Save detection data to JSON for further analysis
        print("Saving detection data...")
        detection_data = []
        for detection in tqdm(all_detections, desc="Processing detection data", unit="detection"):
            # Convert numpy arrays to lists for JSON serialization
            detection_json = {
                'box': detection['box'].tolist(),
                'score': float(detection['score']),
                'crop_index': detection['crop_index'],
                'contours_data': []
            }
            
            for contour_data in detection['contours_data']:
                contour_json = {
                    'area': float(contour_data['area']),
                    'contour_points': contour_data['contour'].reshape(-1, 2).tolist()
                }
                detection_json['contours_data'].append(contour_json)
            
            detection_data.append(detection_json)
        
        json_output_path = f'collembola_detections_{image_name}.json'
        with open(json_output_path, 'w') as f:
            json.dump(detection_data, f, indent=2)
        print(f"Detection data saved to {json_output_path}")
        
    elif all_detections and not save_results:
        print("✅ Detection completed (results not saved - memory-only mode)")
        
    else:
        print("No collembola detections found in the image.")
    
    return all_detections


if __name__ == "__main__":
    # Load and display the image
    image_path = "data/slike/K1_Fe2O3001 (1).jpg"
    
    print("🔬 Starting Collembola Detection Pipeline")
    print(f"📁 Processing image: {image_path}")
    print("=" * 50)
    
    # Run the detection pipeline (memory-only mode)
    detections = detect_collembola_in_image(
        image_path, 
        visualise_objects=False,  # Show the initial object detection
        save_results=True        # Save final results (annotated image + JSON)
    )
    
    print("=" * 50)
    print(f"✅ Pipeline completed! Found {len(detections)} collembola detections.")
        