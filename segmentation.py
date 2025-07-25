from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
import numpy as np

# Define the ontology with the text prompt
ontology = CaptionOntology({"organism": "organism"})

# Initialize the Grounding DINO model (more stable than GroundedSAM2)
base_model = GroundingDINO(ontology=ontology)

# Load the image
image_path = "data/dataset_crops/detected_object_5188.jpg"
image = cv2.imread(image_path)

# Run inference to get bounding boxes
results = base_model.predict(image_path)

# Convert bounding boxes to masks using simple contour detection
def create_masks_from_boxes(image, detections):
    masks = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    for box in detections.xyxy:
        x1, y1, x2, y2 = box.astype(int)
        
        # Extract ROI
        roi = gray[y1:y2, x1:x2]
        
        # Apply threshold to create binary mask
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours in the ROI
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask for the largest contour
        mask = np.zeros_like(gray)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Adjust contour coordinates to full image
            largest_contour[:, :, 0] += x1
            largest_contour[:, :, 1] += y1
            cv2.fillPoly(mask, [largest_contour], 255)
        
        masks.append(mask)
    
    return masks

# Create masks and extract contours
masks = create_masks_from_boxes(image, results)
annotated_image = image.copy()

# Draw contours and get polygon points
for i, mask in enumerate(masks):
    # Find contours from the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter small contours
            # Draw contour
            cv2.drawContours(annotated_image, [contour], -1, (0, 255, 0), 2)
            
            # Simplify contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw polygon points
            for point in polygon:
                cv2.circle(annotated_image, tuple(point[0]), 3, (255, 0, 0), -1)
            
            print(f"Organism {i+1} polygon points: {polygon.reshape(-1, 2)}")
            print(f"Organism {i+1} contour area: {cv2.contourArea(contour)}")

# Save the results
output_path = "output_segmented.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"Results saved to {output_path}")
print(f"Number of detections: {len(results)}")
if len(results) > 0:
    print(f"Detection confidences: {results.confidence}")
    print(f"Bounding boxes: {results.xyxy}")