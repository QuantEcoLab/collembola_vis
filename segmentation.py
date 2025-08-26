from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv
import numpy as np

# Define the ontology with the text prompt
ontology = CaptionOntology({"bug": "bug"})

# Initialize the Grounding DINO model (more stable than GroundedSAM2)
base_model = GroundingDINO(ontology=ontology)


# detect function
def detect_collembola(image):
    results = base_model.predict(image)

    # Directory where templates are stored
    import os
    template_dir = "data/organism_templates"

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


    # --- Template-based filtering of DINO detections ---
    # Load all templates
    import glob
    template_paths = glob.glob(os.path.join(template_dir, '*.jpg'))
    templates = [cv2.imread(tp, cv2.IMREAD_GRAYSCALE) for tp in template_paths]

    match_threshold = 0.8 # Adjust as needed

    masks = create_masks_from_boxes(image, results)
    kept_detections = []


    for i, box in enumerate(results.xyxy):
        x1, y1, x2, y2 = box.astype(int)
        crop = image[y1:y2, x1:x2]
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        best_score = 0
        best_template = None
        for template in templates:
            if template is None or crop_gray.shape[0] < template.shape[0] or crop_gray.shape[1] < template.shape[1]:
                continue
            res = cv2.matchTemplate(crop_gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_template = template
        if best_score >= match_threshold:
            # --- Polygon extraction for segmentation ---
            mask = masks[i] if i < len(masks) else None
            contours_data = []
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 100:
                        # Simplify contour to polygon
                        epsilon = 0.02 * cv2.arcLength(contour, True)
                        polygon = cv2.approxPolyDP(contour, epsilon, True)
                        contours_data.append({
                            'contour': contour,
                            'polygon_points': polygon.reshape(-1, 2),
                            'area': cv2.contourArea(contour)
                        })
            
            kept_detections.append({
                'box': box,
                'score': best_score,
                'contours_data': contours_data
            })

    
    return kept_detections


if __name__ == "__main__":
    
    # Load the image
    image_path = "data/dataset_crops/detected_object_5188.jpg"
    image_path = "data/dataset_crops/detected_object_5973.jpg"
    image = cv2.imread(image_path)
    kept_detections = detect_collembola(image)
    # Annotate the image with the kept detections
    annotated_image = image.copy()
    for detection in kept_detections:
        box = detection['box']
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        score = detection['score']
        cv2.putText(annotated_image, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw contours if available
        for contour_data in detection['contours_data']:
            contour = contour_data['contour']
            cv2.drawContours(annotated_image, [contour], -1, (255, 0, 0), 1)
        # Save contours data for fuarther processing
        contours_data = detection['contours_data']
        print(f"Detection: {box}, Score: {score}, Contours: {len(contours_data)}")
        
    # Save the annotated image
    output_path = "output_segmented_filtered.jpg"
    cv2.imwrite(output_path, annotated_image)
    print(f"Filtered results saved to {output_path}")
    print(f"Number of kept detections: {len(kept_detections)}")