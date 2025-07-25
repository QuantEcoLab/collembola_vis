from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology
import cv2
import supervision as sv

# Define the ontology with the text prompt
ontology = CaptionOntology({"organism": "organism"})

# Initialize the Grounding DINO model (more stable than GroundedSAM2)
base_model = GroundingDINO(ontology=ontology)

# Load the image
image_path = "data/dataset_crops/detected_object_5188.jpg"
image = cv2.imread(image_path)

# Run inference
results = base_model.predict(image_path)

# Create annotated image with bounding boxes
annotator = sv.BoxAnnotator()
annotated_image = annotator.annotate(scene=image, detections=results)

# Save the results (image with bounding boxes)
output_path = "output_image.jpg"
cv2.imwrite(output_path, annotated_image)

print(f"Results saved to {output_path}")
print(f"Number of detections: {len(results)}")
if len(results) > 0:
    print(f"Detection confidences: {results.confidence}")
    print(f"Bounding boxes: {results.xyxy}")