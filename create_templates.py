import cv2
import os
import pandas as pd
from autodistill_grounding_dino import GroundingDINO
from autodistill.detection import CaptionOntology

# Define the ontology with the text prompt
ontology = CaptionOntology({"bug": "bug"})
# Initialize the Grounding DINO model
base_model = GroundingDINO(ontology=ontology)

# Read CSV and filter for images with organisms
csv_path = "data/crops_dataset.csv"
df = pd.read_csv(csv_path)
organism_df = df[df['collembola'] == True]

# Directory to save templates
template_dir = "data/organism_templates"
os.makedirs(template_dir, exist_ok=True)

# Loop through images with organisms
for idx, row in organism_df.iterrows():
    image_path = row['crop_id']
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not read {image_path}")
        continue
    results = base_model.predict(image_path)
    # For each detection, crop and save the region
    for i, box in enumerate(results.xyxy):
        x1, y1, x2, y2 = box.astype(int)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        template_path = os.path.join(template_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_org_{i}.jpg")
        cv2.imwrite(template_path, crop)
        print(f"Saved template: {template_path}")
