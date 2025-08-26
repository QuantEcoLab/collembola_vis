import cv2
import numpy as np
from PIL import Image
import requests
import torch
from segment_anything import SamPredictor, sam_model_registry

# Placeholder for grounded-sam integration
# In practice, you would use the grounded-sam repo and its grounding DINO model
# Here, we simulate prompt-based segmentation with SAM only

def segment_with_sam(image_path, prompt="organism"):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load SAM model (ViT-B by default)
    sam = sam_model_registry["vit_b"]("sam_vit_b_01ec64.pth")
    predictor = SamPredictor(sam)
    predictor.set_image(image_rgb)

    # Simulate a bounding box for the prompt (in real use, use grounded-sam for this)
    h, w, _ = image.shape
    bbox = np.array([w//4, h//4, 3*w//4, 3*h//4])  # center box

    masks, scores, logits = predictor.predict(box=bbox[None, :], multimask_output=True)
    best_mask = masks[np.argmax(scores)]

    # Save mask overlay
    mask_img = (best_mask * 255).astype(np.uint8)
    overlay = image_rgb.copy()
    overlay[best_mask] = [255, 0, 0]  # Red overlay for mask
    result = Image.fromarray(overlay)
    result.save("segmented_result.png")
    print("Segmentation saved to segmented_result.png")

if __name__ == "__main__":
    segment_with_sam("data/dataset_crops/detected_object_3809.jpg", prompt="organism")
