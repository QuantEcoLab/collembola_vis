
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import os

# Add paths for Grounded-SAM-2
grounded_sam2_path = '/home/domagoj/GitHub/Grounded-SAM-2'
sys.path.append(grounded_sam2_path)
sys.path.append(os.path.join(grounded_sam2_path, 'grounding_dino'))
sys.path.append(os.path.join(grounded_sam2_path, 'sam2'))

# Device setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Import Grounded-SAM-2 modules
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Model configurations and checkpoints for Grounded-SAM-2
GROUNDING_DINO_CONFIG = "/home/domagoj/GitHub/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/domagoj/GitHub/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "/home/domagoj/GitHub/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"

# Load image
image_path = 'data/dataset_crops/detected_object_3650.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Text prompt and thresholds
TEXT_PROMPT = "organism"
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.2

print(f"Using text prompt: '{TEXT_PROMPT}'")

def load_grounding_dino_model(config_path, checkpoint_path, device):
    args = SLConfig.fromfile(config_path)
    model = build_model(args)
    args.device = device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    model.eval()
    model = model.to(device)
    return model

print("Loading GroundingDINO model...")
grounding_dino_model = load_grounding_dino_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, DEVICE)
print("GroundingDINO model loaded successfully")

print("Loading SAM2 model...")
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 model loaded successfully")

print("Setting image for SAM2...")
sam2_predictor.set_image(image_rgb)

print("Running GroundingDINO for detection...")
def run_grounding_dino(model, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
    image_source, image_tensor = load_image(image_path)
    image_tensor = image_tensor.to(DEVICE)
    boxes, logits, phrases = predict(
        model=model,
        image=image_tensor,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE
    )
    return image_source, boxes, logits, phrases

image_source, boxes, logits, phrases = run_grounding_dino(
    grounding_dino_model,
    image_path,
    TEXT_PROMPT,
    BOX_THRESHOLD,
    TEXT_THRESHOLD
)

print(f"Detected {len(boxes)} objects: {phrases}")

if len(boxes) > 0:
    print("Running SAM2 for segmentation...")
    H, W = image_rgb.shape[:2]
    boxes_xyxy = boxes * torch.tensor([W, H, W, H], device=DEVICE)
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_xyxy.cpu().numpy(),
        multimask_output=False,
    )
    print(f"Generated {len(masks)} masks")
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    for i, mask in enumerate(masks):
        plt.imshow(mask, alpha=0.5, cmap='viridis')
    plt.title(f'Segmentation: {TEXT_PROMPT}')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    if len(masks) > 0:
        plt.imshow(masks[0], cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    if len(masks) > 0:
        mask_img = Image.fromarray((masks[0] * 255).astype('uint8'))
        mask_img.save('grounded_sam2_mask.png')
        print("Mask saved as grounded_sam2_mask.png")
else:
    print("No objects detected with the given prompt and thresholds")

# Model configurations and checkpoints for Grounded-SAM-2
GROUNDING_DINO_CONFIG = "/home/domagoj/GitHub/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "/home/domagoj/GitHub/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "/home/domagoj/GitHub/Grounded-SAM-2/checkpoints/sam2_hiera_large.pt"
SAM2_MODEL_CONFIG = "sam2_hiera_l.yaml"

# Check if files exist and provide download instructions if not
def check_model_files():
    missing_files = []
    
    if not os.path.exists(GROUNDING_DINO_CONFIG):
        missing_files.append(f"Config: {GROUNDING_DINO_CONFIG}")
    
    if not os.path.exists(GROUNDING_DINO_CHECKPOINT):
        missing_files.append(f"GroundingDINO checkpoint: {GROUNDING_DINO_CHECKPOINT}")
    
    if not os.path.exists(SAM2_CHECKPOINT):
        missing_files.append(f"SAM2 checkpoint: {SAM2_CHECKPOINT}")
    
    if missing_files:
        print("Missing model files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nTo download checkpoints:")
        print("1. GroundingDINO: https://github.com/IDEA-Research/GroundingDINO/releases")
        print("2. SAM2: https://github.com/facebookresearch/sam2#download-checkpoints")
        return False
    
    return True

if not check_model_files():
    print("Using basic segmentation instead...")
    # Fall back to basic segmentation
    
    # Load image
    image_path = 'data/dataset_crops/detected_object_3650.jpg'
    print(f"Loading image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
        
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image loaded successfully: {image_rgb.shape}")
    
    # Convert to grayscale and apply thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try adaptive thresholding first (better for organisms)
    thresh_adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 11, 2)
    
    # Also try Otsu thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Apply morphological operations to both
    kernel = np.ones((3,3), np.uint8)
    cleaned_adaptive = cv2.morphologyEx(thresh_adaptive, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_adaptive = cv2.morphologyEx(cleaned_adaptive, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    cleaned_otsu = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_otsu = cv2.morphologyEx(cleaned_otsu, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours for both methods
    contours_adaptive, _ = cv2.findContours(cleaned_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_otsu, _ = cv2.findContours(cleaned_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Adaptive thresholding found {len(contours_adaptive)} contours")
    print(f"Otsu thresholding found {len(contours_otsu)} contours")
    
    # Choose the method that gives a reasonable number of objects (1-10)
    if 1 <= len(contours_adaptive) <= 10:
        contours = contours_adaptive
        method_name = "Adaptive"
        print("Using adaptive thresholding")
    elif 1 <= len(contours_otsu) <= 10:
        contours = contours_otsu
        method_name = "Otsu"
        print("Using Otsu thresholding")
    else:
        # Default to the one with fewer contours to avoid noise
        if len(contours_adaptive) <= len(contours_otsu):
            contours = contours_adaptive
            method_name = "Adaptive (default)"
        else:
            contours = contours_otsu
            method_name = "Otsu (default)"
        print(f"Using {method_name} - both methods found many contours")
    
    # Create mask from appropriately sized contours
    mask = np.zeros_like(gray)
    img_area = gray.shape[0] * gray.shape[1]
    min_area = 50  # Minimum area in pixels
    max_area = img_area * 0.5  # Maximum 50% of image
    
    valid_objects = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.fillPoly(mask, [contour], 255)
            valid_objects += 1
    
    print(f"Created mask with {valid_objects} objects")
    
    # Visualize results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image - Collembola')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.6, cmap='viridis')
    plt.title(f'Segmentation ({method_name})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Save mask
    mask_img = Image.fromarray(mask)
    mask_img.save('basic_segmentation_mask.png')
    print("Basic segmentation mask saved as basic_segmentation_mask.png")
    
    # Also save a colored overlay version
    overlay = image_rgb.copy()
    overlay[mask > 0] = [0, 255, 0]  # Green overlay
    overlay_img = Image.fromarray(overlay)
    overlay_img.save('segmentation_overlay.png')
    print("Segmentation overlay saved as segmentation_overlay.png")
    
    sys.exit(0)

# Load image
image_path = 'data/dataset_crops/detected_object_3650.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Text prompt and thresholds
TEXT_PROMPT = "organism"
BOX_THRESHOLD = 0.25  # Lower threshold to detect more objects
TEXT_THRESHOLD = 0.2   # Lower threshold for text matching

print(f"Using text prompt: '{TEXT_PROMPT}'")

def load_grounding_dino_model(config_path, checkpoint_path, device):
    """Load GroundingDINO model"""
    try:
        args = SLConfig.fromfile(config_path)
        model = build_model(args)
        # Try CPU first to avoid _C errors
        args.device = torch.device('cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        model.eval()
        
        # Move to target device after loading
        if device.type == 'cuda':
            try:
                model = model.to(device)
                print(f"Model moved to {device}")
            except Exception as e:
                print(f"Warning: Could not move model to GPU ({e}), keeping on CPU")
                device = torch.device('cpu')
        
        return model
    except Exception as e:
        print(f"Error in model loading: {e}")
        raise e

print("Loading GroundingDINO model...")
try:
    # Load GroundingDINO model
    grounding_dino_model = load_grounding_dino_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, DEVICE)
    print("GroundingDINO model loaded successfully")
except Exception as e:
    print(f"Error loading GroundingDINO model: {e}")
    print("Falling back to basic segmentation...")
    
    # Fall back to basic segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.fillPoly(mask, [contour], 255)
    
    # Visualize fallback results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5, cmap='viridis')
    plt.title('Basic Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    mask_img = Image.fromarray(mask)
    mask_img.save('basic_segmentation_mask.png')
    print("Basic segmentation mask saved as basic_segmentation_mask.png")
    sys.exit(0)

print("Loading SAM2 model...")
try:
    # Load SAM2 model
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print("SAM2 model loaded successfully")
except Exception as e:
    print(f"Error loading SAM2 model: {e}")
    print("Falling back to basic segmentation...")
    
    # Fall back to basic segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.fillPoly(mask, [contour], 255)
    
    # Visualize fallback results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5, cmap='viridis')
    plt.title('Basic Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    mask_img = Image.fromarray(mask)
    mask_img.save('basic_segmentation_mask.png')
    print("Basic segmentation mask saved as basic_segmentation_mask.png")
    sys.exit(0)

print("Setting image for SAM2...")
# Set the image for SAM2
sam2_predictor.set_image(image_rgb)

print("Running GroundingDINO for detection...")
# Run GroundingDINO to get bounding boxes
def run_grounding_dino(model, image_path, text_prompt, box_threshold=0.3, text_threshold=0.25):
    """Run GroundingDINO to get bounding boxes"""
    try:
        image_source, image_tensor = load_image(image_path)
        
        # Ensure model and tensor are on the same device
        model_device = next(model.parameters()).device
        image_tensor = image_tensor.to(model_device)
        
        boxes, logits, phrases = predict(
            model=model,
            image=image_tensor,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=model_device
        )
        return image_source, boxes, logits, phrases
    except Exception as e:
        print(f"GroundingDINO inference failed: {e}")
        raise e

try:
    image_source, boxes, logits, phrases = run_grounding_dino(
        grounding_dino_model, 
        image_path, 
        TEXT_PROMPT, 
        BOX_THRESHOLD, 
        TEXT_THRESHOLD
    )
    
    print(f"Detected {len(boxes)} objects: {phrases}")
    
    if len(boxes) > 0:
        print("Running SAM2 for segmentation...")
        # Convert boxes to the format expected by SAM2
        # Assuming boxes are in normalized format, convert to pixel coordinates
        H, W = image_rgb.shape[:2]
        boxes_xyxy = boxes * torch.tensor([W, H, W, H], device=DEVICE)
        
        # Get masks from SAM2
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes_xyxy.cpu().numpy(),
            multimask_output=False,
        )
        
        print(f"Generated {len(masks)} masks")
        
        # Visualize results
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        # Masks overlay
        plt.subplot(1, 3, 2)
        plt.imshow(image_rgb)
        for i, mask in enumerate(masks):
            plt.imshow(mask, alpha=0.5, cmap='viridis')
        plt.title(f'Segmentation: {TEXT_PROMPT}')
        plt.axis('off')
        
        # Just the mask
        plt.subplot(1, 3, 3)
        if len(masks) > 0:
            plt.imshow(masks[0], cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Save the first mask
        if len(masks) > 0:
            mask_img = Image.fromarray((masks[0] * 255).astype('uint8'))
            mask_img.save('grounded_sam2_mask.png')
            print("Mask saved as grounded_sam2_mask.png")
        
    else:
        print("No objects detected with the given prompt and thresholds")
        print("Falling back to basic segmentation...")
        
        # Fall back to basic segmentation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        kernel = np.ones((3,3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.zeros_like(gray)
        for contour in contours:
            if cv2.contourArea(contour) > 100:
                cv2.fillPoly(mask, [contour], 255)
        
        # Visualize fallback results
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(image_rgb)
        plt.imshow(mask, alpha=0.5, cmap='viridis')
        plt.title('Fallback Segmentation')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        mask_img = Image.fromarray(mask)
        mask_img.save('fallback_segmentation_mask.png')
        print("Fallback segmentation mask saved as fallback_segmentation_mask.png")
        
except Exception as e:
    print(f"Error during Grounded-SAM-2 processing: {e}")
    print("Falling back to basic segmentation...")
    
    # Fall back to basic segmentation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.fillPoly(mask, [contour], 255)
    
    # Visualize fallback results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(image_rgb)
    plt.imshow(mask, alpha=0.5, cmap='viridis')
    plt.title('Fallback Segmentation')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    mask_img = Image.fromarray(mask)
    mask_img.save('fallback_segmentation_mask.png')
    print("Fallback segmentation mask saved as fallback_segmentation_mask.png")
