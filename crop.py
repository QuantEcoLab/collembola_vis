import csv
import numpy as np
import skimage.io
from skimage.color      import rgb2gray
from skimage.filters    import gaussian, threshold_otsu
from skimage.measure    import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from pathlib            import Path

BASE_DIR  = Path(__file__).resolve().parent
src_dir   = BASE_DIR / "slike"
dots_dir  = BASE_DIR / "masks"
crops_dir = BASE_DIR / "crops"
csv_path  = BASE_DIR / "crops_K1.csv"

crops_dir.mkdir(exist_ok=True)

slika_path = next(src_dir.glob("K1_*.jpg"))

dots_mask_path = dots_dir / f"{slika_path.stem}_dots.png"
if not dots_mask_path.exists():
    raise RuntimeError(f"nema: {dots_mask_path}")

dots = skimage.io.imread(str(dots_mask_path))
if dots.ndim == 3:
    manual_mask = dots[..., 0] > 0
else:
    manual_mask = dots > 0

img    = skimage.io.imread(str(slika_path))
gray   = rgb2gray(img)
smooth = gaussian(gray, sigma=2)
thresh = threshold_otsu(smooth)
binary = smooth > thresh
binary = remove_small_objects(binary, min_size=20)
binary = clear_border(binary)

lbls  = label(binary)
props = regionprops(lbls)

with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "is_collembola"])

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        cy, cx = map(int, region.centroid)
        coords = region.coords
        dists  = np.sqrt((coords[:,0]-cy)**2 + (coords[:,1]-cx)**2)
        r      = int(dists.max()) + 1

        is_col = bool(manual_mask[cy, cx])

        y0, y1 = max(cy-r,0), min(cy+r, img.shape[0])
        x0, x1 = max(cx-r,0), min(cx+r, img.shape[1])
        crop   = img[y0:y1, x0:x1]

        crop_name = f"{slika_path.stem}_obj{idx}.png"
        skimage.io.imsave(str(crops_dir/crop_name), crop)
        writer.writerow([crop_name, is_col])

print("Odsječci u:", crops_dir, "CSV:", csv_path)
