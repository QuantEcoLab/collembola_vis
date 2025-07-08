import csv
import numpy as np
import pandas as pd
import skimage.io
from skimage.color      import rgb2gray
from skimage.filters    import gaussian, threshold_otsu
from skimage.measure    import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from pathlib            import Path

BASE_DIR      = Path(__file__).resolve().parent
src_dir       = BASE_DIR / "slike"
crops_dir     = BASE_DIR / "crops"
csv_out_path  = BASE_DIR / "crops_K1.csv"
metadata_path = BASE_DIR / "collembolas_metadata.csv"

crops_dir.mkdir(exist_ok=True)

meta_df = pd.read_csv(metadata_path, encoding='utf-8')
meta_df['orig'] = meta_df['path'].apply(lambda p: Path(p).stem.split('_c_')[0])

slika_path = next(src_dir.glob("K1_*.jpg"))
source_id  = slika_path.stem 
boxes_df   = meta_df[meta_df['orig'] == source_id]

boxes = boxes_df[['x','y','w','h']].to_records(index=False)

img    = skimage.io.imread(str(slika_path))
gray   = rgb2gray(img)
smooth = gaussian(gray, sigma=2)
thresh = threshold_otsu(smooth)
binary = smooth > thresh
binary = remove_small_objects(binary, min_size=20)
binary = clear_border(binary)

lbls  = label(binary)
props = regionprops(lbls)

with open(csv_out_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "is_collembola"])

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        cy, cx = map(int, region.centroid)

        is_col = any(
            (cx >= x) and (cx <= x + w) and
            (cy >= y) and (cy <= y + h)
            for x, y, w, h in boxes
        )

        coords = region.coords
        dists  = np.sqrt((coords[:,0]-cy)**2 + (coords[:,1]-cx)**2)
        r      = int(dists.max()) + 1

        y0, y1 = max(cy-r, 0), min(cy+r, img.shape[0])
        x0, x1 = max(cx-r, 0), min(cx+r, img.shape[1])
        crop   = img[y0:y1, x0:x1]

        crop_name = f"{source_id}_obj{idx}.png"
        skimage.io.imsave(str(crops_dir / crop_name), crop)
        writer.writerow([crop_name, is_col])

print(f"odsječci u '{crops_dir}', CSV u '{csv_out_path}'")
