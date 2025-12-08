
import os
import pandas as pd
import json
import skimage.io
from matplotlib.patches import Rectangle

CSV_PATH   = "Slika_1_csv (8).csv"
IMAGE_PATH = r"slike\C1_1_Fe2O3002 (1).jpg"
OUTPUT_DIR = "collembolas"
os.makedirs(OUTPUT_DIR, exist_ok=True)

df      = pd.read_csv(CSV_PATH)
mask    = df["filename"] == "C1_1_Fe2O3002.jpg"
regions = df.loc[mask, "region_shape_attributes"]

img = skimage.io.imread(IMAGE_PATH)

for idx, shape_str in enumerate(regions, start=1):
    d = json.loads(shape_str)
    x,y = int(d["x"]), int(d["y"])
    w,h = int(d["width"]), int(d["height"])
    crop = img[y:y+h, x:x+w]
    out = os.path.join(OUTPUT_DIR, f"C1_1_Fe2O3002_{idx:03d}.jpg")
    skimage.io.imsave(out, crop)

print(f"Gotovo! Spremila sam {idx} izrezaka u '{OUTPUT_DIR}/'.")



