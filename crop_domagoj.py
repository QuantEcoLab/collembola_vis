import os
import pandas as pd
import json
import skimage.io
from matplotlib.patches import Rectangle
from pathlib import Path

slike = list(Path("slike").glob("*.jpg"))
CSV_PATH = "Slika_1_csv (8).csv"
df = pd.read_csv(CSV_PATH)

regions_df = df[["filename", "region_shape_attributes"]]
imena = regions_df["filename"].values 

records = []

for path_slike in slike:
    slika = skimage.io.imread(path_slike)

    ime_slike = ""
    for sl in df["filename"].unique():
        if sl.replace(".jpg", "") in str(path_slike):
            ime_slike = sl.replace(".jpg", "")
    print(ime_slike)

    subset = regions_df[imena == ime_slike + ".jpg"]
    shapes = subset["region_shape_attributes"].values

    for idx, c in enumerate(shapes):
        d = json.loads(c)
        x = d["x"]
        y = d["y"]
        w = d["width"]
        h = d["height"]
        xc = x + w / 2
        yc = y + h / 2

        collembola = slika[y:y+h, x:x+w]
        path = f"collembolas/{ime_slike}_c_{idx:03d}.jpg"
        # skimage.io.imsave(path, collembola)

        records.append({
            "id": idx,
            "path": path,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "xc": xc,
            "yc": yc
        })

out_df = pd.DataFrame.from_dict(records)
out_df.to_csv("collembolas_metadata.csv", index=False)
