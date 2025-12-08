import os
import re
import pandas as pd
import json
import skimage.io
from pathlib import Path

# Kreiraj izlazni direktorij za isječke
out_dir = Path("collembolas")
out_dir.mkdir(exist_ok=True)

# Učitaj CSV
df_csv = pd.read_csv("Slika_1_csv (8).csv")

# Pripremi slike i nazive iz CSV
image_dir = Path("slike")
slike = list(image_dir.glob("*.jpg"))
imena_slika = set(df_csv["filename"].unique())
df_regions = df_csv[["filename", "region_shape_attributes"]]

# Lista za rezultate
rows = []

for path_slike in slike:
    # Učitaj sliku
    slika = skimage.io.imread(str(path_slike))
    stem = path_slike.stem  # npr. 'C1_1_Fe203002 (1)'
    # Ukloni suffix poput ' (1)' ako postoji
    base = re.sub(r"\s*\(\d+\)$", "", stem)
    filename_ext = f"{base}.jpg"
    print(f"Obrađujem {path_slike.name} -> tretiram kao {filename_ext}...")

    # Ako nema tog imena u CSV
    if filename_ext not in imena_slika:
        print(f"  Preskačem, '{filename_ext}' nije u CSV-u.")
        continue

    # Dohvati regije
    image_regions = df_regions[df_regions["filename"] == filename_ext]["region_shape_attributes"].values
    if not len(image_regions):
        print(f"  Nema regija za '{filename_ext}'.")
        continue

    # Obradi i spremi
    for idx, region_str in enumerate(image_regions):
        d = json.loads(region_str)
        x, y, w, h = d["x"], d["y"], d["width"], d["height"]
        isjecak = slika[y:y+h, x:x+w]
        output_path = out_dir / f"{base}_c_{idx:03d}.jpg"
        skimage.io.imsave(str(output_path), isjecak)

        rows.append({
            "id_collembole": f"{base}_c_{idx:03d}",
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "path_slike": str(output_path)
        })

# Spremi tablicu
if rows:
    df_out = pd.DataFrame(rows)
    df_out.to_csv("collembolas_table.csv", index=False)
    print("Gotovo, spremljeno u collembolas_table.csv")
else:
    print("Nema pronadjenih regija.")

