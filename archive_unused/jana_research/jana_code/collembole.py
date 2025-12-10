import pandas as pd
import json
import skimage
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


df = pd.read_csv("Slika_1_csv (8).csv")
slika1 = "slike\C1_1_Fe2O3002 (1).jpg"
slika2 = "slike\C5_2_Fe2O3003 (1).jpg"
slika3 = "slike\K1_Fe2O3001 (1).jpg"

slika1 = skimage.io.imread(slika1)

regions = df[["filename", "region_shape_attributes"]]
imena = regions["filename"]
regions = regions[imena == "C1_1_Fe2O3002.jpg"]

j = regions["region_shape_attributes"][0]

d = json.loads(j)
x = d["x"]
y = d["y"]
w = d["width"]
h = d["height"]

fig, ax = plt.subplots()
ax.imshow(slika1)
ax.add_patch(Rectangle((x,y),w, h, alpha=0.5))
#ax.add_patch(Circle((x,y),10))
plt.show()

collembola1 = slika1[y:y+h, x:x+w]

skimage.io.imsave("collembolas/C1_1_Fe2O3002_"+"1"+".jpg", collembola1)

# def izvuci_dimenzije(podatak):
#     try:
#         d = json.loads(podatak)
#         return d.get("width"), d.get("height")
#     except:
#         return None, None

# df["width"], df["height"] = zip(*df["region_shape_attributes"].map(izvuci_dimenzije))
# df = df.dropna(subset=["width", "height"])

# px_u_mm = 1 / 116.7
# df["width_mm"] = df["width"] * px_u_mm
# df["height_mm"] = df["height"] * px_u_mm
# df["area_mm2"] = df["width_mm"] * df["height_mm"]

# df["petrijevka"] = df["filename"].map({
#     "C1_1_Fe2O3002.jpg": "Petrijevka_1",
#     "C5_2_Fe2O3003.jpg": "Petrijevka_2",
#     "K1_Fe2O3001.jpg": "Petrijevka_3"
# })

# pregled = df.groupby("petrijevka").agg({
#     "width_mm": "mean",
#     "height_mm": "mean",
#     "area_mm2": "mean",
#     "width": "count"

# })

# pregled = pregled.rename(columns={"width": "broj_jedinki"})

# pregled.to_excel("sazetak_collembole.xlsx")
