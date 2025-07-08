import pandas as pd
import json

df = pd.read_csv("Slika_1_csv (8).csv")

def izvuci_dimenzije(podatak):
    try:
        d = json.loads(podatak)
        return d.get("width"), d.get("height")
    except:
        return None, None

df["width"], df["height"] = zip(*df["region_shape_attributes"].map(izvuci_dimenzije))
df = df.dropna(subset=["width", "height"])

px_u_mm = 1 / 116.7
df["width_mm"] = df["width"] * px_u_mm
df["height_mm"] = df["height"] * px_u_mm
df["area_mm2"] = df["width_mm"] * df["height_mm"]


df["petrijevka"] = df["filename"].map({
    "C1_1_Fe2O3002.jpg": "Petrijevka_1",
    "C5_2_Fe2O3003.jpg": "Petrijevka_2",
    "K1_Fe2O3001.jpg": "Petrijevka_3"
})

rezultat = df[["petrijevka", "filename", "width_mm", "height_mm", "area_mm2"]]

rezultat.to_excel("sve_jedinke_collembole.xlsx", index=False)

pregled = df.groupby("petrijevka").agg(
    broj_jedinki=("width", "count"),
    prosirječna_širina_mm=("width_mm", "mean"),
    prosječna_visina_mm=("height_mm", "mean"),
    prosječna_površina_mm2=("area_mm2", "mean")
).reset_index()

std_dev = df.groupby("petrijevka")["area_mm2"].std().reset_index()
std_dev.columns = ["petrijevka", "standardna_devijacija_povrsine"]

pregled = pregled.merge(std_dev, on="petrijevka")

pregled.to_excel("sazetak_collembole_sa_devijacijom.xlsx", index=False)
