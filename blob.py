# %%
import numpy as np
import skimage.io
from skimage.color import rgb2gray
from skimage.feature import blob_log
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

# blob.py (ovo u C:\Users\HP\PycharmProjects\Collembole\blob.py)
BASE_DIR = Path(__file__).resolve().parent

src_dir = BASE_DIR / "slike"
print("Učitavam slike iz:", src_dir.resolve())
slike = list(src_dir.glob("*.jpg"))
print("Pronađene slike:", [p.name for p in slike])

blob_kwargs = dict(
    max_sigma=20,
    num_sigma=10,
    threshold=0.02
)

out_dir = BASE_DIR / "masks"
out_dir.mkdir(exist_ok=True)
print("Rezultati će se spremiti u:", out_dir.resolve())

for slika_path in slike:
    print(f" obrađuje se: {slika_path.name}")
    img = skimage.io.imread(str(slika_path))
    gray = rgb2gray(img)

    # detekcija blobova
    blobs = blob_log(gray, **blob_kwargs)
    blobs[:, 2] *= sqrt(2)

    # crtanje slike i blobova pomoću matplotlib-a
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    for y, x, r in tqdm(blobs):
        circ = Circle((x, y), r, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(circ)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # spremi rezultat
    out_path = out_dir / f"{slika_path.stem}_overlay.png"
    fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"   snimljeno → {out_path.name}")

print("'masks'")
