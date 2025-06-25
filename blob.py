# %%
import numpy as np
import skimage.io

# blob.py (ovo u C:\Users\HP\PycharmProjects\Collembole\blob.py)

import numpy as np
import skimage.io
from skimage.color import rgb2gray
from skimage.feature import blob_log
from math import sqrt
from pathlib import Path

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

    blobs = blob_log(gray, **blob_kwargs)
    blobs[:, 2] *= sqrt(2)

    mask = np.zeros_like(gray, dtype=np.uint8)
    for y, x, r in blobs:
        yy, xx = int(round(y)), int(round(x))
        if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]:
            mask[yy, xx] = 255

    out_path = out_dir / f"{slika_path.stem}_dots.png"
    skimage.io.imsave(str(out_path), mask)
    print(f"   snimljeno → {out_path.name}")

print("\'masks'")



# %%
