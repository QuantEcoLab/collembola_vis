import numpy as np
import skimage.io
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE_DIR = Path(__file__).resolve().parent
src_dir = BASE_DIR / "slike"
slike = list(src_dir.glob("*.jpg"))

out_dir = BASE_DIR / "masks"
out_dir.mkdir(exist_ok=True)

for slika_path in slike:
    img = skimage.io.imread(str(slika_path))
    gray = rgb2gray(img)

    gray_smooth = gaussian(gray, sigma=2)
    thresh = threshold_otsu(gray_smooth)
    binary = gray_smooth > thresh
    binary = remove_small_objects(binary, min_size=20)
    binary = clear_border(binary)

    labels = label(binary)
    props = regionprops(labels)

    # Crtanje
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)

    for region in props:
        if region.area < 10:
            continue
        y, x = region.centroid
        coords = region.coords
        distances = np.sqrt((coords[:, 0] - y)**2 + (coords[:, 1] - x)**2)
        r = distances.max() + 1  

        circ = Circle(
            (x, y), r,
            edgecolor="red",
            facecolor="red",
            alpha=0.2,
            linewidth=1
        )
        ax.add_patch(circ)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    out_path = out_dir / f"{slika_path.stem}_adaptive_circles.png"
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"   Snimljeno {out_path.name}")

print("'masks'")

#sdofijwoi