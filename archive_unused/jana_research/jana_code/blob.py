# %%
import numpy as np
import skimage.io
from skimage.color import rgb2gray
from skimage.feature import blob_log, blob_dog, blob_doh
from math import sqrt
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm
from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border, watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage import exposure

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

    # Enhance contrast using histogram equalization
    print("   pojačavam kontrast...")
    gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)

    # Local binarization (adaptive thresholding) with adjusted parameters
    print("   lokalna binarizacija...")
    block_size = 51  # Larger block size for more global context
    offset = 0.01    # Lower offset for faint features
    local_thresh = threshold_local(gray_eq, block_size, offset=offset)
    binary_local = gray_eq > local_thresh

    # Visualize and save the binary mask for debugging
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(binary_local, cmap='gray')
    ax.set_title('Binary mask after thresholding')
    ax.set_axis_off()
    plt.tight_layout(pad=0)
    out_path = out_dir / f"{slika_path.stem}_binary_debug.png"
    fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"   snimljeno → {out_path.name}")

    # Clean up the binary image with lower min_size
    binary_local = remove_small_objects(binary_local, min_size=10)
    binary_local = clear_border(binary_local)

    # Compute distance map for watershed
    print("   priprema watershed segmentacije...")
    distance = ndi.distance_transform_edt(binary_local)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=binary_local)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=binary_local)

    # Overlay watershed boundaries on the image
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.imshow(labels, alpha=0.3, cmap='nipy_spectral')  # Overlay segmentation

    # Optionally, draw contours or boundaries
    # from skimage.segmentation import find_boundaries
    # boundaries = find_boundaries(labels)
    # ax.imshow(boundaries, cmap='hot', alpha=0.5)

    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # Save result
    out_path = out_dir / f"{slika_path.stem}_watershed.png"
    fig.savefig(str(out_path), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"   snimljeno → {out_path.name}")
    break
print("'masks'")
