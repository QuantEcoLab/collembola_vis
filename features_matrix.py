import numpy as np
import pandas as pd
from pathlib import Path
from skimage.io import imread
from skimage.color import rgb2lab, rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize

def color_histogram(img, bins=16):
   
    lab = rgb2lab(img)
    hL, _ = np.histogram(lab[..., 0], bins=bins, range=(0, 100), density=True)
    hA, _ = np.histogram(lab[..., 1], bins=bins, range=(-128, 127), density=True)
    hB, _ = np.histogram(lab[..., 2], bins=bins, range=(-128, 127), density=True)
    return np.concatenate([hL, hA, hB])

def hog_features(img, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1)):

    gray = rgb2gray(img)
    feats = hog(
        gray,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=False,
        feature_vector=True
    )
    return feats

def lbp_histogram(img, P=8, R=1, bins=None):

    gray = rgb2gray(img)
    lbp = local_binary_pattern(gray, P, R, method="uniform")
    if bins is None:
        bins = int(P + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist

if __name__ == "__main__":
    template_dir = Path("data/templates")
    TARGET_SIZE = (128, 128)  # fiksna veličina za sve izrezke

    paths = sorted(template_dir.glob("*.png"))
    print(f"Found {len(paths)} templates in {template_dir}.")

    feature_list = []
    filenames = []
    for p in paths:
        img = imread(str(p))
        img_r = resize(img, (*TARGET_SIZE, img.shape[2]), preserve_range=True).astype(np.uint8)

        f_color = color_histogram(img_r, bins=16)
        f_hog   = hog_features(img_r, orientations=8, pixels_per_cell=(16,16), cells_per_block=(1,1))
        f_lbp   = lbp_histogram(img_r, P=8, R=1)

        feats = np.concatenate([f_color, f_hog, f_lbp])
        feature_list.append(feats)
        filenames.append(p.name)

    X = np.vstack(feature_list)
    print("Feature matrix X shape:", X.shape)

    df = pd.DataFrame(X)
    df.insert(0, "filename", filenames)
    out_path = template_dir / "features_matrix.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved features matrix to {out_path}")
