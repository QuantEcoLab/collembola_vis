import os
from pathlib import Path
from roifile import ImagejRoi
from skimage.io import imread, imsave


def extract_rois_to_templates(
    image_path: str,
    rois_dir: str,
    output_dir: str
) -> None:
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img = imread(image_path)

    for roi_file in sorted(Path(rois_dir).glob("*.roi")):

        roi = ImagejRoi.fromfile(str(roi_file))
        coords = roi.coordinates()
        xs = [int(c[0]) for c in coords]
        ys = [int(c[1]) for c in coords]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        w = x_max - x_min + 1
        h = y_max - y_min + 1

        crop = img[y_min:y_min+h, x_min:x_min+w]

        base = Path(image_path).stem
        out_name = f"{base}_roi_{roi_file.stem}.png"
        out_path = Path(output_dir) / out_name

        imsave(str(out_path), crop)
        print(f"Saved template: {out_path}")


# npr
if __name__ == '__main__':
    extract_rois_to_templates(
        image_path = 'data/eugen/K_1_HDPE001.jpg',
        rois_dir   = 'data/eugen/K_1_HDPE001',
        output_dir = 'data/templates'
    )
    print("templates in data/templates.")
