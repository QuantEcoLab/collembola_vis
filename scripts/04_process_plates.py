from __future__ import annotations
from pathlib import Path
import json
import os

from collembola_pipeline.analyze_plate import analyze_plate
from collembola_pipeline.config import CLASSIFIER_THRESHOLD
from tqdm.auto import tqdm

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--plates", type=Path, default=Path("data/plates"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.device.startswith("cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    results = []
    img_paths = sorted(args.plates.glob("*.jpg"))
    for img_path in tqdm(img_paths, desc="Plates"):
        res = analyze_plate(
            img_path, device=args.device, threshold=CLASSIFIER_THRESHOLD, verbose=True
        )
        results.append(res)
        print(json.dumps(res, indent=2))

    (Path("outputs") / "summary.json").write_text(json.dumps(results, indent=2))
    print("Wrote outputs/summary.json")
