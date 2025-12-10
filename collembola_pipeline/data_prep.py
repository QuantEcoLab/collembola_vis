from __future__ import annotations
from pathlib import Path
import pandas as pd

from .config import DATA_DIR, CROPS_DIR, ANNOTATIONS_DIR


def build_unified_dataset(
    out_train: Path, out_val: Path, val_frac: float = 0.2, seed: int = 42
) -> None:
    """
    Build a unified organism-vs-junk dataset from crops_dataset.csv and collembolas_table.csv.
    Writes train.csv and val.csv with columns: img_path,label
    """
    crops_csv = ANNOTATIONS_DIR / "crops_dataset.csv"
    gt_csv = ANNOTATIONS_DIR / "collembolas_table.csv"

    df_crops = pd.read_csv(crops_csv)
    # Expect columns: crop_path or id, collembola True/False
    # Try to resolve image path column
    path_col = next(
        (
            c
            for c in df_crops.columns
            if ("path" in c.lower())
            or ("file" in c.lower())
            or (c.lower() in ("crop_id", "img_path"))
        ),
        None,
    )
    if path_col is None:
        # Fallback: assume 'crop' or 'id' that can be joined to CROPS_DIR
        if "crop" in df_crops.columns:
            df_crops["img_path"] = df_crops["crop"].apply(
                lambda x: str((CROPS_DIR / str(x)).as_posix())
            )
        elif "crop_id" in df_crops.columns:
            df_crops["img_path"] = df_crops["crop_id"].astype(str)
        elif "id" in df_crops.columns:
            df_crops["img_path"] = df_crops["id"].apply(
                lambda x: str((CROPS_DIR / f"{x}.png").as_posix())
            )
        else:
            raise ValueError("crops_dataset.csv must contain a path-like column")
    else:
        df_crops["img_path"] = df_crops[path_col].astype(str)

    label_col = next(
        (
            c
            for c in df_crops.columns
            if c.lower() in ("collembola", "label", "is_collembola")
        ),
        None,
    )
    if label_col is None:
        raise ValueError(
            "crops_dataset.csv must contain a boolean/label column for collembola"
        )
    df_crops["label"] = df_crops[label_col].astype(int)

    # Ground truth positives: use generated crops at data/crops/gt_positive/{id}.jpg
    df_gt = pd.read_csv(gt_csv)
    if "id_collembole" not in df_gt.columns:
        raise ValueError("collembolas_table.csv must have id_collembole")
    df_gt["img_path"] = (
        df_gt["id_collembole"]
        .astype(str)
        .apply(lambda s: str((CROPS_DIR / "gt_positive" / f"{s}.jpg").as_posix()))
    )
    df_gt["label"] = 1

    # Build positives from crops where label==1 and all gt rows
    df_pos = pd.concat(
        [
            df_crops[df_crops["label"] == 1][["img_path", "label"]],
            df_gt[["img_path", "label"]],
        ],
        ignore_index=True,
    )

    # Negatives from crops where label==0
    df_neg = df_crops[df_crops["label"] == 0][["img_path", "label"]]

    df_all = pd.concat([df_pos, df_neg], ignore_index=True).drop_duplicates()

    # Train/val split
    df_all = df_all.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_val = int(len(df_all) * val_frac)
    df_val = df_all.iloc[:n_val]
    df_train = df_all.iloc[n_val:]

    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--train", type=Path, default=Path("data/train.csv"))
    p.add_argument("--val", type=Path, default=Path("data/val.csv"))
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    build_unified_dataset(args.train, args.val, args.val_frac, args.seed)
