from __future__ import annotations
from pathlib import Path

from collembola_pipeline.data_prep import build_unified_dataset

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--train', type=Path, default=Path('data/train.csv'))
    p.add_argument('--val', type=Path, default=Path('data/val.csv'))
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    build_unified_dataset(args.train, args.val, args.val_frac, args.seed)
    print(f'Wrote {args.train} and {args.val}')
