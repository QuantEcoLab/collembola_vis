from __future__ import annotations
from pathlib import Path
import json

from collembola_pipeline.config import CALIBRATION_DIR


def save_px_to_um(px_to_um: float):
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    out = CALIBRATION_DIR / 'ruler.json'
    out.write_text(json.dumps({'px_to_um': px_to_um}, indent=2))
    print(f'Saved calibration to {out}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--mm', type=float, required=True, help='Known distance in mm (e.g., 10)')
    p.add_argument('--pixels', type=float, required=True, help='Measured pixels for that distance')
    args = p.parse_args()

    px_to_um = (args.mm / args.pixels) * 1000.0
    save_px_to_um(px_to_um)
    print(f'px_to_um = {px_to_um:.6f}')
