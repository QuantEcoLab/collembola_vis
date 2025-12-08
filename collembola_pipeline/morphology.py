from __future__ import annotations
import numpy as np
from skimage.measure import regionprops

from .config import PX_TO_UM


def mask_measurements(mask: np.ndarray) -> dict:
    props = regionprops(mask.astype(np.uint8))
    if not props:
        return {}
    r = props[0]
    major_axis_px = float(r.major_axis_length)
    minor_axis_px = float(r.minor_axis_length)
    length_um = major_axis_px * PX_TO_UM
    width_um = minor_axis_px * PX_TO_UM

    a = length_um / 2.0
    b = width_um / 2.0
    volume_um3 = (4.0 / 3.0) * np.pi * a * (b**2)

    return dict(
        area_px=int(r.area),
        perimeter_px=float(getattr(r, "perimeter", 0.0)),
        major_axis_px=major_axis_px,
        minor_axis_px=minor_axis_px,
        length_um=length_um,
        width_um=width_um,
        volume_um3=volume_um3,
        eccentricity=float(r.eccentricity),
        solidity=float(r.solidity),
    )
