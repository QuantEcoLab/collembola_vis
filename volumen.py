import numpy as np

def compute_collembola_volume(width_px: float,
                               height_px: float,
                               um_per_pixel: float,
                               model: str = "ellipsoid") -> float:
  
    w_um = width_px * um_per_pixel
    h_um = height_px * um_per_pixel

    if model == "ellipsoid":
        # Elipsoid: polu-osi: a = visina/2, b = c = širina/2
        a = h_um / 2.0
        b = w_um / 2.0
        c = b
        volume = (4.0 / 3.0) * np.pi * a * b * c

    elif model == "cylinder":
        # Valjak: visina = dulja dimenzija, promjer = kraća dimenzija
        radius = w_um / 2.0
        volume = np.pi * (radius ** 2) * h_um

    else:
        raise ValueError(
            f"Nepodržani model '{model}'. Odaberite 'ellipsoid' ili 'cylinder'."
        )

    return volume
