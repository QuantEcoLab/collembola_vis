# Repository Guidelines
## Project Structure & Module Organization
- `mk_dataset.py` orchestrates dataset inspection and collembola detection heuristics; it reads imagery from `data/slike/` and annotations in `data/collembolas_table.csv`.
- `volumen.py` exposes `compute_collembola_volume`; import this helper instead of duplicating formulas in experiments.
- `jana_code/` hosts research prototypes (blob detection, cropping, measurement scripts) and derived assets (`crops/`, `masks/`, `slike/`); keep exploratory work here and note promotion-ready versions in `dz.md`.
- `data/` stores raw imagery, generated crops, and CSV metadata; never overwrite source JPEGs—create new subdirectories for derived datasets or measurements.

## Environment, Build & Dev Commands
- `conda activate collembola` switches into the shared development environment (create once with `conda create -n collembola python=3.11` if missing).
- `pip install pandas scikit-image matplotlib tqdm numpy openpyxl` inside the `collembola` env restores current dependencies.
- `python mk_dataset.py` visualizes detections against annotations and prints coverage stats; rerun after algorithm changes.
- `python jana_code/sve_jedinke.py` regenerates Excel summaries (`sve_jedinke_collembole.xlsx`, `sazetak_collembole_sa_devijacijom.xlsx`) from the raw ROI CSV.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and snake_case for functions and variables; keep module names lowercase (`volumen.py`).
- Use type hints for public helpers (see `compute_collembola_volume`) and add docstrings describing image and unit expectations.
- Prefer `Path` objects and explicit encodings when loading data; replace hard-coded Windows paths with forward-slash relative paths.

## Testing Guidelines
- Ship new utilities with `pytest` coverage under `tests/` (e.g. `tests/test_volumen.py`) covering unit conversions and edge cases.
- Use fixtures or factory functions for sample CSV rows or synthetic images instead of committing large binaries.
- Run `pytest -q` locally before opening a pull request and call out any skipped integration checks.

## Commit & Pull Request Guidelines
- Match existing history: concise, imperative summaries (“zadatak …”, “Update volumen.py”) and stick to one language per commit.
- Reference issues or task IDs in the body, list dataset versions touched, and call out regenerated artifacts.
- Pull requests should report the intent, include screenshots or plots when detection output changes, and note exact reproduction steps (`python mk_dataset.py`, resulting Excel paths).

## Data & Asset Handling
- Keep bulk crops and masks out of version control; track provenance in `jana_code/dz.md` and commit only lightweight metadata.
- When exporting new templates, add README notes under `data/` describing filename patterns and measurement units.
- Store credentials or API keys outside the repo and rely on environment variables for any future cloud pipelines.
