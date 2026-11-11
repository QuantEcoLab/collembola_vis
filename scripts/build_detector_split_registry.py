#!/usr/bin/env python3
"""Read-only inventory of existing split membership; never modifies datasets or trains."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.detector_registry import build_registry
from backend.config import settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, action='append', required=True)
    parser.add_argument('--exclude', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    app_paths = [settings.uploads_dir, settings.outputs_dir, settings.calibration_dir, settings.detector_dataset_dir]
    previous = json.loads(args.output.read_text()) if args.output.is_file() else None
    result = build_registry([p.resolve() for p in args.root],
                            [p.resolve() for p in args.exclude + app_paths] + [args.output.resolve()],
                            ROOT / 'scripts/create_tiled_dataset.py', previous=previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(args.output)
    print(json.dumps({'registry': str(args.output), 'sources': len(result['source_memberships']), 'images': len(result['images'])}))


if __name__ == '__main__':
    main()
