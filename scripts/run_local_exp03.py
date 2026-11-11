#!/usr/bin/env python3
"""Launch the integration branch locally with isolated app data and a verified checkpoint."""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = "3fe97dd43f387815f8ee947eb5d02a6b0a7afadc1be59942a6b4a1f49f9aa0da"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8774)
    parser.add_argument("--frontend-dist", type=Path, help="Explicit immutable preview build directory")
    args = parser.parse_args()
    os.chdir(ROOT)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT).decode().strip()
    if branch not in {"feature/integrate-exp03-no-os", "feature/curated-detector-dataset", "feature/exp03-centerline-and-curation"}:
        raise SystemExit("Local preview must run from an integration feature branch")
    checkpoint = ROOT / "outputs/ruler_v7_reviewed367_ft_exp03_no_os/evaluation/fixed_candidate/best.pt"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != EXPECTED_SHA256:
        raise SystemExit("Checkpoint SHA-256 mismatch; app not started")
    local = ROOT / "outputs/integration_exp03_local"
    active_file = local / "active_frontend.json"
    if not active_file.is_file():
        raise SystemExit("Run scripts/build_local_exp03_frontend.py first; stale legacy dist is not a fallback")
    active = json.loads(active_file.read_text())
    dist = args.frontend_dist.resolve() if args.frontend_dist else Path(active["dist"]).resolve()
    if not (dist / "index.html").is_file():
        raise SystemExit("Versioned frontend index missing")
    manifest = json.loads((dist / "preview-build.json").read_text())
    if manifest["build_id"] != active["build_id"]:
        raise SystemExit("Requested dist differs from the selected active preview build")
    for relative, expected in manifest["files"].items():
        if hashlib.sha256((dist / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit("Preview build hash mismatch: " + relative)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(local / "runtime_dependencies"))
    sys.path.insert(0, str(ROOT))
    for key, suffix in {"DATA_DIR": "data", "UPLOADS_DIR": "uploads", "OUTPUTS_DIR": "web_outputs",
                        "CALIBRATION_DIR": "calibration", "ANNOTATIONS_DIR": "annotations"}.items():
        os.environ["COLLEMBOLA_" + key] = str(local / "app_data" / suffix)
    os.environ["COLLEMBOLA_FRONTEND_DIST"] = str(dist)
    os.environ["COLLEMBOLA_LOCAL_PREVIEW"] = "1"
    os.environ["COLLEMBOLA_DETECTOR_DATASET_DIR"] = str(local / "app_data" / "curated_detector")
    os.environ["COLLEMBOLA_DETECTOR_SPLIT_REGISTRY"] = str(local / "detector_split_registry.json")
    os.environ["ROOT_PATH"] = ""
    os.environ["YOLO_CONFIG_DIR"] = str(local / "runtime_settings")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    if not os.environ.get("AUTH_SECRET_KEY"):
        key_file = local / ".preview_auth_secret"
        if not key_file.exists():
            with os.fdopen(os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
                handle.write(secrets.token_urlsafe(32))
        os.environ["AUTH_SECRET_KEY"] = key_file.read_text().strip()
    import torch
    import uvicorn
    torch.set_num_threads(4)
    metadata = {"branch": branch, "port": args.port, "url": f"http://127.0.0.1:{args.port}/workspace",
                "checkpoint_sha256": digest, "isolated_data": str(local / "app_data"),
                "frontend_dist": str(dist), "frontend_build_id": manifest["build_id"],
                "frontend_built_at_utc": manifest["built_at_utc"], "frontend_entrypoint": manifest["entrypoint_js"],
                "frontend_entrypoint_sha256": manifest["entrypoint_sha256"], "pid": os.getpid(), "torch_threads": 4}
    (local / "server_connection.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
