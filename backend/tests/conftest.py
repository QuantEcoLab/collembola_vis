"""Backend tests must never open the existing app/review databases or output queues."""
import atexit
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_runtime = tempfile.TemporaryDirectory(prefix="exp03-backend-tests-", dir="/tmp/opencode")
atexit.register(_runtime.cleanup)
for key, suffix in {"DATA_DIR": "data", "UPLOADS_DIR": "uploads", "OUTPUTS_DIR": "outputs", "CALIBRATION_DIR": "calibration", "ANNOTATIONS_DIR": "annotations"}.items():
    path = Path(_runtime.name) / suffix
    path.mkdir(exist_ok=True)
    os.environ["COLLEMBOLA_" + key] = str(path)
os.environ["ROOT_PATH"] = ""
os.environ["COLLEMBOLA_FRONTEND_DIST"] = str(Path(_runtime.name) / "no-spa")
os.environ["YOLO_CONFIG_DIR"] = str(Path(_runtime.name) / "yolo-settings")
os.environ["YOLO_AUTOINSTALL"] = "false"
