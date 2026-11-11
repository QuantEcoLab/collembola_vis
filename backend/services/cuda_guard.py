"""Keep GPU jobs working in the shared backend process.

Ultralytics' ``select_device("cpu")`` sets ``CUDA_VISIBLE_DEVICES=""`` for the
whole process. If torch enumerates CUDA devices while the variable is empty, it
caches "no GPU" for the rest of the process and every later GPU detection fails
until the service restarts. To prevent that:

* CPU runs hand ultralytics a ``torch.device`` (``scripts.infer_tiled.ultralytics_device``),
  so the variable is never cleared by our own code;
* app startup calls ``pin_cuda_runtime`` so torch enumerates CUDA once with the
  environment the service was started with; the cached answer then survives
  any later clobbering by other code;
* GPU detections call ``ensure_inference_device`` first, which restores a cleared
  variable and fails with a clear message if CUDA is unusable in this process.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Captured when the backend imports this module at startup, before any job runs.
STARTUP_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
_env_lock = threading.Lock()


def is_cpu_device(device) -> bool:
    return str(device).strip().lower() == "cpu"


def restore_cuda_visible_devices() -> bool:
    """Undo a process-wide ``CUDA_VISIBLE_DEVICES=""`` left behind by a CPU run."""
    with _env_lock:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or STARTUP_CUDA_VISIBLE_DEVICES == "":
            return False
        if STARTUP_CUDA_VISIBLE_DEVICES is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = STARTUP_CUDA_VISIBLE_DEVICES
    logger.warning("CUDA_VISIBLE_DEVICES was cleared by a CPU run; restored to %r", STARTUP_CUDA_VISIBLE_DEVICES)
    return True


def pin_cuda_runtime() -> bool:
    """Let torch enumerate CUDA devices now, with the startup environment.

    torch caches the device count for the life of the process, so a later CPU
    run that clears ``CUDA_VISIBLE_DEVICES`` can no longer hide the GPUs.
    """
    restore_cuda_visible_devices()
    try:
        import torch
    except ImportError:
        return False
    available = torch.cuda.is_available()
    logger.info("CUDA pinned at startup: available=%s, CUDA_VISIBLE_DEVICES=%r", available,
                os.environ.get("CUDA_VISIBLE_DEVICES"))
    return available


def ensure_inference_device(device) -> None:
    """Check a GPU device before inference; CPU devices need no check."""
    if is_cpu_device(device):
        return
    restore_cuda_visible_devices()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"GPU '{device}' nije dostupan u procesu poslužitelja (torch.cuda.is_available() = False; "
            f"CUDA_VISIBLE_DEVICES pri pokretanju: {STARTUP_CUDA_VISIBLE_DEVICES!r}, sada: "
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}). Ako je GPU prije radio, CUDA je u ovom procesu "
            "inicijaliziran bez vidljivih GPU-ova; ponovno pokrenite servis collembola."
        )
