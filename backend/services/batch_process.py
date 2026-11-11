"""Combined project processing delegates to the sole current measurement method."""
from backend.services.batch_measurement import run_batch_measurement


def run_batch_process(job, progress_callback):
    # The common worker detects only if no saved masks/detections are available.
    # It never replaces existing human annotation files with auto-accepted boxes.
    return run_batch_measurement(job, progress_callback)
