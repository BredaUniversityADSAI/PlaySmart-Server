"""
Inference Client — runs on queenbee (pipeline container)
=========================================================

HOW THIS FITS IN THE PIPELINE
──────────────────────────────
This file lives on queenbee, inside the pipeline container (alongside your
auto_merge.py / auto_cleaning_and_dq.py scripts). After the data pipeline has
produced a merged CSV and a gameplay video is available, this client triggers
the GPU PC to run YOLO inference and waits for the results.

The GPU PC never needs to know anything about queenbee's internals — it just
receives file paths and returns JSON. This client handles all the coordination.
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
# The GPU PC's IP on the Hive LAN. Give it a static IP or DHCP reservation
# so this never changes. Port 8000 is where main.py listens.
# INFERENCE_SERVICE_URL = "http://10.4.28.236:8000"  # ← replace X with GPU PC's IP
INFERENCE_SERVICE_URL = os.getenv("GPU_PC_URL", "http://10.4.28.236:8000")

# How long to wait between status polls (seconds)
POLL_INTERVAL = 10

# Maximum total wait time before giving up (seconds). 20 min should cover
# even long gameplay recordings.
MAX_WAIT = 1200

def to_windows_path(linux_path: str) -> str:
    """
    Translate queenbee Linux paths to GPU PC Windows SMB paths.
    /mnt/raid0/esports/ → Z:/
    """
    prefix = "/mnt/raid0/esports/"
    if linux_path.startswith(prefix):
        relative = linux_path[len(prefix):]
        return "Z:/" + relative
    return linux_path

def check_service_health() -> bool:
    """
    Ping the GPU PC's health endpoint before submitting a job.
    Returns True if the service is up and CUDA is available, False otherwise.

    Call this at the start of your pipeline run so you fail fast and loud
    if the GPU PC is off rather than discovering it after a 10-minute wait.
    """
    try:
        resp = requests.get(f"{INFERENCE_SERVICE_URL}/health", timeout=5)
        data = resp.json()
        if data.get("status") == "ok":
            logger.info(f"GPU PC is up. GPU: {data.get('gpu', 'unknown')}")
            return True
        logger.warning(f"GPU PC health check returned unexpected response: {data}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Could not connect to inference service at {INFERENCE_SERVICE_URL}")
        return False


def submit_inference_job(
        video_path: str,
        csv_path: str,
        threshold: float = 0.8,
        batch_size: int = 16,
        use_batch: bool = True,
    ) -> str:
    """
    Submit an inference job to the GPU PC and return the job ID.

    The paths here are from the GPU PC's perspective — they're the SMB-mounted
    paths that the GPU PC uses to read files from queenbee's storage.

    Example:
        video_path = "Z:/uploads/player_01/session_03/gameplay.mp4"
        csv_path   = "Z:/uploads/player_01/session_03/merged.csv"

    These are the same files that live on queenbee — the GPU PC just sees them
    as a network drive rather than a local path.
    """
    payload = {
        "video_path": to_windows_path(video_path),
        "csv_path": to_windows_path(csv_path),
        "threshold": threshold,
        "batch_size": batch_size,
        "use_batch": use_batch,
    }

    resp = requests.post(f"{INFERENCE_SERVICE_URL}/predict", json=payload, timeout=10)
    if not resp.ok:
        logger.error(
            f"GPU service returned {resp.status_code} for /predict.\n"
            f"Payload sent: {payload}\n"
            f"Response body: {resp.text}"
        )
    resp.raise_for_status()  # raises HTTPError if the service returns 4xx/5xx

    job_id = resp.json()["job_id"]
    logger.info(f"Inference job submitted. Job ID: {job_id}")
    return job_id


def wait_for_result(job_id: str) -> dict:
    """
    Poll the GPU PC until the inference job is done, then return the results.

    Polling (repeatedly asking "are you done yet?") is simpler than webhooks
    for this use case — the Hive has one GPU PC and one caller, so there's no
    need for a more complex event-driven setup.

    Returns the full result dict including reaction_data on success.
    Raises RuntimeError on failure or timeout.
    """
    elapsed = 0

    while elapsed < MAX_WAIT:
        resp = requests.get(f"{INFERENCE_SERVICE_URL}/jobs/{job_id}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]

        if status == "complete":
            logger.info(f"Job {job_id} complete. "
                        f"{len(data['reaction_data'])} reaction instances found.")
            return data

        elif status == "failed":
            raise RuntimeError(f"Inference job failed: {data.get('error', 'unknown error')}")

        else:
            # still queued or running — wait and try again
            logger.info(f"Job {job_id} status: {status}. Waiting {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

    raise TimeoutError(f"Inference job {job_id} did not complete within {MAX_WAIT}s")


def run_inference_for_session(
    video_path: str,
    csv_path: str,
    threshold: float = 0.8,
) -> list:
    """
    Full end-to-end helper: health check → submit → poll → return results.

    This is the single function your pipeline container needs to call.
    Drop it after auto_merge.py has finished producing the merged CSV.

    Returns:
        List of reaction time dicts, e.g.:
        [
            {
                "Instance": "Instance 1",
                "Start Time (ms)": 1234.5,
                "Visual Reaction Time (ms)": 387.2,
                "Click Reaction Time (ms)": 412.0,
                ...
            },
            ...
        ]
    """
    if not check_service_health():
        raise ConnectionError("GPU PC inference service is not available.")

    job_id = submit_inference_job(video_path, csv_path, threshold)
    result = wait_for_result(job_id)
    return result["reaction_data"]


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    # These paths are what the GPU PC sees via its SMB mount of queenbee's storage
    reaction_times = run_inference_for_session(
        video_path="Z:/uploads/player_01/session_03/gameplay.mp4",
        csv_path="Z:/uploads/player_01/session_03/merged.csv",
    )

    for instance in reaction_times:
        print(instance)
