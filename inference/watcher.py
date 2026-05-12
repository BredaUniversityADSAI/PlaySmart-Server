"""
Inference Watcher Service
=========================

This script runs inside the inference container on queenbee. It does two things:

1. AUTOMATIC: Watches the merged folder for new CSV+video pairs and triggers
   inference on the GPU PC automatically when a new session is detected.

2. MANUAL: Exposes a FastAPI endpoint so you can trigger inference manually
   for testing or reprocessing without restarting anything.

Folder structure it expects:
    /mnt/raid0/esports/sftp_data/merged/
        11th_game_P045_valorant_22-06-2025_18-01-06_merged.csv
        11th_game_P045_valorant_22-06-2025_18-01-06.mp4
        test_session/
            11th_game_P045_valorant_22-06-2025_18-01-06_merged.csv
            11th_game_P045_valorant_22-06-2025_18-01-06.mp4

Naming convention:
    video:  <base_name>.mp4
    csv:    <base_name>_merged.csv

A session is considered ready when BOTH files are present.
Once processed, a marker file (<base_name>.done) is created so it won't
be processed again on the next watch cycle.
"""

import os
import csv
import time
import logging
import threading
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from inference_client import run_inference_for_session

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
STALE_AFTER = 3600

# ── Config from environment variables ─────────────────────────────────────────
# These are set in docker-compose.yml so you can change them without
# rebuilding the image.
WATCH_FOLDER = os.getenv("WATCH_FOLDER", "/mnt/raid0/esports/sftp_data/merged")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", "/mnt/raid0/esports/sftp_data/video")
RESULTS_FOLDER = os.getenv("RESULTS_FOLDER", "/mnt/raid0/esports/sftp_data/results")
DONE_FOLDER = os.path.join(os.getenv("WATCH_FOLDER", "/mnt/raid0/esports/sftp_data/merged"), "done")
GPU_PC_URL = os.getenv("GPU_PC_URL", "http://10.4.28.236:8000")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "30"))  # seconds between watch cycles

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Inference Watcher",
    description="Watches for new sessions and triggers GPU inference",
    version="1.0.0",
)


# ── Helper functions ──────────────────────────────────────────────────────────

# def find_sessions(merged_folder: str, video_folder: str) -> list:
#     """
#     Scan a folder for complete sessions (paired .mp4 + _merged.csv).
#     Recurses one level deep to support subfolders like test_session/.

#     Returns a list of dicts with video_path, csv_path, and base_name.
#     Skips sessions that already have a .done marker file.
#     """
#     sessions = []

#     # Check both the root folder and one level of subfolders
#     folders_to_check = [folder]
#     for entry in os.scandir(folder):
#         if entry.is_dir():
#             folders_to_check.append(entry.path)

#     for check_folder in folders_to_check:
#         try:
#             files = os.listdir(check_folder)
#         except PermissionError:
#             continue

#         # Find all mp4 files and check if a matching merged CSV exists
#         for f in files:
#             if not f.endswith(".mp4"):
#                 continue

#             base_name = f[:-4]  # strip .mp4
#             csv_name = f"{base_name}_merged.csv"
#             done_marker = f"{base_name}.done"
#             inProgress_maker = f"{base_name}.inProgress"

#             if csv_name in files and done_marker not in files and inProgress_maker not in files:
#                 sessions.append({
#                     "base_name": base_name,
#                     "video_path": os.path.join(check_folder, f),
#                     "csv_path": os.path.join(check_folder, csv_name),
#                     "done_marker": os.path.join(check_folder, done_marker),
#                     "inProgress_maker": os.path.join(check_folder, inProgress_maker),
#                 })

#     return sessions

def find_sessions(merged_folder: str, video_folder: str) -> list:
    os.makedirs(DONE_FOLDER, exist_ok=True)
    sessions = []

    try:
        csv_files = os.listdir(merged_folder)
    except PermissionError:
        return sessions

    for f in csv_files:
        if not f.endswith("_merged.csv"):
            continue

        base_name = f[:-len("_merged.csv")]
        video_name = f"{base_name}.mp4"
        done_marker = os.path.join(DONE_FOLDER, f"{base_name}.done")
        inprogress_marker = os.path.join(merged_folder, f"{base_name}.inprogress")

        video_path = os.path.join(video_folder, video_name)

        if (
            os.path.exists(video_path)
            and not os.path.exists(done_marker)
        ):
            sessions.append({
                "base_name": base_name,
                "video_path": video_path,
                "csv_path": os.path.join(merged_folder, f),
                "done_marker": done_marker,
                "inprogress_marker": inprogress_marker,
            })

    return sessions


def process_session(session: dict):
    """
    Run inference for a single session and write results.
    Creates a .done marker file on success so it won't be reprocessed.
    """
    base_name = session["base_name"]
    inprogress_marker = session["inprogress_marker"]

    # Handle existing in-progress marker
    if os.path.exists(inprogress_marker):
        age = time.time() - os.path.getmtime(inprogress_marker)
        if age > STALE_AFTER:
            logger.warning(f"Reclaiming stale marker (age {age:.0f}s): {inprogress_marker}")
            os.remove(inprogress_marker)
        else:
            logger.info(f"Session already in progress, skipping: {base_name}")
            return {"status": "skipped", "reason": "in_progress"}

    # Claim the session
    with open(inprogress_marker, "x") as f:
        f.write(str(os.getpid()))

    logger.info(f"Processing session: {base_name}")

    try:
        reaction_data = run_inference_for_session(
            video_path=session["video_path"],
            csv_path=session["csv_path"],
        )

        # Write results
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        result_path = os.path.join(RESULTS_FOLDER, f"{base_name}_reaction_times.csv")
        with open(result_path, "w", newline="") as f:
            if reaction_data:
                writer = csv.DictWriter(f, fieldnames=reaction_data[0].keys())
                writer.writeheader()
                writer.writerows(reaction_data)
        logger.info(f"Results written to {result_path}")

        # Mark done
        os.makedirs(DONE_FOLDER, exist_ok=True)
        with open(session["done_marker"], "w") as f:
            f.write("processed")
        logger.info(f"Session {base_name} marked as done.")

        return {"status": "complete", "instances": len(reaction_data), "result_path": result_path}

    except Exception as e:
        logger.exception(f"Failed to process session {base_name}")
        return {"status": "failed", "error": str(e)}

    finally:
        # Always remove the marker, success or failure
        if os.path.exists(inprogress_marker):
            os.remove(inprogress_marker)


# ── Background watcher thread ─────────────────────────────────────────────────

def watch_loop():
    """
    Runs in a background thread. Every POLL_INTERVAL seconds it scans the
    watch folder for new complete sessions and processes them automatically.
    """
    logger.info(f"Watcher started. Watching: {WATCH_FOLDER} every {POLL_INTERVAL}s")
    while True:
        sessions = find_sessions(WATCH_FOLDER, VIDEO_FOLDER)
        if sessions:
            logger.info(f"Found {len(sessions)} new session(s) to process.")
            for session in sessions:
                process_session(session)
        else:
            logger.info("No new sessions found.")
        time.sleep(POLL_INTERVAL)


# Start the watcher in a background thread when the container starts
watcher_thread = threading.Thread(target=watch_loop, daemon=True)
watcher_thread.start()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if the watcher is running."""
    return {
        "status": "ok",
        "watch_folder": WATCH_FOLDER,
        "gpu_pc": GPU_PC_URL,
        "poll_interval_seconds": POLL_INTERVAL,
    }


@app.get("/sessions")
def list_sessions():
    """
    List all sessions that are ready to process (have both video + CSV)
    and haven't been processed yet.
    Useful for checking what's in the watch folder before triggering manually.
    """
    sessions = find_sessions(WATCH_FOLDER, VIDEO_FOLDER)
    return {
        "pending_sessions": [s["base_name"] for s in sessions],
        "count": len(sessions),
    }


class TriggerRequest(BaseModel):
    session_path: str  # full path to the folder containing the session files
    base_name: str     # e.g. "11th_game_P045_valorant_22-06-2025_18-01-06"


@app.post("/trigger-scan")
def trigger_scan():
    """
    Manually trigger a scan of WATCH_FOLDER and VIDEO_FOLDER right now,
    exactly as the background watcher would. No input needed.
    """
    sessions = find_sessions(WATCH_FOLDER, VIDEO_FOLDER)

    if not sessions:
        return {"status": "no_sessions", "message": "No pending sessions found."}

    results = []
    for session in sessions:
        result = process_session(session)
        results.append({"base_name": session["base_name"], "result": result})

    return {"status": "done", "processed": results}


@app.post("/trigger")
def manual_trigger(req: TriggerRequest):
    """
    Manually trigger inference for a specific session.
    Use this for testing or reprocessing without waiting for the watcher cycle.

    Example:
        curl -X POST http://localhost:8001/trigger \\
          -H "Content-Type: application/json" \\
          -d '{
            "session_path": "/mnt/raid0/esports/sftp_data/merged/test_session",
            "base_name": "11th_game_P045_valorant_22-06-2025_18-01-06"
          }'
    """
    video_path = os.path.join(req.session_path, f"{req.base_name}.mp4")
    csv_path = os.path.join(req.session_path, f"{req.base_name}_merged.csv")
    done_marker = os.path.join(req.session_path, f"{req.base_name}.done")

    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail=f"Video not found: {video_path}")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=400, detail=f"CSV not found: {csv_path}")

    session = {
        "base_name": req.base_name,
        "video_path": video_path,
        "csv_path": csv_path,
        "done_marker": done_marker,
        "inprogress_marker": os.path.join(req.session_path, f"{req.base_name}.inprogress"),
    }

    result = process_session(session)
    return result
