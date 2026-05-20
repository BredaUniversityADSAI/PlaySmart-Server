"""
YOLO Inference and Reaction Time Analysis Script

This script uses a YOLO model to detect objects in a video, focusing on two classes:
- Class 1: "Enemy" (drawn with a red bounding box)
- Class 0: "Ally" (drawn with a green bounding box)

The system annotates the video with these bounding boxes and labels, and detects participant reactions
based on:
- Mouse clicks (from an input log)
- Eye gaze (from a gaze log)

For each instance where an "Enemy" appears, it calculates:
- Visual reaction time
- Click reaction time (if any)
- Gaze reaction time (if any)

Annotated video and reaction times are saved as output.

Author: [imani-Jamir Senior]
Edit: [Louie Daans]
Date: 2025-06-26
"""

import cv2
import csv
import pandas as pd
from ultralytics import YOLO
import torch
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_model(model_path):
    """
    Load a YOLO model from a given file path.

    :param model_path: Path to the YOLO model file.
    :return: Loaded YOLO model.
    """
    return YOLO(model_path)


def load_input_log(
    input_log_file,
):  # REMOVED by @Louie - Reason? - We now load both input and gaze data from a merged file
    """
    Load and filter input log data to extract only mouse click events.

    :param input_log_file: Path to the input log CSV file.
    :return: Filtered pandas DataFrame with mouse click events.
    """
    df = pd.read_csv(input_log_file)
    return df[df["event_type"] == "mouse_click"]


def load_gaze_log(gaze_log_file):
    """
    Load gaze data and rename columns to standard names.

    :param gaze_log_file: Path to the gaze log CSV file.
    :return: Cleaned pandas DataFrame with columns ['unix_time', 'x', 'y'].
    :raises ValueError: If required columns are missing.
    """
    df = pd.read_csv(gaze_log_file)
    logging.info(f"Gaze log columns before rename: {df.columns.tolist()}")
    if not {"unix_time", "screen_x", "screen_y"}.issubset(df.columns):
        raise ValueError(
            "Gaze log file must contain columns 'unix_time', 'screen_x', 'screen_y'"
        )
    df = df[["unix_time", "screen_x", "screen_y"]].rename(
        columns={"screen_x": "x", "screen_y": "y"}
    )
    logging.info(f"Gaze log columns after rename: {df.columns.tolist()}")
    return df


def is_gaze_on_box(gaze_x, gaze_y, box, margin=30):
    """
    Check whether gaze coordinates fall within or near a bounding box.

    :param gaze_x: Gaze x-coordinate.
    :param gaze_y: Gaze y-coordinate.
    :param box: Bounding box tuple (xmin, ymin, xmax, ymax).
    :param margin: Pixel margin around the box to consider.
    :return: True if gaze is within the box (with margin), else False.
    """
    xmin, ymin, xmax, ymax = box
    return (xmin - margin <= gaze_x <= xmax + margin) and (
        ymin - margin <= gaze_y <= ymax + margin
    )


# Changed function process_video by @Louie
def process_video(
    cap, model, input_log, gaze_log=None, threshold=0.7, data_file="reaction_times.csv"
):
    """
    Process video with YOLO detection and overlay reaction time annotations.

    :param cap: OpenCV VideoCapture object.
    :param model: Loaded YOLO model.

    :param threshold: Detection confidence threshold.
    :param output_file: Output path for the annotated video.
    :param data_file: Output path for the reaction time CSV.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_ms = 1000 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    video_start_unix = input_log.iloc[0][
        "unix_time"
    ]  # Assuming video starts at the time of the first mouse click ???

    enemy_active = False
    response_start_ms = None
    click_recorded = False
    gaze_recorded = False
    frames_since_enemy = 0
    max_no_enemy_frames = 5
    reaction_data = []
    instance_count = 1
    persistent_text = None
    persistent_text_frames = 60
    current_enemy_box = None
    click_time_ms = gaze_time_ms = None
    click_rt_ms = gaze_rt_ms = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp_ms = current_frame * frame_interval_ms
        current_unix_time = video_start_unix + timestamp_ms / 1000

        if current_frame % 100 == 0:
            logging.info(
                f"Processing frame {current_frame}/{total_frames} at {timestamp_ms:.2f} ms"
            )

        results = model.predict(frame, conf=threshold, verbose=False, device=device)
        annotated = frame.copy()
        enemy_found = False
        click_time_ms = click_rt_ms = gaze_time_ms = gaze_rt_ms = None

        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())

            if conf >= threshold:
                label = "Enemy" if cls_id == 1 else "Ally"
                color = (0, 0, 255) if cls_id == 1 else (0, 255, 0)
                cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(
                    annotated,
                    f"{label} {conf:.2f}",
                    (xmin, ymin - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                if cls_id == 1:
                    enemy_found = True
                    if not enemy_active:
                        enemy_active = True
                        response_start_ms = timestamp_ms
                        click_recorded = False
                        gaze_recorded = False
                        current_enemy_box = (xmin, ymin, xmax, ymax)
                        persistent_text = None
                        logging.info(
                            f"[Frame {current_frame}] Enemy appeared at {timestamp_ms:.2f} ms"
                        )

                        # @Louie
                        # Add boolean value enemy_visible to new merged dataset
                        # Also add enemy appeared timestamp to new merged dataset
                        # This can be done later

        frames_since_enemy = 0 if enemy_found else frames_since_enemy + 1

        # Handle mouse click detection
        if enemy_active and not click_recorded:
            if response_start_ms is None:
                # Defensive: if response_start_ms somehow not set, skip click detection this frame
                logging.debug(
                    f"Skipping click detection at frame {current_frame} because response_start_ms is None"
                )
            else:
                reaction_start_time = video_start_unix + response_start_ms / 1000
                reaction_end_time = reaction_start_time + 1.0
                clicks = input_log[
                    (input_log["unix_time"] >= reaction_start_time)
                    & (input_log["unix_time"] <= reaction_end_time)
                ]
                if not clicks.empty:
                    first_click_unix = clicks.iloc[0]["unix_time"]
                    click_time_ms = (first_click_unix - video_start_unix) * 1000
                    click_rt_ms = click_time_ms - response_start_ms
                    click_recorded = True
                    persistent_text = f"Click RT: {click_rt_ms:.0f} ms"
                    persistent_text_frames = 60
                    logging.info(
                        f"[Frame {current_frame}] Click at {click_time_ms:.2f} ms (RT: {click_rt_ms:.2f} ms)"
                    )

        # Handle gaze detection
        if enemy_active and not gaze_recorded and gaze_log is not None:
            gaze_points = gaze_log[
                (gaze_log["unix_time"] >= current_unix_time - 1.0)
                & (gaze_log["unix_time"] <= current_unix_time + 1.0)
            ]

            for _, gaze in gaze_points.iterrows():
                if is_gaze_on_box(gaze["x"], gaze["y"], current_enemy_box, margin=30):
                    gaze_time_ms = (gaze["unix_time"] - video_start_unix) * 1000
                    gaze_rt_ms = gaze_time_ms - response_start_ms
                    gaze_recorded = True
                    persistent_text = f"Gaze RT: {gaze_rt_ms:.0f} ms"
                    persistent_text_frames = 60
                    logging.info(
                        f"[Frame {current_frame}] Gaze at {gaze_time_ms:.2f} ms (GRT: {gaze_rt_ms:.2f} ms)"
                    )
                    break

        if persistent_text and persistent_text_frames > 0:
            cv2.putText(
                annotated,
                persistent_text,
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            persistent_text_frames -= 1

        if enemy_active and frames_since_enemy >= max_no_enemy_frames:
            visual_rt = timestamp_ms - response_start_ms
            logging.info(
                f"[Frame {current_frame}] Enemy disappeared at {timestamp_ms:.2f} ms (Visual RT: {visual_rt:.2f} ms)"
            )

            reaction_data.append(
                {
                    "Instance": f"Instance {instance_count}",
                    "Start Time (ms)": response_start_ms,
                    "End Time (ms)": timestamp_ms,
                    "Visual Reaction Time (ms)": visual_rt,
                    "Click Time (ms)": click_time_ms if click_recorded else "N/A",
                    "Click Reaction Time (ms)": (
                        click_rt_ms if click_recorded else "N/A"
                    ),
                    "Gaze Time (ms)": gaze_time_ms if gaze_recorded else "N/A",
                    "Gaze Reaction Time (ms)": gaze_rt_ms if gaze_recorded else "N/A",
                    "seconds": (
                        response_start_ms / 1000.0
                        if response_start_ms is not None
                        else None
                    ),  # Added by @Louie
                }
            )

            instance_count += 1
            enemy_active = False
            response_start_ms = None
            frames_since_enemy = 0
            click_recorded = False
            gaze_recorded = False
            current_enemy_box = None
            persistent_text = None

        cv2.putText(
            annotated,
            f"Frame: {current_frame}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        # out.write(annotated)

    cap.release()
    # out.release()

    # Return reaction_data instead of forcing a CSV write here. The caller
    # may choose to persist the results to disk or process them further.
    logging.info(
        f"Processing complete. Total reaction instances recorded: {len(reaction_data)}"
    )
    return reaction_data


def run_yolo_inference(
    model_path: str,
    video_path: str,
    input_log_path: str,
    gaze_log_path: str,
    threshold: float = 0.8,
    # output_video_path: str = "annotated_output_with_gaze.mp4",
    output_csv_path: str = "reaction_times_with_gaze.csv",
):
    """
    Run the full YOLO inference pipeline with input and gaze logs.

    :param model_path: Path to the YOLO model.
    :param video_path: Path to the input video file.
    :param input_log_path: Path to the input log CSV file.
    :param gaze_log_path: Path to the gaze log CSV file.
    :param threshold: Confidence threshold for YOLO detections.
    :param output_video_path: Output path for annotated video.
    :param output_csv_path: Output path for reaction times CSV.
    :return: Dictionary with status and output paths.
    """
    # Resolve model: if caller didn't provide a valid path, try the backend default model location
    default_model_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "Model_YoloV11_4.pt")
    )
    if not model_path or not os.path.exists(model_path):
        if os.path.exists(default_model_path):
            logging.info(
                f"Model not found at provided path. Using default model at {default_model_path}"
            )
            model_path = default_model_path
        else:
            logging.error(
                f"Model file not found. Checked: '{model_path}' and '{default_model_path}'"
            )
            return {
                "error": "Model file not found",
                "checked_paths": [model_path, default_model_path],
            }

    model = load_model(model_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logging.error("Error: Could not open video.")
        return {"error": "Could not open video."}

    # Load input log (required)
    input_log = load_input_log(input_log_path)

    # Load gaze log. If gaze_log_path not provided or file missing, attempt to use the same merged CSV
    gaze_log = None
    if gaze_log_path and os.path.exists(gaze_log_path):
        try:
            gaze_log = load_gaze_log(gaze_log_path)
        except Exception as e:
            logging.warning(f"Failed to load gaze log '{gaze_log_path}': {e}")
            gaze_log = None
    else:
        # Try loading gaze from the input_log_path (merged CSV case)
        try:
            gaze_log = load_gaze_log(input_log_path)
        except Exception as e:
            logging.info(
                f"No separate gaze log provided or failed to load gaze from input log: {e}"
            )
            gaze_log = None

    reaction_data = process_video(
        cap,
        model,
        input_log,
        gaze_log,
        threshold=threshold,
        data_file=output_csv_path,
    )

    # Optionally write CSV if an output path is provided
    if output_csv_path:
        try:
            with open(output_csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Instance",
                        "Start Time (ms)",
                        "End Time (ms)",
                        "Visual Reaction Time (ms)",
                        "Click Time (ms)",
                        "Click Reaction Time (ms)",
                        "Gaze Time (ms)",
                        "Gaze Reaction Time (ms)",
                        "seconds",
                    ],
                )
                writer.writeheader()
                writer.writerows(reaction_data)
            logging.info(f"Saved reaction CSV to {output_csv_path}")
        except Exception as e:
            logging.warning(f"Failed to write CSV to {output_csv_path}: {e}")

    return {
        "message": "YOLO inference completed.",
        "reaction_data": reaction_data,
        "output_csv_path": output_csv_path,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 5:
        print(
            "Usage: python script.py <model_path> <video_path> <input_log_path> <gaze_log_path>"
        )
        sys.exit(1)

    model_path = sys.argv[1]
    video_path = sys.argv[2]
    input_log_path = sys.argv[3]
    gaze_log_path = sys.argv[4]

    result = run_yolo_inference(model_path, video_path, input_log_path, gaze_log_path)
    print(result)
