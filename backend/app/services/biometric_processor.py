import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
import os

# Updated to use the same data directory structure as videos
# CSVs should be in the parent 'data' folder
DATA_DIR = os.getenv("DATA_DIR", "/app/data")


def convert_to_native_types(obj: Any) -> Any:
    if isinstance(obj, (np.integer, np.int64, np.int32)):  # type: ignore
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):  # type: ignore
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    return obj


# def _find_csv_file(
#     filename: str, data_directory: Optional[str] = None
# ) -> Optional[str]:
#     """
#     Find CSV file in possible locations.
#     Returns the full path if found, None otherwise.
#     """
#     # Use provided data directory or fall back to hardcoded path
#     if data_directory:
#         base_data_dir = data_directory
#     else:
#         base_data_dir = r"C:\Users\Endijs\Downloads\research_software_copy\research_software_copy\data"

#     possible_paths = [
#         os.path.join(
#             base_data_dir, "merged", filename
#         ),  # In data/merged subfolder (PRIMARY)
#         os.path.join(base_data_dir, filename),  # Directly in data folder
#         os.path.join(base_data_dir, "csvs", filename),  # In data/csvs subfolder
#         os.path.join("data/csvs", filename),  # Fallback to local data/csvs
#     ]

#     for path in possible_paths:
#         if os.path.exists(path):
#             return path

#     return None



def _find_csv_file(filename: str, data_directory: Optional[str] = None) -> Optional[str]:
    """
    Find CSV file in possible locations.
    Returns the full path if found, None otherwise.
    """
    # Use provided data directory or fall back to hardcoded path
    base_data_dir = data_directory or DATA_DIR
    possible_paths = [
        os.path.join(base_data_dir, "merged", filename), # In data/merged subfolder (PRIMARY)
        os.path.join(base_data_dir, filename), # Directly in data folder
        os.path.join(base_data_dir, "csvs", filename), # In data/csvs subfolder
        os.path.join("data/csvs", filename), # Fallback to local data/csvs
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


# Removed @lru_cache because data_directory parameter can't be cached properly
def load_biometric_data(
    filename: str, data_directory: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Load biometric CSV data from file.

    Args:
        filename: Name of the CSV file
        data_directory: Optional base data directory path from toolkit configuration
    """
    file_path = _find_csv_file(filename, data_directory)

    if not file_path:
        print(f"❌ File not found: {filename}")
        if data_directory:
            print(f"   Searched in data_directory: {data_directory}")
        return None

    print(f"✅ Found CSV file at: {file_path}")

    try:
        df = pd.read_csv(file_path)

        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(
                df["datetime"], format="%M:%S.%f", errors="coerce"
            )

        df_gameplay = df[df["event_type"].notna()].copy()

        if len(df_gameplay) < 100:
            print(f"Warning: Only {len(df_gameplay)} gameplay rows, using all data")
            df_gameplay = df.copy()

        print(
            f"Loaded {len(df)} total rows, {len(df_gameplay)} gameplay rows from {filename}"
        )
        return df_gameplay

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_emotion_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze emotion data
    """
    if df is None or df.empty:
        return {"error": "No data"}

    high_conf = df[df["confidence"] > 0.5].copy()

    if high_conf.empty:
        return {"error": "No high-confidence emotion data"}

    emotion_counts = high_conf["emotion"].value_counts()
    total = len(high_conf)

    emotion_distribution = {}
    chart_data = []

    for emotion, count in emotion_counts.items():
        percentage = float(round((count / total) * 100, 1))
        emotion_distribution[str(emotion)] = {
            "count": int(count),
            "percentage": percentage,
        }
        chart_data.append({"name": str(emotion), "value": percentage})

    dominant_emotion = (
        str(emotion_counts.idxmax()) if not emotion_counts.empty else "Unknown"
    )
    dominant_percentage = (
        float(round((emotion_counts.max() / total) * 100, 1))
        if not emotion_counts.empty
        else 0
    )

    # Average confidence
    avg_confidence = float(round(high_conf["confidence"].mean() * 100, 1))

    return {
        "dominant_emotion": dominant_emotion,
        "dominant_percentage": dominant_percentage,
        "emotion_distribution": emotion_distribution,
        "average_confidence": avg_confidence,
        "total_samples": int(total),
        "chart_data": chart_data,
    }


def get_gaze_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze gaze/eye tracking data
    """
    if df is None or df.empty:
        return {"error": "No data"}

    df["avg_gaze_x"] = (df["left_gaze_x"] + df["right_gaze_x"]) / 2
    df["avg_gaze_y"] = (df["left_gaze_y"] + df["right_gaze_y"]) / 2

    screen_width = 2560
    screen_height = 1440

    df["gaze_pixel_x"] = df["avg_gaze_x"] * screen_width
    df["gaze_pixel_y"] = df["avg_gaze_y"] * screen_height

    grid_size = 10
    df["grid_x"] = (df["avg_gaze_x"] * grid_size).astype(int).clip(0, grid_size - 1)
    df["grid_y"] = (df["avg_gaze_y"] * grid_size).astype(int).clip(0, grid_size - 1)

    unique_cells = df[["grid_x", "grid_y"]].drop_duplicates()
    coverage_percentage = float(
        round((len(unique_cells) / (grid_size * grid_size)) * 100, 1)
    )

    # Minimap attention (bottom-right 15%)
    # minimap_gazes = df[
    #     (df['gaze_pixel_x'] > screen_width * 0.85) &
    #     (df['gaze_pixel_y'] > screen_height * 0.85)
    # ]
    minimap_gazes = df[
        (df["gaze_pixel_x"] < screen_width * 0.15)
        & (df["gaze_pixel_y"] < screen_height * 0.15)
    ]

    estimated_duration_minutes = len(df) / 1000

    minimap_checks_per_min = (
        float(round(len(minimap_gazes) / estimated_duration_minutes, 1))
        if estimated_duration_minutes > 0
        else 0
    )

    center_gazes = df[
        (df["avg_gaze_x"] > 0.3)
        & (df["avg_gaze_x"] < 0.7)
        & (df["avg_gaze_y"] > 0.3)
        & (df["avg_gaze_y"] < 0.7)
    ]
    center_percentage = float(round((len(center_gazes) / len(df)) * 100, 1))

    heatmap_sample = df[["gaze_pixel_x", "gaze_pixel_y"]].iloc[::200].dropna()
    heatmap_data = [
        {"x": float(row["gaze_pixel_x"]), "y": float(row["gaze_pixel_y"])}
        for _, row in heatmap_sample.head(400).iterrows()
    ]

    return {
        "screen_coverage_percentage": coverage_percentage,
        "minimap_checks_per_minute": minimap_checks_per_min,
        "center_focus_percentage": center_percentage,
        "screen_resolution": {"width": screen_width, "height": screen_height},
        "total_gaze_points": int(len(df)),
        "minimap_gaze_count": int(len(minimap_gazes)),
        "heatmap_data": heatmap_data,
    }


def get_input_analysis(df: pd.DataFrame) -> Dict:
    """
    Analyze keyboard and mouse input data with realistic timing.
    Expects a column with Unix timestamp in milliseconds as the first column,
    and columns 'event_type' and 'details_x'.
    """
    if df is None or df.empty:
        return {"error": "No data"}

    df["unix_time"] = pd.to_datetime(df.iloc[:, 0], unit="ms")

    # Calculate actual duration in minutes
    duration_minutes = (
        df["unix_time"].max() - df["unix_time"].min()
    ).total_seconds() / 60
    if duration_minutes <= 0.01:
        duration_minutes = 0.01

    key_presses = df[df["event_type"] == "key_press"].copy()
    mouse_clicks = df[df["event_type"] == "mouse_click"].copy()
    mouse_moves = df[df["event_type"] == "mouse_move"].copy()

    total_actions = len(key_presses) + len(mouse_clicks)
    apm = round(total_actions / duration_minutes, 1)

    if key_presses.empty:
        return {
            "apm": apm,
            "total_key_presses": 0,
            "total_mouse_clicks": int(len(mouse_clicks)),
            "total_mouse_moves": int(len(mouse_moves)),
            "clicks_per_minute": round(len(mouse_clicks) / duration_minutes, 1),
            "duration_minutes": round(duration_minutes, 1),
            "wasd_distribution": {
                k: {"count": 0, "percentage": 0.0} for k in ["W", "A", "S", "D"]
            },
            "ability_usage": {},
            "top_keys": [],
        }

    # Count all keys
    all_keys = key_presses["details_x"].value_counts()

    # WASD mapping and distribution
    wasd_map = {
        "'w'": "W",
        "'W'": "W",
        "'a'": "A",
        "'A'": "A",
        "'s'": "S",
        "'S'": "S",
        "'d'": "D",
        "'D'": "D",
    }
    wasd_presses = key_presses[key_presses["details_x"].isin(wasd_map.keys())].copy()
    wasd_presses["normalized_key"] = wasd_presses["details_x"].map(wasd_map)
    wasd_counts = wasd_presses["normalized_key"].value_counts()

    total_wasd = wasd_counts.sum()
    wasd_distribution = {}
    for key in ["W", "A", "S", "D"]:
        count = int(wasd_counts.get(key, 0))
        percentage = round((count / total_wasd) * 100, 1) if total_wasd > 0 else 0.0
        wasd_distribution[key] = {"count": count, "percentage": percentage}

    ability_map = {
        "'q'": "Q",
        "'e'": "E",
        "'c'": "C",
        "'x'": "X",
        "'Q'": "Q",
        "'E'": "E",
        "'C'": "C",
        "'X'": "X",
        "'1'": "1",
        "'2'": "2",
        "'3'": "3",
        "'4'": "4",
        "'5'": "5",
    }
    ability_presses = key_presses[
        key_presses["details_x"].isin(ability_map.keys())
    ].copy()
    if not ability_presses.empty:
        ability_presses["normalized_key"] = ability_presses["details_x"].map(
            ability_map
        )
        ability_counts = ability_presses["normalized_key"].value_counts()
        ability_usage = {str(key): int(count) for key, count in ability_counts.items()}
    else:
        ability_usage = {}

    top_keys = [
        {"key": str(key), "count": int(count)}
        for key, count in all_keys.head(10).items()
    ]

    clicks_per_min = round(len(mouse_clicks) / duration_minutes, 1)

    return {
        "apm": apm,
        "total_key_presses": int(len(key_presses)),
        "total_mouse_clicks": int(len(mouse_clicks)),
        "total_mouse_moves": int(len(mouse_moves)),
        "clicks_per_minute": clicks_per_min,
        "duration_minutes": round(duration_minutes, 1),
        "wasd_distribution": wasd_distribution,
        "ability_usage": ability_usage,
        "top_keys": top_keys,
    }


def get_complete_analysis(filename: str, data_directory: Optional[str] = None) -> Dict:
    """
    Get complete biometric analysis for a match file.

    Args:
        filename: Name of the CSV file
        data_directory: Optional base data directory path (overrides hardcoded path)
    """
    df = load_biometric_data(filename, data_directory)

    if df is None:
        return {"error": "Could not load file"}

    # Try to get total row count from the actual file
    file_path = _find_csv_file(filename, data_directory)
    if file_path:
        try:
            total_df = pd.read_csv(file_path)
            total_rows = len(total_df)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            total_rows = len(df)
    else:
        total_rows = len(df)

    has_gameplay = bool(df["event_type"].notna().sum() > 0)

    # NEW: Extract raw emotion data for timeline markers
    raw_emotion_data = []
    if (
        "unix_time" in df.columns
        and "emotion" in df.columns
        and "confidence" in df.columns
    ):
        # Only include high-confidence emotion data to reduce payload size
        emotion_df = df[df["confidence"] > 0.5].copy()

        # Get the start time to calculate relative timestamps
        if len(emotion_df) > 0:
            # Check if we have a proper datetime column to use instead
            if "datetime" in emotion_df.columns:
                # Try to parse datetime column (format: "1900-01-01 HH:MM:SS.ffffff")
                emotion_df["datetime_parsed"] = pd.to_datetime(
                    emotion_df["datetime"], errors="coerce"
                )

                # Check if datetime parsing worked and has variation
                if emotion_df["datetime_parsed"].notna().sum() > 0:
                    unique_datetimes = emotion_df["datetime_parsed"].nunique()

                    # Check if we have enough variation (more than 10 unique values)
                    if unique_datetimes > 10:
                        print(
                            f"✅ Using datetime column for timestamps ({unique_datetimes} unique values)"
                        )
                        emotion_df = emotion_df[
                            emotion_df["datetime_parsed"].notna()
                        ].copy()

                        if len(emotion_df) > 0:
                            start_time = emotion_df["datetime_parsed"].iloc[0]

                            # Debug: print first and last timestamps
                            last_time = emotion_df["datetime_parsed"].iloc[-1]
                            duration = (last_time - start_time).total_seconds()
                            print(
                                f"   Start: {start_time}, End: {last_time}, Duration: {duration:.1f}s"
                            )

                            for _, row in emotion_df.iterrows():
                                current_time = row["datetime_parsed"]
                                seconds_from_start = (
                                    current_time - start_time
                                ).total_seconds()

                                raw_emotion_data.append(
                                    {
                                        "unix_time": str(row["unix_time"]),
                                        "emotion": str(row["emotion"]),
                                        "confidence": str(int(row["confidence"] * 100)),
                                        "datetime": str(row.get("datetime", "")),
                                        "video_timestamp": float(seconds_from_start),
                                    }
                                )

                        print(
                            f"✅ Generated {len(raw_emotion_data)} emotion data points using datetime column"
                        )

                        # Early return with datetime-based data
                        return {
                            "file_info": {
                                "filename": filename,
                                "total_rows": int(total_rows),
                                "gameplay_rows": int(len(df)),
                                "has_gameplay_data": has_gameplay,
                            },
                            "emotion_analysis": get_emotion_analysis(df),
                            "gaze_analysis": get_gaze_analysis(df),
                            "input_analysis": get_input_analysis(df),
                            "raw_emotion_data": raw_emotion_data,
                        }
                    else:
                        print(
                            f"⚠️ datetime column has only {unique_datetimes} unique values, falling back to unix_time"
                        )

            # Fallback to unix_time if datetime doesn't work
            print("Falling back to unix_time column for timestamps")
            emotion_df["unix_time_numeric"] = pd.to_numeric(
                emotion_df["unix_time"], errors="coerce"
            )
            emotion_df["unix_time_parsed"] = pd.to_datetime(
                emotion_df["unix_time_numeric"], unit="ms", errors="coerce"
            )

            # Remove rows where parsing failed
            emotion_df = emotion_df[emotion_df["unix_time_parsed"].notna()].copy()

            if len(emotion_df) > 0:
                start_time = emotion_df["unix_time_parsed"].iloc[0]

                for _, row in emotion_df.iterrows():
                    # Calculate seconds from start
                    current_time = row["unix_time_parsed"]
                    seconds_from_start = (current_time - start_time).total_seconds()

                    raw_emotion_data.append(
                        {
                            "unix_time": str(row["unix_time"]),
                            "emotion": str(row["emotion"]),
                            "confidence": str(int(row["confidence"] * 100)),
                            "datetime": str(row.get("datetime", "")),
                            "video_timestamp": float(seconds_from_start),
                        }
                    )

    return {
        "file_info": {
            "filename": filename,
            "total_rows": int(total_rows),
            "gameplay_rows": int(len(df)),
            "has_gameplay_data": has_gameplay,
        },
        "emotion_analysis": get_emotion_analysis(df),
        "gaze_analysis": get_gaze_analysis(df),
        "input_analysis": get_input_analysis(df),
        "raw_emotion_data": raw_emotion_data,
    }
