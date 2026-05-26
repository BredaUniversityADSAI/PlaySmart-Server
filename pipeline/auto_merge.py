import os
import pandas as pd
from collections import defaultdict

# ---------------- CONFIG ----------------
GAZE_DIR = "/mnt/raid0/esports/sftp_data/gaze"
INPUT_DIR = "/mnt/raid0/esports/sftp_data/input"
EMOTION_DIR = "/mnt/raid0/esports/sftp_data/emotion"
OUT_DIR = "/mnt/raid0/esports/sftp_data/merged"
MERGE_TOLERANCE_MS = 100
AUTO_ALIGN_EMOTION = True
# --------------------------------------

def load_csv_clean_time(path, label):
    """Load a CSV and clean the unix_time column."""
    if not path or not os.path.isfile(path):
        print(f"[WARN] {label} file not found: {path}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})

    if os.path.getsize(path) == 0:
        print(f"[WARN] {label} file is empty: {path}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[WARN] {label} has no data: {path}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})
    except Exception as e:
        print(f"[ERROR] Failed to read {label}: {e}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})

    if 'unix_time' not in df.columns:
        print(f"[WARN] `unix_time` missing in {label}: {path}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})

    df['unix_time'] = pd.to_numeric(df['unix_time'], errors='coerce')
    df.dropna(subset=['unix_time'], inplace=True)

    if df.empty:
        print(f"[WARN] {label} empty after cleaning: {path}")
        return pd.DataFrame({'unix_time': pd.Series([], dtype='int64')})

    if df['unix_time'].max() < 1e12:
        df['unix_time'] *= 1000

    df['unix_time'] = df['unix_time'].astype('int64')
    return df.sort_values('unix_time')


def collect_files():
    """Collect all CSV files by prefix."""
    groups = defaultdict(dict)

    def scan(folder, key, suffix):
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            return
        for f in os.listdir(folder):
            if f.endswith(suffix):
                prefix = f.replace(suffix, "")
                groups[prefix][key] = os.path.join(folder, f)

    scan(GAZE_DIR, "gaze", "_gaze.csv")
    scan(INPUT_DIR, "input", "_input.csv")
    scan(EMOTION_DIR, "emotion", "_emotion.csv")

    return groups


def merge_group(prefix, paths):
    print(f"\n=== Merging {prefix} ===")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{prefix}_merged.csv")

    # --- Skip if already merged ---
    if os.path.isfile(out_path):
        print(f"[SKIP] {prefix}: merged file already exists: {out_path}")
        print("[INFO] Test #2 successful")
        return

    df_gaze = load_csv_clean_time(paths.get("gaze"), "Gaze") if "gaze" in paths else None
    df_input = load_csv_clean_time(paths.get("input"), "Input") if "input" in paths else None
    df_emo = load_csv_clean_time(paths.get("emotion"), "Emotion") if "emotion" in paths else None

    # ---- Auto-align emotion ----
    if AUTO_ALIGN_EMOTION and df_emo is not None and not df_emo.empty:
        refs = []
        if df_gaze is not None and not df_gaze.empty:
            refs.append(df_gaze['unix_time'].min())
        if df_input is not None and not df_input.empty:
            refs.append(df_input['unix_time'].min())

        if refs:
            offset = min(refs) - df_emo['unix_time'].min()
            if offset != 0:
                df_emo = df_emo.copy()
                df_emo['unix_time'] += int(offset)

    # ---- Choose base timeline ----
    base = None
    base_label = None
    if df_gaze is not None and not df_gaze.empty:
        base = df_gaze
        base_label = "gaze"
    elif df_input is not None and not df_input.empty:
        base = df_input
        base_label = "input"
    elif df_emo is not None and not df_emo.empty:
        base = df_emo
        base_label = "emotion"
    else:
        print(f"[SKIP] {prefix}: all streams empty")
        return

    merged = base.copy()

    def asof(left, right):
        if right is None or right.empty:
            return left
        return pd.merge_asof(
            left.sort_values('unix_time'),
            right.sort_values('unix_time'),
            on='unix_time',
            direction='nearest',
            tolerance=MERGE_TOLERANCE_MS
        )

    if base_label != "gaze":
        merged = asof(merged, df_gaze)
    if base_label != "input":
        merged = asof(merged, df_input)
    if base_label != "emotion":
        merged = asof(merged, df_emo)

    merged.drop_duplicates(subset="unix_time", inplace=True)
    merged["datetime"] = pd.to_datetime(merged["unix_time"], unit="ms")

    # Ensure emotion columns exist if missing
    for col in ["emotion", "confidence"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    # --- Save merged CSV ---
    try:
        merged.to_csv(out_path, index=False)
        print(f"[OK] Saved {out_path}")
    except PermissionError:
        print(f"[ERROR] Permission denied when saving {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save {out_path}: {e}")


def main():
    groups = collect_files()
    if not groups:
        print("No matching files found.")
        return

    for prefix, paths in groups.items():
        merge_group(prefix, paths)


if __name__ == "__main__":
    main()
