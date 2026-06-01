"""
Match analytics router — server-side analytics computed from merged session CSVs.

WHY SERVER-SIDE:
    The merged CSV for a session can contain 50 000+ rows. Processing that volume
    in the browser caused severe performance issues (slow load, UI freezes). All
    heavy computation now happens here with pandas, returning only the ~200 rows
    of pre-computed results the frontend needs.

CACHING:
    Results are cached in-process for CACHE_TTL_SECONDS (5 minutes). The cache
    is keyed by source + stem/path, so repeated visits to the same session are
    near-instant. The cache lives only in memory — it resets when the server
    restarts and is not shared across workers.

DATA SOURCES:
    Each endpoint comes in a hive/local pair:
      - hive:  reads the _merged.csv from the Hive SFTP server (queenbee)
      - local: reads the _merged.csv from a local file path on disk

ENDPOINTS:
    GET /hive                    — Valorant analytics from Hive
    GET /local                   — Valorant analytics from local disk
    GET /hive/league             — League of Legends analytics from Hive
    GET /local/league            — League of Legends analytics from local disk
    GET /emotion-markers/hive    — Emotion timeline markers from Hive
    GET /emotion-markers/local   — Emotion timeline markers from local disk

MERGED CSV COLUMNS (expected, all optional with graceful fallback):
    emotion       — detected emotion label per row
    confidence    — model confidence (0–1 decimal or 0–100 integer)
    screen_x      — gaze X coordinate (pixels, 2560px wide screen assumed)
    screen_y      — gaze Y coordinate (pixels, 1440px tall screen assumed)
    event_type    — input event type ("key_press", "mouse_click", etc.)
    details_x     — key name for keyboard events
    datetime      — ISO timestamp (preferred for duration/timeline calculations)
    timestamp     — fallback timestamp column
    unix_time     — Unix epoch in milliseconds (fallback for video sync)
"""

import io
import os
import time
from pathlib import Path

import pandas as pd
import paramiko
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query

load_dotenv()

router = APIRouter()

# ---------------------------------------------------------------------------
# In-process cache
# ---------------------------------------------------------------------------

_cache: dict = {}
CACHE_TTL_SECONDS = 300  # 5 minutes — adjust if data changes more frequently


def _get_cached(key: str):
    """Return cached result if it exists and has not expired, otherwise None."""
    entry = _cache.get(key)
    if entry and (time.time() - entry["cached_at"]) < CACHE_TTL_SECONDS:
        return entry["result"]
    return None


def _set_cached(key: str, result: dict):
    """Store a result in the cache with the current timestamp."""
    _cache[key] = {"result": result, "cached_at": time.time()}


# ---------------------------------------------------------------------------
# Connection config — values come from .env; defaults are fallback only.
# ---------------------------------------------------------------------------

HIVE_HOST      = os.getenv("HIVE_HOST",      "10.4.28.2")
HIVE_USER      = os.getenv("HIVE_USER",      "kowalski")
HIVE_PASSWORD  = os.getenv("HIVE_PASSWORD",  "bg1337#@!")
HIVE_BASE_PATH = os.getenv("HIVE_BASE_PATH", "/mnt/raid0/esports/sftp_data")


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_csv_local(file_path: str) -> pd.DataFrame:
    """Read a merged CSV from the local filesystem."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return pd.read_csv(file_path, low_memory=False)


def _read_csv_hive(stem: str) -> pd.DataFrame:
    """
    Download and parse a merged CSV from the Hive SFTP server.

    Expects the file at: HIVE_BASE_PATH/merged/<stem>_merged.csv
    The file is read fully into memory as bytes, then parsed by pandas.
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(HIVE_HOST, username=HIVE_USER, password=HIVE_PASSWORD, timeout=15)
        sftp = ssh.open_sftp()
        remote_path = f"{HIVE_BASE_PATH}/merged/{stem}_merged.csv"
        with sftp.open(remote_path, "r") as f:
            content = f.read()
        sftp.close()
        return pd.read_csv(io.BytesIO(content), low_memory=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"Merged CSV not found on Hive: {stem}_merged.csv")
    finally:
        ssh.close()


# ---------------------------------------------------------------------------
# Analytics computation
# ---------------------------------------------------------------------------

def _compute_analytics(df: pd.DataFrame) -> dict:
    """
    Compute all Valorant match analytics from a merged session DataFrame.

    Sections computed:
      1. Emotion distribution — percentage breakdown of all detected emotions.
      2. Emotion progress     — emotion counts split across 5 Valorant match
                                stages (Pistol Round → Final Rounds), each
                                covering an equal slice of total rows.
      3. Gaze points          — every 20th row sampled to keep the payload small
                                (~500 points for a 10k-row session).
      4. Screen zones         — gaze bucketed into HUD regions at 2560×1440.
                                Coordinates are Valorant-specific.
      5. Input metrics        — APM, key presses, mouse clicks, ability usage,
                                top 10 keys by frequency.
      6. Data quality flags   — booleans indicating which data streams are
                                present and non-empty in this session.

    Returns a dict ready to be JSON-serialised and returned by the endpoint.
    """
    total_rows = len(df)
    df.columns = df.columns.str.strip()  # Remove accidental whitespace in column names.

    emotion_col = "emotion" if "emotion" in df.columns else None

    # --- 1. Emotion distribution ---
    emotion_distribution = []
    dominant_emotion    = "Unknown"
    dominant_percentage = 0.0
    if emotion_col:
        counts = df[emotion_col].fillna("Unknown").value_counts()
        emotion_distribution = [
            {"name": str(k), "value": round((v / total_rows) * 100, 1), "count": int(v)}
            for k, v in counts.items()
        ]
        if emotion_distribution:
            dominant_emotion    = emotion_distribution[0]["name"]
            dominant_percentage = emotion_distribution[0]["value"]

    # --- 2. Emotion progress across match stages ---
    # Rows are divided into 5 equal segments. Valorant-specific stage names.
    stages       = ["Pistol Round", "Early Rounds", "Mid Game", "Late Rounds", "Final Rounds"]
    emotions_list = ["Neutral", "Happiness", "Anger", "Sadness", "Surprise", "Disgust", "Fear"]
    seg_size     = max(1, total_rows // 5)
    emotion_progress = []
    for i, stage in enumerate(stages):
        start = i * seg_size
        end   = total_rows if i == 4 else (i + 1) * seg_size
        seg   = df.iloc[start:end]
        row: dict = {"stage": stage}
        if emotion_col:
            seg_counts = seg[emotion_col].fillna("Unknown").value_counts()
            for e in emotions_list:
                row[e] = int(seg_counts.get(e, 0))
        else:
            for e in emotions_list:
                row[e] = 0
        emotion_progress.append(row)

    # --- 3. Gaze points ---
    # Sample every 20th row to reduce payload size while preserving distribution.
    gaze_points = []
    if "screen_x" in df.columns and "screen_y" in df.columns:
        gaze_df = df[["screen_x", "screen_y", emotion_col or "emotion"]].iloc[::20].copy()
        gaze_df = gaze_df.dropna(subset=["screen_x", "screen_y"])
        gaze_df["screen_x"] = pd.to_numeric(gaze_df["screen_x"], errors="coerce")
        gaze_df["screen_y"] = pd.to_numeric(gaze_df["screen_y"], errors="coerce")
        gaze_df = gaze_df.dropna(subset=["screen_x", "screen_y"])
        gaze_points = [
            {"x": int(r.screen_x), "y": int(r.screen_y), "emotion": str(getattr(r, emotion_col, "Unknown"))}
            for r in gaze_df.itertuples(index=False)
        ]

    # --- 4. Screen zones — Valorant HUD at 2560×1440 ---
    # Zone boundaries are pixel ranges tuned for the Valorant UI layout.
    # If the game or resolution changes these will need updating.
    screen_zones = []
    if "screen_x" in df.columns and "screen_y" in df.columns:
        sx = pd.to_numeric(df["screen_x"], errors="coerce").fillna(0)
        sy = pd.to_numeric(df["screen_y"], errors="coerce").fillna(0)
        zone_counts = {
            "Minimap":   int(((sx >= 0)    & (sx <= 300)  & (sy >= 0)    & (sy <= 300)).sum()),
            "Abilities": int(((sx >= 900)  & (sx <= 1660) & (sy >= 1300) & (sy <= 1440)).sum()),
            "Killfeed":  int(((sx >= 2200) & (sx <= 2560) & (sy >= 0)    & (sy <= 300)).sum()),
            "Center":    int(((sx >= 300)  & (sx <= 2200) & (sy >= 300)  & (sy <= 1300)).sum()),
        }
        # Any gaze point that falls outside all named zones is counted as "Other".
        zone_counts["Other"] = total_rows - sum(zone_counts.values())
        screen_zones = [
            {"zone": z, "count": c, "percentage": round((c / total_rows) * 100, 1)}
            for z, c in zone_counts.items()
        ]

    # --- 5. Input metrics ---
    input_metrics: dict = {
        "apm": 0, "total_key_presses": 0, "total_mouse_clicks": 0,
        "clicks_per_minute": 0, "duration_minutes": 0.0,
        "ability_usage": {k: 0 for k in ["Q", "E", "C", "X", "1", "2", "3", "4", "5"]},
        "top_keys": [],
    }
    if "event_type" in df.columns:
        # Prefer "datetime" column; fall back to "timestamp" for duration calculation.
        ts_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        duration_minutes = 0.0
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
            if len(ts) >= 2:
                duration_minutes = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60

        kb_mask          = df["event_type"].isin(["key_press", "keyboard", "key"])
        kb_df            = df[kb_mask].copy()
        total_key_presses = int(kb_mask.sum())

        key_counts: dict = {}
        if "details_x" in df.columns and total_key_presses > 0:
            # Strip quotes that appear in some toolkit output formats.
            keys = (
                kb_df["details_x"]
                .dropna()
                .astype(str)
                .str.replace(r"['\"]", "", regex=True)
                .str.strip()
                .str.upper()
            )
            key_counts = keys.value_counts().to_dict()

        # Valorant ability keys: Q, E, C, X (ultimate) plus item slots 1–5.
        ability_usage    = {k: int(key_counts.get(k, 0)) for k in ["Q", "E", "C", "X", "1", "2", "3", "4", "5"]}
        top_keys         = [
            {"key": str(k), "count": int(v)}
            for k, v in sorted(key_counts.items(), key=lambda x: -x[1])[:10]
        ]
        mouse_mask        = df["event_type"].isin(["mouse_click", "click"])
        total_mouse_clicks = int(mouse_mask.sum())
        apm = round(total_key_presses  / duration_minutes) if duration_minutes > 0 else 0
        cpm = round(total_mouse_clicks / duration_minutes) if duration_minutes > 0 else 0
        input_metrics = {
            "apm":               apm,
            "total_key_presses": total_key_presses,
            "total_mouse_clicks": total_mouse_clicks,
            "clicks_per_minute": cpm,
            "duration_minutes":  round(duration_minutes, 1),
            "ability_usage":     ability_usage,
            "top_keys":          top_keys,
        }

    # --- 6. Data quality flags ---
    data_quality = {
        "has_emotion": bool(emotion_col and df[emotion_col].notna().any()),
        "has_gaze":    bool("screen_x" in df.columns and df["screen_x"].notna().any()),
        "has_input":   bool("event_type" in df.columns and df["event_type"].notna().any()),
    }

    return {
        "total_rows":           total_rows,
        "emotion_distribution": emotion_distribution,
        "dominant_emotion":     dominant_emotion,
        "dominant_percentage":  dominant_percentage,
        "emotion_progress":     emotion_progress,
        "gaze_points":          gaze_points,
        "screen_zones":         screen_zones,
        "input_metrics":        input_metrics,
        "data_quality":         data_quality,
    }


def _compute_analytics_league(df: pd.DataFrame) -> dict:
    """
    Compute all League of Legends match analytics from a merged session DataFrame.

    Identical structure to _compute_analytics() but with LoL-specific differences:
      - Stage names: Early Game → Final Push (instead of Valorant round names).
      - Screen zones: LoL HUD layout at 2560×1440.
          Minimap:    bottom-right corner (opposite side from Valorant).
          Abilities:  bottom-center Q/W/E/R bar.
          Scoreboard: top-center strip.
          Center:     main game area.
      - Ability keys: Q, W, E, R (spells) + D, F (summoner spells) + item slots 1–7.

    See _compute_analytics() for full documentation of shared logic.
    """
    total_rows = len(df)
    df.columns = df.columns.str.strip()

    emotion_col = "emotion" if "emotion" in df.columns else None

    # --- 1. Emotion distribution ---
    emotion_distribution = []
    dominant_emotion    = "Unknown"
    dominant_percentage = 0.0
    if emotion_col:
        counts = df[emotion_col].fillna("Unknown").value_counts()
        emotion_distribution = [
            {"name": str(k), "value": round((v / total_rows) * 100, 1), "count": int(v)}
            for k, v in counts.items()
        ]
        if emotion_distribution:
            dominant_emotion    = emotion_distribution[0]["name"]
            dominant_percentage = emotion_distribution[0]["value"]

    # --- 2. Emotion progress — LoL-specific stage names ---
    stages        = ["Early Game", "Mid Game", "Teamfights", "Late Game", "Final Push"]
    emotions_list = ["Neutral", "Happiness", "Anger", "Sadness", "Surprise", "Disgust", "Fear"]
    seg_size      = max(1, total_rows // 5)
    emotion_progress = []
    for i, stage in enumerate(stages):
        start = i * seg_size
        end   = total_rows if i == 4 else (i + 1) * seg_size
        seg   = df.iloc[start:end]
        row: dict = {"stage": stage}
        if emotion_col:
            seg_counts = seg[emotion_col].fillna("Unknown").value_counts()
            for e in emotions_list:
                row[e] = int(seg_counts.get(e, 0))
        else:
            for e in emotions_list:
                row[e] = 0
        emotion_progress.append(row)

    # --- 3. Gaze points ---
    gaze_points = []
    if "screen_x" in df.columns and "screen_y" in df.columns:
        gaze_df = df[["screen_x", "screen_y", emotion_col or "emotion"]].iloc[::20].copy()
        gaze_df = gaze_df.dropna(subset=["screen_x", "screen_y"])
        gaze_df["screen_x"] = pd.to_numeric(gaze_df["screen_x"], errors="coerce")
        gaze_df["screen_y"] = pd.to_numeric(gaze_df["screen_y"], errors="coerce")
        gaze_df = gaze_df.dropna(subset=["screen_x", "screen_y"])
        gaze_points = [
            {"x": int(r.screen_x), "y": int(r.screen_y), "emotion": str(getattr(r, emotion_col, "Unknown"))}
            for r in gaze_df.itertuples(index=False)
        ]

    # --- 4. Screen zones — LoL HUD at 2560×1440 ---
    # LoL minimap is bottom-RIGHT (opposite of Valorant which is bottom-left).
    # Scoreboard is a narrow strip at the very top-center.
    screen_zones = []
    if "screen_x" in df.columns and "screen_y" in df.columns:
        sx = pd.to_numeric(df["screen_x"], errors="coerce").fillna(0)
        sy = pd.to_numeric(df["screen_y"], errors="coerce").fillna(0)
        zone_counts = {
            "Minimap":    int(((sx >= 2160) & (sx <= 2560) & (sy >= 1040) & (sy <= 1440)).sum()),
            "Abilities":  int(((sx >= 870)  & (sx <= 1690) & (sy >= 1200) & (sy <= 1440)).sum()),
            "Scoreboard": int(((sx >= 1010) & (sx <= 1550) & (sy >= 0)    & (sy <= 60)).sum()),
            "Center":     int(((sx >= 260)  & (sx <= 2300) & (sy >= 60)   & (sy <= 1160)).sum()),
        }
        zone_counts["Other"] = max(0, total_rows - sum(zone_counts.values()))
        screen_zones = [
            {"zone": z, "count": c, "percentage": round((c / total_rows) * 100, 1)}
            for z, c in zone_counts.items()
        ]

    # --- 5. Input metrics — LoL ability keys ---
    # Q/W/E/R = champion spells, D/F = summoner spells, 1-7 = item actives.
    league_ability_keys = ["Q", "W", "E", "R", "D", "F", "1", "2", "3", "4", "5", "6", "7"]
    input_metrics: dict = {
        "apm": 0, "total_key_presses": 0, "total_mouse_clicks": 0,
        "clicks_per_minute": 0, "duration_minutes": 0.0,
        "ability_usage": {k: 0 for k in league_ability_keys},
        "top_keys": [],
    }
    if "event_type" in df.columns:
        ts_col = "datetime" if "datetime" in df.columns else ("timestamp" if "timestamp" in df.columns else None)
        duration_minutes = 0.0
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
            if len(ts) >= 2:
                duration_minutes = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 60

        kb_mask           = df["event_type"].isin(["key_press", "keyboard", "key"])
        kb_df             = df[kb_mask].copy()
        total_key_presses = int(kb_mask.sum())

        key_counts: dict = {}
        if "details_x" in df.columns and total_key_presses > 0:
            keys = (
                kb_df["details_x"]
                .dropna()
                .astype(str)
                .str.replace(r"['\"]", "", regex=True)
                .str.strip()
                .str.upper()
            )
            key_counts = keys.value_counts().to_dict()

        ability_usage      = {k: int(key_counts.get(k, 0)) for k in league_ability_keys}
        top_keys           = [
            {"key": str(k), "count": int(v)}
            for k, v in sorted(key_counts.items(), key=lambda x: -x[1])[:10]
        ]
        mouse_mask         = df["event_type"].isin(["mouse_click", "click"])
        total_mouse_clicks = int(mouse_mask.sum())
        apm = round(total_key_presses  / duration_minutes) if duration_minutes > 0 else 0
        cpm = round(total_mouse_clicks / duration_minutes) if duration_minutes > 0 else 0
        input_metrics = {
            "apm":                apm,
            "total_key_presses":  total_key_presses,
            "total_mouse_clicks": total_mouse_clicks,
            "clicks_per_minute":  cpm,
            "duration_minutes":   round(duration_minutes, 1),
            "ability_usage":      ability_usage,
            "top_keys":           top_keys,
        }

    # --- 6. Data quality flags ---
    data_quality = {
        "has_emotion": bool(emotion_col and df[emotion_col].notna().any()),
        "has_gaze":    bool("screen_x" in df.columns and df["screen_x"].notna().any()),
        "has_input":   bool("event_type" in df.columns and df["event_type"].notna().any()),
    }

    return {
        "total_rows":           total_rows,
        "emotion_distribution": emotion_distribution,
        "dominant_emotion":     dominant_emotion,
        "dominant_percentage":  dominant_percentage,
        "emotion_progress":     emotion_progress,
        "gaze_points":          gaze_points,
        "screen_zones":         screen_zones,
        "input_metrics":        input_metrics,
        "data_quality":         data_quality,
    }


def _compute_emotion_markers(df: pd.DataFrame) -> dict:
    """
    Detect significant emotion change events for the video timeline.

    This logic was ported from the original frontend (calculateVideoTimestamps +
    processEmotionMarkers in ValorantFeedbackReview.tsx) to avoid sending 50k+
    rows to the browser for processing.

    Algorithm:
      1. Filter to rows with a non-empty emotion and confidence above threshold
         (50% — auto-detected as decimal 0–1 or integer 0–100).
      2. Determine the video timestamp source in priority order:
           a. datetime/timestamp column — converted to seconds since session start.
           b. unix_time column          — milliseconds; divided by 1000.
              Special case: if the first 100 unix values are all the same rounded
              integer, the column is likely a frame counter, not a real epoch, so
              row index is used instead.
           c. Row index fallback        — when no timestamp column exists.
      3. Walk through rows, emitting a marker when:
           - The emotion changes from the previous row.
           - The previous emotion lasted at least MIN_DURATION seconds.
           - Confidence is at least MIN_CONFIDENCE percent.
           - At least MIN_TIME_GAP seconds have passed since the last marker
             (prevents marker spam on rapid fluctuations).

    Constants (match the original frontend values):
        MIN_TIME_GAP   = 5s
        MIN_CONFIDENCE = 70%
        MIN_DURATION   = 2s

    Returns:
        markers        — list of marker dicts (id, timestamp, emotion, label, type)
        emotions_found — deduplicated set of emotion labels that appear in markers
        total_markers  — count of markers
    """
    df.columns = df.columns.str.strip()
    emotion_col = "emotion"    if "emotion"    in df.columns else None
    conf_col    = "confidence" if "confidence" in df.columns else None

    if not emotion_col or df[emotion_col].dropna().empty:
        return {"markers": [], "emotions_found": [], "total_markers": 0}

    edf = df[df[emotion_col].notna() & (df[emotion_col] != "")].copy()

    # --- Confidence filtering ---
    # The toolkit may output confidence as 0.0–1.0 or 0–100; detect which.
    if conf_col and conf_col in edf.columns:
        edf[conf_col] = pd.to_numeric(edf[conf_col], errors="coerce").fillna(0)
        is_decimal = edf[conf_col].iloc[0] <= 1.0 if len(edf) > 0 else True
        threshold  = 0.5 if is_decimal else 50
        edf = edf[edf[conf_col] > threshold].copy()
        if is_decimal:
            edf[conf_col] = edf[conf_col] * 100  # Normalise to 0–100 for MIN_CONFIDENCE check.

    if edf.empty:
        return {"markers": [], "emotions_found": [], "total_markers": 0}

    # --- Timestamp resolution ---
    ts_col = None
    for col in ("datetime", "timestamp"):
        if col in edf.columns:
            ts_col = col
            break

    use_datetime = False
    if ts_col:
        edf["_ts_parsed"] = pd.to_datetime(edf[ts_col], errors="coerce")
        # Only use datetime if it actually varies (>10 unique values); otherwise
        # it may be a placeholder column with a single constant value.
        if edf["_ts_parsed"].dropna().nunique() > 10:
            use_datetime = True

    if use_datetime:
        edf = edf.dropna(subset=["_ts_parsed"]).copy()
        start_time = edf["_ts_parsed"].iloc[0]
        edf["video_timestamp"] = (edf["_ts_parsed"] - start_time).dt.total_seconds()

    elif "unix_time" in edf.columns:
        edf["_unix"] = pd.to_numeric(edf["unix_time"], errors="coerce")
        edf = edf.dropna(subset=["_unix"]).copy()
        # Heuristic: if the first 100 unix values round to the same integer,
        # the column is probably a frame counter, not a real millisecond epoch.
        unique_unix = edf["_unix"].iloc[:100].round(0).nunique()
        start_unix  = edf["_unix"].iloc[0]
        if unique_unix == 1:
            # Frame counter — use row index divided by assumed 60fps as seconds.
            edf["video_timestamp"] = (edf.index - edf.index[0]) / 60.0
        else:
            # Real epoch in milliseconds — convert to seconds relative to start.
            edf["video_timestamp"] = (edf["_unix"] - start_unix) / 1000.0
    else:
        # No timestamp data at all — use row index as a proxy for time.
        edf["video_timestamp"] = range(len(edf))

    # --- Marker detection ---
    MIN_TIME_GAP   = 5   # Minimum seconds between consecutive markers.
    MIN_CONFIDENCE = 70  # Minimum confidence % to emit a marker.
    MIN_DURATION   = 2   # Minimum seconds an emotion must hold to be recorded.

    markers       = []
    last_emotion  = ""
    last_marker_t = -MIN_TIME_GAP
    emotion_start = 0.0

    for _, row in edf.iterrows():
        emotion = str(row[emotion_col])
        ts      = float(row["video_timestamp"])
        conf    = float(row[conf_col]) if conf_col and conf_col in row else 100.0

        # Skip if too soon after the last marker.
        if ts - last_marker_t < MIN_TIME_GAP:
            continue

        if emotion != last_emotion:
            duration = ts - emotion_start
            # Emit marker for the emotion that just ended.
            if conf >= MIN_CONFIDENCE and duration >= MIN_DURATION and last_emotion != "":
                markers.append({
                    "id":        f"emotion-{len(markers)}",
                    "timestamp": round(emotion_start, 1),
                    "emotion":   last_emotion,
                    "label":     f"{last_emotion} ({round(duration)}s duration)",
                    "type":      "emotion",
                })
                last_marker_t = emotion_start
            last_emotion  = emotion
            emotion_start = ts

    # Emit a final marker for the last emotion segment if it meets the threshold.
    if last_emotion and len(edf) > 0:
        last_ts  = float(edf["video_timestamp"].iloc[-1])
        duration = last_ts - emotion_start
        if duration >= MIN_DURATION:
            markers.append({
                "id":        f"emotion-{len(markers)}",
                "timestamp": round(emotion_start, 1),
                "emotion":   last_emotion,
                "label":     f"{last_emotion} ({round(duration)}s duration)",
                "type":      "emotion",
            })

    emotions_found = list({m["emotion"] for m in markers})
    return {
        "markers":        markers,
        "emotions_found": emotions_found,
        "total_markers":  len(markers),
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/hive")
def hive_match_analytics(
    stem: str = Query(..., description="Base filename stem e.g. 5th_game_P038_valorant_17-11-2025_15-24-34")
):
    """Valorant match analytics computed from the merged CSV on the Hive server."""
    cache_key = f"hive::{stem}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_hive(stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed")
    except paramiko.SSHException as e:
        raise HTTPException(status_code=503, detail=f"SSH error: {e}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Hive connection timed out — are you on the university network?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive analytics error: {str(e)}")

    result = _compute_analytics(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}


@router.get("/local")
def local_match_analytics(
    file_path: str = Query(..., description="Full path to the merged CSV on local disk")
):
    """Valorant match analytics computed from a merged CSV on the local filesystem."""
    cache_key = f"local::{file_path}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_local(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local analytics error: {str(e)}")

    result = _compute_analytics(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}


@router.get("/hive/league")
def hive_league_analytics(stem: str = Query(..., description="Base filename stem for the LoL session")):
    """League of Legends match analytics computed from the merged CSV on the Hive server."""
    cache_key = f"hive::league::{stem}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_hive(stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed")
    except paramiko.SSHException as e:
        raise HTTPException(status_code=503, detail=f"SSH error: {e}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Hive connection timed out — are you on the university network?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive League analytics error: {str(e)}")

    result = _compute_analytics_league(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}


@router.get("/local/league")
def local_league_analytics(file_path: str = Query(..., description="Full path to the merged CSV on local disk")):
    """League of Legends match analytics computed from a merged CSV on the local filesystem."""
    cache_key = f"local::league::{file_path}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_local(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local League analytics error: {str(e)}")

    result = _compute_analytics_league(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}


@router.get("/emotion-markers/hive")
def hive_emotion_markers(stem: str = Query(..., description="Base filename stem for the session")):
    """Compute emotion timeline markers from the merged CSV on the Hive server."""
    cache_key = f"emotion_markers::hive::{stem}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_hive(stem)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed")
    except paramiko.SSHException as e:
        raise HTTPException(status_code=503, detail=f"SSH error: {e}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Hive connection timed out — are you on the university network?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive emotion markers error: {str(e)}")

    result = _compute_emotion_markers(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}


@router.get("/emotion-markers/local")
def local_emotion_markers(file_path: str = Query(..., description="Full path to the merged CSV on local disk")):
    """Compute emotion timeline markers from a merged CSV on the local filesystem."""
    cache_key = f"emotion_markers::local::{file_path}"
    cached = _get_cached(cache_key)
    if cached:
        return {**cached, "cached": True}

    try:
        df = _read_csv_local(file_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local emotion markers error: {str(e)}")

    result = _compute_emotion_markers(df)
    _set_cached(cache_key, result)
    return {**result, "cached": False}