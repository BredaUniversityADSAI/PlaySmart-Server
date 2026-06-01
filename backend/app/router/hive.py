"""
Hive SFTP router — all endpoints that read data from the Hive server (queenbee).

The Hive server (queenbee, 10.4.28.2) is a Linux NAS that stores all recorded
session files uploaded by the toolkit after each session. This router opens a
fresh SFTP connection per request and closes it in a finally block.

Connection credentials are loaded from the .env file. Required keys:
    HIVE_HOST      — server IP (default: 10.4.28.2)
    HIVE_USER      — SFTP username
    HIVE_PASSWORD  — SFTP password
    HIVE_BASE_PATH — absolute path on the server to the esports data root

Hive folder structure under HIVE_BASE_PATH:
    /merged/    — _merged.csv files (primary data, one per session)
    /video/     — .mp4 gameplay recordings
    /emotion/   — emotion CSV files
    /gaze/      — gaze/eye-tracking CSV files
    /input/     — keyboard & mouse input CSV files
    /results/   — inference output CSVs (e.g. reaction_times)
    /json/      — IGN mapping files (ign_mapping.json, per-PC files)
"""

import csv
import io
import os
import json
from pathlib import PurePosixPath

import paramiko
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse

load_dotenv()

router = APIRouter()

# ---------------------------------------------------------------------------
# Connection config — values come from .env; defaults are fallback only.
# ---------------------------------------------------------------------------
HIVE_HOST      = os.getenv("HIVE_HOST",      "10.4.28.2")
HIVE_USER      = os.getenv("HIVE_USER",      "kowalski")
HIVE_PASSWORD  = os.getenv("HIVE_PASSWORD",  "bg1337#@!")
HIVE_BASE_PATH = os.getenv("HIVE_BASE_PATH", "/mnt/raid0/esports/sftp_data")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_sftp() -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    """
    Open a new SSH + SFTP connection to the Hive server.

    Always call sftp.close() and ssh.close() after use (use a finally block).
    A fresh connection is created per request; there is no connection pooling.
    """
    ssh = paramiko.SSHClient()
    # AutoAddPolicy accepts unknown host keys automatically. Acceptable here
    # because we control the server and are on a trusted university network.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HIVE_HOST, username=HIVE_USER, password=HIVE_PASSWORD, timeout=10)
    return ssh, ssh.open_sftp()


def _load_ign_mapping_raw(sftp: paramiko.SFTPClient) -> dict:
    """
    Fetch the merged IGN mapping JSON from the Hive server.

    The merged file (ign_mapping.json inside the json/ folder) is the single
    source of truth combining per-PC mapping files. It is written by a
    server-side merge script whenever a PC uploads its local ign_mapping file.

    Returns an empty dict if the file does not exist or cannot be parsed,
    so callers can treat a missing mapping as a graceful degradation rather
    than an error.
    """
    remote_path = f"{HIVE_BASE_PATH}/json/merged/merged_json.json"
    try:
        with sftp.open(remote_path, "r") as f:
            raw = f.read().decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return {}


def _resolve_player_ids(riot_id: str, mapping: dict) -> tuple[list[str], str, str]:
    """
    Look up a Riot ID in the IGN mapping and return the matching player ID(s).

    The mapping has two sections:
      - "single_player": list of {ign: player_id} dicts — one ID per player.
      - "multiplayer":   list of {player: ign, pc01: id, pc02: id, ...} dicts
                         — multiple IDs per player (one per PC station they
                         have used). All PC IDs are returned so filename
                         filtering works regardless of which PC the session
                         was recorded on.

    Matching is case-insensitive on the game name portion of the Riot ID
    (the part before the # tag).

    Returns:
        (player_ids, matched_ign, mapping_type)
        player_ids   — list of player ID strings (e.g. ["P001"]) or [] if not found
        matched_ign  — the IGN as stored in the mapping, or "" if not found
        mapping_type — "single_player", "multiplayer", or "" if not found
    """
    game_name = riot_id.split("#")[0].strip().lower()

    # Check single_player section first.
    for entry in mapping.get("single_player", []):
        for ign, pid in entry.items():
            if ign.lower() == game_name:
                return [pid], ign, "single_player"

    # Check multiplayer section — collect all PC-assigned IDs for this player.
    for entry in mapping.get("multiplayer", []):
        player_name = entry.get("player", "").lower()
        if player_name == game_name:
            # All keys except "player" are PC identifiers (pc01, pc02, ...).
            pids = list({v for k, v in entry.items() if k != "player"})
            return pids, entry.get("player", game_name), "multiplayer"

    return [], "", ""


def _listdir_safe(sftp: paramiko.SFTPClient, path: str) -> list[str]:
    """
    List a remote directory, returning an empty list if it does not exist.

    Prevents a missing subfolder (e.g. no /results yet) from crashing the
    whole list-matches endpoint.
    """
    try:
        return sftp.listdir(path)
    except (FileNotFoundError, Exception):
        return []


def _parse_filename_date(filename: str) -> str:
    """
    Extract a human-readable date/time string from a session filename stem.

    Session files follow the naming convention:
        <game>_<player_id>_<game_type>_<DD-MM-YYYY>_<HH-MM-SS>
    e.g. "1st_game_P001_valorant_27-05-2026_14-30-00"

    Returns "DD/MM/YYYY HH:MM", or the original filename if parsing fails.
    """
    parts = filename.split("_")
    if len(parts) >= 2:
        date_str = parts[-2]
        time_str = parts[-1]
        if "-" in date_str and "-" in time_str:
            try:
                day, month, year = date_str.split("-")
                hour, minute, _ = time_str.split("-")
                return f"{day}/{month}/{year} {hour}:{minute}"
            except ValueError:
                pass
    return filename


def _normalize_video_filename(filename: str) -> str:
    """
    Ensure a filename refers to a .mp4 file.

    The frontend sometimes passes a merged CSV filename (e.g. session_merged.csv)
    when navigating to the video endpoint. This normalises those to .mp4 so the
    correct video file is served from the Hive /video/ folder.
    """
    for old, new in [
        ("_merged.csv", ".mp4"),
        ("_merged.mp4", ".mp4"),
        (".csv",        ".mp4"),
    ]:
        if filename.endswith(old):
            return filename[: -len(old)] + new
    if not filename.endswith(".mp4"):
        return filename + ".mp4"
    return filename


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/resolve-player")
def resolve_player(riot_id: str):
    """
    Resolve a Riot ID to one or more Hive player IDs via the IGN mapping.

    Player IDs (e.g. P001) are embedded in session filenames on the Hive
    server. This endpoint maps a human-readable Riot ID back to those IDs so
    the frontend can filter the match list to show only a specific player's
    sessions.

    Returns:
        found             — whether the Riot ID was found in the mapping
        player_ids        — list of matching player IDs (may be multiple for
                            multiplayer sessions recorded across several PCs)
        ign               — the IGN as stored in the mapping
        mapping_type      — "single_player" or "multiplayer"
        mapping_available — False if the mapping file could not be fetched at all
    """
    ssh, sftp = None, None
    try:
        ssh, sftp = _get_sftp()
        mapping = _load_ign_mapping_raw(sftp)

        if not mapping:
            return {
                "found": False, "player_ids": [], "ign": None,
                "mapping_type": None, "mapping_available": False,
            }

        player_ids, ign, mapping_type = _resolve_player_ids(riot_id, mapping)
        if player_ids:
            return {
                "found": True,
                "player_ids": player_ids,
                "ign": ign,
                "mapping_type": mapping_type,
                "mapping_available": True,
            }
        return {
            "found": False, "player_ids": [], "ign": None,
            "mapping_type": None, "mapping_available": True,
        }

    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive resolve-player error: {str(e)}")
    finally:
        if sftp: sftp.close()
        if ssh:  ssh.close()


@router.get("/list-matches")
def list_hive_matches(game_type: str = "valorant"):
    """
    List all session matches on the Hive server for a given game type.

    Scans the /merged/ folder for CSV files matching the game type, then
    cross-references stems against the other data folders to determine which
    supplementary files exist for each session.

    IMPORTANT — base_stem() / filename matching:
        Merged files carry a "_merged" suffix (e.g. session_P001_merged.csv).
        All other folders store files without that suffix
        (e.g. session_P001_emotion.csv). The inner base_stem() helper strips
        these type suffixes before comparison, otherwise all has_* flags would
        always be False.

    Returns a list of match dicts sorted newest-first, plus directory_stats
    showing raw file counts per folder for debugging.
    """
    ssh, sftp = None, None
    try:
        ssh, sftp = _get_sftp()

        merged_dir   = f"{HIVE_BASE_PATH}/merged"
        videos_dir   = f"{HIVE_BASE_PATH}/video"
        emotions_dir = f"{HIVE_BASE_PATH}/emotion"
        gaze_dir     = f"{HIVE_BASE_PATH}/gaze"
        input_dir    = f"{HIVE_BASE_PATH}/input"
        results_dir  = f"{HIVE_BASE_PATH}/results"

        merged_files   = _listdir_safe(sftp, merged_dir)
        video_files    = _listdir_safe(sftp, videos_dir)
        emotions_files = _listdir_safe(sftp, emotions_dir)
        gaze_files     = _listdir_safe(sftp, gaze_dir)
        input_files    = _listdir_safe(sftp, input_dir)
        results_files  = _listdir_safe(sftp, results_dir)

        def base_stem(filename: str) -> str:
            """
            Strip the data-type suffix from a session filename to get the
            shared base stem used across all data folders.
            e.g. "session_P001_valorant_merged.csv" → "session_P001_valorant"
            """
            stem = PurePosixPath(filename).stem
            for suffix in ("_merged", "_emotion", "_gaze", "_input", "_reaction_times"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            return stem

        # Build sets of stems per folder for O(1) has_* lookups below.
        video_stems    = {base_stem(f) for f in video_files}
        emotions_stems = {base_stem(f) for f in emotions_files}
        gaze_stems     = {base_stem(f) for f in gaze_files}
        input_stems    = {base_stem(f) for f in input_files}
        results_stems  = {base_stem(f) for f in results_files if f.endswith("_reaction_times.csv")}

        filtered = [f for f in merged_files if game_type.lower() in f.lower()]

        matches = []
        for filename in filtered:
            stem = base_stem(filename)
            matches.append({
                "filename":             filename,
                "display_name":         stem,
                "game_type":            game_type,
                "date":                 _parse_filename_date(stem),
                "has_video":            stem in video_stems,
                "has_merged_data":      True,
                "has_emotions":         stem in emotions_stems,
                "has_gaze":             stem in gaze_stems,
                "has_input":            stem in input_stems,
                "has_reaction_results": stem in results_stems,
                # Paths are intentionally None — the frontend constructs API
                # URLs from the filename directly rather than using file paths.
                "video_path":       None,
                "merged_data_path": None,
            })

        def _sort_key(m: dict) -> int:
            """Parse the display date back to a timestamp for sorting."""
            try:
                date_part, time_part = m["date"].split(" ")
                day, month, year = date_part.split("/")
                hour, minute = time_part.split(":")
                from datetime import datetime
                return datetime(int(year), int(month), int(day),
                                int(hour), int(minute)).timestamp()
            except Exception:
                return 0

        matches.sort(key=_sort_key, reverse=True)

        return {
            "success": True,
            "source":  "hive",
            "matches": matches,
            # Raw file counts per folder — useful for debugging missing files.
            "directory_stats": {
                "merged":   len(merged_files),
                "videos":   len(video_files),
                "emotions": len(emotions_files),
                "gaze":     len(gaze_files),
                "input":    len(input_files),
                "results":  len(results_files),
            },
        }

    except paramiko.AuthenticationException:
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed — check credentials in .env")
    except paramiko.SSHException as e:
        raise HTTPException(status_code=503, detail=f"SSH error: {e}")
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Hive connection timed out — are you on the university network?")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive error: {str(e)}")
    finally:
        if sftp: sftp.close()
        if ssh:  ssh.close()


@router.get("/video/{filename}")
async def get_hive_video(filename: str, request: Request):
    """
    Stream an MP4 video from the Hive server with HTTP range request support.

    The browser's <video> element sends range requests to seek within a video
    without downloading the entire file. This endpoint reads only the requested
    byte range from the SFTP file handle and streams it in 1 MB chunks, so the
    full file is never loaded into memory.

    The SFTP connection is closed inside the streaming generator (generate())
    rather than in a finally block, because the response body is streamed
    after this function returns — closing early would kill the stream.
    """
    # Sanitise filename to prevent path traversal, then ensure .mp4 extension.
    safe        = os.path.basename(filename)
    safe        = _normalize_video_filename(safe)
    remote_path = f"{HIVE_BASE_PATH}/video/{safe}"

    ssh, sftp = None, None
    try:
        ssh, sftp = _get_sftp()

        try:
            file_size = sftp.stat(remote_path).st_size or 0
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Video not found on Hive: {safe}")

        range_header = request.headers.get("range")
        chunk_size   = 1024 * 1024  # 1 MB per SFTP read — balances memory vs round-trips

        if not range_header:
            start, end = 0, file_size - 1
        else:
            try:
                unit, _, range_part = range_header.partition("=")
                if unit != "bytes":
                    raise ValueError()
                start_str, _, end_str = range_part.partition("-")
                start = int(start_str) if start_str else 0
                end   = int(end_str)   if end_str   else file_size - 1
                if end >= file_size: end = file_size - 1
                if start > end:      raise ValueError()
            except Exception:
                # 416 Range Not Satisfiable
                sftp.close(); ssh.close()
                return Response(status_code=416)

        content_length = end - start + 1

        def generate():
            """Yield byte chunks from the SFTP file, then close the connection."""
            try:
                fh = sftp.open(remote_path, "rb")
                fh.seek(start)
                remaining = content_length
                while remaining > 0:
                    data = fh.read(min(chunk_size, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data
                fh.close()
            finally:
                # Connection must be closed here, not in the outer finally,
                # because streaming continues after get_hive_video() returns.
                try: sftp.close()
                except Exception: pass
                try: ssh.close()
                except Exception: pass

        headers = {
            "Content-Range":  f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges":  "bytes",
            "Content-Length": str(content_length),
            "Cache-Control":  "no-cache",
            "Content-Type":   "video/mp4",
        }
        return StreamingResponse(
            generate(),
            status_code=206 if range_header else 200,
            headers=headers,
        )

    except HTTPException:
        raise
    except paramiko.AuthenticationException:
        if sftp: sftp.close()
        if ssh:  ssh.close()
        raise HTTPException(status_code=401, detail="Hive SFTP authentication failed")
    except Exception as e:
        if sftp: sftp.close()
        if ssh:  ssh.close()
        raise HTTPException(status_code=500, detail=f"Hive video error: {str(e)}")


@router.get("/csv-summary")
def hive_csv_summary(subdir: str, filename: str):
    """
    Return metadata and a sample of rows from a CSV file on the Hive server.

    Used by the frontend to inspect a file before deciding whether to load it
    in full. Downloads the entire file into memory — only suitable for CSV
    files (not video). Avoid calling this on large files in a hot path.

    Args:
        subdir:   Subfolder under HIVE_BASE_PATH (e.g. "merged", "emotion").
        filename: Name of the CSV file within that subfolder.

    Returns:
        file_size_mb, total_rows, columns, column_count, sample_rows (first 5).
    """
    ssh, sftp = None, None
    try:
        ssh, sftp = _get_sftp()
        remote_path = f"{HIVE_BASE_PATH}/{subdir}/{filename}"

        try:
            file_size_mb = round((sftp.stat(remote_path).st_size or 0) / (1024 * 1024), 2)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found on Hive: {subdir}/{filename}")

        with sftp.open(remote_path, "r") as f:
            content = f.read().decode("utf-8", errors="replace")

        reader  = csv.DictReader(io.StringIO(content))
        headers = reader.fieldnames or []
        rows    = list(reader)

        return {
            "file_size_mb": file_size_mb,
            "total_rows":   len(rows),
            "columns":      list(headers),
            "column_count": len(headers),
            "sample_rows":  rows[:5],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive csv-summary error: {str(e)}")
    finally:
        if sftp: sftp.close()
        if ssh:  ssh.close()


@router.get("/read-csv")
def read_hive_csv(
    subdir:    str,
    filename:  str,
    max_rows:  int = 500,
    skip_rows: int = 0,
):
    """
    Read and paginate rows from a CSV file on the Hive server.

    Downloads the full file into memory then applies skip/limit pagination.
    This is acceptable for the CSV sizes in this project (session data files),
    but should not be used for very large files.

    Args:
        subdir:    Subfolder under HIVE_BASE_PATH (e.g. "merged").
        filename:  Name of the CSV file.
        max_rows:  Maximum number of rows to return (default 500).
        skip_rows: Number of rows to skip from the start (default 0).

    Returns:
        headers, data (paged rows), rows_returned, total_rows.
    """
    ssh, sftp = None, None
    try:
        ssh, sftp = _get_sftp()
        remote_path = f"{HIVE_BASE_PATH}/{subdir}/{filename}"

        try:
            with sftp.open(remote_path, "r") as f:
                content = f.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"File not found on Hive: {subdir}/{filename}")

        reader   = csv.DictReader(io.StringIO(content))
        headers  = reader.fieldnames or []
        all_rows = list(reader)
        paged    = all_rows[skip_rows: skip_rows + max_rows]

        return {
            "headers":       list(headers),
            "data":          paged,
            "rows_returned": len(paged),
            "total_rows":    len(all_rows),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hive read-csv error: {str(e)}")
    finally:
        if sftp: sftp.close()
        if ssh:  ssh.close()


@router.get("/reaction-results/check")
def check_reaction_results(stem: str = Query(...)):
    """
    Check whether a reaction_times CSV exists in the Hive results folder for a given stem.

    The stem is the base session name without any suffix
    (e.g. "1st_game_P001_valorant_27-05-2026_14-30-00").
    The expected file is: results/<stem>_reaction_times.csv

    Returns:
        exists — True/False
        path   — full remote path (only present when exists is True)
    """
    ssh, sftp = _get_sftp()
    try:
        path = str(PurePosixPath(HIVE_BASE_PATH) / "results" / f"{stem}_reaction_times.csv")
        try:
            sftp.stat(path)
            return {"exists": True, "path": path}
        except FileNotFoundError:
            return {"exists": False}
    finally:
        sftp.close()
        ssh.close()


@router.get("/reaction-results")
def get_reaction_results(stem: str = Query(...)):
    """
    Fetch and parse the reaction_times CSV for a session from the Hive results folder.

    Reaction time CSVs are written by the inference pipeline after the YOLO
    model processes a session VOD. This endpoint returns the parsed rows so the
    frontend can render the reaction time timeline without re-running inference.

    Args:
        stem: Base session name (e.g. "1st_game_P001_valorant_27-05-2026_14-30-00").

    Returns:
        rows — list of dicts, one per detected reaction event.

    Raises:
        404 if the results file does not exist yet (inference not run).
    """
    ssh, sftp = _get_sftp()
    try:
        path = str(PurePosixPath(HIVE_BASE_PATH) / "results" / f"{stem}_reaction_times.csv")
        try:
            with sftp.open(path, "r") as f:
                content = f.read().decode("utf-8")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"Reaction results not found: {path}")

        reader = csv.DictReader(io.StringIO(content))
        rows = list(reader)
        return {"rows": rows}
    finally:
        sftp.close()
        ssh.close()