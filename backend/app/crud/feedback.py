"""
CRUD operations for feedback entries.
"""

from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.database_models import Feedback
from app.models.feedback import FeedbackCreate


def create_feedback(db: Session, feedback: FeedbackCreate) -> Feedback:
    """
    Create a new feedback entry in the database.

    Args:
        db (Session): SQLAlchemy database session.
        feedback (FeedbackCreate): Feedback data to be stored.

    Returns:
        Feedback: The newly created feedback entry.
    """
    db_feedback = Feedback(
        riot_id=feedback.riot_id,
        coach_username=feedback.coach_username,
        match_id=feedback.match_id,
        timestamp=feedback.timestamp,
        category=feedback.category,
        error_code=feedback.error_code,
        feedback_text=feedback.feedback_text,
        game=feedback.game,
    )

    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)

    return db_feedback


def get_feedback_by_riot_id(
    db: Session, riot_id: str, game: str, match_id: Optional[str] = None
) -> List[Feedback]:
    """
    Retrieve all feedback for a specific riot_id and game, optionally filtered by match_id.

    Args:
        db (Session): SQLAlchemy database session.
        riot_id (str): Riot ID of the player.
        game (str): Name of the game.
        match_id (Optional[str]): Optional match ID to filter by.

    Returns:
        List[Feedback]: List of feedback entries ordered by timestamp ascending.
    """
    query = db.query(Feedback).filter(Feedback.riot_id == riot_id, Feedback.game == game)

    if match_id:
        query = query.filter(Feedback.match_id == match_id)

    return query.order_by(Feedback.timestamp.asc()).all()


def delete_feedback(db: Session, feedback_id: int) -> bool:
    """
    Delete a feedback entry by its ID.

    Args:
        db (Session): SQLAlchemy database session.
        feedback_id (int): ID of the feedback to delete.

    Returns:
        bool: True if the feedback was deleted, False if it was not found.
    """
    feedback = db.query(Feedback).filter(Feedback.id == feedback_id).first()

    if feedback:
        db.delete(feedback)
        db.commit()
        return True

    return False


def get_feedback_by_id(db: Session, feedback_id: int) -> Optional[Feedback]:
    """
    Retrieve a single feedback entry by its ID.

    Args:
        db (Session): SQLAlchemy database session.
        feedback_id (int): ID of the feedback to retrieve.

    Returns:
        Feedback | None: The feedback entry if found, otherwise None.
    """
    return db.query(Feedback).filter(Feedback.id == feedback_id).first()


def get_error_patterns(db: Session, riot_id: str, game: str) -> list[dict]:
    """
    Analyse recurring coaching errors for a player across all their matches.

    For each unique error_code found in the player's feedback, this function
    computes:
      - total_occurrences: how many times the error was flagged across all matches
      - matches_affected: how many distinct matches contained the error
      - occurrences: list of individual feedback entries (match_id, timestamp, text)
      - trend: whether the error is "improving", "worsening", or "stable"
      - is_recurring: True if the error appears in 3 or more matches

    Trend logic (only applied when 4+ matches are affected):
      - Matches are sorted chronologically by match_id.
      - Occurrence counts are split into a first half and second half.
      - If the second half has fewer occurrences than the first → "improving".
      - If more → "worsening". If equal → "stable".
      - Fewer than 4 affected matches defaults to "stable" (not enough data).

    Results are sorted by total_occurrences descending so the most frequent
    errors appear first.

    Args:
        db (Session): SQLAlchemy database session.
        riot_id (str): Riot ID of the player.
        game (str): Game name (e.g. "valorant", "lol").

    Returns:
        list[dict]: One dict per error_code, each containing:
            error_code, category, total_occurrences, matches_affected,
            occurrences, trend, is_recurring.
    """
    # Fetch all feedback rows that have an error_code assigned, ordered so
    # grouping by code is straightforward.
    rows = (
        db.query(Feedback)
        .filter(Feedback.riot_id == riot_id, Feedback.game == game)
        .filter(Feedback.error_code.isnot(None))
        .order_by(Feedback.error_code, Feedback.match_id, Feedback.timestamp)
        .all()
    )

    # Group rows by error_code. Private keys (_match_ids, _match_timestamps)
    # are used for trend calculation and are stripped before returning.
    grouped: dict[str, dict] = {}
    for row in rows:
        code = row.error_code
        if code not in grouped:
            grouped[code] = {
                "error_code":        code,
                "category":          row.category,
                "total_occurrences": 0,
                "matches_affected":  0,
                "occurrences":       [],
                "_match_ids":        set(),
                "_match_timestamps": {},  # match_id → list of timestamps
            }
        entry = grouped[code]
        entry["total_occurrences"] += 1
        entry["_match_ids"].add(row.match_id)
        entry["occurrences"].append({
            "match_id":      row.match_id,
            "timestamp":     row.timestamp,
            "feedback_text": row.feedback_text,
        })
        # Track per-match timestamps to count occurrences in each half.
        if row.match_id not in entry["_match_timestamps"]:
            entry["_match_timestamps"][row.match_id] = []
        entry["_match_timestamps"][row.match_id].append(row.timestamp)

    patterns = []
    for entry in grouped.values():
        # Sort match IDs so the split into first/second half is chronological.
        match_ids = sorted(entry["_match_ids"])
        entry["matches_affected"] = len(match_ids)

        # Trend: compare occurrence density in the first half of matches vs
        # the second half. Requires at least 4 affected matches to be meaningful.
        mid = len(match_ids) // 2
        if len(match_ids) >= 4:
            first_half  = sum(len(entry["_match_timestamps"][m]) for m in match_ids[:mid])
            second_half = sum(len(entry["_match_timestamps"][m]) for m in match_ids[mid:])
            if second_half < first_half:
                trend = "improving"
            elif second_half > first_half:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            # Not enough match history to determine a meaningful trend.
            trend = "stable"

        entry["trend"] = trend
        entry["is_recurring"] = entry["matches_affected"] >= 3

        # Remove internal tracking keys before returning to the caller.
        del entry["_match_ids"]
        del entry["_match_timestamps"]

        patterns.append(entry)

    # Most frequent errors first so the caller can display highest-priority issues at the top.
    patterns.sort(key=lambda x: x["total_occurrences"], reverse=True)
    return patterns