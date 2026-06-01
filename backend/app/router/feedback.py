"""
FastAPI routes for managing player feedback.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, Depends, Query
from sqlalchemy.orm import Session

from app.crud import feedback as feedback_crud
from app.database import get_db
from app.models.feedback import FeedbackCreate, FeedbackResponse

router = APIRouter()


# -------------------------------
# Create Feedback
# -------------------------------
@router.post(
    "/feedback", response_model=FeedbackResponse, status_code=status.HTTP_201_CREATED
)
async def create_feedback(feedback: FeedbackCreate, db: Session = Depends(get_db)):
    """
    Coach submits feedback for a player at a specific timestamp.

    Args:
        feedback (FeedbackCreate): Feedback data to store.
        db (Session): Database session.

    Raises:
        HTTPException: If feedback creation fails.

    Returns:
        FeedbackResponse: Created feedback object.
    """
    try:
        db_feedback = feedback_crud.create_feedback(db=db, feedback=feedback)
        return db_feedback
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating feedback: {str(e)}",
        )


# -------------------------------
# Get Feedback
# -------------------------------
@router.get("/feedback/{riot_id}/{game}", response_model=List[FeedbackResponse])
async def get_feedback(
    riot_id: str,
    game: str,
    match_id: Optional[str] = Query(
        None, description="Optional match ID to filter feedback"
    ),
    db: Session = Depends(get_db),
):
    """
    Fetch all feedback for a player's Riot ID and game, optionally filtered by match_id.

    Args:
        riot_id (str): Riot ID of the player.
        game (str): Name of the game (e.g., "valorant" or "lol").
        match_id (Optional[str]): Optional match ID to filter feedback.
        db (Session): Database session.

    Raises:
        HTTPException: If fetching feedback fails.

    Returns:
        List[FeedbackResponse]: List of feedback objects.
    """
    try:
        feedback_list = feedback_crud.get_feedback_by_riot_id(
            db=db, riot_id=riot_id, game=game, match_id=match_id
        )
        return feedback_list
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching feedback: {str(e)}",
        )


# -------------------------------
# Delete Feedback
# -------------------------------
@router.delete("/feedback/{feedback_id}", status_code=status.HTTP_200_OK)
async def delete_feedback(feedback_id: int, db: Session = Depends(get_db)):
    """
    Delete a specific feedback by its ID.

    Args:
        feedback_id (int): ID of the feedback to delete.
        db (Session): Database session.

    Raises:
        HTTPException: If feedback not found or deletion fails.

    Returns:
        dict: Success message.
    """
    try:
        success = feedback_crud.delete_feedback(db=db, feedback_id=feedback_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Feedback not found"
            )
        return {"message": "Feedback deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting feedback: {str(e)}",
        )
        
        
        
        

# -------------------------------
# Get Error Patterns
# -------------------------------
@router.get("/feedback/{riot_id}/{game}/patterns")
async def get_error_patterns(
    riot_id: str,
    game: str,
    db: Session = Depends(get_db),
):
    """
    Fetch cross-match error patterns for a player, grouped by error_code.

    Args:
        riot_id (str): Riot ID of the player.
        game (str): Name of the game (e.g., "valorant" or "lol").
        db (Session): Database session.

    Raises:
        HTTPException: If fetching patterns fails.

    Returns:
        dict: { patterns: [...], total_annotations, unique_errors,
                recurring_count, matches_analysed }
    """
    try:
        patterns = feedback_crud.get_error_patterns(
            db=db, riot_id=riot_id, game=game
        )

        # Compute summary stats for the header cards
        total_annotations = sum(p["total_occurrences"] for p in patterns)
        unique_errors      = len(patterns)
        recurring_count    = sum(1 for p in patterns if p["is_recurring"])
        matches_analysed   = len({
            occ["match_id"]
            for p in patterns
            for occ in p["occurrences"]
            if occ["match_id"]
        })

        return {
            "patterns":          patterns,
            "total_annotations": total_annotations,
            "unique_errors":     unique_errors,
            "recurring_count":   recurring_count,
            "matches_analysed":  matches_analysed,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching error patterns: {str(e)}",
        )