"""
FastAPI routes for managing coach configuration, including
selecting data directory and CRUD operations.
"""

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
import tkinter as tk
from tkinter import filedialog

from app.crud import coach_config as coach_crud
from app.database import get_db
from app.models.coach_config import CoachConfig, CoachConfigUpdate
from app.utils.auth import get_current_user_id

router = APIRouter()


# -------------------------------
# Select Data Directory
# -------------------------------
@router.get("/select-data-directory")
async def select_data_directory():
    """
    Open a directory picker dialog for coach to select a data directory.

    Returns:
        dict: {
            "success": bool,
            "directoryPath": str (if success),
            "message": str (if cancelled)
        }

    Raises:
        HTTPException: If the dialog fails to open.
    """
    try:
        # Create a hidden Tk root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        # Open directory picker dialog
        directory_path = filedialog.askdirectory(
            title="Select Data Directory - Coach",
            mustexist=True
        )

        root.destroy()

        if directory_path:
            return {"success": True, "directoryPath": directory_path}
        else:
            return {"success": False, "message": "Directory selection cancelled"}

    except Exception as e:
        print(f"Error in select_data_directory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to open directory dialog: {str(e)}"
        )


# -------------------------------
# Get Coach Configuration
# -------------------------------
@router.get("/config", response_model=CoachConfig)
async def get_coach_config(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Retrieve coach configuration for the current user.

    Args:
        user_id (int): ID of the current user (from auth dependency).
        db (Session): Database session.

    Raises:
        HTTPException: If no configuration is found.

    Returns:
        CoachConfig: Coach configuration object.
    """
    config = coach_crud.get_coach_config_by_user(db, user_id)
    if not config:
        raise HTTPException(status_code=404, detail="Coach configuration not found")
    return config


# -------------------------------
# Save or Update Coach Configuration
# -------------------------------
@router.post("/config", response_model=CoachConfig)
async def save_coach_config(
    config: CoachConfigUpdate,
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Save a new coach configuration or update an existing one for the current user.

    Args:
        config (CoachConfigUpdate): Configuration data to save or update.
        user_id (int): Current user ID.
        db (Session): Database session.

    Returns:
        CoachConfig: Updated or newly created configuration.
    """
    updated_config = coach_crud.update_coach_config(db, user_id, config)
    return updated_config


# -------------------------------
# Delete Coach Configuration
# -------------------------------
@router.delete("/config")
async def delete_coach_config(
    user_id: int = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    """
    Delete the coach configuration for the current user.

    Args:
        user_id (int): Current user ID.
        db (Session): Database session.

    Returns:
        dict: Success message if deleted.

    Raises:
        HTTPException: If no configuration is found to delete.
    """
    deleted = coach_crud.delete_coach_config(db, user_id)
    if deleted:
        return {"success": True, "message": "Coach configuration deleted"}
    raise HTTPException(status_code=404, detail="Coach configuration not found")