from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import httpx  # type: ignore
import os
import pytz
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

router = APIRouter(prefix="/api/valorant", tags=["valorant"])

"""
This application fetches and displays Valorant-related data using the HenrikDev Valorant API.

API Provider:
- HenrikDev
- Website: https://henrikdev.xyz
- API Documentation: https://docs.henrikdev.xyz

All Valorant data (agents, matches, MMR, stats, etc.) is provided by
HenrikDev's public API and is not owned or maintained by this project.

This project uses a HenrikDev API key strictly for data retrieval
and presentation purposes.

"""

HENRIK_BASE_URL = "https://api.henrikdev.xyz"


def load_api_key() -> str:
    """
    Load Henrik API key from environment variable or config file.
    
    Returns:
        str: The API key
        
    Raises:
        ValueError: If API key is not found in either location
    """
    api_key = os.getenv("HENRIK_API_KEY")
    
    if not api_key:
        try:
            with open("config/henrik_api_key.txt", "r") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise ValueError("Henrik API key not found")
    
    return api_key


async def fetch_account_info(name: str, tag: str, force: bool = False) -> Dict[str, Any]:
    """
    Fetch Valorant account information from Henrik API.
    
    Args:
        name: Player name (without tag)
        tag: Player tag (without # symbol)
        force: If True, force refresh the cached data
        
    Returns:
        Dict containing account data from the API
        
    Raises:
        HTTPException: For various API errors (400, 403, 404, 408, 429, 503)
    """
    api_key = load_api_key()
    url = f"{HENRIK_BASE_URL}/valorant/v2/account/{name}/{tag}"
    
    headers = {
        "Authorization": api_key,
        "Accept": "*/*"
    }
    
    params = {}
    if force:
        params["force"] = "true"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid request parameters"
                )
            elif response.status_code == 403:
                raise HTTPException(
                    status_code=403,
                    detail="API access forbidden - possible maintenance or rate limit"
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Player {name}#{tag} not found"
                )
            elif response.status_code == 408:
                raise HTTPException(
                    status_code=408,
                    detail="Request timeout while fetching data"
                )
            elif response.status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later"
                )
            elif response.status_code == 503:
                raise HTTPException(
                    status_code=503,
                    detail="Riot API unavailable"
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Unexpected error occurred"
                )
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail="Request timeout")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Connection error: {str(e)}")


async def fetch_match_history(
    region: str,
    name: str,
    tag: str,
    mode: Optional[str] = None,
    map_name: Optional[str] = None,
    size: int = 5
) -> Dict[str, Any]:
    """
    Fetch match history for a player from Henrik API.
    
    Args:
        region: Player's region (eu, na, ap, kr, latam, br)
        name: Player name (without tag)
        tag: Player tag (without # symbol)
        mode: Optional game mode filter
        map_name: Optional map filter
        size: Number of matches to return (max 10)
        
    Returns:
        Dict containing match history data
        
    Raises:
        HTTPException: For various API errors
    """
    api_key = load_api_key()
    url = f"{HENRIK_BASE_URL}/valorant/v3/matches/{region}/{name}/{tag}"
    
    headers = {
        "Authorization": api_key,
        "Accept": "*/*"
    }
    
    params = {}
    if mode:
        params["mode"] = mode
    if map_name:
        params["map"] = map_name
    if size:
        params["size"] = min(size, 10)  # Enforce max size of 10
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                raise HTTPException(status_code=400, detail="Invalid request parameters")
            elif response.status_code == 403:
                raise HTTPException(status_code=403, detail="API access forbidden")
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"No matches found for {name}#{tag}")
            elif response.status_code == 408:
                raise HTTPException(status_code=408, detail="Request timeout")
            elif response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            elif response.status_code == 503:
                raise HTTPException(status_code=503, detail="Riot API unavailable")
            else:
                raise HTTPException(status_code=response.status_code, detail="Unexpected error")
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail="Request timeout")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Connection error: {str(e)}")


async def fetch_match_details(match_id: str) -> Dict[str, Any]:
    """
    Fetch detailed information for a specific match.
    
    Args:
        match_id: Unique match identifier
        
    Returns:
        Dict containing detailed match data
        
    Raises:
        HTTPException: For various API errors
    """
    api_key = load_api_key()
    url = f"{HENRIK_BASE_URL}/valorant/v2/match/{match_id}"
    
    headers = {
        "Authorization": api_key,
        "Accept": "*/*"
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                raise HTTPException(status_code=400, detail="Invalid match ID")
            elif response.status_code == 403:
                raise HTTPException(status_code=403, detail="API access forbidden")
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
            elif response.status_code == 408:
                raise HTTPException(status_code=408, detail="Request timeout")
            elif response.status_code == 429:
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
            elif response.status_code == 501:
                raise HTTPException(status_code=501, detail="API version not implemented")
            elif response.status_code == 503:
                raise HTTPException(status_code=503, detail="Riot API unavailable")
            else:
                raise HTTPException(status_code=response.status_code, detail="Unexpected error")
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=408, detail="Request timeout")
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Connection error: {str(e)}")


# ============================================================================
# API Endpoints
# ============================================================================

@router.get("/account/{riot_id}")
async def get_account_by_riot_id(
    riot_id: str,
    force: bool = Query(False, description="Force data update")
):
    """
    Get Valorant account information using Riot ID format (name#tag).
    
    Args:
        riot_id: Riot ID in format "name#tag"
        force: Force refresh of cached data
        
    Returns:
        Account information including level, rank, etc.
    """
    try:
        # Validate and split Riot ID format
        if "#" not in riot_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid Riot ID format. Use name#tag (e.g., skywalker17#17605)"
            )
        
        name, tag = riot_id.split("#", 1)
        
        if not name or not tag:
            raise HTTPException(
                status_code=400,
                detail="Both name and tag are required"
            )
        
        account_data = await fetch_account_info(name, tag, force)
        
        return {
            "success": True,
            "data": account_data.get("data", {}),
            "status": account_data.get("status", 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/account/{name}/{tag}")
async def get_account_by_parts(
    name: str,
    tag: str,
    force: bool = Query(False, description="Force data update")
):
    """
    Get Valorant account information using separate name and tag parameters.
    
    Args:
        name: Player name (without # or tag)
        tag: Player tag (without # symbol)
        force: Force refresh of cached data
        
    Returns:
        Account information including level, rank, etc.
    """
    try:
        account_data = await fetch_account_info(name, tag, force)
        
        return {
            "success": True,
            "data": account_data.get("data", {}),
            "status": account_data.get("status", 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/matches/{region}/{name}/{tag}")
async def get_match_history(
    region: Literal["eu", "na", "ap", "kr", "latam", "br"],
    name: str,
    tag: str,
    mode: Optional[Literal[
        "competitive", "custom", "deathmatch", "escalation",
        "teamdeathmatch", "newmap", "replication", "snowballfight",
        "spikerush", "swiftplay", "unrated"
    ]] = None,
    map: Optional[Literal[
        "Ascent", "Split", "Fracture", "Bind", "Breeze",
        "District", "Kasbah", "Piazza", "Lotus", "Pearl",
        "Icebox", "Haven"
    ]] = None,
    size: int = Query(5, ge=1, le=10, description="Number of matches to return (1-10)")
):
    """
    Get match history for a player using separate name and tag.
    
    Args:
        region: Player's region
        name: Player name (without # or tag)
        tag: Player tag (without # symbol)
        mode: Optional filter by game mode
        map: Optional filter by map
        size: Number of matches to return (1-10)
        
    Returns:
        List of recent matches with stats
    """
    try:
        match_data = await fetch_match_history(region, name, tag, mode, map, size)
        
        return {
            "success": True,
            "data": match_data.get("data", []),
            "status": match_data.get("status", 200)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/matches/{riot_id}")
async def get_match_history_by_riot_id(
    riot_id: str,
    region: Literal["eu", "na", "ap", "kr", "latam", "br"] = "eu",
    mode: Optional[Literal[
        "competitive", "custom", "deathmatch", "escalation",
        "teamdeathmatch", "newmap", "replication", "snowballfight",
        "spikerush", "swiftplay", "unrated"
    ]] = None,
    map: Optional[Literal[
        "Ascent", "Split", "Fracture", "Bind", "Breeze",
        "District", "Kasbah", "Piazza", "Lotus", "Pearl",
        "Icebox", "Haven"
    ]] = None,
    size: int = Query(5, ge=1, le=10, description="Number of matches to return (1-10)")
):
    """
    Get match history for a player using Riot ID format (name#tag).
    
    Args:
        riot_id: Riot ID in format "name#tag"
        region: Player's region (defaults to EU)
        mode: Optional filter by game mode
        map: Optional filter by map
        size: Number of matches to return (1-10)
        
    Returns:
        List of recent matches with stats
    """
    try:
        # Validate and split Riot ID
        if "#" not in riot_id:
            raise HTTPException(
                status_code=400,
                detail="Invalid Riot ID format. Use name#tag"
            )
        
        name, tag = riot_id.split("#", 1)
        
        if not name or not tag:
            raise HTTPException(
                status_code=400,
                detail="Both name and tag are required"
            )
        
        match_data = await fetch_match_history(region, name, tag, mode, map, size)
        
        return {
            "success": True,
            "data": match_data.get("data", []),
            "status": match_data.get("status", 200)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/match/{match_id}")
async def get_match_details(match_id: str):
    """
    Get detailed information for a specific match.
    
    Args:
        match_id: Unique match identifier
        
    Returns:
        Detailed match data including player stats, rounds, etc.
    """
    try:
        match_data = await fetch_match_details(match_id)
        
        return {
            "success": True,
            "data": match_data.get("data", {}),
            "status": match_data.get("status", 200)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# ============================================================================
# Match Enrichment Models and Endpoint
# ============================================================================

class MatchToEnrich(BaseModel):
    """Model for a match file to be enriched with API data."""
    filename: str


class EnrichMatchesRequest(BaseModel):
    """Request model for enriching multiple matches with API data."""
    riot_id: str
    matches: List[MatchToEnrich]
    region: str = "eu"
    timezone: str = "Europe/Amsterdam"


@router.post("/enrich-matches")
async def enrich_matches_with_api(request: EnrichMatchesRequest):
    """
    Enrich local match recordings with online API data by matching timestamps.
    
    This endpoint correlates local gameplay recordings with API match data by:
    1. Parsing timestamps from recording filenames (format: DD-MM-YYYY_HH-MM-SS)
    2. Converting local timestamps to UTC for accurate comparison
    3. Finding API matches within a 20-minute window of recording end time
    4. Extracting player statistics and match details
    
    Args:
        request: Contains Riot ID, match list, region, and timezone
        
    Returns:
        Enriched match data with API statistics for each recording
    """
    try:
        # Validate Riot ID format
        if "#" not in request.riot_id:
            raise HTTPException(status_code=400, detail="Invalid Riot ID format")
        
        name, tag = request.riot_id.split("#", 1)
        
        # Load timezone
        try:
            local_tz = pytz.timezone(request.timezone)
        except pytz.UnknownTimeZoneError:
            local_tz = pytz.UTC
        
        # Fetch recent matches from API
        match_history = await fetch_match_history(request.region, name, tag, size=10)
        api_matches = match_history.get("data", [])
        
        if not api_matches:
            return {
                "success": True,
                "riot_id": request.riot_id,
                "enriched_matches": [],
                "message": "No recent matches found"
            }
        
        # Track which API matches have been used to avoid duplicates
        used_api_matches = set()
        enriched_results = []
        
        # Process each recording
        for match in request.matches:
            filename = match.filename
            parts = filename.split('_')
            
            # Validate filename format
            if len(parts) < 2:
                enriched_results.append({"filename": filename, "match_data": None})
                continue
            
            time_str = parts[-1]  # Expected: HH-MM-SS
            date_str = parts[-2]  # Expected: DD-MM-YYYY
            
            if not (date_str.count('-') == 2 and time_str.count('-') == 2):
                enriched_results.append({"filename": filename, "match_data": None})
                continue
            
            # Parse timestamp from filename
            try:
                day, month, year = date_str.split('-')
                hour, minute, second = time_str.split('-')
                
                if not all(p.isdigit() for p in [day, month, year, hour, minute, second]):
                    raise ValueError("Non-numeric values in timestamp")
                
                # Create naive datetime in local timezone
                recording_time_naive = datetime(
                    int(year), int(month), int(day),
                    int(hour), int(minute), int(second)
                )
                
                # Convert local time to UTC for comparison
                recording_time_local = local_tz.localize(recording_time_naive)
                recording_time_utc = recording_time_local.astimezone(pytz.UTC)
                recording_time = recording_time_utc.replace(tzinfo=None)
                
            except (ValueError, IndexError):
                enriched_results.append({"filename": filename, "match_data": None})
                continue
            
            # Find best matching API match within time window
            best_match = None
            best_match_id = None
            min_time_diff = float('inf')
            
            for idx, api_match in enumerate(api_matches):
                # Skip already matched API entries
                if idx in used_api_matches:
                    continue
                
                metadata = api_match.get("metadata", {})
                game_start = metadata.get("game_start")
                game_length = metadata.get("game_length")
                
                if not game_start or not game_length:
                    continue
                
                # Calculate game end time (both in UTC)
                game_end_timestamp = game_start + game_length
                game_end_time = datetime.utcfromtimestamp(game_end_timestamp)
                time_diff_seconds = abs((recording_time - game_end_time).total_seconds())
                
                # Match within 20-minute window (1200 seconds)
                if time_diff_seconds <= 1200 and time_diff_seconds < min_time_diff:
                    # Find player stats in match data
                    player_stats = None
                    for player in api_match.get("players", {}).get("all_players", []):
                        if (player.get("name", "").lower() == name.lower() and
                            player.get("tag", "").lower() == tag.lower()):
                            player_stats = player
                            break
                    
                    # Only match if player was in this game
                    if player_stats:
                        min_time_diff = time_diff_seconds
                        best_match = {
                            "match_id": metadata.get("matchid"),
                            "map": metadata.get("map"),
                            "mode": metadata.get("mode"),
                            "rounds_played": metadata.get("rounds_played"),
                            "game_start": game_start,
                            "game_end": game_end_timestamp,
                            "game_length": game_length,
                            "time_difference_seconds": int(time_diff_seconds),
                            "player_stats": {
                                "agent": player_stats.get("character"),
                                "kills": player_stats.get("stats", {}).get("kills", 0),
                                "deaths": player_stats.get("stats", {}).get("deaths", 0),
                                "assists": player_stats.get("stats", {}).get("assists", 0),
                                "score": player_stats.get("stats", {}).get("score", 0),
                                "headshots": player_stats.get("stats", {}).get("headshots", 0),
                                "bodyshots": player_stats.get("stats", {}).get("bodyshots", 0),
                                "legshots": player_stats.get("stats", {}).get("legshots", 0),
                                "team": player_stats.get("team", "Unknown"),
                            }
                        }
                        best_match_id = idx
            
            # Mark API match as used if found
            if best_match and best_match_id is not None:
                used_api_matches.add(best_match_id)
            
            enriched_results.append({
                "filename": filename,
                "match_data": best_match
            })
        
        # Calculate success metrics
        matched_count = sum(1 for r in enriched_results if r.get("match_data"))
        
        return {
            "success": True,
            "riot_id": request.riot_id,
            "total_matches": len(enriched_results),
            "matched_count": matched_count,
            "enriched_matches": enriched_results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enriching matches: {str(e)}")