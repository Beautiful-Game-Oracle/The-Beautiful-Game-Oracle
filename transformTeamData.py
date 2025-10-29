import asyncio
import aiohttp
import ast
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from understat import Understat

BASE_DIR = Path("understat_data")
MAX_CONCURRENCY = 6
YEAR_MIN = 2022
YEAR_MAX = 2025


def safe_parse_dict(val: Any) -> Optional[Dict[str, Any]]:
    """Parse a dict-like string from CSV into a real dict."""
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip()
        try:
            return ast.literal_eval(val)
        except Exception:
            return None
    return None


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None or pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def to_int(x: Any) -> Optional[int]:
    try:
        if x is None or pd.isna(x):
            return None
        return int(float(x))
    except Exception:
        return None


def sanitize_name(name: str) -> str:
    """Sanitize team name for use as filename."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def result_to_points(res: Any) -> Optional[int]:
    """Convert result (w/d/l) to points (3/1/0)."""
    if res is None or (isinstance(res, float) and pd.isna(res)):
        return None
    s = str(res).strip().lower()
    if not s:
        return None
    c = s[0]
    if c == "w":
        return 3
    if c == "d":
        return 1
    if c == "l":
        return 0
    return None


async def fetch_shots_counts(
    understat: Understat,
    match_id: int,
    sem: asyncio.Semaphore,
    cache: Dict[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """Fetch (home_shots, away_shots) for a match with caching."""
    if match_id in cache:
        return cache[match_id]
    
    async with sem:
        try:
            data = await understat.get_match_shots(match_id)
            h_list = data.get("h", []) or []
            a_list = data.get("a", []) or []
            h_shots = len(h_list)
            a_shots = len(a_list)
            cache[match_id] = (h_shots, a_shots)
            return h_shots, a_shots
        except Exception:
            cache[match_id] = (0, 0)
            return 0, 0


def build_team_rows_from_league_results(
    df_league: pd.DataFrame,
    league: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process league_results.csv and build team-centric rows.
    Returns a dict mapping team_name -> list of match rows.
    """
    team_rows_map: Dict[str, List[Dict[str, Any]]] = {}
    
    # Filter to only completed matches
    if "isResult" in df_league.columns:
        df_league = df_league[df_league["isResult"].astype(str).str.lower() == "true"]
    
    for _, row in df_league.iterrows():
        match_id = to_int(row.get("id"))
        if match_id is None:
            continue
        
        season = to_int(row.get("Season"))
        if season is None or season < YEAR_MIN or season > YEAR_MAX:
            continue
        
        h_dict = safe_parse_dict(row.get("h"))
        a_dict = safe_parse_dict(row.get("a"))
        goals_dict = safe_parse_dict(row.get("goals"))
        xg_dict = safe_parse_dict(row.get("xG"))
        
        if not h_dict or not a_dict:
            continue
        
        home_team = h_dict.get("title", "")
        away_team = a_dict.get("title", "")
        
        if not home_team or not away_team:
            continue
        
        date_str = row.get("datetime") or ""
        
        # Extract goals
        h_goals = to_int(goals_dict.get("h")) if goals_dict else None
        a_goals = to_int(goals_dict.get("a")) if goals_dict else None
        
        # Extract xG
        h_xg = to_float(xg_dict.get("h")) if xg_dict else None
        a_xg = to_float(xg_dict.get("a")) if xg_dict else None
        
        # Determine results
        if h_goals is not None and a_goals is not None:
            if h_goals > a_goals:
                h_result, a_result = "w", "l"
            elif h_goals < a_goals:
                h_result, a_result = "l", "w"
            else:
                h_result, a_result = "d", "d"
        else:
            h_result = a_result = None
        
        # Create row for home team
        home_row = {
            "match_id": match_id,
            "date": date_str,
            "league": league,
            "season": season,
            "team": home_team,
            "opponent": away_team,
            "venue": "Home",
            "goals_for": h_goals,
            "goals_against": a_goals,
            "xg_for": h_xg,
            "xg_against": a_xg,
            "shots_for": None,
            "shots_against": None,
            "result": h_result,
        }
        
        # Create row for away team
        away_row = {
            "match_id": match_id,
            "date": date_str,
            "league": league,
            "season": season,
            "team": away_team,
            "opponent": home_team,
            "venue": "Away",
            "goals_for": a_goals,
            "goals_against": h_goals,
            "xg_for": a_xg,
            "xg_against": h_xg,
            "shots_for": None,
            "shots_against": None,
            "result": a_result,
        }
        
        team_rows_map.setdefault(home_team, []).append(home_row)
        team_rows_map.setdefault(away_team, []).append(away_row)
    
    return team_rows_map


async def main():
    # Find league_results.csv in the understat_data folder
    results_file = BASE_DIR / "league_results.csv"
    
    if not results_file.exists():
        print(f"league_results.csv not found at {results_file}")
        return
    
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        print(f"Processing {results_file}...")
        
        try:
            df_league = pd.read_csv(results_file, low_memory=False)
        except Exception as e:
            print(f"Error reading {results_file}: {e}")
            return
        
        # Group by league
        if "League" not in df_league.columns:
            print("League column not found in league_results.csv")
            return
        
        leagues = df_league["League"].unique()
        
        for league in leagues:
            print(f"\nProcessing {league}...")
            df_league_subset = df_league[df_league["League"] == league]
            
            # Build team-centric rows from league results
            team_rows_map = build_team_rows_from_league_results(df_league_subset, league)
            
            if not team_rows_map:
                print(f"No valid data found for {league}")
                continue
            
            # Fetch shots for all unique matches
            all_rows = [r for rows in team_rows_map.values() for r in rows]
            unique_match_ids = sorted({int(r["match_id"]) for r in all_rows})
            print(f"Fetching shots for {len(unique_match_ids)} matches...")
            
            sem = asyncio.Semaphore(MAX_CONCURRENCY)
            cache: Dict[int, Tuple[int, int]] = {}
            tasks = [
                asyncio.create_task(fetch_shots_counts(understat, mid, sem, cache))
                for mid in unique_match_ids
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Assign shots and compute points
            for rows in team_rows_map.values():
                for r in rows:
                    h_shots, a_shots = cache.get(int(r["match_id"]), (0, 0))
                    if r["venue"] == "Home":
                        r["shots_for"] = h_shots
                        r["shots_against"] = a_shots
                    else:
                        r["shots_for"] = a_shots
                        r["shots_against"] = h_shots
                    r["points"] = result_to_points(r.get("result"))
            
            # Output directory
            out_dir = BASE_DIR / league / "Team_Results"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Write per-team files and collect for league aggregate
            league_frames: List[pd.DataFrame] = []
            for team_name, rows in sorted(team_rows_map.items(), key=lambda kv: kv[0].lower()):
                df_out = pd.DataFrame(rows)
                
                # Deduplicate
                df_out = df_out.drop_duplicates(subset=["match_id", "team"])
                
                # Order columns
                cols = [
                    "match_id",
                    "date",
                    "league",
                    "season",
                    "team",
                    "opponent",
                    "venue",
                    "goals_for",
                    "goals_against",
                    "xg_for",
                    "xg_against",
                    "shots_for",
                    "shots_against",
                    "result",
                    "points",
                ]
                df_out = df_out[cols]
                
                # Sort by season and date
                try:
                    df_out["date_parsed"] = pd.to_datetime(df_out["date"])
                    df_out = df_out.sort_values(["season", "date_parsed"]).drop(columns=["date_parsed"])
                except Exception:
                    df_out = df_out.sort_values(["season", "date"])
                
                # Save per-team file
                team_file = out_dir / f"{sanitize_name(team_name)}.csv"
                df_out.to_csv(team_file, index=False, quoting=csv.QUOTE_MINIMAL)
                league_frames.append(df_out)
                print(f"  Wrote {team_file.name}")
            
            # League-wide aggregate
            league_out = pd.concat(league_frames, ignore_index=True)
            league_out.to_csv(out_dir / "team_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
            print(f"  Wrote team_results.csv ({len(league_frames)} teams)")
    
    print("\n✓ Done creating team-centric CSVs for seasons 2022–2025.")


if __name__ == "__main__":
    asyncio.run(main())