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
MAX_CONCURRENCY = 6  # be nice to the API


def safe_parse_dict(val: Any) -> Optional[Dict[str, Any]]:
    """
    Parse a dict-like string from CSV into a real dict.
    Return None if parsing fails.
    """
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return None
    if isinstance(val, str):
        val = val.strip()
        # Normalize quotes if necessary (in case CSV quoted JSON differently)
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
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


async def fetch_shots_counts(
    understat: Understat,
    match_id: int,
    sem: asyncio.Semaphore,
    cache: Dict[int, Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Return (home_shots, away_shots) for match_id.
    Uses in-memory cache and concurrency limit.
    """
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
            # Fallback to zeros if API fails
            cache[match_id] = (0, 0)
            return 0, 0


def build_team_rows_for_season(
    df_results: pd.DataFrame,
    team_name_folder: str,
    league: str,
    season: int,
) -> List[Dict[str, Any]]:
    """
    From team_results.csv (possibly nested fields), build a list of rows with
    team-centric columns ready to augment with shots.
    """
    rows: List[Dict[str, Any]] = []

    # Filter out non-played matches if column exists
    if "isResult" in df_results.columns:
        df_results = df_results[df_results["isResult"].astype(str).str.lower() == "true"]

    for _, row in df_results.iterrows():
        match_id = to_int(row.get("id"))
        if match_id is None:
            continue

        h_dict = safe_parse_dict(row.get("h"))
        a_dict = safe_parse_dict(row.get("a"))
        goals_dict = safe_parse_dict(row.get("goals"))
        xg_dict = safe_parse_dict(row.get("xG"))

        # Figure out venue/side for this team
        side = row.get("side")  # preferred if present
        if side not in ("h", "a"):
            # Infer it by comparing folder name with h/a title (case-insensitive, sanitized)
            team_folder_norm = sanitize_name(str(team_name_folder)).lower()
            h_title = sanitize_name(str(h_dict.get("title"))) if h_dict else ""
            a_title = sanitize_name(str(a_dict.get("title"))) if a_dict else ""
            if team_folder_norm == sanitize_name(h_title).lower():
                side = "h"
            elif team_folder_norm == sanitize_name(a_title).lower():
                side = "a"
            else:
                # Fallback: skip if cannot determine
                continue

        # Opponent
        opponent_title = ""
        if side == "h":
            opponent_title = (a_dict or {}).get("title", "")
        else:
            opponent_title = (h_dict or {}).get("title", "")

        # Goals for/against
        gf = ga = None
        if goals_dict:
            if side == "h":
                gf = to_int(goals_dict.get("h"))
                ga = to_int(goals_dict.get("a"))
            else:
                gf = to_int(goals_dict.get("a"))
                ga = to_int(goals_dict.get("h"))

        # xG for/against
        xgf = xga = None
        if xg_dict:
            if side == "h":
                xgf = to_float(xg_dict.get("h"))
                xga = to_float(xg_dict.get("a"))
            else:
                xgf = to_float(xg_dict.get("a"))
                xga = to_float(xg_dict.get("h"))

        # Result (derive if not provided)
        result = row.get("result", None)
        if not isinstance(result, str) and gf is not None and ga is not None:
            if gf > ga:
                result = "w"
            elif gf < ga:
                result = "l"
            else:
                result = "d"

        date_str = row.get("datetime") or row.get("date") or ""

        rows.append(
            {
                "match_id": match_id,
                "date": date_str,
                "league": league,
                "season": season,
                "team": team_name_folder,  # use folder (sanitized) for stable naming
                "opponent": opponent_title,
                "venue": "Home" if side == "h" else "Away",
                "goals_for": gf,
                "goals_against": ga,
                "xg_for": xgf,
                "xg_against": xga,
                # shots_for/against to be filled later
                "shots_for": None,
                "shots_against": None,
                "result": result,
            }
        )

    return rows


async def process_team_season(
    understat: Understat, league: str, season: int, team_dir: Path
) -> Optional[Path]:
    """
    Read team_results.csv in {team_dir}, build flattened team-centric CSV with shots,
    and write to understat_data/{league}/{season}/Team/{TeamName}.csv
    """
    team_name_folder = team_dir.name
    results_path = team_dir / "team_results.csv"
    if not results_path.exists():
        return None

    try:
        df_results = pd.read_csv(results_path, low_memory=False)
    except Exception:
        return None

    base_rows = build_team_rows_for_season(df_results, team_name_folder, league, season)
    if not base_rows:
        return None

    # Fetch shots for all matches with concurrency + cache
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    cache: Dict[int, Tuple[int, int]] = {}
    tasks: List[asyncio.Task] = []

    for r in base_rows:
        mid = int(r["match_id"])
        tasks.append(asyncio.create_task(fetch_shots_counts(understat, mid, sem, cache)))

    shots_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Assign shots counts
    for r, shot_res in zip(base_rows, shots_results):
        if isinstance(shot_res, Exception):
            h_shots, a_shots = 0, 0
        else:
            h_shots, a_shots = shot_res

        if r["venue"] == "Home":
            r["shots_for"] = h_shots
            r["shots_against"] = a_shots
        else:
            r["shots_for"] = a_shots
            r["shots_against"] = h_shots

    # Build DataFrame and set column order
    df_out = pd.DataFrame(base_rows)
    df_out = df_out[
        [
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
        ]
    ]

    # Sort by date if possible
    if "date" in df_out.columns:
        try:
            df_out["date_parsed"] = pd.to_datetime(df_out["date"])
            df_out = df_out.sort_values("date_parsed").drop(columns=["date_parsed"])
        except Exception:
            pass

    out_dir = BASE_DIR / league / str(season) / "Team"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{team_name_folder}.csv"

    # Ensure consistent CSV quoting
    df_out.to_csv(out_path, index=False, quoting=csv.QUOTE_MINIMAL)
    return out_path


async def main():
    # Scan understat_data/{league}/{season}/{TeamName}/team_results.csv
    leagues = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    if not leagues:
        print("No leagues found under understat_data")
        return

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        for league_path in leagues:
            league = league_path.name
            season_dirs = [p for p in league_path.iterdir() if p.is_dir()]
            for season_dir in season_dirs:
                try:
                    season = int(season_dir.name)
                except ValueError:
                    continue

                # Team directories that have team_results.csv
                team_dirs = [
                    p
                    for p in season_dir.iterdir()
                    if p.is_dir() and (p / "team_results.csv").exists()
                ]
                if not team_dirs:
                    continue

                print(f"Processing {league} {season} ({len(team_dirs)} teams)...")
                for team_dir in sorted(team_dirs, key=lambda d: d.name.lower()):
                    out_path = await process_team_season(understat, league, season, team_dir)
                    if out_path:
                        print(f"  Wrote {out_path}")

    print("\nDone creating team-centric CSVs.")


if __name__ == "__main__":
    asyncio.run(main())