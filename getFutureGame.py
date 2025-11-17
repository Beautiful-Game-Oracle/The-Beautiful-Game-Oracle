import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Base path where league data may already exist
BASE_DIR = Path("understat_data")

# Target leagues: these should match directory names under understat_data
TARGET_LEAGUES = ["EPL", "Bundesliga", "La_liga", "Ligue_1", "Serie_A"]


async def fetch_league_fixtures(understat: Understat, league: str, season: int) -> Optional[List[Dict[str, Any]]]:
    """Try common Understat client methods to fetch fixtures/matches for a league-season.
    Returns a list of match dicts or None on failure.
    """
    candidates = [
        "get_league_matches",
        "get_league_fixtures",
        "get_matches",
        "get_fixtures",
    ]

    for name in candidates:
        fn = getattr(understat, name, None)
        if not fn:
            continue
        try:
            res = await fn(league, season)
            # Some clients return a dict wrapper: try to extract list
            if isinstance(res, dict):
                for key in ("matches", "fixtures", "data", "result"):
                    if key in res and isinstance(res[key], (list, tuple)):
                        return list(res[key])
                # if dict but looks like a match list directly, try to coerce
                # fallback continues
            if isinstance(res, (list, tuple)):
                return list(res)
        except Exception:
            continue
    return None


def extract_field(m: Dict[str, Any], *keys):
    for k in keys:
        if k in m and m[k] is not None:
            return m[k]
    return None


def parse_match_record(m: Dict[str, Any], season: int) -> Dict[str, Any]:
    # sample fields covered: id, isResult, h, a, goals, xG, datetime
    match_id = extract_field(m, "id", "match_id", "_id")
    is_result = m.get("isResult")

    h = m.get("h") or {}
    a = m.get("a") or {}

    home_title = extract_field(h, "title", "team", "name")
    home_short = extract_field(h, "short_title", "short")
    away_title = extract_field(a, "title", "team", "name")
    away_short = extract_field(a, "short_title", "short")

    goals = m.get("goals") or {}
    xg = m.get("xG") or {}

    home_goals = goals.get("h") if isinstance(goals, dict) else None
    away_goals = goals.get("a") if isinstance(goals, dict) else None
    home_xg = xg.get("h") if isinstance(xg, dict) else None
    away_xg = xg.get("a") if isinstance(xg, dict) else None

    date_str = extract_field(m, "datetime", "date", "time")
    parsed = None
    if date_str:
        try:
            ds = str(date_str)
            if ds.endswith("Z"):
                ds = ds.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(ds)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
        except Exception:
            try:
                parsed = datetime.fromtimestamp(int(float(date_str)), tz=timezone.utc)
            except Exception:
                parsed = None

    return {
        "match_id": match_id,
        "season": season,
        "is_result": is_result,
        "date_str": date_str,
        "date_parsed": parsed.isoformat() if parsed else None,
        "home_team": home_title,
        "home_short": home_short,
        "away_team": away_title,
        "away_short": away_short,
        "home_goals": home_goals,
        "away_goals": away_goals,
        "home_xg": home_xg,
        "away_xg": away_xg,
    }


async def process_league_year(understat: Understat, league: str, season: int) -> None:
    print(f"Processing {league} {season}...")
    matches = await fetch_league_fixtures(understat, league, season)
    if not matches:
        print(f"  No fixtures returned for {league} {season}")
        return

    parsed_rows = [parse_match_record(m, season) for m in matches]

    # Filter future matches: is_result is falsy or goals are None, and date in future (if available)
    now = datetime.now(timezone.utc)
    future_rows = []
    for r in parsed_rows:
        is_result = r.get("is_result")
        home_goals = r.get("home_goals")
        away_goals = r.get("away_goals")
        date_iso = r.get("date_parsed")
        date_dt = None
        if date_iso:
            try:
                date_dt = datetime.fromisoformat(date_iso)
                if date_dt.tzinfo is None:
                    date_dt = date_dt.replace(tzinfo=timezone.utc)
            except Exception:
                date_dt = None

        # consider future if not is_result (False/None) and either no goals recorded and date in future or no date
        if is_result is False or is_result in ("false", "False", 0):
            # treat as future if date in future or no goals
            if date_dt:
                if date_dt > now:
                    future_rows.append(r)
            else:
                # no date but flagged as not result — include
                future_rows.append(r)
        else:
            # if goals are None and date is future, include
            if home_goals in (None, "", "null") and away_goals in (None, "", "null"):
                if date_dt and date_dt > now:
                    future_rows.append(r)

    # Output path: one CSV per league (include season in filename to avoid collisions)
    out_dir = BASE_DIR / league
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"future_games_{league}_{season}.csv"

    if future_rows:
        df = pd.DataFrame(future_rows)
        # normalize columns order
        cols = [
            "match_id",
            "season",
            "date_str",
            "date_parsed",
            "home_team",
            "home_short",
            "away_team",
            "away_short",
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "is_result",
        ]
        df = df[cols]
        df = df.sort_values(["date_parsed", "match_id"], na_position="last")
        df.to_csv(out_path, index=False)
        print(f"  Wrote {out_path} ({len(df)} future matches)")
    else:
        # write empty skeleton
        df = pd.DataFrame(columns=["match_id", "season", "date_str", "date_parsed", "home_team", "away_team", "home_goals", "away_goals", "is_result"])
        df.to_csv(out_path, index=False)
        print(f"  No future matches for {league} {season}; wrote empty {out_path}")


async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        # For each target league, if directory exists under BASE_DIR, scan seasons and process
        for league in TARGET_LEAGUES:
            league_dir = BASE_DIR / league
            if not league_dir.exists() or not league_dir.is_dir():
                print(f"Skipping {league}: directory {league_dir} not found")
                continue

            # iterate season folders and pick the most recent (largest numeric folder name)
            season_dirs = [p for p in league_dir.iterdir() if p.is_dir()]
            if not season_dirs:
                # No local season folders: attempt a single recent-season fetch (current year)
                print(f"No season folders for {league}; attempting single fetch for current year only")
                current_year = datetime.now().year
                await process_league_year(understat, league, current_year)
            else:
                # filter folders whose names are integer-like, pick the max (most recent) and process only that
                season_nums = []
                for sd in season_dirs:
                    try:
                        season_nums.append(int(sd.name))
                    except Exception:
                        continue
                if not season_nums:
                    # fallback to current year if no numeric season folders
                    current_year = datetime.now().year
                    await process_league_year(understat, league, current_year)
                else:
                    latest = max(season_nums)
                    print(f"  Found seasons: {sorted(season_nums)}; processing latest: {latest}")
                    await process_league_year(understat, league, latest)

            # After processing seasons, aggregate per-season CSVs into one per-league file
            agg_files = sorted((BASE_DIR / league).glob(f"future_games_{league}_*.csv"))
            if agg_files:
                parts = []
                for p in agg_files:
                    try:
                        parts.append(pd.read_csv(p))
                    except Exception:
                        continue
                if parts:
                    big = pd.concat(parts, ignore_index=True, sort=False)
                    big = big.sort_values(["date_parsed", "match_id"], na_position="last")
                    agg_out = BASE_DIR / league / f"future_games_{league}.csv"
                    big.to_csv(agg_out, index=False)
                    print(f"  Wrote aggregated {agg_out} ({len(big)} rows)")


if __name__ == "__main__":
    asyncio.run(main())
