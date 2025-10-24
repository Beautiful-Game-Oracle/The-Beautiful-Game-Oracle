import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path
import re

# Base path where league data already exists
BASE_DIR = Path("understat_data")

async def fetch_team_data(understat, team_name, season):
    """Fetch detailed data for a specific team."""
    print(f"Fetching {team_name} {season}...")

    try:
        team_data = {
            "stats": await understat.get_team_stats(team_name, season),
            "players": await understat.get_team_players(team_name, season),
            "results": await understat.get_team_results(team_name, season),
            "fixtures": await understat.get_team_fixtures(team_name, season)
        }
        return team_data
    except Exception as e:
        print(f"Error fetching data for {team_name} ({season}): {e}")
        return None


async def save_team_data(league, season, team_name, data):
    """Save a team's data in its own folder as CSV files."""
    # Safe folder name (remove spaces/special chars)
    team_folder = re.sub(r"[^A-Za-z0-9_]+", "_", team_name)
    output_dir = BASE_DIR / league / str(season) / team_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Team stats (nested dicts)
    if data["stats"]:
        try:
            df_stats = pd.json_normalize(data["stats"], sep="_")
            df_stats.to_csv(output_dir / "team_stats.csv", index=False)
        except Exception:
            with open(output_dir / "team_stats_raw.json", "w", encoding="utf-8") as f:
                import json; json.dump(data["stats"], f, indent=2)

    # Players, results, fixtures
    for key, filename in [
        ("players", "team_players.csv"),
        ("results", "team_results.csv"),
        ("fixtures", "team_fixtures.csv")
    ]:
        if data[key]:
            pd.DataFrame(data[key]).to_csv(output_dir / filename, index=False)

    print(f"Saved {team_name} {season} data in {output_dir}")


async def process_league_year(understat, league, season):
    """Read the league_teams.csv and gather team-level data."""
    league_path = BASE_DIR / league / str(season) / "league_teams.csv"
    if not league_path.exists():
        print(f"Skipping {league} {season}, no team file found.")
        return

    df_teams = pd.read_csv(league_path)
    team_names = df_teams["title"].dropna().unique().tolist()

    for team_name in team_names:
        team_data = await fetch_team_data(understat, team_name, season)
        if team_data:
            await save_team_data(league, season, team_name, team_data)


async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        leagues = [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
        for league in leagues:
            for season_folder in (BASE_DIR / league).iterdir():
                if not season_folder.is_dir():
                    continue
                try:
                    season = int(season_folder.name)
                except ValueError:
                    continue
                await process_league_year(understat, league, season)

    print("\nTeam-level data successfully collected and saved.")

if __name__ == "__main__":
    asyncio.run(main())
