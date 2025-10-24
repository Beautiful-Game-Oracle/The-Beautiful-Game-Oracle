import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path

# Leagues and seasons to collect
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Ligue_1", "Serie_A"]
SEASONS = [2022, 2023, 2024, 2025]

# Base directory for data
OUTPUT_DIR = Path("understat_data")
OUTPUT_DIR.mkdir(exist_ok=True)

async def fetch_league_data(understat, league, season):
    """Fetch all relevant league data for a given league and season."""
    print(f"Fetching {league} {season}...")

    try:
        data = {
            "table": await understat.get_league_table(league, season),
            "results": await understat.get_league_results(league, season),
            "players": await understat.get_league_players(league, season),
            "teams": await understat.get_teams(league, season)
        }
        return data
    except Exception as e:
        print(f"Error fetching {league} {season}: {e}")
        return None

async def save_as_csv(league, season, data):
    """Save each data type into a separate CSV file in organized folders."""
    league_dir = OUTPUT_DIR / league / str(season)
    league_dir.mkdir(parents=True, exist_ok=True)

    if data["table"]:
        df_table = pd.DataFrame(data["table"][1:], columns=data["table"][0])
        df_table.to_csv(league_dir / "league_table.csv", index=False)

    if data["results"]:
        pd.DataFrame(data["results"]).to_csv(league_dir / "league_results.csv", index=False)

    if data["players"]:
        pd.DataFrame(data["players"]).to_csv(league_dir / "league_players.csv", index=False)

    if data["teams"]:
        pd.DataFrame(data["teams"]).to_csv(league_dir / "league_teams.csv", index=False)

    print(f"Saved {league} {season} data to {league_dir}")

async def main():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        for league in LEAGUES:
            for season in SEASONS:
                data = await fetch_league_data(understat, league, season)
                if data:
                    await save_as_csv(league, season, data)

    print("\nAll data successfully collected and saved.")

if __name__ == "__main__":
    asyncio.run(main())
