import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path
from datetime import datetime
import calendar

# Leagues to update
LEAGUES = ["EPL", "La_liga", "Bundesliga", "Ligue_1", "Serie_A"]

# Base directory for data
BASE_DIR = Path("understat_data")
BASE_DIR.mkdir(exist_ok=True)


def get_current_season() -> int:
    """
    Determine current football season based on current date.
    Football seasons typically run from August to May of the following year.
    - August 2024 to May 2025 = 2024 season
    - August 2025 to May 2026 = 2025 season
    """
    now = datetime.now()
    year = now.year
    month = now.month
    
    # If we're in Jan-July, we're in the second half of the season
    # So the season year is the previous year
    if month <= 7:
        return year - 1
    else:
        # If we're in Aug-Dec, we're in the first half of the season
        # So the season year is the current year
        return year


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
    league_dir = BASE_DIR / league / str(season)
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


async def update_league_results():
    """Update the combined league_results.csv with current season data."""
    current_season = get_current_season()
    
    # Load existing league_results.csv if it exists
    results_file = BASE_DIR / "league_results.csv"
    if results_file.exists():
        print(f"Loading existing league_results.csv...")
        existing_df = pd.read_csv(results_file)
        # Remove current season data to replace with fresh data
        existing_df = existing_df[existing_df["Season"] != current_season]
        print(f"Removed existing {current_season} season data")
    else:
        existing_df = pd.DataFrame()
        print("No existing league_results.csv found, creating new one")

    # Collect new data for current season
    new_data = []
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        for league in LEAGUES:
            print(f"Fetching {league} {current_season}...")
            try:
                results = await understat.get_league_results(league, current_season)
                if results:
                    df_results = pd.DataFrame(results)
                    df_results["League"] = league
                    df_results["Season"] = current_season
                    new_data.append(df_results)
                    print(f"  Added {len(df_results)} matches for {league}")
                else:
                    print(f"  No results found for {league} {current_season}")
            except Exception as e:
                print(f"  Error fetching {league} {current_season}: {e}")

    # Combine and save
    if new_data:
        new_df = pd.concat(new_data, ignore_index=True)
        
        if not existing_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            final_df = new_df
            
        # Sort by season and date
        try:
            final_df["datetime"] = pd.to_datetime(final_df["datetime"], errors="coerce")
            final_df = final_df.sort_values(["Season", "datetime"])
        except Exception:
            final_df = final_df.sort_values(["Season"])
        
        final_df.to_csv(results_file, index=False)
        print(f"Updated league_results.csv with {len(new_df)} new matches")
        print(f"Total matches in file: {len(final_df)}")
    else:
        print("No new data to add")


async def main():
    """Main update function - fetches current season data only."""
    current_season = get_current_season()
    
    print(f"Updating data for current season: {current_season}")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Leagues: {', '.join(LEAGUES)}")
    
    # Update individual league folders with current season data
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)

        for league in LEAGUES:
            data = await fetch_league_data(understat, league, current_season)
            if data:
                await save_as_csv(league, current_season, data)

    # Update combined league_results.csv
    await update_league_results()

    print(f"\n Update complete for season {current_season}!")
    print("\n Next steps:")
    print("1. Run transformTeamData.py to process team data")
    print("2. Run getTeamEloV2.py to update Elo ratings")


if __name__ == "__main__":
    asyncio.run(main())