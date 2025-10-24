import asyncio
import aiohttp
import pandas as pd
from understat import Understat
from pathlib import Path
import re
import random
import asyncio
import ast

BASE_DIR = Path("understat_data")
PLAYER_DIR = BASE_DIR / "Players"
PLAYER_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# Step 1: Combine & Clean League Players
# ------------------------------------------------
def combine_league_players():
    print("🔄 Combining all league_players.csv files...")

    all_files = list(BASE_DIR.glob("*/[0-9][0-9][0-9][0-9]/league_players.csv"))
    dfs = []
    for f in all_files:
        league, season = f.parts[-3], f.parts[-2]
        try:
            df = pd.read_csv(f)
            df["League"] = league
            df["Season"] = season
            dfs.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if not dfs:
        raise RuntimeError("No league_players.csv files found!")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["id", "Season", "League"])
    combined.to_csv(BASE_DIR / "combined_players_raw.csv", index=False)

    grouped = (
        combined.groupby(["id", "player_name"])
        .agg({
            "team_title": lambda x: list(set(x.dropna())),
            "League": lambda x: list(set(x.dropna())),
            "Season": lambda x: list(set(x.dropna())),
            "games": "sum",
            "time": "sum",
            "goals": "sum",
            "assists": "sum",
            "shots": "sum",
            "xG": "sum",
            "xA": "sum"
        })
        .reset_index()
    )

    grouped.to_csv(BASE_DIR / "combined_players_cleaned.csv", index=False)
    print(f"Saved combined player list with {len(grouped)} unique players.")
    return grouped


# ------------------------------------------------
# Step 2: Fetch and Save Player Stats
# ------------------------------------------------
async def fetch_player_stats(understat, player_id, player_name):
    """Fetch detailed grouped stats for each player."""
    try:
        stats = await understat.get_player_grouped_stats(player_id)
        return stats
    except Exception as e:
        print(f"Error fetching stats for {player_name} ({player_id}): {e}")
        return None


def flatten_player_stats(player_id, player_name, stats):
    """Flatten nested player grouped stats JSON safely."""
    if not stats:
        return []

    if isinstance(stats, str):
        print(f"Skipping malformed data for {player_name} ({player_id}) [string response]")
        return []

    # Wrap single dicts into a list
    if isinstance(stats, dict):
        stats = [stats]

    if not isinstance(stats, list):
        print(f"Unexpected format for {player_name} ({player_id}): {type(stats)}")
        return []

    rows = []
    for record in stats:
        if not isinstance(record, dict):
            # Skip anything that isn’t a dict (this prevents AttributeError)
            continue

        base = {
            "player_id": player_id,
            "player_name": player_name,
            "season": record.get("season"),
            "position": record.get("position")
        }

        nested_stats = record.get("stats", {})
        if isinstance(nested_stats, dict):
            flat_row = {**base, **nested_stats}
            rows.append(flat_row)
        else:
            rows.append(base)

    return rows


async def save_player_stats(player_id, player_name, stats):
    """Save player stats into CSV."""
    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", player_name.strip())
    file_path = PLAYER_DIR / f"{safe_name}_{player_id}.csv"

    flat_records = flatten_player_stats(player_id, player_name, stats)

    if flat_records:
        pd.DataFrame(flat_records).to_csv(file_path, index=False)
        print(f"Saved stats for {player_name} ({player_id}) → {file_path}")
    else:
        print(f"No usable records for {player_name} ({player_id})")


# ------------------------------------------------
# Async orchestration with concurrency limit
# ------------------------------------------------
async def process_player(sem, understat, pid, name):
    """Fetch and save player stats with concurrency limit."""
    async with sem:
        stats = await fetch_player_stats(understat, pid, name)
        if stats:
            await save_player_stats(pid, name, stats)
        # Small random sleep to avoid hammering the API
        await asyncio.sleep(random.uniform(0.1, 0.3))


async def collect_all_player_stats(players_df, max_concurrent=8):
    """Collect all player stats in parallel with concurrency control."""
    sem = asyncio.Semaphore(max_concurrent)

    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        tasks = [
            asyncio.create_task(process_player(sem, understat, str(row["id"]), row["player_name"]))
            for _, row in players_df.iterrows()
        ]
        await asyncio.gather(*tasks)

    print("\nAll player stats successfully collected.")


# ------------------------------------------------
# Main
# ------------------------------------------------
def main():
    combined_df = combine_league_players()

    # Filter fringe players
    combined_df = combined_df[
        (combined_df["games"] >= 5) & (combined_df["time"] >= 90)
    ]

    # Filter players that played in 2024 or 2025 seasons
    # Parse the Season column from string to actual list
    combined_df['Season'] = combined_df['Season'].apply(ast.literal_eval)

    # Filter players that have '2024' or '2025' in their Season list
    combined_df = combined_df[
    combined_df['Season'].apply(lambda seasons: any(s in ['2024', '2025'] for s in seasons))
]

    combined_df.to_csv(BASE_DIR / "combined_players_filtered.csv", index=False)

    # Run with concurrency
    asyncio.run(collect_all_player_stats(combined_df, max_concurrent=8))


if __name__ == "__main__":
    main()
