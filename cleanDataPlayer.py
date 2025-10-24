import pandas as pd
from pathlib import Path
import ast
import json

BASE_DIR = Path("understat_data")
PLAYER_DIR = BASE_DIR / "Players"
CLEANED_DIR = BASE_DIR / "Players_Cleaned"
POSITIONS_DIR = CLEANED_DIR / "player_positions"
SEASONS_DIR = CLEANED_DIR / "player_seasons"

# Create output directories
CLEANED_DIR.mkdir(parents=True, exist_ok=True)
POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
SEASONS_DIR.mkdir(parents=True, exist_ok=True)


def parse_value(value):
    """Safely parse a string that might be a list or dict."""
    if pd.isna(value) or value == '':
        return None
    
    if isinstance(value, (list, dict)):
        return value
    
    if isinstance(value, str):
        try:
            # Try ast.literal_eval first (safer)
            return ast.literal_eval(value)
        except:
            try:
                # Fallback to json
                return json.loads(value.replace("'", '"'))
            except:
                return None
    return None


def extract_season_position_data(df):
    """
    Extract season-position breakdown from the nested data.
    Expected columns in df: player_id, player_name, season, position (which contain nested dicts/lists)
    """
    rows = []
    
    for idx, row in df.iterrows():
        player_id = row.get('player_id')
        player_name = row.get('player_name')
        
        # Try to parse the 'position' column which contains the nested season->position data
        position_data = parse_value(row.get('position'))
        
        if isinstance(position_data, dict):
            # Structure: {season: {position: stats}}
            for season, positions in position_data.items():
                if isinstance(positions, dict):
                    for pos, stats in positions.items():
                        if isinstance(stats, dict):
                            flat_row = {
                                'player_id': player_id,
                                'player_name': player_name,
                                'season': stats.get('season', season),
                                'team': stats.get('team', ''),
                                'position': stats.get('position', pos),
                                'games': stats.get('games', 0),
                                'time': stats.get('time', 0),
                                'goals': stats.get('goals', 0),
                                'assists': stats.get('assists', 0),
                                'xG': stats.get('xG', 0),
                                'xA': stats.get('xA', 0),
                                'shots': stats.get('shots', 0),
                                'key_passes': stats.get('key_passes', 0),
                                'yellow': stats.get('yellow', 0),
                                'red': stats.get('red', 0),
                                'npg': stats.get('npg', 0),
                                'npxG': stats.get('npxG', 0),
                                'xGChain': stats.get('xGChain', 0),
                                'xGBuildup': stats.get('xGBuildup', 0)
                            }
                            rows.append(flat_row)
        
        # Also check the 'season' column which might contain a list of season stats
        season_data = parse_value(row.get('season'))
        
        if isinstance(season_data, list):
            # Structure: [{season: X, position: Y, stats...}, ...]
            for item in season_data:
                if isinstance(item, dict):
                    flat_row = {
                        'player_id': player_id,
                        'player_name': player_name,
                        'season': item.get('season', ''),
                        'team': item.get('team', ''),
                        'position': item.get('position', ''),
                        'games': item.get('games', 0),
                        'time': item.get('time', 0),
                        'goals': item.get('goals', 0),
                        'assists': item.get('assists', 0),
                        'xG': item.get('xG', 0),
                        'xA': item.get('xA', 0),
                        'shots': item.get('shots', 0),
                        'key_passes': item.get('key_passes', 0),
                        'yellow': item.get('yellow', 0),
                        'red': item.get('red', 0),
                        'npg': item.get('npg', 0),
                        'npxG': item.get('npxG', 0),
                        'xGChain': item.get('xGChain', 0),
                        'xGBuildup': item.get('xGBuildup', 0)
                    }
                    rows.append(flat_row)
    
    return rows


def clean_player_files():
    """Process all player CSV files and create cleaned versions split by team presence."""
    all_files = list(PLAYER_DIR.glob("*.csv"))
    
    if not all_files:
        print("No CSV files found in Players directory")
        return
    
    print(f"Processing {len(all_files)} player files...")
    
    success_count = 0
    error_count = 0
    
    for f in all_files:
        try:
            # Read the raw CSV
            df = pd.read_csv(f)
            
            # Extract and flatten the data
            rows = extract_season_position_data(df)
            
            if not rows:
                print(f"No data extracted from {f.name}")
                error_count += 1
                continue
            
            # Create cleaned DataFrame
            cleaned_df = pd.DataFrame(rows)
            
            # Ensure correct column order
            column_order = [
                'player_id', 'player_name', 'season', 'team', 'position', 
                'games', 'time', 'goals', 'assists', 'xG', 'xA', 'shots', 
                'key_passes', 'yellow', 'red', 'npg', 'npxG', 'xGChain', 'xGBuildup'
            ]
            
            # Add any columns that exist in the data but not in our order
            existing_cols = [col for col in column_order if col in cleaned_df.columns]
            extra_cols = [col for col in cleaned_df.columns if col not in column_order]
            final_cols = existing_cols + extra_cols
            
            cleaned_df = cleaned_df[final_cols]
            
            # Convert numeric columns
            numeric_cols = ['games', 'time', 'goals', 'assists', 'xG', 'xA', 'shots', 
                          'key_passes', 'yellow', 'red', 'npg', 'npxG', 'xGChain', 'xGBuildup']
            for col in numeric_cols:
                if col in cleaned_df.columns:
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce').fillna(0)
            
            # Split data based on team presence
            has_team = cleaned_df[cleaned_df['team'].notna() & (cleaned_df['team'] != '')]
            no_team = cleaned_df[cleaned_df['team'].isna() | (cleaned_df['team'] == '')]
            
            # Save to player_seasons folder (entries WITH team)
            if not has_team.empty:
                output_file = SEASONS_DIR / f.name
                has_team.to_csv(output_file, index=False)
                print(f"Seasons: {len(has_team)} rows (with team)")
            
            # Save to player_positions folder (entries WITHOUT team)
            if not no_team.empty:
                output_file = POSITIONS_DIR / f.name
                no_team.to_csv(output_file, index=False)
                print(f"Positions: {len(no_team)} rows (no team)")
            
            success_count += 1
            print(f"Cleaned {f.name}")
            
        except Exception as e:
            error_count += 1
            print(f"Error processing {f.name}: {e}")
    
    print(f"\n Successfully cleaned {success_count} files, {error_count} errors")
    print(f"\n Output:")
    print(f"  - Season data: {SEASONS_DIR}")
    print(f"  - Position data: {POSITIONS_DIR}")


if __name__ == "__main__":
    clean_player_files()