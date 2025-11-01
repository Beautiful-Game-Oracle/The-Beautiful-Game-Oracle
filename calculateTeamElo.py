import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BASE_DIR = Path("understat_data")
START_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 35  # Elo points added to home side when computing expected score

# New knobs to amplify upsets and weight by margin
UPSET_WEIGHT = 2.0    # larger -> bigger reward/penalty for upsets
GOAL_WEIGHT = 1.0     # scales margin-of-victory effect


def expected_score(elo_a: float, elo_b: float, home_adv_a: float = 0.0) -> float:
    """
    Expected score for team A vs B.
    Optionally add home advantage to A (in Elo points).
    """
    return 1.0 / (1.0 + 10 ** (((elo_b - (elo_a + home_adv_a)) / 400.0)))


def result_to_score(home_goals: int, away_goals: int) -> Tuple[float, float]:
    """Return (S_home, S_away) as 1/0.5/0"""
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def load_matches_from_league_file(path: Path) -> pd.DataFrame:
    """
    Read a league's Team_Results/team_results.csv (rows are team-centric)
    and reconstruct one row per match with home/away teams.
    Returns a DataFrame with one row per match and columns:
      match_id, date, season, home_team, away_team, home_goals, away_goals
    """
    df = pd.read_csv(path, low_memory=False)
    # ensure only played matches
    if "match_id" not in df.columns and "match_id" in df.columns:
        pass

    # The aggregated file produced by transform script has match_id + venue
    # Group rows by match_id and collect home/away rows
    matches = []
    grouped = df.groupby("match_id")
    for match_id, g in grouped:
        # need exactly one home and one away row to process
        try:
            home_row = g[g["venue"].str.lower().isin(["home", "h"])].iloc[0]
            away_row = g[g["venue"].str.lower().isin(["away", "a"])].iloc[0]
        except Exception:
            # fallback: try to infer by comparing venues or skip
            if len(g) >= 2:
                # pick the row whose venue==Home/Away string-insensitively
                rows = g.to_dict("records")
                home_row = None
                away_row = None
                for r in rows:
                    v = str(r.get("venue", "")).strip().lower()
                    if v == "home" and home_row is None:
                        home_row = r
                    elif v == "away" and away_row is None:
                        away_row = r
                if home_row is None or away_row is None:
                    continue
            else:
                continue

        # parse goals safely
        try:
            h_goals = int(home_row.get("goals_for"))
        except Exception:
            h_goals = None
        try:
            a_goals = int(away_row.get("goals_for"))
        except Exception:
            a_goals = None

        date = home_row.get("date") or away_row.get("date") or ""
        season = home_row.get("season") or away_row.get("season")

        if h_goals is None or a_goals is None:
            # skip unfinished matches
            continue

        matches.append(
            {
                "match_id": int(match_id),
                "date": date,
                "season": int(season) if pd.notna(season) else None,
                "home_team": home_row.get("team"),
                "away_team": away_row.get("team"),
                "home_goals": h_goals,
                "away_goals": a_goals,
            }
        )

    df_matches = pd.DataFrame(matches)
    # parse date to datetime for sorting; keep invalid as NaT
    if "date" in df_matches.columns:
        df_matches["date_parsed"] = pd.to_datetime(df_matches["date"], errors="coerce")
    else:
        df_matches["date_parsed"] = pd.NaT
    df_matches = df_matches.sort_values(["season", "date_parsed"]).reset_index(drop=True)
    return df_matches


def process_league(league_dir: Path) -> None:
    """
    Compute Elo for all teams in a league. Reads:
      understat_data/{league}/Team_Results/team_results.csv
    Writes:
      understat_data/{league}/team_elos.csv  (final Elo + stats)
      understat_data/{league}/Team_Results/team_elos_timeseries.csv (optional timeseries)
    """
    team_results_path = league_dir / "Team_Results" / "team_results.csv"
    if not team_results_path.exists():
        print(f"  Skipping {league_dir.name}: {team_results_path} not found")
        return

    df_matches = load_matches_from_league_file(team_results_path)
    if df_matches.empty:
        print(f"  No matches found for {league_dir.name}")
        return

    # Elo state and stats
    elos: Dict[str, float] = {}
    stats: Dict[str, Dict[str, int]] = {}  # wins/draws/losses/played
    timeseries_records: List[Dict] = []

    def ensure_team(team: str):
        if team not in elos:
            elos[team] = START_ELO
            stats[team] = {"played": 0, "wins": 0, "draws": 0, "losses": 0}

    for _, m in df_matches.iterrows():
        home = m["home_team"]
        away = m["away_team"]
        h_goals = int(m["home_goals"])
        a_goals = int(m["away_goals"])
        match_id = int(m["match_id"])
        date = m["date"]

        ensure_team(home)
        ensure_team(away)

        elo_home = elos[home]
        elo_away = elos[away]

        # expected scores (consider home advantage)
        E_home = expected_score(elo_home, elo_away, HOME_ADVANTAGE)
        E_away = 1.0 - E_home

        # actual scores
        S_home, S_away = result_to_score(h_goals, a_goals)

        # margin and upset handling:
        goal_diff = abs(h_goals - a_goals)
        # goal multiplier: grows with margin but not linearly
        goal_mult = 1.0 + GOAL_WEIGHT * math.log1p(goal_diff)

        # Determine winner's expected probability for upset multiplier
        if S_home == 1.0:
            p_winner = E_home
        elif S_away == 1.0:
            p_winner = E_away
        else:
            # draw: use the favorite's expected prob to scale (smaller reward if favorite draws)
            p_winner = max(E_home, E_away)

        upset_mult = 1.0 + UPSET_WEIGHT * (1.0 - p_winner)

        # Combined dynamic K
        dynamic_K = K_FACTOR * goal_mult * upset_mult

        # update
        delta_home = dynamic_K * (S_home - E_home)
        delta_away = dynamic_K * (S_away - E_away)

        elos[home] = elo_home + delta_home
        elos[away] = elo_away + delta_away

        # update stats
        stats[home]["played"] += 1
        stats[away]["played"] += 1
        if S_home == 1.0:
            stats[home]["wins"] += 1
            stats[away]["losses"] += 1
        elif S_away == 1.0:
            stats[away]["wins"] += 1
            stats[home]["losses"] += 1
        else:
            stats[home]["draws"] += 1
            stats[away]["draws"] += 1

        # record timeseries after match (include deltas and expectations for debugging)
        timeseries_records.append(
            {
                "match_id": match_id,
                "date": date,
                "league": league_dir.name,
                "home_team": home,
                "away_team": away,
                "home_elo_pre": round(elo_home, 2),
                "away_elo_pre": round(elo_away, 2),
                "E_home": round(E_home, 3),
                "E_away": round(E_away, 3),
                "goal_diff": goal_diff,
                "dynamic_K": round(dynamic_K, 3),
                "delta_home": round(delta_home, 3),
                "delta_away": round(delta_away, 3),
                "home_elo_post": round(elos[home], 2),
                "away_elo_post": round(elos[away], 2),
            }
        )

    # prepare final elos DF
    rows = []
    for team, elo in sorted(elos.items(), key=lambda kv: kv[1], reverse=True):
        s = stats[team]
        rows.append(
            {
                "team": team,
                "final_elo": round(elo, 2),
                "played": s["played"],
                "wins": s["wins"],
                "draws": s["draws"],
                "losses": s["losses"],
            }
        )

    out_elos = pd.DataFrame(rows)
    out_elos_path = league_dir / "team_elos.csv"
    out_elos.to_csv(out_elos_path, index=False)
    # timeseries
    ts_df = pd.DataFrame(timeseries_records)
    ts_out_path = league_dir / "Team_Results" / "team_elos_timeseries.csv"
    ts_df.to_csv(ts_out_path, index=False)

    print(f"  Wrote {out_elos_path} and {ts_out_path} (teams: {len(rows)})")


def main():
    leagues = [p for p in BASE_DIR.iterdir() if p.is_dir()]
    if not leagues:
        print("No leagues found in understat_data/")
        return

    print("Calculating Elo ratings for each league...")
    for league in sorted(leagues, key=lambda p: p.name.lower()):
        print(f"Processing league: {league.name}")
        process_league(league)
    print("Done.")


if __name__ == "__main__":
    main()