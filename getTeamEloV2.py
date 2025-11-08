import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

BASE_DIR = Path("understat_data")

# Core parameters (tune per league for best accuracy)
START_ELO = 1500
K_BASE = 25                # base K
SCALE = 300                # logistic scale 's' in 10^(dr/s)
HOME_ADVANTAGE = 60        # Elo points added to home side
HALF_LIFE_DAYS = 365       # recency half-life for decay
DRAW_NU = 1.25             # Davidson drawness parameter
REGRESS_GAMMA = 0.2        # preseason regression toward league mean (0..1)
MOV_CAP = 4                # cap goal diff for MOV if you want (None to disable)


def davidson_probs(dr: float) -> Tuple[float, float, float]:
    """
    Davidson model to produce 3-way probabilities given rating diff dr.
    dr = (R_home + H) - R_away
    """
    q = 10 ** (dr / SCALE)
    r = 10 ** (-dr / SCALE)
    mid = DRAW_NU * math.sqrt(q * r)
    denom = 1.0 + q + mid
    p_h = q / denom
    p_d = mid / denom
    p_a = 1.0 / denom - p_h  # or r/denom
    return p_h, p_d, p_a


def mov_multiplier(gd: int, dr: float) -> float:
    """
    World Football Elo MOV with rating-gap damping.
    MOV = ln(gd+1) * (2.2 / (2.2 + 0.001*|dr|))
    """
    if gd <= 0:
        return 1.0
    if MOV_CAP is not None:
        gd = min(gd, MOV_CAP)
    return math.log(gd + 1.0) * (2.2 / (2.2 + 0.001 * abs(dr)))


def recency_weight(delta_days: float) -> float:
    """
    Exponential decay by age of match relative to the most recent match in the dataset.
    """
    if delta_days is None or delta_days <= 0:
        return 1.0
    return math.exp(-delta_days / HALF_LIFE_DAYS)


def expected_score_2way(p_h: float, p_d: float) -> float:
    """
    Convert 3-way probs to 2-way expected score for the home team:
    E2 = P(Home) + 0.5*P(Draw)
    """
    return p_h + 0.5 * p_d


def result_to_score(home_goals: int, away_goals: int) -> Tuple[float, float]:
    if home_goals > away_goals:
        return 1.0, 0.0
    if home_goals < away_goals:
        return 0.0, 1.0
    return 0.5, 0.5


def load_matches_from_team_results_folder(folder: Path) -> pd.DataFrame:
    """
    Read all team-centric CSVs and reconstruct one row per match:
      match_id, date, season, home_team, away_team, home_goals, away_goals
    """
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        return pd.DataFrame()
    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f, low_memory=False)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    all_teams = pd.concat(dfs, ignore_index=True, sort=False)
    all_teams = all_teams[all_teams["match_id"].notna()]
    all_teams = all_teams[all_teams["venue"].notna()]

    matches = []
    grouped = all_teams.groupby("match_id")
    for match_id, g in grouped:
        try:
            home_row = g[g["venue"].astype(str).str.lower().isin(["home", "h"])].iloc[0]
            away_row = g[g["venue"].astype(str).str.lower().isin(["away", "a"])].iloc[0]
        except Exception:
            if len(g) >= 2:
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
        try:
            h_goals = int(home_row.get("goals_for"))
            a_goals = int(away_row.get("goals_for"))
        except Exception:
            continue

        date = home_row.get("date") or away_row.get("date") or ""
        season = home_row.get("season") or away_row.get("season") or None

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

    if not matches:
        return pd.DataFrame()

    df_matches = pd.DataFrame(matches)
    df_matches["date_parsed"] = pd.to_datetime(df_matches["date"], errors="coerce")
    df_matches = df_matches.sort_values(["season", "date_parsed"]).reset_index(drop=True)
    return df_matches


def compute_elos_for_league(league_dir: Path) -> None:
    team_results_dir = league_dir / "Team_Results"
    if not team_results_dir.exists():
        print(f"  Skip {league_dir.name}: {team_results_dir} missing")
        return

    df_matches = load_matches_from_team_results_folder(team_results_dir)
    if df_matches.empty:
        print(f"  No matches for {league_dir.name}")
        return

    # For recency weights: age relative to the most recent match in this league
    max_date = df_matches["date_parsed"].max()
    # Elo state and stats
    elos: Dict[str, float] = {}
    stats: Dict[str, Dict[str, int]] = {}

    def ensure(team: str):
        if team not in elos:
            elos[team] = START_ELO
            stats[team] = {"played": 0, "wins": 0, "draws": 0, "losses": 0}

    timeseries: List[Dict] = []
    last_season = None

    for _, m in df_matches.iterrows():
        season = m["season"]
        # Apply preseason regression when season changes (after the first season)
        if pd.notna(season) and last_season is not None and season != last_season and len(elos) > 0:
            league_mean = sum(elos.values()) / len(elos)
            for t in list(elos.keys()):
                elos[t] = (1.0 - REGRESS_GAMMA) * elos[t] + REGRESS_GAMMA * league_mean

        last_season = season if pd.notna(season) else last_season

        home = m["home_team"]
        away = m["away_team"]
        h_goals = int(m["home_goals"])
        a_goals = int(m["away_goals"])
        match_id = int(m["match_id"])
        date = m.get("date", "")
        date_parsed = m.get("date_parsed", pd.NaT)

        ensure(home)
        ensure(away)

        elo_home = elos[home]
        elo_away = elos[away]

        # Rating diff with home advantage
        dr = (elo_home + HOME_ADVANTAGE) - elo_away

        # 3-way expectations
        p_h, p_d, p_a = davidson_probs(dr)
        e2 = expected_score_2way(p_h, p_d)  # expected 2-way score for update

        # Actual scores
        s_home, s_away = result_to_score(h_goals, a_goals)

        # Multipliers
        gd = abs(h_goals - a_goals)
        mov = mov_multiplier(gd, dr)
        # recency weight by age of match relative to max_date
        if isinstance(date_parsed, pd.Timestamp) and pd.notna(date_parsed) and isinstance(max_date, pd.Timestamp):
            delta_days = (max_date - date_parsed).days
            rec = recency_weight(delta_days)
        else:
            rec = 1.0

        k_eff = K_BASE * mov * rec

        # Update using 2-way expectation derived from 3-way probs
        delta = k_eff * (s_home - e2)
        elos[home] = elo_home + delta
        elos[away] = elo_away - delta

        # Stats
        stats[home]["played"] += 1
        stats[away]["played"] += 1
        if s_home == 1.0:
            stats[home]["wins"] += 1
            stats[away]["losses"] += 1
        elif s_away == 1.0:
            stats[away]["wins"] += 1
            stats[home]["losses"] += 1
        else:
            stats[home]["draws"] += 1
            stats[away]["draws"] += 1

        timeseries.append(
            {
                "match_id": match_id,
                "date": date,
                "home_team": home,
                "away_team": away,
                "home_goals": h_goals,
                "away_goals": a_goals,
                "dr_pre": round(dr, 2),
                "p_home": round(p_h, 4),
                "p_draw": round(p_d, 4),
                "p_away": round(p_a, 4),
                "k_eff": round(k_eff, 3),
                "home_elo_post": round(elos[home], 2),
                "away_elo_post": round(elos[away], 2),
            }
        )

    # Final table
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

    # Export as team_elos_v2.csv
    out_elos_path = league_dir / "team_elos_v2.csv"
    out_elos.to_csv(out_elos_path, index=False)

    # timeseries unchanged
    ts_df = pd.DataFrame(timeseries)
    ts_out_path = team_results_dir / "team_elos_timeseries.csv"
    ts_df.to_csv(ts_out_path, index=False)

    print(f"  Wrote {out_elos_path} ({len(rows)} teams)")
    

def main():
    if not BASE_DIR.exists():
        print("understat_data not found")
        return
    
    # Only process the main football leagues
    target_leagues = ["Bundesliga", "EPL", "La_liga", "Ligue_1", "Serie_A"]
    
    print("Computing Elo for leagues...")
    for league_name in target_leagues:
        league_dir = BASE_DIR / league_name
        if league_dir.exists() and league_dir.is_dir():
            print(f"Processing {league_name}")
            compute_elos_for_league(league_dir)
        else:
            print(f"  Skip {league_name}: directory not found")
    print("Done.")


if __name__ == "__main__":
    main()