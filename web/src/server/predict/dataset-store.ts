import fs from "node:fs";
import path from "node:path";

import { parse } from "csv-parse/sync";

import {
  getDatasetRoot,
  getFeatureDatasetVersion,
  getTeamCacheDir,
  getPredictionTargetSeason,
} from "@/config/env";
import {
  FINANCIAL_FEATURE_COLUMNS,
  getFinancialFeatureMap,
  buildApproximateFinancialSnapshot,
} from "@/server/predict/financial-data";
import { buildFixtureKey } from "@/server/predict/fixture-key";

// dataset-store.ts centralizes CSV loading, feature derivation, and synthetic fixture creation for the web API.

const DEFAULT_DATASET_VERSION = "7";
const EPS = 1e-3;
const DATASET_ROOT = getDatasetRoot();
const TEAM_CACHE_DIR = getTeamCacheDir();
const TARGET_SEASON = getPredictionTargetSeason();

export type FixtureRow = {
  season: string;
  league: string;
  home: string;
  away: string;
  values: Record<string, number>;
};

type DatasetRecord = {
  matchId: number;
  season: string;
  league: string;
  home: string;
  away: string;
  values: Record<string, number>;
  orderKey: number;
};

type RollingConfig = {
  valueField: string;
  teamField: "home" | "away";
  window: number;
};

export type TeamCacheEntry = {
  name: string;
  canonical: string;
  shortName: string;
};

export type TeamCachePayload = {
  league: string;
  season: string;
  teams: TeamCacheEntry[];
};

const datasetCache = new Map<string, Map<string, FixtureRow>>();
const latestSeasonByVersion = new Map<string, string>();
type SnapshotKey = string;

type TeamSnapshot = {
  league: string;
  season: string;
  homeRecord: DatasetRecord | null;
  awayRecord: DatasetRecord | null;
};

const teamSnapshotsByVersion = new Map<string, Map<SnapshotKey, TeamSnapshot>>();

function datasetPathFor(version: string) {
  return path.resolve(DATASET_ROOT, `Dataset_Version_${version}.csv`);
}

function resolveDatasetVersion(
  input?: string | number | null,
): string {
  const override = getFeatureDatasetVersion();
  const candidate = override ?? input;
  if (!candidate) return DEFAULT_DATASET_VERSION;
  const text = String(candidate);
  const match = text.match(/(\d+)/);
  return match ? match[1] : text;
}

function loadDataset(version: string) {
  let cache = datasetCache.get(version);
  if (cache) return cache;
  const datasetPath = datasetPathFor(version);
  if (!fs.existsSync(datasetPath)) {
    throw new Error(
      `Dataset not found at ${datasetPath}. Run build_dataset_version${version}.py or provide FEATURE_DATASET_VERSION.`,
    );
  }
  const csv = fs.readFileSync(datasetPath, "utf-8");
  const records = parse(csv, {
    columns: true,
    skip_empty_lines: true,
  }) as Array<Record<string, string>>;
  const parsed = records.map((record, idx) => buildRecord(record, idx));
  enrichRecords(parsed);
  mergeFinancialRecords(parsed, version);
  const latestSeason = determineLatestSeason(parsed);
  latestSeasonByVersion.set(version, latestSeason);
  ensureTeamCaches(parsed);
  cache = buildFixtureMap(parsed);
  datasetCache.set(version, cache);
  const snapshots = buildTeamSnapshots(parsed);
  teamSnapshotsByVersion.set(version, snapshots);
  return cache;
}

function buildRecord(record: Record<string, string>, idx: number): DatasetRecord {
  const season = String(record.season ?? "");
  const home = String(record.home_team_name ?? "");
  const away = String(record.away_team_name ?? "");
  if (!home || !away) {
    throw new Error(`Fixture missing team names at row ${idx}`);
  }
  const league = String(record.league ?? "EPL");
  const matchId = Number(record.match_id ?? idx);
  const values: Record<string, number> = {};
  for (const [key, value] of Object.entries(record)) {
    if (value === "" || value === null || value === undefined) continue;
    const num = Number(value);
    if (!Number.isNaN(num)) {
      values[key] = num;
    }
  }
  const dateText = record.match_datetime_utc ?? record.match_date ?? "";
  const dateValue = Date.parse(dateText);
  const orderKey = Number.isNaN(dateValue) ? idx : dateValue;
  return { matchId, season, league, home, away, values, orderKey };
}

function buildFixtureMap(records: DatasetRecord[]) {
  const map = new Map<string, FixtureRow>();
  for (const record of records) {
    const key = buildFixtureKey(record.season, record.home, record.away);
    map.set(key, {
      season: record.season,
      league: record.league,
      home: record.home,
      away: record.away,
      values: record.values,
    });
  }
  return map;
}

function determineLatestSeason(records: DatasetRecord[]) {
  let latest: string | null = null;
  for (const record of records) {
    const season = String(record.season ?? "");
    if (!season) continue;
    if (!latest || Number(season) > Number(latest)) {
      latest = season;
    }
  }
  return latest ?? DEFAULT_DATASET_VERSION;
}

export function getFixtureRow(
  season: string,
  home: string,
  away: string,
  datasetVersion?: string | number | null,
): FixtureRow {
  const version = resolveDatasetVersion(datasetVersion);
  const map = loadDataset(version);
  const requestedSeason = String(season);
  const key = buildFixtureKey(requestedSeason, home, away);
  const match = map.get(key);
  if (match) {
    return match;
  }
  const fallbackSeason = latestSeasonByVersion.get(version);
  if (fallbackSeason && fallbackSeason !== requestedSeason) {
    const fallbackMatch = map.get(buildFixtureKey(fallbackSeason, home, away));
    if (fallbackMatch) {
      return {
        ...fallbackMatch,
        season: requestedSeason,
      };
    }
  }
  const synthetic = buildSyntheticFixture(version, requestedSeason, home, away);
  if (synthetic) {
    return synthetic;
  }
  throw new Error(
    `Fixture ${home} vs ${away} (${requestedSeason}) not found in dataset version ${version}.`,
  );
}

export function getLatestSeason(
  datasetVersion?: string | number | null,
): string {
  const version = resolveDatasetVersion(datasetVersion);
  loadDataset(version);
  if (TARGET_SEASON) {
    return TARGET_SEASON;
  }
  return latestSeasonByVersion.get(version) ?? DEFAULT_DATASET_VERSION;
}

export function getTeamCache(
  league: string,
  season?: string | number | null,
  datasetVersion?: string | number | null,
): TeamCachePayload {
  const version = resolveDatasetVersion(datasetVersion);
  const resolvedSeason = season ? String(season) : getLatestSeason(version);
  loadDataset(version);
  const direct = loadTeamCachePayload(league, resolvedSeason);
  if (direct) {
    return normalizeTeamCacheSeason(direct, resolvedSeason);
  }
  const fallbackSeason = latestSeasonByVersion.get(version);
  if (fallbackSeason) {
    const fallback = loadTeamCachePayload(league, fallbackSeason);
    if (fallback) {
      return normalizeTeamCacheSeason(fallback, resolvedSeason);
    }
  }
  throw new Error(
    `Team cache missing for ${league} ${resolvedSeason}. Verify dataset contains fixtures.`,
  );
}

function loadTeamCachePayload(
  league: string,
  season?: string | number | null,
): TeamCachePayload | null {
  if (!season) return null;
  const cachePath = teamCachePath(league, String(season));
  if (!fs.existsSync(cachePath)) {
    return null;
  }
  const json = fs.readFileSync(cachePath, "utf-8");
  return JSON.parse(json) as TeamCachePayload;
}

function normalizeTeamCacheSeason(payload: TeamCachePayload, season: string): TeamCachePayload {
  if (payload.season === season) {
    return payload;
  }
  return {
    ...payload,
    season,
  };
}

function enrichRecords(records: DatasetRecord[], window = 5) {
  records.sort((a, b) => a.orderKey - b.orderKey);
  computeProbEdge(records);
  computeRecentGamesFrac(records, window);
  computeSmoothedAverages(records, window);
  computeShotFeatures(records, window);
}

function computeProbEdge(records: DatasetRecord[]) {
  for (const record of records) {
    const homeProb = record.values.forecast_home_win ?? 0;
    const awayProb = record.values.forecast_away_win ?? 0;
    record.values.prob_edge = homeProb - awayProb;
  }
}

function computeRecentGamesFrac(records: DatasetRecord[], window: number) {
  const homeCounts = new Map<string, number>();
  const awayCounts = new Map<string, number>();
  for (const record of records) {
    const homeKey = `${record.season}|${record.home.toLowerCase()}`;
    const awayKey = `${record.season}|${record.away.toLowerCase()}`;
    const homeGames = Math.min(homeCounts.get(homeKey) ?? 0, window);
    const awayGames = Math.min(awayCounts.get(awayKey) ?? 0, window);
    record.values.home_recent_games_frac = homeGames / window;
    record.values.away_recent_games_frac = awayGames / window;
    homeCounts.set(homeKey, (homeCounts.get(homeKey) ?? 0) + 1);
    awayCounts.set(awayKey, (awayCounts.get(awayKey) ?? 0) + 1);
  }
}

function computeSmoothedAverages(records: DatasetRecord[], window: number) {
  assignSmoothedAverage(records, "home_goals_for_last_5", "home_recent_games_frac", "home_goals_for_avg5", window);
  assignSmoothedAverage(records, "home_goals_against_last_5", "home_recent_games_frac", "home_goals_against_avg5", window);
  assignSmoothedAverage(records, "home_xg_for_last_5", "home_recent_games_frac", "home_xg_for_avg5", window);
  assignSmoothedAverage(records, "home_xg_against_last_5", "home_recent_games_frac", "home_xg_against_avg5", window);
  assignSmoothedAverage(records, "home_points_last_5", "home_recent_games_frac", "home_points_avg5", window);

  assignSmoothedAverage(records, "away_goals_for_last_5", "away_recent_games_frac", "away_goals_for_avg5", window);
  assignSmoothedAverage(records, "away_goals_against_last_5", "away_recent_games_frac", "away_goals_against_avg5", window);
  assignSmoothedAverage(records, "away_xg_for_last_5", "away_recent_games_frac", "away_xg_for_avg5", window);
  assignSmoothedAverage(records, "away_xg_against_last_5", "away_recent_games_frac", "away_xg_against_avg5", window);
  assignSmoothedAverage(records, "away_points_last_5", "away_recent_games_frac", "away_points_avg5", window);

  for (const record of records) {
    const values = record.values;
    values.att_gap_avg5 = (values.home_goals_for_avg5 ?? 0) - (values.away_goals_for_avg5 ?? 0);
    values.def_gap_avg5 = (values.away_goals_against_avg5 ?? 0) - (values.home_goals_against_avg5 ?? 0);
    values.points_gap_avg5 = (values.home_points_avg5 ?? 0) - (values.away_points_avg5 ?? 0);
    values.xg_att_gap_avg5 = (values.home_xg_for_avg5 ?? 0) - (values.away_xg_for_avg5 ?? 0);
    values.xg_def_gap_avg5 = (values.away_xg_against_avg5 ?? 0) - (values.home_xg_against_avg5 ?? 0);
    const homeXg = values.home_xg_for_avg5 ?? 0;
    const awayXg = values.away_xg_for_avg5 ?? 0;
    const ratio = Math.log((homeXg + EPS) / (awayXg + EPS));
    values.log_xg_ratio_avg5 = Number.isFinite(ratio) ? ratio : 0;
  }
}

function assignSmoothedAverage(
  records: DatasetRecord[],
  sumColumn: string,
  fracColumn: string,
  targetColumn: string,
  window: number,
) {
  const sums = records.map((record) => record.values[sumColumn] ?? 0);
  const fracs = records.map((record) => clamp(record.values[fracColumn] ?? 0, 0, 1));
  const averages = smoothedAverage(sums, fracs, window);
  records.forEach((record, idx) => {
    record.values[targetColumn] = averages[idx];
  });
}

function smoothedAverage(
  sums: number[],
  gamesFrac: number[],
  window: number,
) {
  const perMatch = sums.map((sum, idx) => {
    const games = gamesFrac[idx] * window;
    if (games <= 0) {
      return Number.NaN;
    }
    return sum / games;
  });
  const valid = perMatch.filter((value) => Number.isFinite(value));
  const prior =
    valid.length > 0
      ? valid.reduce((acc, value) => acc + value, 0) / valid.length
      : 0;
  return perMatch.map((value, idx) => {
    const alpha = clamp(gamesFrac[idx], 0, 1);
    const perGame = Number.isFinite(value) ? value : prior;
    return alpha * perGame + (1 - alpha) * prior;
  });
}

function computeShotFeatures(records: DatasetRecord[], window: number) {
  fillMedian(records, "home_shots_for");
  fillMedian(records, "away_shots_for");
  records.forEach((record) => {
    record.values.home_shots_allowed = record.values.away_shots_for ?? 0;
    record.values.away_shots_allowed = record.values.home_shots_for ?? 0;
  });

  const homeShotsAvg5 = priorRollingMean(records, {
    valueField: "home_shots_for",
    teamField: "home",
    window,
  });
  const awayShotsAvg5 = priorRollingMean(records, {
    valueField: "away_shots_for",
    teamField: "away",
    window,
  });
  const homeAllowedAvg5 = priorRollingMean(records, {
    valueField: "home_shots_allowed",
    teamField: "home",
    window,
  });
  const awayAllowedAvg5 = priorRollingMean(records, {
    valueField: "away_shots_allowed",
    teamField: "away",
    window,
  });

  const shortWindow = Math.min(3, window);
  const homeShotsAvg3 = priorRollingMean(records, {
    valueField: "home_shots_for",
    teamField: "home",
    window: shortWindow,
  });
  const awayShotsAvg3 = priorRollingMean(records, {
    valueField: "away_shots_for",
    teamField: "away",
    window: shortWindow,
  });
  const homeAllowedAvg3 = priorRollingMean(records, {
    valueField: "home_shots_allowed",
    teamField: "home",
    window: shortWindow,
  });
  const awayAllowedAvg3 = priorRollingMean(records, {
    valueField: "away_shots_allowed",
    teamField: "away",
    window: shortWindow,
  });

  records.forEach((record, idx) => {
    const values = record.values;
    values.home_shots_for_avg5 = homeShotsAvg5[idx];
    values.away_shots_for_avg5 = awayShotsAvg5[idx];
    values.home_shots_allowed_avg5 = homeAllowedAvg5[idx];
    values.away_shots_allowed_avg5 = awayAllowedAvg5[idx];

    values.home_shots_for_avg3 = homeShotsAvg3[idx];
    values.away_shots_for_avg3 = awayShotsAvg3[idx];
    values.home_shots_allowed_avg3 = homeAllowedAvg3[idx];
    values.away_shots_allowed_avg3 = awayAllowedAvg3[idx];

    values.shot_vol_gap_avg5 = values.home_shots_for_avg5 - values.away_shots_for_avg5;
    values.shot_suppress_gap_avg5 = values.away_shots_allowed_avg5 - values.home_shots_allowed_avg5;
    const ratio = Math.log(
      (values.home_shots_for_avg5 + EPS) / (values.away_shots_for_avg5 + EPS),
    );
    values.log_shot_ratio_avg5 = Number.isFinite(ratio) ? ratio : 0;
    values.shots_tempo_avg5 = (values.home_shots_for_avg5 + values.away_shots_for_avg5) / 2;

    values.shot_volume_gap_avg3 = values.home_shots_for_avg3 - values.away_shots_for_avg3;
    values.shot_suppress_gap_avg3 = values.away_shots_allowed_avg3 - values.home_shots_allowed_avg3;
    values.shots_tempo_avg3 = (values.home_shots_for_avg3 + values.away_shots_for_avg3) / 2;
  });

  seasonZScore(records, "shot_volume_gap_avg3", "shot_volume_gap_avg3_season_z");
  seasonZScore(records, "shot_suppress_gap_avg3", "shot_suppress_gap_avg3_season_z");
  seasonZScore(records, "shots_tempo_avg3", "shots_tempo_avg3_season_z");
}

function priorRollingMean(
  records: DatasetRecord[],
  config: RollingConfig,
) {
  const result = new Array(records.length).fill(0);
  const history = new Map<string, number[]>();
  const fallback = new Map<string, number>();
  const columnValues = records
    .map((record) => record.values[config.valueField])
    .filter((value) => typeof value === "number" && Number.isFinite(value)) as number[];
  const globalMedian = median(columnValues) ?? 0;
  records.forEach((record, idx) => {
    const teamName = config.teamField === "home" ? record.home : record.away;
    const key = `${record.season}|${teamName.toLowerCase()}`;
    const queue = history.get(key) ?? [];
    const sample = queue.slice(-config.window);
    if (sample.length === 0) {
      const last = fallback.get(key);
      result[idx] = last ?? globalMedian;
    } else {
      const sum = sample.reduce((acc, value) => acc + value, 0);
      result[idx] = sum / sample.length;
    }
    const current = record.values[config.valueField];
    if (typeof current === "number" && Number.isFinite(current)) {
      queue.push(current);
      if (queue.length > config.window) {
        queue.shift();
      }
      history.set(key, queue);
      fallback.set(key, current);
    }
  });
  return result;
}

function seasonZScore(
  records: DatasetRecord[],
  source: string,
  target: string,
) {
  const groups = new Map<string, number[]>();
  records.forEach((record) => {
    const value = record.values[source];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      return;
    }
    const list = groups.get(record.season) ?? [];
    list.push(value);
    groups.set(record.season, list);
  });
  const stats = new Map<string, { mean: number; std: number }>();
  for (const [season, values] of groups.entries()) {
    if (values.length === 0) {
      stats.set(season, { mean: 0, std: 1 });
      continue;
    }
    const mean = values.reduce((acc, value) => acc + value, 0) / values.length;
    const variance =
      values.reduce((acc, value) => acc + (value - mean) ** 2, 0) / values.length;
    const std = Math.sqrt(variance) || 1;
    stats.set(season, { mean, std });
  }
  records.forEach((record) => {
    const value = record.values[source];
    const stat = stats.get(record.season);
    if (!stat || typeof value !== "number" || !Number.isFinite(value)) {
      record.values[target] = 0;
      return;
    }
    const z = (value - stat.mean) / stat.std;
    record.values[target] = Number.isFinite(z) ? z : 0;
  });
}

function mergeFinancialRecords(records: DatasetRecord[], version: string) {
  const financialMap = getFinancialFeatureMap(version);
  let merged = 0;
  for (const record of records) {
    const key = buildFixtureKey(record.season, record.home, record.away);
    let snapshot = financialMap.get(key);
    if (!snapshot) {
      snapshot = buildApproximateFinancialSnapshot(version, record.season, record.home, record.away);
      if (snapshot) {
        financialMap.set(key, snapshot);
      }
    }
    if (!snapshot) {
      continue;
    }
    merged += 1;
    for (const column of FINANCIAL_FEATURE_COLUMNS) {
      const value = snapshot[column];
      if (typeof value === "number" && Number.isFinite(value)) {
        record.values[column] = value;
      }
    }
    record.values.valuationGap = normalizeFinancialGap(snapshot.squad_value_diff, 1e8);
    record.values.wageGap = normalizeFinancialGap(snapshot.wage_bill_diff, 1e7);
    record.values.netSpendGap = normalizeFinancialGap(snapshot.avg_salary_diff, 1e6);
  }
  if (merged === 0) {
    throw new Error(
      `[dataset-store] Financial dataset loaded but no fixtures merged for version ${version}. Ensure financial_dataset.csv aligns with main fixture CSV.`,
    );
  }
}

function normalizeFinancialGap(value: number | undefined, scale: number) {
  if (typeof value !== "number" || Number.isNaN(value) || !Number.isFinite(value)) {
    return 0;
  }
  return value / scale;
}

function fillMedian(records: DatasetRecord[], column: string) {
  const values = records
    .map((record) => record.values[column])
    .filter((value) => typeof value === "number" && Number.isFinite(value)) as number[];
  const medianValue = median(values) ?? 0;
  records.forEach((record) => {
    if (typeof record.values[column] !== "number" || Number.isNaN(record.values[column])) {
      record.values[column] = medianValue;
    }
  });
}

function median(values: number[]) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

function ensureTeamCaches(records: DatasetRecord[]) {
  const perLeague = new Map<string, Map<string, Set<string>>>();
  for (const record of records) {
    const league = String(record.league ?? "EPL");
    const season = String(record.season ?? "");
    if (!season) continue;
    const seasonMap = perLeague.get(league) ?? new Map<string, Set<string>>();
    perLeague.set(league, seasonMap);
    const teamSet = seasonMap.get(season) ?? new Set<string>();
    teamSet.add(record.home);
    teamSet.add(record.away);
    seasonMap.set(season, teamSet);
  }
  for (const [league, seasons] of perLeague.entries()) {
    for (const [season, teams] of seasons.entries()) {
      writeTeamCacheIfMissing(league, season, Array.from(teams));
    }
  }
}

function writeTeamCacheIfMissing(league: string, season: string, teamNames: string[]) {
  fs.mkdirSync(TEAM_CACHE_DIR, { recursive: true });
  const cachePath = teamCachePath(league, season);
  if (fs.existsSync(cachePath)) return;
  const payload: TeamCachePayload = {
    league,
    season,
    teams: teamNames
      .map((name) => ({
        name,
        canonical: slugify(name),
        shortName: deriveShortName(name),
      }))
      .sort((a, b) => a.name.localeCompare(b.name)),
  };
  fs.writeFileSync(cachePath, JSON.stringify(payload, null, 2));
}

function teamCachePath(league: string, season: string) {
  const safeLeague = slugify(league).toUpperCase();
  const safeSeason = String(season ?? "").replace(/[^0-9a-z]+/gi, "_") || "latest";
  return path.join(TEAM_CACHE_DIR, `${safeLeague}_${safeSeason}.json`);
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
}

function deriveShortName(name: string) {
  const cleaned = name.replace(/[^A-Za-z ]+/g, "").trim();
  if (!cleaned) {
    return name.slice(0, 3).toUpperCase();
  }
  const parts = cleaned.split(/\s+/);
  if (parts.length === 1) {
    return parts[0].slice(0, 3).toUpperCase();
  }
  return parts
    .slice(0, 3)
    .map((part) => part[0].toUpperCase())
    .join("");
}
function buildTeamSnapshots(records: DatasetRecord[]) {
  const map = new Map<SnapshotKey, TeamSnapshot>();
  for (const record of records) {
    const season = String(record.season ?? "");
    const league = String(record.league ?? "EPL");
    const homeKey = snapshotKey(season, record.home);
    const awayKey = snapshotKey(season, record.away);
    const homeSnapshot = map.get(homeKey) ?? {
      league,
      season,
      homeRecord: null,
      awayRecord: null,
    };
    homeSnapshot.league = league;
    homeSnapshot.season = season;
    homeSnapshot.homeRecord = record;
    map.set(homeKey, homeSnapshot);

    const awaySnapshot = map.get(awayKey) ?? {
      league,
      season,
      homeRecord: null,
      awayRecord: null,
    };
    awaySnapshot.league = league;
    awaySnapshot.season = season;
    awaySnapshot.awayRecord = record;
    map.set(awayKey, awaySnapshot);
  }
  return map;
}

function snapshotKey(season: string, team: string) {
  return `${season}|${team.toLowerCase()}`;
}

function findTeamSnapshot(
  version: string,
  season: string,
  team: string,
): TeamSnapshot | null {
  const snapshots = teamSnapshotsByVersion.get(version);
  if (!snapshots) return null;
  const key = snapshotKey(season, team);
  const direct = snapshots.get(key);
  if (direct) {
    return direct;
  }
  const fallbackSeason = latestSeasonByVersion.get(version);
  if (fallbackSeason && fallbackSeason !== season) {
    return snapshots.get(snapshotKey(fallbackSeason, team)) ?? null;
  }
  return null;
}

function buildSyntheticFixture(
  version: string,
  season: string,
  home: string,
  away: string,
) {
  const providedSeason = season && season.toString().trim().length > 0 ? season : null;
  const latestSeason = providedSeason ?? latestSeasonByVersion.get(version) ?? DEFAULT_DATASET_VERSION;
  const homeSnapshot = findTeamSnapshot(version, latestSeason, home);
  const awaySnapshot = findTeamSnapshot(version, latestSeason, away);
  if (!homeSnapshot || !awaySnapshot) {
    return null;
  }
  const league = homeSnapshot.league || awaySnapshot.league || "EPL";
  const homeRecord = homeSnapshot.homeRecord ?? homeSnapshot.awayRecord;
  const awayRecord = awaySnapshot.awayRecord ?? awaySnapshot.homeRecord;
  if (!homeRecord || !awayRecord) {
    return null;
  }
  const values: Record<string, number> = {};
  values.match_id = Number(
    `${Date.now().toString().slice(-6)}${Math.floor(Math.random() * 1000)}`,
  );
  if (typeof homeRecord.values.home_team_id === "number") {
    values.home_team_id = homeRecord.values.home_team_id;
  }
  if (typeof awayRecord.values.away_team_id === "number") {
    values.away_team_id = awayRecord.values.away_team_id;
  }
  values.home_points_actual = 0;
  values.away_points_actual = 0;
  applyPrefixedValues(values, homeRecord.values, "home_");
  applyPrefixedValues(values, awayRecord.values, "away_");
  applyElo(values, homeRecord.values, awayRecord.values);
  applyMarket(values, homeRecord.values, awayRecord.values);
  applyDerivedGaps(values);
  return {
    season: latestSeason,
    league,
    home,
    away,
    values,
  };
}

function applyPrefixedValues(
  target: Record<string, number>,
  source: Record<string, number>,
  prefix: "home_" | "away_",
) {
  for (const [key, value] of Object.entries(source)) {
    if (!key.startsWith(prefix) || typeof value !== "number") continue;
    target[key] = value;
  }
}

function applyElo(
  target: Record<string, number>,
  homeSource: Record<string, number>,
  awaySource: Record<string, number>,
) {
  const homeElo = homeSource.elo_home_pre ?? 1500;
  const awayElo = awaySource.elo_away_pre ?? 1500;
  const homeExp = homeSource.elo_home_expectation ?? 0.5;
  const awayExp = awaySource.elo_away_expectation ?? (1 - homeExp);
  target.elo_home_pre = homeElo;
  target.elo_away_pre = awayElo;
  target.elo_mean_pre = (homeElo + awayElo) / 2;
  target.elo_gap_pre = homeElo - awayElo;
  target.elo_home_expectation = homeExp;
  target.elo_expectation_gap = homeExp - (1 - awayExp);
}

function applyMarket(
  target: Record<string, number>,
  homeSource: Record<string, number>,
  awaySource: Record<string, number>,
) {
  const homeProb = homeSource.forecast_home_win ?? 0.5;
  const awayProb = awaySource.forecast_away_win ?? 0.25;
  const drawProb =
    homeSource.forecast_draw ??
    awaySource.forecast_draw ??
    Math.max(0, 1 - homeProb - awayProb);
  target.forecast_home_win = homeProb;
  target.forecast_away_win = awayProb;
  target.forecast_draw = drawProb;
  target.marketEdge = homeProb - awayProb;
  target.market_entropy = homeSource.market_entropy ?? target.market_entropy ?? 0;
  target.market_home_edge = homeSource.market_home_edge ?? target.marketEdge;
  target.market_logit_home = homeSource.market_logit_home ?? 0;
  target.market_max_prob = Math.max(homeProb, awayProb, drawProb);
}

function applyDerivedGaps(values: Record<string, number>) {
  const derivedPairs: Array<[string, string, string]> = [
    ["goal_diff_std_gap5", "home_goal_diff_std5", "away_goal_diff_std5"],
    ["goal_diff_exp_decay_gap", "home_goal_diff_exp_decay", "away_goal_diff_exp_decay"],
    ["xg_diff_std_gap5", "home_xg_diff_std5", "away_xg_diff_std5"],
    ["xg_diff_exp_decay_gap", "home_xg_diff_exp_decay", "away_xg_diff_exp_decay"],
    ["shot_diff_std_gap5", "home_shot_diff_std5", "away_shot_diff_std5"],
    ["shot_diff_exp_decay_gap", "home_shot_diff_exp_decay", "away_shot_diff_exp_decay"],
    ["att_gap_avg5", "home_goals_for_avg5", "away_goals_for_avg5"],
    ["def_gap_avg5", "away_goals_against_avg5", "home_goals_against_avg5"],
    ["points_gap_avg5", "home_points_avg5", "away_points_avg5"],
    ["xg_att_gap_avg5", "home_xg_for_avg5", "away_xg_for_avg5"],
    ["xg_def_gap_avg5", "away_xg_against_avg5", "home_xg_against_avg5"],
  ];
  for (const [gap, homeKey, awayKey] of derivedPairs) {
    const homeVal = values[homeKey] ?? 0;
    const awayVal = values[awayKey] ?? 0;
    values[gap] = homeVal - awayVal;
  }
  const EPS = 1e-3;
  const shotRatio = Math.log(
    ((values.home_shots_for_avg5 ?? 0) + EPS) / ((values.away_shots_for_avg5 ?? 0) + EPS),
  );
  values.log_shot_ratio_avg5 = Number.isFinite(shotRatio) ? shotRatio : 0;
  const xgRatio = Math.log(
    ((values.home_xg_for_avg5 ?? 0) + EPS) / ((values.away_xg_for_avg5 ?? 0) + EPS),
  );
  values.log_xg_ratio_avg5 = Number.isFinite(xgRatio) ? xgRatio : 0;
  values.shots_tempo_avg5 = ((values.home_shots_for_avg5 ?? 0) + (values.away_shots_for_avg5 ?? 0)) / 2;
}
