import fs from "node:fs";

import { parse } from "csv-parse/sync";

import { getFinancialDatasetPath } from "@/config/env";
import { buildFixtureKey } from "@/server/predict/fixture-key";

export const FINANCIAL_FEATURE_COLUMNS = [
  "squad_value_ratio",
  "squad_value_diff",
  "avg_player_value_ratio",
  "avg_player_value_diff",
  "wage_bill_ratio",
  "wage_bill_diff",
  "avg_salary_ratio",
  "avg_salary_diff",
] as const;

export type FinancialFeatureColumn = (typeof FINANCIAL_FEATURE_COLUMNS)[number];
export type FinancialFeatureSnapshot = Partial<Record<FinancialFeatureColumn, number>>;

type FinancialMap = Map<string, FinancialFeatureSnapshot>;

type TeamFinancialProfile = {
  season: string;
  updatedAt: number;
  squadValue: number;
  avgPlayerValue: number;
  wageBill: number;
  avgSalary: number;
};

type FinancialCache = {
  fixtures: FinancialMap;
  profiles: Map<string, TeamFinancialProfile>;
};

const cache = new Map<string, FinancialCache>();

export function getFinancialFeatureMap(version: string): FinancialMap {
  return ensureCache(version).fixtures;
}

export function buildApproximateFinancialSnapshot(
  version: string,
  season: string,
  home: string,
  away: string,
): FinancialFeatureSnapshot | undefined {
  const { profiles } = ensureCache(version);
  const normalizedHome = home.toLowerCase();
  const normalizedAway = away.toLowerCase();
  const homeProfile = findTeamProfile(profiles, normalizedHome, season);
  const awayProfile = findTeamProfile(profiles, normalizedAway, season);
  if (!homeProfile || !awayProfile) {
    return undefined;
  }
  return snapshotFromProfiles(homeProfile, awayProfile);
}

function ensureCache(version: string): FinancialCache {
  const cacheKey = version || "default";
  const existing = cache.get(cacheKey);
  if (existing) {
    return existing;
  }
  const datasetPath = getFinancialDatasetPath(version);
  if (!datasetPath || !fs.existsSync(datasetPath)) {
    throw new Error(
      `[financial-data] No financial dataset found for version ${version}. Set FINANCIAL_DATASET_PATH or regenerate financial_dataset.csv.`,
    );
  }
  const csv = fs.readFileSync(datasetPath, "utf-8");
  const records = parse(csv, {
    columns: true,
    skip_empty_lines: true,
  }) as Array<Record<string, string>>;
  const fixtures: FinancialMap = new Map();
  const profiles = new Map<string, TeamFinancialProfile>();
  for (const row of records) {
    const season = (row.season ?? "").toString().trim();
    const home = (row.home_team ?? "").toString().trim();
    const away = (row.away_team ?? "").toString().trim();
    if (season && home && away) {
      const snapshot: FinancialFeatureSnapshot = {};
      for (const column of FINANCIAL_FEATURE_COLUMNS) {
        const value = parseNumber(row[column]);
        if (value !== null) {
          snapshot[column] = value;
        }
      }
      if (Object.keys(snapshot).length > 0) {
        fixtures.set(buildFixtureKey(season, home, away), snapshot);
      }
      registerProfile(profiles, season, home, row, "home");
      registerProfile(profiles, season, away, row, "away");
    }
  }
  if (fixtures.size === 0) {
    throw new Error(
      `[financial-data] Financial dataset ${datasetPath} contained no keyed fixtures.`,
    );
  }
  const loaded: FinancialCache = { fixtures, profiles };
  cache.set(cacheKey, loaded);
  return loaded;
}

function registerProfile(
  profiles: Map<string, TeamFinancialProfile>,
  season: string,
  team: string,
  row: Record<string, string>,
  prefix: "home" | "away",
) {
  const normalizedTeam = team.toLowerCase();
  const profile: TeamFinancialProfile = {
    season,
    updatedAt: parseDate(row.date ?? row.match_date ?? row.match_datetime_utc ?? null),
    squadValue: parseNumber(row[`${prefix}_squad_value_eur`]) ?? 0,
    avgPlayerValue: parseNumber(row[`${prefix}_avg_player_value_eur`]) ?? 0,
    wageBill: parseNumber(row[`${prefix}_wage_bill_eur`]) ?? 0,
    avgSalary: parseNumber(row[`${prefix}_avg_salary_eur`]) ?? 0,
  };
  const seasonKey = profileKey(season, normalizedTeam);
  const currentSeasonProfile = profiles.get(seasonKey);
  if (!currentSeasonProfile || profile.updatedAt >= currentSeasonProfile.updatedAt) {
    profiles.set(seasonKey, profile);
  }
  const latestKey = profileKey("latest", normalizedTeam);
  const currentLatest = profiles.get(latestKey);
  if (!currentLatest || profile.updatedAt >= currentLatest.updatedAt) {
    profiles.set(latestKey, profile);
  }
}

function findTeamProfile(
  profiles: Map<string, TeamFinancialProfile>,
  team: string,
  season?: string,
): TeamFinancialProfile | undefined {
  if (season) {
    const seasonProfile = profiles.get(profileKey(season, team));
    if (seasonProfile) {
      return seasonProfile;
    }
  }
  return profiles.get(profileKey("latest", team));
}

function profileKey(season: string, team: string) {
  return `${season}|${team}`;
}

function snapshotFromProfiles(
  home: TeamFinancialProfile,
  away: TeamFinancialProfile,
): FinancialFeatureSnapshot {
  return {
    squad_value_ratio: safeRatio(home.squadValue, away.squadValue),
    squad_value_diff: home.squadValue - away.squadValue,
    avg_player_value_ratio: safeRatio(home.avgPlayerValue, away.avgPlayerValue),
    avg_player_value_diff: home.avgPlayerValue - away.avgPlayerValue,
    wage_bill_ratio: safeRatio(home.wageBill, away.wageBill),
    wage_bill_diff: home.wageBill - away.wageBill,
    avg_salary_ratio: safeRatio(home.avgSalary, away.avgSalary),
    avg_salary_diff: home.avgSalary - away.avgSalary,
  };
}

function safeRatio(numerator: number, denominator: number) {
  const EPS = 1e-6;
  if (Math.abs(denominator) < EPS) {
    return numerator >= 0 ? 1 : -1;
  }
  return numerator / denominator;
}

function parseNumber(value: unknown) {
  if (value === null || value === undefined) {
    return null;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : null;
  }
  const text = String(value).trim();
  if (!text) {
    return null;
  }
  const parsed = Number(text);
  return Number.isFinite(parsed) ? parsed : null;
}

function parseDate(value: string | null | undefined) {
  if (!value) return 0;
  const parsed = Date.parse(String(value));
  return Number.isNaN(parsed) ? 0 : parsed;
}
