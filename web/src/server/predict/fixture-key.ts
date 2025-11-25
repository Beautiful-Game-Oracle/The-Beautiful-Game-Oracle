// Shared helper for generating canonical fixture keys.

export function buildFixtureKey(season: string, home: string, away: string) {
  const normalizedSeason = String(season ?? "").toLowerCase();
  const normalizedHome = String(home ?? "").toLowerCase();
  const normalizedAway = String(away ?? "").toLowerCase();
  return `${normalizedSeason}|${normalizedHome}|${normalizedAway}`;
}
