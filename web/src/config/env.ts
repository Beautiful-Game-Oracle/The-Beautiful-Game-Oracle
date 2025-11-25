import fs from "node:fs";
import path from "node:path";

const DEFAULT_TARGET_SEASON = "2025";

export function getManifestSource(): string {
  const value = process.env.MODEL_MANIFEST_SOURCE;
  if (value && value.trim().length > 0) {
    return value;
  }
  const fallback = path.resolve(process.cwd(), "./public/fixtures/mock_manifest.json");
  if (fs.existsSync(fallback)) {
    return fallback;
  }
  throw new Error(
    "MODEL_MANIFEST_SOURCE is not set and no fallback manifest was found. Provide a local file path or remote URL pointing to model_manifest.json.",
  );
}

export function getLocalArtefactRoot(): string {
  const root = process.env.LOCAL_ARTEFACT_ROOT;
  if (root && root.trim().length > 0) {
    return path.resolve(root);
  }
  return path.resolve(process.cwd(), "..");
}

export function getDatasetRoot(): string {
  const root = process.env.FEATURE_DATASET_ROOT;
  if (root && root.trim().length > 0) {
    return path.resolve(root);
  }
  return path.resolve(process.cwd(), "./public/fixtures");
}

export function getFinancialDatasetPath(version?: string | number | null): string | null {
  const override = process.env.FINANCIAL_DATASET_PATH;
  if (override && override.trim().length > 0) {
    return path.resolve(override.trim());
  }
  const processedRoot = path.resolve(process.cwd(), "../data/processed");
  const candidates: string[] = [];
  if (version) {
    const normalized = String(version).replace(/[^0-9a-z]/gi, "");
    if (normalized) {
      candidates.push(`financial_dataset_v${normalized}.csv`);
      candidates.push(`financial_dataset_${normalized}.csv`);
    }
  }
  candidates.push("financial_dataset.csv");
  for (const candidate of candidates) {
    const target = path.resolve(processedRoot, candidate);
    if (fs.existsSync(target)) {
      return target;
    }
  }
  return null;
}

export function getTeamCacheDir(): string {
  const dir = process.env.FEATURE_TEAM_CACHE_DIR;
  if (dir && dir.trim().length > 0) {
    return path.resolve(dir);
  }
  return path.resolve(getDatasetRoot(), "team_cache");
}

export function getReloadToken(): string | null {
  return process.env.RELOAD_TOKEN ?? null;
}

export function getFeatureDatasetVersion(): string | null {
  const value = process.env.FEATURE_DATASET_VERSION;
  if (value && value.trim().length > 0) {
    return value.trim();
  }
  return null;
}

export function getPredictionTargetSeason(): string {
  const value = process.env.PREDICTION_TARGET_SEASON;
  if (value && value.trim().length > 0) {
    return value.trim();
  }
  return DEFAULT_TARGET_SEASON;
}

export function getEnvSummary() {
  return {
    manifestSource: process.env.MODEL_MANIFEST_SOURCE ?? null,
    localArtefactRoot: getLocalArtefactRoot(),
    reloadTokenConfigured: Boolean(process.env.RELOAD_TOKEN),
    featureDatasetVersion: getFeatureDatasetVersion(),
    predictionTargetSeason: getPredictionTargetSeason(),
    financialDatasetPath: getFinancialDatasetPath(getFeatureDatasetVersion()) ?? null,
  };
}
