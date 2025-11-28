import { promises as fs } from "node:fs";

import type { ResourceHandle } from "@/server/models/loader";

export type PreprocessingBundle = {
  feature_names: string[];
  scaling?: {
    mean: number[];
    std: number[];
  };
};

const cache = new Map<string, PreprocessingBundle>();

export async function loadPreprocessing(handle: ResourceHandle | undefined) {
  if (!handle || !handle.location) {
    return null;
  }
  const key = handle.location.kind === "local" ? handle.location.path : handle.location.uri;
  const cached = cache.get(key);
  if (cached) return cached;

  let bundle: PreprocessingBundle;
  if (handle.location.kind === "local") {
    const data = await fs.readFile(handle.location.path, "utf-8");
    bundle = JSON.parse(data) as PreprocessingBundle;
  } else {
    console.log(`[preprocessing] Fetching remote bundle: ${handle.location.uri}`);
    const response = await fetch(handle.location.uri);
    if (!response.ok) {
      throw new Error(`Failed to fetch remote bundle ${handle.location.uri}: ${response.statusText}`);
    }
    bundle = (await response.json()) as PreprocessingBundle;
  }

  cache.set(key, bundle);
  return bundle;
}

export function matchPreprocessing(
  model: ResourceHandle,
  preprocessingHandles: ResourceHandle[],
): ResourceHandle | undefined {
  const matches = preprocessingHandles.find((handle) => {
    if (handle.entry.id === model.id) return true;
    return handle.entry.id.startsWith(`${model.id}_`);
  });
  return matches;
}
