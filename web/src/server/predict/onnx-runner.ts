import path from "node:path";

import * as ort from "onnxruntime-node";

import type { FixtureFeatureVector } from "@/server/predict/types";
import type { ResourceHandle, ResourceLocation } from "@/server/models/loader";
import { loadPreprocessing, matchPreprocessing } from "@/server/predict/preprocessing";

const sessionCache = new Map<string, ort.InferenceSession>();

export async function runOnnxModel(
  modelHandle: ResourceHandle,
  featureVector: FixtureFeatureVector,
  preprocessingHandles: ResourceHandle[],
) {
  if (!modelHandle.location) {
    throw new Error(`Model ${modelHandle.id} missing location for ONNX inference`);
  }
  const session = await getSession(modelHandle.location);
  const preprocessingHandle = matchPreprocessing(modelHandle, preprocessingHandles);
  const preprocessing = await loadPreprocessing(preprocessingHandle);
  if (!preprocessing) {
    throw new Error(`Preprocessing bundle not found for model ${modelHandle.id}`);
  }
  const features = buildFeatureArray(featureVector, preprocessing.feature_names);
  const inputTensor = new ort.Tensor("float32", features, [1, features.length]);
  const outputs = await session.run({ input: inputTensor });
  const probabilityTensor = selectProbabilityTensor(outputs, modelHandle.id);
  const dataArray = Array.from(probabilityTensor.data as Iterable<number | bigint>, (value) =>
    typeof value === "bigint" ? Number(value) : value,
  );
  return normalizeProbabilities(dataArray);
}

async function getSession(location: ResourceLocation) {
  const key = location.kind === "local" ? location.path : location.uri;
  let session = sessionCache.get(key);
  if (!session) {
    if (location.kind === "local") {
      const abs = path.resolve(location.path);
      session = await ort.InferenceSession.create(abs);
      sessionCache.set(abs, session);
    } else {
      console.log(`[onnx-runner] Fetching remote model: ${location.uri}`);
      const response = await fetch(location.uri);
      if (!response.ok) {
        throw new Error(`Failed to fetch remote model ${location.uri}: ${response.statusText}`);
      }
      const arrayBuffer = await response.arrayBuffer();
      const buffer = new Uint8Array(arrayBuffer);
      session = await ort.InferenceSession.create(buffer);
      sessionCache.set(location.uri, session);
    }
  }
  return session;
}

function buildFeatureArray(
  vector: FixtureFeatureVector,
  featureNames: string[],
): Float32Array {
  const values = featureNames.map((name) => {
    const value = (vector as Record<string, number>)[name];
    if (value === undefined) {
      console.warn(`[onnx-runner] Missing feature '${name}', defaulting to 0.`);
      return 0;
    }
    return value;
  });
  return Float32Array.from(values);
}

function normalizeProbabilities(data: number[]) {
  const sum = data.reduce((acc, value) => acc + value, 0) || 1;
  return {
    home: data[0] / sum,
    draw: data[1] / sum,
    away: data[2] / sum,
  };
}

function selectProbabilityTensor(
  outputs: Record<string, ort.Tensor | ort.Tensor[]>,
  modelId: string,
) {
  const tensors = Object.entries(outputs)
    .map(([name, value]) => ({ name, value }))
    .filter((entry): entry is { name: string; value: ort.Tensor } => entry.value instanceof ort.Tensor);

  const preferred = tensors.find(
    (entry) =>
      entry.name.toLowerCase().includes("prob") &&
      typeof entry.value.data[0] !== "undefined",
  );
  if (preferred) {
    return preferred.value;
  }
  if (tensors.length === 0) {
    throw new Error(`ONNX model ${modelId} produced no tensor outputs`);
  }
  return tensors[0].value;
}
