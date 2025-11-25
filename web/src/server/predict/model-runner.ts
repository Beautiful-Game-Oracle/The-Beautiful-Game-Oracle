import { ensureManifestLoaded, getLoaderState, ResourceHandle } from "@/server/models/loader";
import { runOnnxModel } from "@/server/predict/onnx-runner";
import {
  FixtureFeatureVector,
  ProbabilityTriplet,
} from "@/server/predict/types";

const EPS = 1e-6;

export type ModelPrediction = {
  id: string;
  format: string | null;
  location: ReturnType<typeof serializeLocation>;
  view: string | null;
  logits: ProbabilityTriplet;
  probs: ProbabilityTriplet;
  note: string;
};

export type EnsemblePrediction = {
  method: "log_prob_avg";
  probs: ProbabilityTriplet;
};

export async function runModelPredictions(features: FixtureFeatureVector) {
  await ensureManifestLoaded();
  const state = getLoaderState();
  const predictions = await Promise.all(
    state.models.map((handle) =>
      producePrediction(handle, features, state.preprocessing),
    ),
  );
  const ensemble = computeEnsemble(predictions);
  return { predictions, ensemble };
}

async function producePrediction(
  handle: ResourceHandle,
  vector: FixtureFeatureVector,
  preprocessingHandles: ResourceHandle[],
): Promise<ModelPrediction> {
  const view = inferView(handle);
  if (handle.entry.format === "onnx") {
    try {
      const probs = await runOnnxModel(handle, vector, preprocessingHandles);
      const logits = logProbs(probs);
      return {
        id: handle.id,
        format: "onnx",
        location: serializeLocation(handle),
        view,
        logits,
        probs,
        note: "Inference via onnxruntime-node",
      };
    } catch (error) {
      console.error(`ONNX inference failed for ${handle.id}`, error);
    }
  }

  const score = pseudoScore(handle.id, vector);
  const volatility = vector.volatility;
  const drawScore = -Math.abs(score) + 0.15 * volatility;

  const logits: ProbabilityTriplet = {
    home: score,
    draw: drawScore,
    away: -score,
  };
  const probs = softmax(logits);
  return {
    id: handle.id,
    format: handle.entry.format ?? null,
    location: serializeLocation(handle),
    view,
    logits,
    probs,
    note: "Heuristic fallback (model export not available).",
  };
}

function logProbs(probs: ProbabilityTriplet): ProbabilityTriplet {
  return {
    home: Math.log(probs.home + EPS),
    draw: Math.log(probs.draw + EPS),
    away: Math.log(probs.away + EPS),
  };
}

function pseudoScore(modelId: string, vector: FixtureFeatureVector) {
  const weights = {
    attGap: weightFor(modelId, "attGap"),
    defGap: weightFor(modelId, "defGap"),
    xgGap: weightFor(modelId, "xgGap"),
    pointsGap: weightFor(modelId, "pointsGap"),
    wageGap: weightFor(modelId, "wageGap"),
    netSpendGap: weightFor(modelId, "netSpendGap"),
    valuationGap: weightFor(modelId, "valuationGap"),
    marketEdge: weightFor(modelId, "marketEdge"),
  };

  const signal =
    vector.attGap * weights.attGap +
    vector.defGap * weights.defGap +
    vector.xgGap * weights.xgGap +
    vector.pointsGap * weights.pointsGap +
    vector.wageGap * weights.wageGap +
    vector.netSpendGap * weights.netSpendGap +
    vector.valuationGap * weights.valuationGap +
    vector.marketEdge * weights.marketEdge;

  return Math.max(Math.min(signal, 6), -6);
}

function weightFor(modelId: string, feature: string) {
  const input = `${modelId}:${feature}`;
  let hash = 0;
  for (let i = 0; i < input.length; i += 1) {
    hash = (hash * 31 + input.charCodeAt(i)) % 1000;
  }
  const normalized = (hash / 1000) * 2 - 1; // [-1, 1)
  return round(normalized * 0.8);
}

function softmax(logits: ProbabilityTriplet): ProbabilityTriplet {
  const values = [logits.home, logits.draw, logits.away];
  const max = Math.max(...values);
  const expVals = values.map((v) => Math.exp(v - max));
  const sum = expVals.reduce((acc, val) => acc + val, 0) + EPS;
  return {
    home: expVals[0] / sum,
    draw: expVals[1] / sum,
    away: expVals[2] / sum,
  };
}

function computeEnsemble(predictions: ModelPrediction[]): EnsemblePrediction {
  if (predictions.length === 0) {
    return { method: "log_prob_avg", probs: { home: 1 / 3, draw: 1 / 3, away: 1 / 3 } };
  }

  const logs = predictions.map((p) => ({
    home: Math.log(p.probs.home + EPS),
    draw: Math.log(p.probs.draw + EPS),
    away: Math.log(p.probs.away + EPS),
  }));

  const avgLogs = logs.reduce(
    (acc, curr) => ({
      home: acc.home + curr.home / predictions.length,
      draw: acc.draw + curr.draw / predictions.length,
      away: acc.away + curr.away / predictions.length,
    }),
    { home: 0, draw: 0, away: 0 },
  );

  const expHome = Math.exp(avgLogs.home);
  const expDraw = Math.exp(avgLogs.draw);
  const expAway = Math.exp(avgLogs.away);
  const denom = expHome + expDraw + expAway + EPS;

  return {
    method: "log_prob_avg",
    probs: {
      home: expHome / denom,
      draw: expDraw / denom,
      away: expAway / denom,
    },
  };
}

function serializeLocation(handle: ResourceHandle) {
  if (!handle.location) {
    return null;
  }
  if (handle.location.kind === "local") {
    return { kind: "local", path: handle.location.path };
  }
  return { kind: "remote", uri: handle.location.uri };
}

function round(value: number) {
  return Math.round(value * 1000) / 1000;
}

function inferView(handle: ResourceHandle) {
  if (handle.entry.view) {
    return handle.entry.view;
  }
  const id = handle.id.toLowerCase();
  if (id.includes("financial")) return "financial";
  if (id.includes("market")) return "market";
  if (id.includes("performance")) return "performance";
  return null;
}
