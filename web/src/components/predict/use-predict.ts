"use client";

import { useState, useCallback } from "react";

export type PredictionResponse = {
  fixture: {
    season: string;
    home: { name: string; shortName: string };
    away: { name: string; shortName: string };
  };
  models: Array<{
    id: string;
    format: string | null;
    location: { kind: "local"; path: string } | { kind: "remote"; uri: string } | null;
    view: string | null;
    probs: { home: number; draw: number; away: number };
    logits: { home: number; draw: number; away: number };
    note: string;
  }>;
  ensemble: {
    method: string;
    probs: { home: number; draw: number; away: number };
  };
};

type PredictState =
  | { status: "idle"; data: null; error: null }
  | { status: "loading"; data: PredictionResponse | null; error: null }
  | { status: "success"; data: PredictionResponse; error: null }
  | { status: "error"; data: PredictionResponse | null; error: string };

export function usePredict() {
  const [state, setState] = useState<PredictState>({
    status: "idle",
    data: null,
    error: null,
  });
  const [lastRequest, setLastRequest] = useState<{ home: string; away: string } | null>(
    null,
  );

  const predict = useCallback(async (home: string, away: string) => {
    if (!home || !away) return;
    setLastRequest({ home, away });
    setState((prev) => ({
      status: "loading",
      data: prev.data,
      error: null,
    }));
    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ homeTeam: home, awayTeam: away }),
      });
      const payload = await response.json();
      if (!response.ok || !payload.ok) {
        throw new Error(payload.error ?? "Prediction failed.");
      }
      setState({ status: "success", data: payload, error: null });
    } catch (error) {
      setState({
        status: "error",
        data: null,
        error: (error as Error).message,
      });
    }
  }, []);

  return {
    ...state,
    lastRequest,
    predict,
  };
}
