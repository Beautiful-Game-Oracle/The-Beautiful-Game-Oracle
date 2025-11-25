"use client";

import type { PredictionResponse } from "@/components/predict/use-predict";
import { Card, CardBody, CardTitle } from "@/components/ui/card";
import { ProbabilityBar } from "./probability-bar";

type Props = {
  model: PredictionResponse["models"][number];
  isEnsemble?: boolean;
};

export function ModelCard({ model, isEnsemble }: Props) {
  const locationLabel = model.location
    ? model.location.kind === "local"
      ? `Local (${trimPath(model.location.path)})`
      : "Kaggle / Remote"
    : "Unknown source";
  const viewLabel = !isEnsemble && model.view ? formatViewLabel(model.view) : null;

  return (
    <Card className="bg-panel/70">
      <CardTitle className="flex items-start justify-between text-base">
        <div>
          <span className="font-semibold">
            {isEnsemble ? "Ensemble" : model.id}
          </span>
          {viewLabel && (
            <span className="ml-0.5 mt-1 block text-[0.65rem] uppercase tracking-[0.2em] text-muted">
              {viewLabel}
            </span>
          )}
        </div>
        {!isEnsemble && (
          <span className="rounded-full bg-white/5 px-2 py-1 text-xs font-medium uppercase tracking-wider text-muted">
            {model.format ?? "n/a"}
          </span>
        )}
      </CardTitle>
      <CardBody className="mt-2 text-xs text-muted">{locationLabel}</CardBody>
      <div className="mt-5 space-y-3">
        <ProbabilityBar
          label="Home"
          value={model.probs.home}
          highlight={model.probs.home >= model.probs.away && model.probs.home >= model.probs.draw}
        />
        <ProbabilityBar
          label="Draw"
          value={model.probs.draw}
          highlight={model.probs.draw >= model.probs.home && model.probs.draw >= model.probs.away}
        />
        <ProbabilityBar
          label="Away"
          value={model.probs.away}
          highlight={model.probs.away >= model.probs.home && model.probs.away >= model.probs.draw}
        />
      </div>
      <p className="mt-4 text-xs text-muted/80">{model.note}</p>
    </Card>
  );
}

function trimPath(path: string) {
  const parts = path.split("/");
  if (parts.length <= 2) return path;
  return `…/${parts.slice(-2).join("/")}`;
}

function formatViewLabel(view: string) {
  return view
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((segment) => segment.charAt(0).toUpperCase() + segment.slice(1))
    .join(" ");
}
