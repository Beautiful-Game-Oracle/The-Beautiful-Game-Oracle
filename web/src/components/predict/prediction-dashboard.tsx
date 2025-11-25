"use client";

import { useEffect, useState } from "react";

import { useLoaderStatus } from "@/components/predict/use-loader-status";
import { usePredict } from "@/components/predict/use-predict";
import { useTeamOptions } from "@/components/predict/use-team-options";
import { Card, CardBody, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/cn";
import { ModelCard } from "./model-card";
import { TeamSelect } from "./team-select";

// PredictionDashboard wires together the team selector, prediction hook, and manifest status panel.

type Props = {
  onSelectionChange?: (home: string, away: string) => void;
};

export function PredictionDashboard({ onSelectionChange }: Props) {
  const { teams, loading: teamsLoading, season, error: teamError } = useTeamOptions("EPL");
  const [home, setHome] = useState("");
  const [away, setAway] = useState("");
  const loaderStatus = useLoaderStatus();
  const prediction = usePredict();
  const predict = prediction.predict;

  useEffect(() => {
    if (teams.length >= 2 && !home && !away) {
      const [first, second] = teams;
      setHome(first.canonical);
      setAway(second.canonical);
      predict(first.canonical, second.canonical);
    }
  }, [teams, home, away, predict]);

  useEffect(() => {
    if (onSelectionChange) {
      onSelectionChange(home, away);
    }
  }, [home, away, onSelectionChange]);

  const isSameTeam = home === away;

  return (
    <section className="space-y-8">
      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <Card className="bg-panel/80">
          <CardTitle className="text-xl">Fixture Selector</CardTitle>
          <CardBody className="mt-2 text-sm text-muted">
            Choose two Premier League clubs and run the prediction suite above the
          latest manifest-exported models. Target season:{" "}
          <span className="text-foreground font-semibold">
            {season ?? "loading…"}
          </span>
        </CardBody>
        <form
          className="mt-6 space-y-5"
          onSubmit={(event) => {
            event.preventDefault();
            if (isSameTeam) return;
            predict(home, away);
          }}
        >
          <TeamSelect
            label="Home team"
            value={home}
            onChange={setHome}
            disabled={teamsLoading || teams.length === 0}
            teams={teams}
          />
          <TeamSelect
            label="Away team"
            value={away}
            onChange={setAway}
            disabled={teamsLoading || teams.length === 0}
            teams={teams}
          />
          {teamError && (
            <p className="text-sm text-danger">
              Unable to load club list: {teamError}
            </p>
          )}
          {isSameTeam && (
            <p className="text-sm text-danger">
              Home and away teams must be different.
            </p>
          )}
          <button
            type="submit"
            disabled={
              prediction.status === "loading" ||
              isSameTeam ||
              !home ||
              !away ||
              teams.length < 2
            }
            className={cn(
              "w-full rounded-2xl bg-brand px-4 py-3 text-center text-base font-semibold text-black transition hover:bg-brand/80 disabled:opacity-70",
            )}
          >
            {prediction.status === "loading" ? "Predicting…" : "Run Prediction"}
          </button>
        </form>
          <div className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-4 text-sm">
            <p className="font-semibold text-foreground">Active Manifest</p>
            {loaderStatus.status === "ready" && loaderStatus.data ? (
              <ul className="mt-2 space-y-1 text-muted">
                <li>
                  Run ID:{" "}
                  <span className="text-foreground">
                    {loaderStatus.data.run_id ?? "n/a"}
                  </span>
                </li>
                <li>
                  Dataset version:{" "}
                  <span className="text-foreground">
                    {loaderStatus.data.dataset_version ?? "n/a"}
                  </span>
                </li>
                <li>
                  Source:{" "}
                  <span className="text-foreground">
                    {loaderStatus.data.manifest_source?.kind}
                  </span>
                </li>
              </ul>
            ) : loaderStatus.status === "error" ? (
              <p className="text-danger">{loaderStatus.error}</p>
            ) : (
              <p className="text-muted">Loading manifest status…</p>
            )}
          </div>
        </Card>
        <div className="flex flex-col gap-4">
          {prediction.data && (
            <div className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <p className="text-xs uppercase tracking-[0.3em] text-muted">
                {prediction.data.fixture.season}
              </p>
              <h2 className="mt-2 text-2xl font-semibold text-foreground">
                {prediction.data.fixture.home.name} vs {prediction.data.fixture.away.name}
              </h2>
              <p className="text-sm text-muted">
                Ensemble method: {prediction.data.ensemble.method}
              </p>
            </div>
          )}
          {prediction.data ? (
            <ModelCard
              model={{
                id: "Ensemble",
                format: prediction.data.ensemble.method,
                location: null,
                view: null,
                probs: prediction.data.ensemble.probs,
                logits: prediction.data.ensemble.probs,
                note: "Log probability average across loaded models.",
              }}
              isEnsemble
            />
          ) : (
            <Card className="h-full min-h-[220px] bg-panel/60">
              <CardTitle>Ensemble</CardTitle>
              <CardBody className="mt-3 text-sm text-muted">
                Run a prediction to view ensemble probabilities alongside the
                individual model cards.
              </CardBody>
            </Card>
          )}
        </div>
      </div>

      <div className="space-y-5">
        {prediction.status === "error" && (
          <Card className="border-danger/20 bg-danger/10">
            <CardTitle className="text-danger">Prediction failed</CardTitle>
            <CardBody className="mt-2 text-danger/80">
              {prediction.error ?? "Unknown error"}
            </CardBody>
          </Card>
        )}
        {prediction.status === "idle" && (
          <Card>
            <CardTitle>Awaiting prediction</CardTitle>
            <CardBody className="mt-2 text-muted">
              Choose teams and tap <strong>Run Prediction</strong> to see model
              probabilities.
            </CardBody>
          </Card>
        )}
        {prediction.data && (
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {prediction.data.models.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
              />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}
