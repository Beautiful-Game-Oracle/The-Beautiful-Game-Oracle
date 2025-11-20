export type ProbabilityTriplet = {
  home: number;
  draw: number;
  away: number;
};

export type FixtureTeam = {
  canonical: string;
  name: string;
  shortName: string;
  crest: string;
  league: string;
};

export type FinancialFeatures = {
  wageGap: number;
  netSpendGap: number;
  valuationGap: number;
  squad_value_ratio?: number;
  squad_value_diff?: number;
  avg_player_value_ratio?: number;
  avg_player_value_diff?: number;
  wage_bill_ratio?: number;
  wage_bill_diff?: number;
  avg_salary_ratio?: number;
  avg_salary_diff?: number;
};

export type MarketFeatures = {
  marketEdge: number;
  volatility: number;
};

export type PerformanceFeatures = {
  attGap: number;
  defGap: number;
  xgGap: number;
  xgDefGap: number;
  pointsGap: number;
};

export type FixtureFeatureVector = PerformanceFeatures &
  FinancialFeatures &
  MarketFeatures & {
    homeForm: number;
    awayForm: number;
    [feature: string]: number;
  };

export type FixtureContext = {
  season: string;
  home: FixtureTeam;
  away: FixtureTeam;
};
