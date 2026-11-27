from dataclasses import dataclass
from src.config import CFG
from scipy.optimize import minimize
from scipy.special import gammaln
from _collections_abc import Mapping

import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    attack: Mapping[str, float]
    defense: Mapping[str, float]
    home_advantage: float
    rho: float

    def predict_expected_goals(self, home_team: str, away_team: str):
        lambda_home = float(
            np.exp(self.attack[home_team] + self.defense[away_team] + self.home_advantage)
        )
        mu_away = float(
            np.exp(self.attack[away_team] + self.defense[home_team])
        )
        return lambda_home, mu_away


@dataclass(frozen=True)
class _TrainingData:
    home_idx: np.ndarray
    away_idx: np.ndarray
    home_goals: np.ndarray
    away_goals: np.ndarray
    y_home: np.ndarray
    y_away: np.ndarray
    weights: np.ndarray
    teams: list[str]


def _prepare_training_data(played_df: pd.DataFrame) -> _TrainingData:
    teams: list[str] = sorted(set(played_df["home_team"].astype(str)).union(played_df["away_team"].astype(str)))
    team_to_idx = {team: idx for idx, team in enumerate(teams)}

    home_idx = played_df["home_team"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = played_df["away_team"].map(team_to_idx).to_numpy(dtype=int)

    home_goals = played_df["home_goals"].to_numpy(dtype=int)
    away_goals = played_df["away_goals"].to_numpy(dtype=int)

    home_xg = played_df["home_xg"].to_numpy(dtype=float)
    away_xg = played_df["away_xg"].to_numpy(dtype=float)

    y_home = (1 - CFG.XG_WEIGHT) * home_goals + CFG.XG_WEIGHT * home_xg
    y_away = (1 - CFG.XG_WEIGHT) * away_goals + CFG.XG_WEIGHT * away_xg

    dates = pd.Series(pd.to_datetime(played_df["date"], utc=True))
    time_deltas = dates.max() - dates
    days_since_match = time_deltas.dt.total_seconds().to_numpy(dtype=float) / 86400.0
    weights = np.asarray(np.exp(-CFG.TIME_DECAY_XI * days_since_match), dtype=float)

    return _TrainingData(
        home_idx=home_idx,
        away_idx=away_idx,
        home_goals=home_goals,
        away_goals=away_goals,
        y_home=y_home,
        y_away=y_away,
        weights=weights,
        teams=teams,
    )


def _dicon_coles_tau(home_goals: np.ndarray, away_goals: np.ndarray, lambda_home: np.ndarray,
                     mu_away: np.ndarray, rho: float) -> np.ndarray:
    conditions = [
        (home_goals == 0) & (away_goals == 0),
        (home_goals == 0) & (away_goals == 1),
        (home_goals == 1) & (away_goals == 0),
        (home_goals == 1) & (away_goals == 1),
    ]
    choices = [
        1.0 - lambda_home * mu_away * rho,
        1.0 + lambda_home * rho,
        1.0 + mu_away * rho,
        1.0 - rho,
    ]

    return np.asarray(np.select(conditions, choices, default=1.0), dtype=float)


def _negative_log_likelihood(params: np.ndarray, data: _TrainingData) -> float:
    n_teams = len(data.teams)
    attack = params[:n_teams]
    defense = params[n_teams:2 * n_teams]
    home_advantage = float(params[2 * n_teams])
    rho = float(params[2 * n_teams + 1])

    lambda_home = np.exp(attack[data.home_idx] + defense[data.away_idx] + home_advantage)
    mu_away = np.exp(attack[data.away_idx] + defense[data.home_idx])

    tau = _dicon_coles_tau(
        data.home_goals,
        data.away_goals,
        lambda_home,
        mu_away,
        rho,
    )

    if np.any(tau <= 0.0):
        return 1e10

    log_lik_home = data.y_home * np.log(lambda_home) - lambda_home - gammaln(data.y_home + 1.0)
    log_lik_away = data.y_away * np.log(mu_away) - mu_away - gammaln(data.y_away + 1.0)
    total_log_lik = np.sum(data.weights * (log_lik_home + log_lik_away + np.log(tau)))

    return float(-total_log_lik)


def fit_dixon_coles_model(played_df: pd.DataFrame) -> ModelParameters:
    data = _prepare_training_data(played_df)
    n_teams = len(data.teams)

    x0 = np.zeros(2 * n_teams + 2, dtype=float)
    x0[2 * n_teams] = CFG.HOME_ADVANTAGE_INIT
    x0[2 * n_teams + 1] = CFG.RHO_INIT

    bounds = (
            [(-3.0, 3.0)] * (2 * n_teams)
            + [(-1.0, 1.5), (-0.35, 0.35)]
    )

    constraints = {"type": "eq", "fun": lambda p: float(np.mean(p[:n_teams]))}

    result = minimize(
        fun=_negative_log_likelihood,
        x0=x0,
        args=(data,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Dixon-Coles optimization failed: {result.message}")

    attack_params = {
        team: float(result.x[idx]) for idx, team in enumerate(data.teams)
    }
    defense_params = {
        team: float(result.x[n_teams + idx]) for idx, team in enumerate(data.teams)
    }

    return ModelParameters(
        attack=attack_params,
        defense=defense_params,
        home_advantage=float(result.x[2 * n_teams]),
        rho=float(result.x[2 * n_teams + 1]),
    )


if __name__ == "__main__":
    played = pd.read_csv(CFG.PLAYED_MATCHES_FILE)
    model = fit_dixon_coles_model(played)

    print(f"Home advantage (gamma): {model.home_advantage:.4f}")
    print(f"Low-score correlation (rho)_: {model.rho:.4f}")

    ranking = pd.DataFrame(
        [
            {
                "team": team,
                "attack": model.attack[team],
                "defence": model.defense[team],
                "net_strength": model.attack[team] - model.defense[team],
            }
            for team in model.attack
        ]
    ).sort_values("net_strength", ascending=False).reset_index(drop=True)
    ranking.index += 1
    print(ranking.round(3))
