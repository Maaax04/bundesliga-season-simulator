from dataclasses import dataclass
from scipy.stats import poisson

import numpy as np
import pandas as pd

from src.config import CFG
from src.model import ModelParameters


@dataclass(frozen=True)
class _CurrentStandings:
    teams: list[str]
    team_to_idx: dict[str, int]
    points: np.ndarray
    goals_for: np.ndarray
    goals_against: np.ndarray
    goal_diff: np.ndarray


def _compute_current_standings(played_df: pd.DataFrame, teams: list[str]) -> _CurrentStandings:
    team_to_idx = {team: i for i, team in enumerate(teams)}
    n_teams = len(teams)

    points = np.zeros(n_teams, dtype=int)
    goals_for = np.zeros(n_teams, dtype=int)
    goals_against = np.zeros(n_teams, dtype=int)

    home_idx = played_df["home_team"].map(team_to_idx).to_numpy(dtype=int)
    away_idx = played_df["away_team"].map(team_to_idx).to_numpy(dtype=int)
    home_goals = played_df["home_goals"].to_numpy(dtype=int)
    away_goals = played_df["away_goals"].to_numpy(dtype=int)

    home_points = np.where(
        home_goals > away_goals,
        3,
        np.where(home_goals == away_goals, 1, 0),
    )
    away_points = np.where(
        away_goals > home_goals,
        3,
        np.where(away_goals == home_goals, 1, 0),
    )

    np.add.at(points, home_idx, home_points)
    np.add.at(points, away_idx, away_points)

    np.add.at(goals_for, home_idx, home_goals)
    np.add.at(goals_for, away_idx, away_goals)

    np.add.at(goals_against, home_idx, away_goals)
    np.add.at(goals_against, away_idx, home_goals)

    goal_diff = goals_for - goals_against

    return _CurrentStandings(
        teams=teams,
        team_to_idx=team_to_idx,
        points=points,
        goals_for=goals_for,
        goals_against=goals_against,
        goal_diff=goal_diff,
    )


def _compute_match_score_matrix(lambda_home: float, mu_away: float, rho: float,
                                max_goals: int = CFG.MAX_GOALS) -> np.ndarray:
    goals = np.arange(max_goals + 1)
    home_probs = poisson.pmf(goals, mu=lambda_home)
    away_probs = poisson.pmf(goals, mu=mu_away)

    score_matrix = np.outer(home_probs, away_probs)

    score_matrix[0, 0] *= max(0.0, 1.0 - lambda_home * mu_away * rho)
    score_matrix[0, 1] *= max(0.0, 1.0 + lambda_home * rho)
    score_matrix[1, 0] *= max(0.0, 1.0 + mu_away * rho)
    score_matrix[1, 1] *= max(0.0, 1.0 - rho)

    total_prob = float(np.sum(score_matrix))
    return np.asarray(score_matrix / total_prob, dtype=float)


def _build_summary_table(
        teams: list[str],
        current: _CurrentStandings,
        sim_points: np.ndarray,
        sim_goal_diff: np.ndarray,
        ranks: np.ndarray,
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "team": teams,
            "cur_pts": current.points,
            "cur_gd": current.goal_diff,
            "exp_pts": np.mean(sim_points, axis=0),
            "exp_gd": np.mean(sim_goal_diff, axis=0),
            "exp_rank": np.mean(ranks, axis=0),
            "p_champion": np.mean(ranks == 1, axis=0) * 100.0,
            "p_ucl": np.mean(ranks <= 4, axis=0) * 100.0,
            "p_uel": np.mean(ranks == 5, axis=0) * 100.0,
            "p_uecl": np.mean(ranks == 6, axis=0) * 100.0,
            "p_relegation": np.mean(ranks == 16, axis=0) * 100.0,
            "p_direct_relegation": np.mean(ranks >= 17, axis=0) * 100.0,
        }
    )
    summary = summary.sort_values(
        by=["exp_pts", "exp_gd"], ascending=[False, False]
    ).reset_index(drop=True)
    summary.index += 1
    return summary


@dataclass(frozen=True)
class SimulationResult:
    summary: pd.DataFrame
    final_points: np.ndarray
    final_goal_diff: np.ndarray
    ranks: np.ndarray


def simulate_season(
        played_df: pd.DataFrame,
        remaining_df: pd.DataFrame,
        model: ModelParameters,
        n_simulations: int = CFG.N_SIMULATIONS,
        seed: int = CFG.RANDOM_SEED,
        max_goals: int = CFG.MAX_GOALS,
) -> SimulationResult:
    teams: list[str] = sorted(model.attack.keys())
    standings = _compute_current_standings(played_df, teams)
    n_teams = len(teams)

    sim_points = np.broadcast_to(standings.points, (n_simulations, n_teams)).copy()
    sim_goals_for = np.broadcast_to(standings.goals_for, (n_simulations, n_teams)).copy()
    sim_goals_against = np.broadcast_to(standings.goals_against, (n_simulations, n_teams)).copy()

    rng = np.random.default_rng(seed)
    n_outcomes = (max_goals + 1) ** 2
    sim_indices = np.arange(n_simulations)

    for row in remaining_df.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        h_idx = standings.team_to_idx[home_team]
        a_idx = standings.team_to_idx[away_team]

        lambda_home, mu_away = model.predict_expected_goals(home_team, away_team)
        score_matrix = _compute_match_score_matrix(
            lambda_home=lambda_home,
            mu_away=mu_away,
            rho=model.rho,
            max_goals=max_goals,
        )

        drawn_flat = rng.choice(n_outcomes, size=n_simulations, p=score_matrix.ravel())
        h_goals = drawn_flat // (max_goals + 1)
        a_goals = drawn_flat % (max_goals + 1)

        h_pts = np.where(h_goals > a_goals, 3, np.where(h_goals == a_goals, 1, 0))
        a_pts = np.where(a_goals > h_goals, 3, np.where(a_goals == h_goals, 1, 0))

        sim_points[sim_indices, h_idx] += h_pts
        sim_points[sim_indices, a_idx] += a_pts

        sim_goals_for[sim_indices, h_idx] += h_goals
        sim_goals_for[sim_indices, a_idx] += a_goals

        sim_goals_against[sim_indices, h_idx] += a_goals
        sim_goals_against[sim_indices, a_idx] += h_goals

    sim_goal_diff = sim_goals_for - sim_goals_against

    order = np.lexsort((-sim_goals_for, -sim_goal_diff, -sim_points), axis=1)

    ranks = np.empty_like(order)
    ranks[sim_indices[:, None], order] = np.arange(1, n_teams + 1)

    summary = _build_summary_table(
        teams=teams,
        current=standings,
        sim_points=sim_points,
        sim_goal_diff=sim_goal_diff,
        ranks=ranks,
    )

    return SimulationResult(
        summary=summary,
        final_points=sim_points,
        final_goal_diff=sim_goal_diff,
        ranks=ranks,
    )


if __name__ == "__main__":
    from src.model import fit_dixon_coles_model

    played = pd.read_csv(CFG.PLAYED_MATCHES_FILE)
    remaining = pd.read_csv(CFG.REMAINING_MATCHES_FILE)

    fitted_model = fit_dixon_coles_model(played)
    sim_result = simulate_season(played, remaining, fitted_model)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(sim_result.summary.round(2))
