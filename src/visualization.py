from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CFG
from src.model import ModelParameters
from src.simulation import SimulationResult


def plot_team_strengths(
        model: ModelParameters,
        output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = CFG.RESULTS_DIR / "team_strengths.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    teams = sorted(model.attack.keys())
    attack = np.array([model.attack[t] for t in teams], dtype=float)

    defense_strength = np.array([-model.defense[t] for t in teams], dtype=float)
    net_strength = attack + defense_strength

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    ax.axvline(0.0, color="#9E9E9E", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(float(np.mean(defense_strength)), color="#9E9E9E", linestyle="--", linewidth=1.0, zorder=1)

    scatter = ax.scatter(
        attack,
        defense_strength,
        c=net_strength,
        cmap="RdYlGn",
        s=140,
        edgecolors="#212121",
        linewidths=0.9,
        zorder=3,
    )

    for idx, team in enumerate(teams):
        ax.annotate(
            team,
            (attack[idx], defense_strength[idx]),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="medium",
            color="#1A1A1A",
            zorder=4,
        )

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Net Strength (Attack − Defense, log-scale)", fontsize=9.5)

    ax.set_title(
        f"Bundesliga {CFG.SEASON} — Dixon-Coles Team Strength Matrix",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Attack Strength α (higher = stronger offense)", fontsize=10.5)
    ax.set_ylabel("Defensive Strength −β (higher = fewer goals conceded)", fontsize=10.5)
    ax.grid(True, linestyle=":", alpha=0.5, zorder=0)

    x_pad = (attack.max() - attack.min()) * 0.12
    y_pad = (defense_strength.max() - defense_strength.min()) * 0.12
    ax.set_xlim(attack.min() - x_pad, attack.max() + x_pad)
    ax.set_ylim(defense_strength.min() - y_pad, defense_strength.max() + y_pad)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_season_probabilities(
        result: SimulationResult,
        output_path: Path | None = None,
) -> Path:
    if output_path is None:
        output_path = CFG.RESULTS_DIR / "season_probabilities.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = result.summary.iloc[::-1].reset_index(drop=True)
    teams = df["team"].tolist()
    y_pos = np.arange(len(teams))

    p_champ = df["p_champion"].to_numpy(dtype=float)
    p_ucl_rest = np.maximum(0.0, df["p_ucl"].to_numpy(dtype=float) - p_champ)
    p_uel = df["p_uel"].to_numpy(dtype=float)
    p_uecl = df["p_uecl"].to_numpy(dtype=float)
    p_rel = df["p_relegation"].to_numpy(dtype=float)
    p_direct_rel = df["p_direct_relegation"].to_numpy(dtype=float)
    p_mid = np.maximum(
        0.0,
        100.0 - (p_champ + p_ucl_rest + p_uel + p_uecl + p_rel + p_direct_rel),
    )

    segments = [
        (p_champ, "Champion (1st)", "#1B5E20"),
        (p_ucl_rest, "Champions League (2nd–4th)", "#43A047"),
        (p_uel, "Europa League (5th)", "#1E88E5"),
        (p_uecl, "Conference League (6th)", "#64B5F6"),
        (p_mid, "Mid-table (7th–15th)", "#E0E0E0"),
        (p_rel, "Relegation Play-off (16th)", "#FB8C00"),
        (p_direct_rel, "Direct Relegation (17th–18th)", "#D32F2F"),
    ]

    fig, ax = plt.subplots(figsize=(12, 8.5), dpi=300)
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FFFFFF")

    left = np.zeros(len(teams), dtype=float)
    for values, label, color in segments:
        ax.barh(
            y_pos,
            values,
            left=left,
            height=0.72,
            label=label,
            color=color,
            edgecolor="#FFFFFF",
            linewidth=0.5,
        )
        left += values

    for idx, exp_pts in enumerate(df["exp_pts"]):
        ax.text(
            101.2,
            y_pos[idx],
            f"{exp_pts:.1f} pts",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#212121",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(teams, fontsize=9.5, fontweight="medium")
    ax.set_xlim(0, 111)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_xlabel("Simulated Outcome Probability (%)", fontsize=10.5)
    ax.set_title(
        f"Bundesliga {CFG.SEASON} — Monte Carlo Season Outcome Probabilities",
        fontsize=13,
        fontweight="bold",
        pad=28,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.46, 1.06),
        ncol=4,
        frameon=False,
        fontsize=8.2,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    import pandas as pd

    from src.model import fit_dixon_coles_model
    from src.simulation import simulate_season

    played = pd.read_csv(CFG.PLAYED_MATCHES_FILE)
    remaining = pd.read_csv(CFG.REMAINING_MATCHES_FILE)

    fitted_model = fit_dixon_coles_model(played)
    sim_result = simulate_season(played, remaining, fitted_model)

    strength_plot = plot_team_strengths(fitted_model)
    prob_plot = plot_season_probabilities(sim_result)
    print(f"Saved: {strength_plot}")
    print(f"Saved: {prob_plot}")
