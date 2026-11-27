from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # League & API settings (FotMob Bundesliga ID = 54)
    LEAGUE_ID: int = 54
    SEASON: str = "2026/2027"
    FOTMOB_BASE_URL: str = "https://www.fotmob.com/api/data"
    REQUEST_DELAY_SEC: float = 0.25
    CACHE_TTL_HOURS: int = 6

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    MATCH_DETAILS_DIR: Path = DATA_RAW_DIR / "matches"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    RESULTS_DIR: Path = PROJECT_ROOT / "results"

    RAW_LEAGUE_FILE: Path = DATA_RAW_DIR / "fotmob_league_54.json"
    PLAYED_MATCHES_FILE: Path = DATA_PROCESSED_DIR / "played_matches.csv"
    REMAINING_MATCHES_FILE: Path = DATA_PROCESSED_DIR / "remaining_matches.csv"

    # Dixon-Coles & xG hyperparameters
    XG_WEIGHT: float = 0.70  # Blend: 70% xG, 30% actual goals
    TIME_DECAY_XI: float = 0.0035  # Exponential time decay per day
    HOME_ADVANTAGE_INIT: float = 0.25
    RHO_INIT: float = -0.05

    # Monte Carlo simulation settings
    N_SIMULATIONS: int = 10_000
    RANDOM_SEED: int = 42
    MAX_GOALS: int = 10

    # Bundesliga table zones (1-indexed)
    CHAMPIONS_LEAGUE_SPOTS: tuple[int, ...] = (1, 2, 3, 4)
    EUROPA_LEAGUE_SPOT: int = 5
    CONFERENCE_LEAGUE_SPOT: int = 6
    RELEGATION_PLAYOFF_SPOT: int = 16
    DIRECT_RELEGATION_SPOTS: tuple[int, ...] = (17, 18)


CFG = Config()
