"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

import math

SCALE = 400.0 / math.log(10.0)              # Elo-Punkte je logistische Einheit θ (400 Punkte = Faktor 10 der Gewinnchancen), 173,72
BASE = 1500.0                               # Mitte der Elo-Skala in der Anzeige
BETA = 1.702 / math.sqrt(2.0)               # TrueSkill-Leistungsrauschen in θ-Einheiten: Φ(d/(√2 β)) = Φ(d/1,702) ≈ σ(d), damit das Probit-Modell zur logistischen Wahrheit passt
BURN = 30                                   # Runden, die beim Verfolgungsfehler nicht zählen (Kaltstart der Filter)
W_MIN = 0.01                                # kleinste Drift-Annahme des Glätters (θ je Runde): w = 0 ist das statische Bradley-Terry-Modell, numerisch ein sehr steifes Problem
RIDGE = 1e-6                                # fixiert das Niveau (die Wertungen sind nur bis auf eine Konstante bestimmt)
PHI0_INFORMED = 0.7                         # Anfangsunsicherheit des Filters Glicko (θ), wie im Vorgängerstück der informierte Wert
PHI0_DEFAULT = 350.0 / SCALE                # die übliche Voreinstellung von Glicko (350 Punkte, 2,01 θ), nur im Neuling-Experiment
EXP_PRIOR = 1.0                             # Prior auf die Anfangswertung in den Experimenten (wie die Voreinstellung der App)

N_MIN, N_MAX, DEFAULT_N = 4, 12, 8
DELTA_MIN, DELTA_MAX, DELTA_STEP, DEFAULT_DELTA = 0.05, 1.0, 0.05, 0.3
ROUNDS_MIN, ROUNDS_MAX, DEFAULT_ROUNDS = 40, 200, 150
SIGMA_MIN, SIGMA_MAX, SIGMA_STEP, DEFAULT_SIGMA = 0.0, 0.3, 0.01, 0.06       # Schritt der Irrfahrt je Runde (logistische Einheiten)

SCENARIOS = ("Alle spielen jede Runde", "Ungleiche Aktivität", "Neuling kommt später", "Pause und Rückkehr")
DEFAULT_SCENARIO = SCENARIOS[0]
P_RARE_MIN, P_RARE_MAX, P_RARE_STEP, DEFAULT_P_RARE = 0.05, 0.8, 0.05, 0.2    # Spielwahrscheinlichkeit je Runde der seltenen Hälfte
JOIN_MIN, DEFAULT_JOIN = 2, 60                                                # Runde, in der der Neuling kommt (Obergrenze: Runden − 2)
GAP_MIN, GAP_MAX, GAP_STEP, DEFAULT_GAP = -3.0, 3.0, 0.5, 2.0                 # Abstand des Neulings vom Feld (logistische Einheiten)
PAUSE_START_MIN, DEFAULT_PAUSE_START = 5, 40
PAUSE_LEN_MIN, PAUSE_LEN_MAX, DEFAULT_PAUSE_LEN = 5, 150, 60


def pause_start_max(rounds: int) -> int:
    """Spätester Beginn der Pause: danach bleiben mindestens fünf Runden Pause (plus eine, damit der Längenregler eine Spanne hat) und fünf Runden zum Wiedereinspielen."""
    return rounds - PAUSE_LEN_MIN - 6


def pause_len_max(rounds: int, start: int) -> int:
    """Längste Pause bei gegebenem Beginn: fünf Runden bleiben zum Wiedereinspielen."""
    return min(PAUSE_LEN_MAX, rounds - start - 5)


K_MIN, K_MAX, DEFAULT_K = 1, 200, 28                                          # Elo, in Punkten
C_MIN, C_MAX, DEFAULT_C = 1, 60, 8                                            # Zuwachs der Unsicherheit je Runde in Punkten (Glicko c)
W_FACTOR_MIN, W_FACTOR_MAX, W_FACTOR_STEP, DEFAULT_W_FACTOR = 0.0, 3.0, 0.25, 1.0   # Drift-Annahme des Glätters als Vielfaches der wahren Drift σ (0 = statisch)
PRIOR_MIN, PRIOR_MAX, PRIOR_STEP, DEFAULT_PRIOR = 0.3, 3.0, 0.1, 1.0          # Prior auf die Anfangswertung (Standardabweichung, θ-Einheiten)
LAG_ALL = -1                                                                  # Vorschau „alle späteren Runden“
LAG_CHOICES = (0, 2, 5, 10, 20, LAG_ALL)
DEFAULT_LAG = LAG_ALL
SEED_MAX, DEFAULT_SEED = 999999, 7

# Feste Instanzen für alle Experimente: nie ein einzelner Seed als Befund (dieselben wie im Vorgängerstück).
SWEEP_SEEDS = tuple(range(100000, 100060))
EXP_N, EXP_DELTA, EXP_ROUNDS = 8, 0.3, 150
SIGMA_SWEEP = (0.03, 0.06, 0.12)
K_GRID = (10.0, 20.0, 28.0, 40.0, 56.0, 80.0, 112.0)                          # Punkte
C_GRID = (0.01, 0.02, 0.03, 0.045, 0.06, 0.09, 0.12, 0.18)                    # θ-Einheiten je Runde
W_FACTORS = (0.5, 1.0, 2.0)
LAG_T0 = 75                                                                   # Zeitpunkt, an dem die Vorschau gemessen wird
LAG_SIGMAS = (0.06, 0.12)
P_RARE_EXP = 0.2
NEW_JOIN, NEW_ROUNDS, NEW_GAP = 60, 100, 2.0
NEW_PRIORS = (None, 2.0, 1.0, 0.7)                                            # Prior auf die Anfangswertung (None = keiner)
PAUSE_START, PAUSE_LEN, PAUSE_ROUNDS = 40, 60, 125
PAUSE_SIGMAS = (0.06, 0.12)
POS_SIGMA = 0.06
POS_BINS = 10

COLOR_FILTER = "#8e6bb0"
COLOR_ELO = "#2F6B65"
COLOR_WHR = "#d98c1f"
COLOR_TTT = "#3d7ab5"
COLOR_TRUTH = "#e67e22"
PALETTE = ("#2F6B65", "#8e6bb0", "#d98c1f", "#c0392b", "#3d7ab5", "#7a8f3c", "#b0577e", "#5a5a5a", "#1f9d8a", "#a0522d", "#6c5fc7", "#b8860b")

_BASE = {"n": DEFAULT_N, "delta": DEFAULT_DELTA, "rounds": DEFAULT_ROUNDS, "sigma": DEFAULT_SIGMA, "scenario": DEFAULT_SCENARIO, "p_rare": DEFAULT_P_RARE,
         "join": DEFAULT_JOIN, "gap": DEFAULT_GAP, "pause_start": DEFAULT_PAUSE_START, "pause_len": DEFAULT_PAUSE_LEN, "k": DEFAULT_K, "c": DEFAULT_C,
         "w_factor": DEFAULT_W_FACTOR, "prior": DEFAULT_PRIOR, "lag": DEFAULT_LAG, "seed": DEFAULT_SEED}
PRESETS = {
    "Eingespieltes Feld": dict(_BASE),
    "Ungleiche Aktivität": {**_BASE, "scenario": SCENARIOS[1], "p_rare": 0.2},
    "Lange Pause": {**_BASE, "scenario": SCENARIOS[3], "sigma": 0.12, "rounds": 150, "pause_start": 40, "pause_len": 60},
    "Neuling mit Abstand 2": {**_BASE, "scenario": SCENARIOS[2], "sigma": 0.0, "rounds": 100, "join": 60, "gap": 2.0},
}
PRESET_HELP = {
    "Eingespieltes Feld": "8 Spieler, alle spielen jede Runde, die Stärken wandern mit Schritt 0,06: Der Glätter liegt rund ein Viertel unter dem Fehler der Filter.",
    "Ungleiche Aktivität": "Die Hälfte der Spieler spielt nur mit Wahrscheinlichkeit 0,2 je Runde: Wer selten spielt, wird aus den späteren Partien mitgeschätzt; hier bleibt der Gewinn etwa so groß wie bei gleicher Aktivität.",
    "Lange Pause": "Spieler 0 pausiert 60 Runden bei starker Drift (0,12): Der Glätter interpoliert zwischen vor und nach der Pause, der Filter friert ein.",
    "Neuling mit Abstand 2": "Ein Neuling kommt in Runde 60, 2 Einheiten über dem Feld: Am Ende der Aufzeichnung hilft im Rückblick nur ein passender Prior; ohne ihn läuft die Schätzung weg.",
}
