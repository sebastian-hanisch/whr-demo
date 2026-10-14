"""Maße (von Hand gerechnete Läufe), Analyse des gezeigten Laufs, Verdikt-Zweige und die Struktur der Experimente."""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_drift as D
import whr_evaluation as ev
import whr_filter as F
import whr_smoother as S
from conftest import build
from whr_scenario import Settings, make_matches


def run_of(mu, sd=None):
    mu = np.asarray(mu, dtype=float)
    return D.Run(mu, None if sd is None else np.asarray(sd, dtype=float), mu[-1], 0.0)


# --- Maße gegen die Handrechnung --------------------------------------------------------------------------------------------------------------

def test_tracking_error_by_hand():
    truth = [[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
    m = build(3, 2, [[], []], truth=truth)
    run = run_of([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    # Runde 0: Abweichungen (0,5; 0; −0,5) -> RMSE √(0,5/3); Runde 1: (1; 0; −1) -> √(2/3)
    assert ev.tracking_error(run, m, burn=0) == pytest.approx((math.sqrt(0.5 / 3) + math.sqrt(2.0 / 3)) / 2)
    assert ev.tracking_error(run, m, burn=1) == pytest.approx(math.sqrt(2.0 / 3))
    assert ev.round_error(run, m, 0) == pytest.approx(math.sqrt(0.5 / 3))


def test_tracking_error_is_invariant_to_a_common_shift_of_the_ratings():
    m = build(3, 2, [[], []], truth=[[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]])
    a = run_of([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]])
    b = run_of([[11.0, 10.0, 9.0], [12.0, 10.0, 8.0]])
    assert ev.tracking_error(a, m, burn=0) == pytest.approx(ev.tracking_error(b, m, burn=0))


def test_tracking_error_over_a_subset_centres_inside_the_subset():
    m = build(4, 1, [[]], truth=[[0.0, 1.0, 2.0, 3.0]])
    run = run_of([[0.0, 5.0, 2.0, 3.0]])
    # Teilmenge {0, 2}: Wertungen (0, 2), Stärken (0, 2): zentriert gleich -> Fehler 0
    assert ev.tracking_error(run, m, burn=0, subset=(0, 2)) == pytest.approx(0.0)
    assert ev.tracking_error(run, m, burn=0) > 0.5


def test_only_players_in_the_field_count():
    m = build(3, 2, [[], []], join_round=1, truth=[[0.0, 1.0, -1.0], [0.0, 1.0, -1.0]])
    mu = [[np.nan, 1.0, -1.0], [0.0, 1.0, -1.0]]
    assert ev.round_error(run_of(mu), m, 0) == pytest.approx(0.0)           # Spieler 0 fehlt: nur 1 und 2 zählen, beide exakt
    assert math.isnan(ev.round_error(run_of([[np.nan, np.nan, 0.0]] * 2), m, 0))     # ein einzelner Spieler: nichts zu zentrieren


def test_coverage_by_hand():
    truth = [[0.5, 0.0, -0.5], [1.0, 0.0, -1.0]]
    m = build(3, 2, [[], []], truth=truth)
    mu = [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0]]
    sd = [[0.3] * 3, [0.3] * 3]
    # 1,96 · 0,3 = 0,588: Runde 0 Abweichungen (0,5; 0; 0,5) -> 3 von 3; Runde 1 (1; 0; 1) -> 1 von 3
    assert ev.coverage(run_of(mu, sd), m, burn=0) == pytest.approx(4.0 / 6.0)
    assert ev.coverage(run_of(mu, sd), m, burn=1) == pytest.approx(1.0 / 3.0)
    assert math.isnan(ev.coverage(run_of(mu), m, burn=0))                  # Elo hat keine Unsicherheit


def test_player_rounds_and_error_after_games():
    games = [[(0, 1, True)], [(2, 3, True)], [(0, 2, False)], [(0, 3, True)]]
    m = build(4, 4, games, truth=np.zeros((4, 4)))
    assert ev.player_rounds(m, 0) == [0, 2, 3]
    assert ev.player_rounds(m, 0, after=1) == [2, 3]
    mu = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
    run = run_of(mu)
    assert ev.error_after_games(run, m, 1) == pytest.approx(0.0)
    assert ev.error_after_games(run, m, 2) == pytest.approx(1.0 - 0.25)    # nach der zweiten Partie (Runde 2): µ0 = 1, Feldmittel 0,25
    assert math.isnan(ev.error_after_games(run, m, 4))


def test_kendall_tau_values():
    assert ev.kendall_tau([3, 2, 1], [30, 20, 10]) == 1.0
    assert ev.kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert ev.kendall_tau([1, 1, 2], [1, 2, 3]) == pytest.approx(1.0)      # Gleichstände zählen nicht


# --- Vorschau, Analyse, Verdikte --------------------------------------------------------------------------------------------------------------

def test_lagged_error_with_all_history_is_the_tracking_error_of_the_smoother():
    m = ev._matches("equal", 0.06, 100000)
    assert ev.lagged_error(m, C.LAG_ALL, 0.06, 1.0) == pytest.approx(ev.tracking_error(ev.smooth_run(m, 0.06, 1.0), m))


def test_lagged_error_improves_with_more_lookahead_on_average():
    errs = {L: np.mean([ev.lagged_error(ev._matches("equal", 0.06, sd), L, 0.06, 1.0) for sd in C.SWEEP_SEEDS[:8]]) for L in (0, 10, C.LAG_ALL)}
    assert errs[0] > errs[10] > errs[C.LAG_ALL]


def test_lagged_error_uses_the_given_checkpoints():
    m = ev._matches("equal", 0.06, 100000)
    x = S.lagged_estimate(m, 50, 5, 0.06, 1.0)
    run = D.Run(np.tile(x, (m.rounds, 1)), None, x, float("nan"))
    assert ev.lagged_error(m, 5, 0.06, 1.0, checkpoints=[50]) == pytest.approx(ev.round_error(run, m, 50))


def test_effective_drift_assumption_of_the_settings():
    assert Settings(sigma=0.06, w_factor=2.0).w == pytest.approx(0.12)
    assert Settings(sigma=0.06, w_factor=0.0).w == C.W_MIN and Settings(sigma=0.0, w_factor=1.0).w == C.W_MIN


def test_analysis_matches_direct_runs_and_reports_the_lag_error():
    s = Settings(n=6, rounds=50, sigma=0.08, lag=5, seed=3)
    a = ev.analyse(s)
    m = make_matches(s)
    x, _ = S.whr_fit(m, s.w, s.prior)
    assert a.error["WHR"] == pytest.approx(ev.tracking_error(S.as_run(m, x), m))
    assert a.error["Elo"] == pytest.approx(ev.tracking_error(F.run_elo(m, s.k), m))
    assert a.lag_error == pytest.approx(ev.lagged_error(m, 5, s.w, s.prior)) and a.lag_error > a.error["WHR"]
    assert a.raw_whr.shape == (50, 6) and a.games.sum() == 2 * len(m.rnd)


def test_analysis_reports_the_scenario_specific_errors():
    a = ev.analyse(Settings(n=6, rounds=50, scenario=C.SCENARIOS[1], p_rare=0.3))
    assert set(a.error_rare) == set(ev.METHODS) and set(a.error_regular) == set(ev.METHODS)
    b = ev.analyse(Settings(n=6, rounds=60, scenario=C.SCENARIOS[2], join=30, sigma=0.0))
    assert set(b.player0) == set(ev.METHODS) and len(b.player0["WHR"]) == 3
    assert ev.analyse(Settings(n=6, rounds=40)).player0 == {}


def _analysis(whr, elo, glicko, lag=C.LAG_ALL, lag_error=0.0):
    m = build(2, 2, [[], []])
    err = {"Elo": elo, "Glicko": glicko, "WHR": whr, "TTT": whr}
    return ev.Analysis(Settings(lag=lag), m, {}, err, {}, {}, lag_error, np.zeros(2, dtype=int), {}, 5, 1.0, 0.0)


def test_verdict_analysis_boundary_between_gain_and_no_gain():
    gain = ev.verdict_analysis(_analysis(0.2, 0.3, 0.25))
    none = ev.verdict_analysis(_analysis(0.25, 0.3, 0.25))
    loss = ev.verdict_analysis(_analysis(0.26, 0.3, 0.25))
    assert gain[0] == "info" and "20 % unter dem besseren Filter (Glicko)" in gain[1]
    assert none[0] == "info" and "0 %" in none[1]
    assert loss[0] == "warning" and "bringt hier nichts" in loss[1] and "über dem besseren Filter" in loss[1]


def test_verdict_analysis_names_the_better_filter_and_the_lag():
    assert "(Elo)" in ev.verdict_analysis(_analysis(0.2, 0.25, 0.3))[1]
    assert "Mit nur 5 Runden Vorschau sind es 0,234" in ev.verdict_analysis(_analysis(0.2, 0.3, 0.25, lag=5, lag_error=0.234))[1]
    assert "Vorschau" not in ev.verdict_analysis(_analysis(0.2, 0.3, 0.25))[1]


def test_experiment_rows_have_the_documented_shape():
    rows = ev.sweep_hindsight()
    assert [r["sigma"] for r in rows] == list(C.SIGMA_SWEEP) and all(r["gain"] == pytest.approx(1 - r["whr"] / min(r["elo"], r["glicko"])) for r in rows)
    lag = ev.sweep_lag()
    assert [r["sigma"] for r in lag] == list(C.LAG_SIGMAS) and list(lag[0]["lags"]) == list(C.LAG_CHOICES)
    pos = ev.sweep_position()
    assert len(pos["WHR"]) == C.POS_BINS and pos["sigma"] == C.POS_SIGMA
    ww = ev.sweep_wrong_w()
    assert set(ww[0]) == {"sigma", "filter", "w0.5", "w1.0", "w2.0", "static"}
    nc = ev.sweep_newcomer()
    assert [r["prior"] for r in nc if r["kind"] == "WHR"] == list(C.NEW_PRIORS) and len(nc) == len(C.NEW_PRIORS) + 3
    ps = ev.sweep_pause()
    assert [r["sigma"] for r in ps] == list(C.PAUSE_SIGMAS) and set(ps[0]["mid"]) == {"WHR", "Elo", "Glicko"} and len(ps[0]["after"]["WHR"]) == 3
    assert set(ev.sweep_activity()) >= {"whr_total", "elo_rare", "glicko_regular"}


def test_best_filter_picks_the_grid_value_with_the_smallest_mean_error():
    p, e = ev.best_filter("equal", 0.06, "Elo", seeds=C.SWEEP_SEEDS[:5])
    errs = {k: np.mean([ev._filter_error("equal", 0.06, sd, "Elo", k, C.PHI0_INFORMED, "total") for sd in C.SWEEP_SEEDS[:5]]) for k in C.K_GRID}
    assert p == min(errs, key=errs.get) and e == pytest.approx(min(errs.values()))


def test_verdicts_mention_the_numbers_of_their_rows():
    h = ev.sweep_hindsight()
    assert ev.de(h[1]["whr"], 3) in ev.verdict_hindsight(h)[1] and f"{ev.pct(h[1]['gain'])} /" in ev.verdict_hindsight(h)[1]
    w = ev.sweep_wrong_w()
    text = ev.verdict_wrong_w(w)[1]
    assert ev.de(w[2]["static"], 3) in text and ev.de(w[2]["filter"], 3) in text
