"""Glätter: Blocklöser, Gradient, Newton-Verfahren, Vorschau, TTT-Aktualisierung und -Durchlauf an Handwerten und Randfällen."""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_smoother as S
from conftest import build


def _dense_whr_system(T, n, q, diag_blocks):
    """Dichte Matrix zum Blocksystem (für die Gegenprobe)."""
    A = np.zeros((T * n, T * n))
    for t in range(T):
        A[t * n:(t + 1) * n, t * n:(t + 1) * n] = diag_blocks[t]
        if t + 1 < T:
            for i in range(n):
                A[t * n + i, (t + 1) * n + i] = -q
                A[(t + 1) * n + i, t * n + i] = -q
    return A


def test_block_tridiagonal_solver_equals_the_dense_solve():
    rng = np.random.default_rng(3)
    for T, n in ((1, 3), (2, 2), (7, 4), (30, 5)):
        q = 3.0
        D = np.array([a @ a.T + (2 * q + 1.0) * np.eye(n) for a in rng.normal(0, 0.3, (T, n, n))])
        r = rng.normal(size=(T, n))
        x = S.solve_block_tridiagonal(D, q, r)
        ref = np.linalg.solve(_dense_whr_system(T, n, q, D), r.reshape(-1)).reshape(T, n)
        assert x == pytest.approx(ref, abs=1e-9), (T, n)


def test_gradient_equals_the_finite_difference_of_the_objective():
    m = build(3, 4, [[(0, 1, True)], [(1, 2, False)], [(0, 2, True)], [(0, 1, False)]])
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.5, (4, 3))
    g = S.whr_gradient(m, x, 0.3, prior_sd=0.8)
    num = np.zeros_like(x)
    eps = 1e-6
    for idx in np.ndindex(*x.shape):
        up, dn = x.copy(), x.copy()
        up[idx] += eps
        dn[idx] -= eps
        num[idx] = (S.whr_objective(m, up, 0.3, 0.8) - S.whr_objective(m, dn, 0.3, 0.8)) / (2 * eps)
    assert g == pytest.approx(num, abs=1e-6)


def test_newton_ends_at_a_stationary_point():
    m = build(3, 5, [[(0, 1, True)], [(1, 2, True)], [(0, 2, False)], [(0, 1, True)], [(2, 1, False)]])
    x, it = S.whr_fit(m, 0.2, prior_sd=1.0)
    assert np.abs(S.whr_gradient(m, x, 0.2, 1.0)).max() < 1e-8 and it <= 15


def test_two_players_one_game_without_dynamics_and_with_a_prior_by_hand():
    # Ein Spiel, x0 − x1 = d; Prior N(0, s²) auf beide Anfangswerte, T = 1: Stationarität 1 − σ(d) = x0/s² = −x1/s²  ⇒ d = 2 x0, x0 = (1 − σ(2 x0)) s²
    m = build(2, 1, [[(0, 1, True)]])
    x, _ = S.whr_fit(m, 1.0, prior_sd=1.0, ridge=0.0)
    d = x[0, 0] - x[0, 1]
    assert x[0, 0] == pytest.approx(-x[0, 1], abs=1e-9)
    assert x[0, 0] == pytest.approx((1.0 - 1.0 / (1.0 + math.exp(-d))) * 1.0, abs=1e-9)
    assert 0.0 < x[0, 0] < 0.5


def test_static_model_gives_equal_ratings_in_every_round():
    m = build(2, 6, [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)], [(0, 1, True)], [(0, 1, True)], [(0, 1, False)]])
    x, _ = S.whr_fit(m, 0.0, prior_sd=1.0)                         # w → W_MIN
    assert np.ptp(x[:, 0]) < 0.01 and x[0, 0] > x[0, 1]


def test_a_huge_drift_decouples_the_rounds():
    m = build(2, 2, [[(0, 1, True)], [(0, 1, False)]])
    x, _ = S.whr_fit(m, 50.0, prior_sd=1.0)
    assert x[0, 0] > x[0, 1] and x[1, 0] < x[1, 1]


def test_smoother_without_games_returns_the_prior_mean_zero():
    m = build(3, 4, [[], [], [], []])
    x, it = S.whr_fit(m, 0.1, prior_sd=1.0)
    assert np.abs(x).max() < 1e-12


def test_a_player_without_games_in_a_pause_is_interpolated_linearly():
    # Spieler 0 gewinnt vor und verliert nach der Pause: seine Schätzung geht in der Pause monoton und geradlinig von hoch nach niedrig
    games = [[(0, 1, True)], [(0, 1, True)], [], [], [], [], [(0, 1, False)], [(0, 1, False)]]
    m = build(2, 8, games)
    x, _ = S.whr_fit(m, 0.3, prior_sd=2.0)
    mid = x[2:6, 0] - x[2:6, 1]
    assert np.all(np.diff(mid) < 0)
    second = np.diff(mid, 2)
    assert np.abs(second).max() < 0.02 * np.abs(np.diff(mid)).max() + 1e-9    # nahezu linear


def test_lagged_estimate_uses_only_games_up_to_t_plus_lag():
    games = [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)], [(0, 1, False)], [(0, 1, False)], [(0, 1, False)]]
    m = build(2, 6, games)
    x_all = S.lagged_estimate(m, 1, C.LAG_ALL, 0.3, 1.0)
    x_0 = S.lagged_estimate(m, 1, 0, 0.3, 1.0)
    x_2 = S.lagged_estimate(m, 1, 2, 0.3, 1.0)
    full = S.whr_fit(m, 0.3, 1.0)[0][1]
    assert x_all == pytest.approx(full)
    assert x_0 == pytest.approx(S.whr_fit(m, 0.3, 1.0, T=2)[0][1]) and x_2 == pytest.approx(S.whr_fit(m, 0.3, 1.0, T=4)[0][1])
    assert x_0[0] - x_0[1] > x_2[0] - x_2[1] > x_all[0] - x_all[1]         # spätere Niederlagen ziehen die frühere Schätzung herunter


def test_lag_larger_than_the_rest_is_the_full_history_and_zero_rounds_is_rejected():
    m = build(2, 3, [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)]])
    assert S.lagged_estimate(m, 1, 99, 0.3, None) == pytest.approx(S.whr_fit(m, 0.3, None)[0][1])
    with pytest.raises(ValueError):
        S.whr_fit(m, 0.3, T=0)


def test_effective_w_has_a_floor():
    assert S.effective_w(0.0) == C.W_MIN and S.effective_w(0.2) == 0.2


# --- TTT --------------------------------------------------------------------------------------------------------------------------------------

def test_probit_update_two_new_players_by_hand():
    # Zwei gleiche Spieler µ = 0, σ² = 1, β² = 1: c² = 4, t = 0, v = φ(0)/Φ(0) = 0,79788, w = v² = 0,63662
    mw, vw, ml, vl = S.probit_update(np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), beta=1.0)
    v = (1.0 / math.sqrt(2 * math.pi)) / 0.5
    assert mw[0] == pytest.approx(v / 2.0) and ml[0] == pytest.approx(-v / 2.0)
    assert vw[0] == pytest.approx(1.0 - v * v / 4.0) and vl[0] == pytest.approx(vw[0])


def test_probit_update_upset_moves_more_than_expected_win():
    up = S.probit_update(np.array([-1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
    exp = S.probit_update(np.array([1.0]), np.array([1.0]), np.array([-1.0]), np.array([1.0]))
    assert up[0][0] - (-1.0) > exp[0][0] - 1.0


def test_probit_update_stays_finite_for_huge_gaps():
    out = S.probit_update(np.array([-60.0]), np.array([1.0]), np.array([60.0]), np.array([1.0]))
    assert all(np.isfinite(o).all() for o in out) and out[1][0] > 0 and out[3][0] > 0


def test_ttt_without_games_is_the_prior():
    m = build(3, 4, [[], [], [], []])
    mu, sd, used = S.ttt_fit(m, 0.1, sd0=1.5)
    assert np.abs(mu).max() < 1e-12 and sd[0] == pytest.approx(1.5) and used <= 3
    assert (sd[1:] > sd[:-1] - 1e-12).all()                          # ohne Partien wächst die Unsicherheit mit der Zeit


def test_ttt_single_round_equals_the_trueskill_update_for_two_new_players():
    m = build(2, 1, [[(0, 1, True)]])
    mu, sd, _ = S.ttt_fit(m, 0.1, beta=1.0, sd0=1.0)
    mw, vw, ml, vl = S.probit_update(np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([1.0]), beta=1.0)
    assert mu[0, 0] == pytest.approx(mw[0], abs=1e-9) and mu[0, 1] == pytest.approx(ml[0], abs=1e-9)
    assert sd[0, 0] == pytest.approx(math.sqrt(vw[0]), abs=1e-9)


def test_ttt_later_games_change_earlier_estimates():
    early = build(2, 4, [[(0, 1, True)], [], [], []])
    late = build(2, 4, [[(0, 1, True)], [], [], [(0, 1, True)]])
    mu_e, _, _ = S.ttt_fit(early, 0.3, sd0=1.0)
    mu_l, _, _ = S.ttt_fit(late, 0.3, sd0=1.0)
    assert mu_l[0, 0] - mu_l[0, 1] > mu_e[0, 0] - mu_e[0, 1]


def test_as_run_masks_players_outside_the_field():
    m = build(3, 5, [[(1, 2, True)] for _ in range(5)], join_round=3)
    r = S.as_run(m, np.arange(15.0).reshape(5, 3))
    assert np.isnan(r.mu[:3, 0]).all() and not np.isnan(r.mu[3:, 0]).any() and r.final.tolist() == [12.0, 13.0, 14.0]


# --- Randfälle und Zähler (aus dem Mutationstest) ---------------------------------------------------------------------------------------------

def test_objective_and_gradient_ignore_games_after_the_fitted_rounds():
    m = build(2, 4, [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)], [(0, 1, False)]])
    x = np.zeros((2, 2))
    g = S.whr_gradient(m, x, 0.3, 1.0)
    assert g.shape == (2, 2) and g[0, 0] == pytest.approx(0.5 - 0.0) and g[1, 0] == pytest.approx(0.5)      # nur die Partien der Runden 0 und 1 zählen
    assert S.whr_objective(m, x, 0.3, 1.0) == pytest.approx(2 * math.log(0.5))


def test_newton_iteration_count_is_reported_exactly():
    m = build(3, 4, [[], [], [], []])
    assert S.whr_fit(m, 0.2, prior_sd=1.0)[1] == 1                      # ohne Partien ist der Startpunkt schon die Lösung
    m2 = build(2, 3, [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)]])
    assert S.whr_fit(m2, 0.2, prior_sd=1.0, iters=1)[1] == 1
    assert S.whr_fit(m2, 0.2, prior_sd=1.0, iters=2)[1] == 2
    assert 3 <= S.whr_fit(m2, 0.2, prior_sd=1.0)[1] <= 8


def test_single_round_has_no_coupling_term():
    m = build(2, 1, [[(0, 1, True)]])
    x_small, _ = S.whr_fit(m, 0.01, prior_sd=1.0, ridge=0.0)
    x_large, _ = S.whr_fit(m, 100.0, prior_sd=1.0, ridge=0.0)
    assert x_small == pytest.approx(x_large, abs=1e-12)


def test_ttt_prior_variance_grows_by_the_drift_each_round_without_games():
    m = build(2, 4, [[], [], [], []])
    mu, sd, used = S.ttt_fit(m, 0.3, sd0=1.5)
    assert used == 2
    assert sd[:, 0] == pytest.approx(np.sqrt(1.5 ** 2 + 0.3 ** 2 * np.arange(4)), abs=1e-12)
    assert S.ttt_fit(m, 0.3, sd0=1.5, sweeps=1)[2] == 1


def test_ttt_converges_well_before_the_sweep_limit_on_a_normal_instance():
    m = build(3, 6, [[(0, 1, True)], [(1, 2, True)], [(0, 2, False)], [(0, 1, True)], [(1, 2, False)], [(0, 2, True)]])
    used = S.ttt_fit(m, 0.2, sd0=1.0)[2]
    assert 3 <= used <= 25
