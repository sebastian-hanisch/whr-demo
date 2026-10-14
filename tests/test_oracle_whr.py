"""Orakel für die Glätter: WHR gegen einen unabhängigen Maximierer (scipy, BFGS), das dichte System gegen den Blocklöser, die Grenzfälle w → 0 und w → ∞, die TTT-Aktualisierung gegen
das Paket `trueskill` und gegen die exakte Posteriori auf einem Gitter, die Vorschau gegen abgeschnittene Daten, Filter und Glätter mit Vorschau 0."""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_drift as D
import whr_evaluation as ev
import whr_smoother as S
from conftest import build


def _instance(seed, n=3, rounds=6):
    return D.simulate_matches(n, 0.3, rounds, 0.15, 1.0, 0, 0.0, 0, 0, seed)


def test_whr_equals_the_scipy_maximiser_on_random_instances():
    optimize = pytest.importorskip("scipy.optimize")
    for seed in range(6):
        m = _instance(seed)
        w, s0 = 0.25, 1.2
        x, _ = S.whr_fit(m, w, s0)
        T, n = x.shape

        def neg(v, m=m, w=w, s0=s0, T=T, n=n):
            return -S.whr_objective(m, v.reshape(T, n), w, s0)

        res = optimize.minimize(neg, np.zeros(T * n), method="BFGS", options={"gtol": 1e-9, "maxiter": 500})
        assert x.reshape(-1) == pytest.approx(res.x, abs=2e-4), seed
        assert S.whr_objective(m, x, w, s0) >= -res.fun - 1e-9


def test_dense_newton_equals_the_block_solver():
    """Unabhängige Fassung: dichte Hesse-Matrix über alle (Runde, Spieler)-Variablen, Newton mit np.linalg.solve."""
    m = _instance(11, n=4, rounds=8)
    w, s0 = 0.3, 1.0
    T, n = 8, 4
    q = 1.0 / w ** 2
    x = np.zeros(T * n)
    for _ in range(40):
        H = np.zeros((T * n, T * n))
        g = np.zeros(T * n)
        for r, i, j, y in zip(m.rnd, m.i, m.j, m.win):
            a, b = r * n + i, r * n + j
            p = 1.0 / (1.0 + math.exp(-(x[a] - x[b])))
            g[a] += y - p
            g[b] -= y - p
            h = p * (1 - p)
            H[a, a] += h
            H[b, b] += h
            H[a, b] -= h
            H[b, a] -= h
        for t in range(T - 1):
            for i in range(n):
                u, v = t * n + i, (t + 1) * n + i
                H[u, u] += q
                H[v, v] += q
                H[u, v] -= q
                H[v, u] -= q
                g[u] -= q * (x[u] - x[v])
                g[v] -= q * (x[v] - x[u])
        for i in range(n):
            H[i, i] += 1.0 / s0 ** 2
            g[i] -= x[i] / s0 ** 2
        H[np.diag_indices(T * n)] += C.RIDGE
        g -= C.RIDGE * x
        step = np.linalg.solve(H, g)
        x += step
        if np.abs(step).max() < 1e-11:
            break
    ours, _ = S.whr_fit(m, w, s0)
    assert ours.reshape(-1) == pytest.approx(x, abs=1e-8)


def test_w_to_zero_gives_the_static_bradley_terry_estimate():
    """Statisches Bradley-Terry mit Prior N(0, s²) auf der Anfangswertung: ein Wert je Spieler, Newton auf der Summe der Partien."""
    m = _instance(5, n=3, rounds=10)
    s0 = 1.5
    r = np.zeros(3)
    for _ in range(50):
        g = -r / s0 ** 2
        H = np.eye(3) / s0 ** 2
        for i, j, y in zip(m.i, m.j, m.win):
            p = 1.0 / (1.0 + math.exp(-(r[i] - r[j])))
            g[i] += y - p
            g[j] -= y - p
            h = p * (1 - p)
            H[i, i] += h
            H[j, j] += h
            H[i, j] -= h
            H[j, i] -= h
        r += np.linalg.solve(H, g)
    x, _ = S.whr_fit(m, 1e-3, s0)
    assert np.ptp(x, axis=0).max() < 5e-3
    assert x.mean(axis=0) == pytest.approx(r, abs=0.02)


def test_w_to_infinity_gives_independent_rounds():
    """Ohne Kopplung ist jede Runde für sich ein Bradley-Terry-Problem mit Prior; Partie (0 gegen 1) mit Prior s: x0 = −x1 = (1 − σ(2 x0)) s²."""
    games = [[(0, 1, True)], [(0, 1, False)], [(0, 1, True)]]
    m = build(2, 3, games)
    x, _ = S.whr_fit(m, 1e3, prior_sd=1.0)
    t0 = x[0, 0]
    assert x[0, 0] == pytest.approx(-x[0, 1], abs=1e-6) and x[0, 0] > 0
    # die Prior-Kette bindet nur Runde 0 an den Prior; ohne Kopplung bleiben Runden 1 und 2 durch den Ridge fast frei: Vorzeichen folgen dem Ausgang
    assert x[1, 0] < x[1, 1] and x[2, 0] > x[2, 1] and t0 > 0


def test_truncated_history_equals_a_fit_on_the_truncated_games():
    m = _instance(2, n=3, rounds=10)
    cut = D.Matches(m.n, m.truth[:6], m.rnd[m.rnd < 6], m.i[m.rnd < 6], m.j[m.rnd < 6], m.win[m.rnd < 6], m.join_round, m.pause, m.rare)
    a = S.whr_fit(m, 0.2, 1.0, T=6)[0]
    b = S.whr_fit(cut, 0.2, 1.0)[0]
    assert a == pytest.approx(b, abs=1e-10)


def test_probit_update_equals_the_trueskill_package():
    trueskill = pytest.importorskip("trueskill")
    env = trueskill.TrueSkill(mu=0.0, sigma=1.5, beta=0.8, tau=0.0, draw_probability=0.0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        mw, ml = rng.normal(0, 1.5, 2)
        sw, sl = rng.uniform(0.3, 2.0, 2)
        a, b = env.create_rating(mw, sw), env.create_rating(ml, sl)
        a2, b2 = trueskill.rate_1vs1(a, b, env=env)
        nmw, nvw, nml, nvl = S.probit_update(np.array([mw]), np.array([sw ** 2]), np.array([ml]), np.array([sl ** 2]), beta=0.8)
        assert nmw[0] == pytest.approx(a2.mu, abs=1e-6) and nml[0] == pytest.approx(b2.mu, abs=1e-6)
        assert math.sqrt(nvw[0]) == pytest.approx(a2.sigma, abs=1e-6) and math.sqrt(nvl[0]) == pytest.approx(b2.sigma, abs=1e-6)


def test_ttt_is_close_to_the_exact_posterior_on_a_tiny_instance():
    """Zwei Spieler, zwei Runden, je eine Partie: exakter Posteriori-Mittelwert auf einem Vierdimensional-Gitter (Probit-Likelihood, Wiener-Prior), gegen EP bis auf die
    Näherung (die Faktorgraph-Schleife über beide Spieler und beide Runden ist nicht exakt)."""
    m = build(2, 2, [[(0, 1, True)], [(0, 1, True)]])
    gamma, sd0, beta = 0.5, 1.0, C.BETA
    mu, _, _ = S.ttt_fit(m, gamma, beta=beta, sd0=sd0, tol=1e-9, sweeps=200)
    grid = np.linspace(-4.0, 4.0, 41)
    x00, x01, x10, x11 = np.meshgrid(grid, grid, grid, grid, indexing="ij")

    def phi(v):
        return 0.5 * np.vectorize(math.erfc)(-v / math.sqrt(2.0))

    logp = (-0.5 * (x00 ** 2 + x10 ** 2) / sd0 ** 2 - 0.5 * ((x01 - x00) ** 2 + (x11 - x10) ** 2) / gamma ** 2
            + np.log(phi((x00 - x10) / (math.sqrt(2.0) * beta))) + np.log(phi((x01 - x11) / (math.sqrt(2.0) * beta))))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    exact = np.array([[(w * x00).sum(), (w * x10).sum()], [(w * x01).sum(), (w * x11).sum()]])
    assert mu == pytest.approx(exact, abs=0.08)
    assert mu[0, 0] > mu[0, 1] and mu[1, 0] > mu[1, 1]


def test_smoother_with_no_lookahead_is_a_filter_of_the_same_quality():
    """Vorschau 0 ist der Filter derselben Dynamik: im Mittel über 20 Läufe liegt sein Fehler bei dem des besten Filters (Toleranz 10 %)."""
    errs, filt = [], []
    for sd in C.SWEEP_SEEDS[:20]:
        m = ev._matches("equal", 0.06, sd)
        errs.append(np.mean([ev.round_error(_run_at(m, t), m, t) for t in (40, 70, 100)]))
        filt.append(ev.tracking_error(ev.run_method(m, "Glicko", 0.06, C.PHI0_INFORMED), m))
    assert abs(np.mean(errs) / np.mean(filt) - 1.0) < 0.10


def _run_at(m, t):
    x = S.lagged_estimate(m, t, 0, 0.06, 1.0)
    return D.Run(np.where(m.presence, np.tile(x, (m.rounds, 1)), np.nan), None, x, float("nan"))


def _dense_newton(m, w, s0, steps):
    """Unabhängige dichte Newton-Schritte (Python-Schleifen) von x = 0 aus, genau `steps` Schritte."""
    T, n = m.rounds, m.n
    q = 1.0 / w ** 2
    x = np.zeros(T * n)
    for _ in range(steps):
        H = np.zeros((T * n, T * n))
        g = np.zeros(T * n)
        for r, i, j, y in zip(m.rnd, m.i, m.j, m.win):
            a, b = r * n + i, r * n + j
            p = 1.0 / (1.0 + math.exp(-(x[a] - x[b])))
            g[a] += y - p
            g[b] -= y - p
            h = p * (1 - p)
            H[a, a] += h
            H[b, b] += h
            H[a, b] -= h
            H[b, a] -= h
        for t_ in range(T - 1):
            for i in range(n):
                u, v = t_ * n + i, (t_ + 1) * n + i
                H[u, u] += q
                H[v, v] += q
                H[u, v] -= q
                H[v, u] -= q
                g[u] -= q * (x[u] - x[v])
                g[v] -= q * (x[v] - x[u])
        for i in range(n):
            H[i, i] += 1.0 / s0 ** 2
            g[i] -= x[i] / s0 ** 2
        H[np.diag_indices(T * n)] += C.RIDGE
        g -= C.RIDGE * x
        x += np.linalg.solve(H, g)
    return x.reshape(T, n)


@pytest.mark.parametrize("steps", [1, 2, 3])
def test_each_newton_step_equals_the_dense_step(steps):
    """Die ersten Schritte sind noch nicht konvergiert: Sie prüfen die Hesse-Matrix, nicht nur den Fixpunkt."""
    m = _instance(7, n=3, rounds=6)
    ours, _ = S.whr_fit(m, 0.25, 0.6, iters=steps, tol=0.0)
    assert ours == pytest.approx(_dense_newton(m, 0.25, 0.6, steps), abs=1e-9)


def test_ttt_equals_the_exact_posterior_on_a_tree():
    """Eine einzige Partie (kein Kreis im Faktorgraphen): Erwartungsabgleich ist exakt. Exakte Posteriori-Mittel auf einem Gitter (Probit-Likelihood, Wiener-Prior)."""
    m = build(2, 2, [[(0, 1, True)], []])
    gamma, sd0, beta = 0.5, 1.0, C.BETA
    mu, _, _ = S.ttt_fit(m, gamma, beta=beta, sd0=sd0, tol=1e-10, sweeps=200)
    g = np.linspace(-5.0, 5.0, 61)
    x00, x01, x10, x11 = np.meshgrid(g, g, g, g, indexing="ij")
    phi = np.vectorize(lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)))
    logp = (-0.5 * (x00 ** 2 + x10 ** 2) / sd0 ** 2 - 0.5 * ((x01 - x00) ** 2 + (x11 - x10) ** 2) / gamma ** 2 + np.log(phi((x00 - x10) / (math.sqrt(2.0) * beta))))
    w = np.exp(logp - logp.max())
    w /= w.sum()
    exact = np.array([[(w * x00).sum(), (w * x10).sum()], [(w * x01).sum(), (w * x11).sum()]])
    assert mu == pytest.approx(exact, abs=2e-3)


def test_each_newton_step_equals_the_dense_step_on_a_strong_signal():
    """Große Werte nach dem ersten Schritt: Die Hesse-Matrix der folgenden Schritte hängt dann deutlich von den Differenzen der Stärken ab."""
    games = [[(0, 1, True), (2, 3, True)] for _ in range(5)] + [[(0, 2, True), (1, 3, True)] for _ in range(3)]
    m = build(4, 8, games)
    for steps in (2, 3, 4):
        ours, _ = S.whr_fit(m, 1.0, 3.0, iters=steps, tol=0.0)
        assert np.abs(ours).max() > 1.0
        assert ours == pytest.approx(_dense_newton(m, 1.0, 3.0, steps), abs=1e-9), steps


def test_ttt_equals_the_exact_posterior_on_a_path_with_two_games():
    """Drei Spieler, zwei Runden: A schlägt B in Runde 0, B schlägt C in Runde 1. Der Faktorgraph ist ein Pfad A1 – A0 – B0 – B1 – C1 – C0 (kein Kreis): Erwartungsabgleich ist exakt.
    Exakte Posteriori-Mittel durch Vorwärts-Rückwärts-Rechnung auf einem feinen Gitter; die Rückwärtsnachrichten tragen die Partie der Runde 1 in die Runde 0."""
    m = build(3, 2, [[(0, 1, True)], [(1, 2, True)]])
    gamma, sd0, beta = 0.5, 1.0, C.BETA
    mu, _, _ = S.ttt_fit(m, gamma, beta=beta, sd0=sd0, tol=1e-10, sweeps=300)
    g = np.linspace(-7.0, 7.0, 281)
    phi = np.vectorize(lambda v: 0.5 * math.erfc(-v / math.sqrt(2.0)))
    gauss = lambda d, s: np.exp(-0.5 * (d / s) ** 2)                      # noqa: E731
    diff = g[:, None] - g[None, :]
    trans = gauss(diff, gamma)                                           # (v_k, v_{k+1}): Übergang in der Zeit
    game = phi(diff / (math.sqrt(2.0) * beta))                           # (Sieger, Verlierer)
    prior = gauss(g, sd0)
    # Pfad: A1 – A0 – B0 – B1 – C1 – C0; Kanten: trans(A1, A0), game(A0, B0), trans(B0, B1), game(B1, C1), trans(C1, C0); Priors auf A0, B0, C0
    unary = [np.ones_like(g), prior, prior, np.ones_like(g), np.ones_like(g), prior]
    edges = [trans, game, trans, game, trans]
    fwd = [unary[0]]
    for k in range(5):
        fwd.append(unary[k + 1] * (fwd[k] @ edges[k]))
    bwd = [None] * 6
    bwd[5] = np.ones_like(g)
    for k in range(4, -1, -1):
        bwd[k] = edges[k] @ (unary[k + 1] * bwd[k + 1])
    means = []
    for k in range(6):
        marg = (fwd[k] * bwd[k]) if k > 0 else unary[0] * bwd[0]
        marg = marg / marg.sum()
        means.append(float((marg * g).sum()))
    a1, a0, b0, b1, c1, c0 = means
    assert mu[0] == pytest.approx([a0, b0, c0], abs=3e-3)
    assert mu[1] == pytest.approx([a1, b1, c1], abs=3e-3)
    assert mu[0, 0] > mu[0, 1] > mu[0, 2] - 1.0                          # A vor B vor C (C über die Kette von Runde 1 zurück)
