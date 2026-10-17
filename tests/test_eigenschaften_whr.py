"""Metamorphe Eigenschaftstests für die Glätter (WHR und TTT): Man verändert die Eingabe auf eine Weise, deren Wirkung auf die Ausgabe bekannt ist, und prüft diese Beziehung
(ORACLE-PLAYBOOK §5.2). Kein Einzelwert wird erwartet, nur Beziehungen zwischen zwei Läufen.

Belegt sind die Aussagen durch das Modell (siehe Kopf von whr_smoother.py): Die Partien gehen nur über die Differenzen x_i − x_j in die Likelihood ein, die Drift ist ein
Wiener-Prozess (symmetrisch in der Zeit), der Prior sitzt nur auf der Anfangswertung.

Eigenschaften:
  1. Umnummerieren der Spieler, Umdrehen der Notation einer Partie ((i, j, i gewinnt) = (j, i, j gewinnt)) und Umsortieren der Partien innerhalb einer Runde ändern nur die Namen.
  2. Gemeinsame Verschiebung aller Stärken ändert Zielfunktion und Gradient ohne Prior/Ridge nicht; das Niveau legt nur der Prior (Anfangswertung) bzw. der Ridge fest.
  3. Spiegeln aller Ergebnisse negiert alle Stärken.
  4. Drift gegen 0 liefert die statische Bradley-Terry-Schätzung (Bayeselo ist der Sonderfall w² = 0, Coulom 2008, Abschnitt 5; hier mit Normal-Prior auf der Anfangswertung).
  5. Die Schätzung zum Zeitpunkt t₀ mit Vorschau L hängt nur von den Partien bis t₀ + L ab (Kausalität), und sie hängt von den Partien innerhalb des Fensters ab.
  6. Zeitumkehr der Runden spiegelt den Verlauf, wenn kein Prior auf der Anfangswertung sitzt (Wiener-Prozess zeitsymmetrisch; mit Prior nicht, dann gilt die Aussage nicht).
  7. Spieler ohne Partien bleiben bei 0, Randfälle (eine Runde, keine Partie).
  8. TrueSkill Through Time: Spiegeln der Ergebnisse negiert die Mittel, Umnummerieren permutiert sie.
"""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_drift as D
import whr_smoother as S
from conftest import build

N_INST = 30


def _inst(k, n=None, rounds=None, sigma=None, p_rare=1.0):
    """k-te Zufallsinstanz (fester Seed): kleine Felder, wenige Runden, damit die Läufe schnell bleiben."""
    rng = np.random.default_rng(1000 + k)
    n = int(rng.integers(3, 8)) if n is None else n
    rounds = int(rng.integers(6, 16)) if rounds is None else rounds
    sigma = float(rng.choice([0.0, 0.05, 0.15])) if sigma is None else sigma
    return D.simulate_matches(n, 0.3, rounds, sigma, p_rare, 0, 0.0, 0, 0, seed=500 + k)


def _replace(m, i=None, j=None, win=None, rnd=None):
    return D.Matches(m.n, m.truth, m.rnd if rnd is None else rnd, m.i if i is None else i, m.j if j is None else j, m.win if win is None else win, m.join_round, m.pause, m.rare)


def _params(k):
    rng = np.random.default_rng(77 + k)
    return float(rng.uniform(0.05, 0.6)), float(rng.uniform(0.5, 2.5))     # w, prior_sd


# --- 1. Umnummerieren, Notation, Reihenfolge in der Runde ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_relabeling_players_only_permutes_the_estimate(k):
    m = _inst(k, p_rare=0.6 if k % 3 == 0 else 1.0)
    w, s0 = _params(k)
    perm = np.random.default_rng(k).permutation(m.n)                       # Spieler a heißt künftig perm[a]
    m2 = _replace(m, i=perm[m.i], j=perm[m.j])
    x = S.whr_fit(m, w, s0)[0]
    x2 = S.whr_fit(m2, w, s0)[0]
    assert x2[:, perm] == pytest.approx(x, abs=1e-7)


@pytest.mark.parametrize("k", range(N_INST))
def test_game_notation_and_order_within_a_round_do_not_matter(k):
    m = _inst(k)
    w, s0 = _params(k)
    rng = np.random.default_rng(900 + k)
    flip = rng.random(len(m.rnd)) < 0.5
    i2, j2 = np.where(flip, m.j, m.i), np.where(flip, m.i, m.j)
    win2 = np.where(flip, 1 - m.win, m.win)
    order = np.argsort(m.rnd + rng.random(len(m.rnd)) * 0.5, kind="stable")   # Runde bleibt Hauptschlüssel, innerhalb der Runde zufällig
    m2 = _replace(m, i=i2[order], j=j2[order], win=win2[order], rnd=m.rnd[order])
    x = S.whr_fit(m, w, s0)[0]
    assert S.whr_fit(m2, w, s0)[0] == pytest.approx(x, abs=1e-8)


# --- 2. Verschiebung des Niveaus ----------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_common_shift_leaves_objective_and_gradient_unchanged_without_prior_and_ridge(k):
    m = _inst(k)
    w = _params(k)[0]
    rng = np.random.default_rng(300 + k)
    x = rng.normal(0.0, 1.0, (m.rounds, m.n))
    c = float(rng.uniform(-5.0, 5.0))
    f0 = S.whr_objective(m, x, w, None, ridge=0.0)
    f1 = S.whr_objective(m, x + c, w, None, ridge=0.0)
    assert f1 == pytest.approx(f0, abs=1e-8)
    assert S.whr_gradient(m, x + c, w, None, ridge=0.0) == pytest.approx(S.whr_gradient(m, x, w, None, ridge=0.0), abs=1e-9)
    # Mit Prior auf der Anfangswertung ist die Verschiebung dagegen nicht mehr folgenlos: das Niveau hängt am Prior.
    assert S.whr_objective(m, x + c, w, 1.0, ridge=0.0) != pytest.approx(S.whr_objective(m, x, w, 1.0, ridge=0.0), abs=1e-6)


@pytest.mark.parametrize("k", range(N_INST))
def test_level_is_fixed_by_the_prior_or_the_ridge_only(k):
    m = _inst(k)
    w, s0 = _params(k)
    # Mit Prior: Summe der Gradientenbedingungen über alle (Runde, Spieler) ergibt  Σ x[0]/s₀² + ridge Σ x = 0, das Mittel der Anfangswertung liegt also bei 0.
    x = S.whr_fit(m, w, s0)[0]
    assert abs(x[0].sum()) < 1e-4
    # Ohne Prior bestimmt allein der Ridge das Niveau: Σ x = 0.
    y = S.whr_fit(m, w, None)[0]
    assert abs(y.sum()) < 1e-5
    # Ein 10-fach größerer Ridge verschiebt (fast) nur das Niveau: Die Differenzen zwischen den Spielern ändern sich kaum. Nur sinnvoll, wo die Schätzung ohne Prior endlich ist
    # (bei vollständiger Trennung läuft sie ohne Prior gegen unendlich, der Ridge bestimmt dann die Größe); dafür genügen genug Partien.
    m = _inst(k, n=4, rounds=40, sigma=0.1)
    y = S.whr_fit(m, w, None)[0]
    assert np.abs(y).max() < 6.0
    z = S.whr_fit(m, w, None, ridge=1e-5)[0]
    assert np.ptp(y - z) < 2e-2


# --- 3. Spiegeln ----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_mirroring_all_results_negates_the_estimate(k):
    m = _inst(k)
    w, s0 = _params(k)
    m2 = _replace(m, win=1 - m.win)
    x = S.whr_fit(m, w, s0)[0]
    assert S.whr_fit(m2, w, s0)[0] == pytest.approx(-x, abs=1e-8)
    # ohne Prior ebenso
    assert S.whr_fit(m2, w, None)[0] == pytest.approx(-S.whr_fit(m, w, None)[0], abs=1e-6)


# --- 4. Drift gegen 0: statisches Bradley-Terry --------------------------------------------------------------------------------------------------

def _static_bt(m, s0):
    """Statisches Bradley-Terry mit Normal-Prior N(0, s₀²): Newton auf der Summe aller Partien, unabhängig von whr_smoother geschrieben."""
    r = np.zeros(m.n)
    for _ in range(60):
        g = -r / s0 ** 2
        H = np.eye(m.n) / s0 ** 2
        for a, b, y in zip(m.i, m.j, m.win):
            p = 1.0 / (1.0 + math.exp(r[b] - r[a]))
            g[a] += y - p
            g[b] -= y - p
            h = p * (1.0 - p)
            H[a, a] += h
            H[b, b] += h
            H[a, b] -= h
            H[b, a] -= h
        step = np.linalg.solve(H, g)
        r += step
        if np.abs(step).max() < 1e-12:
            break
    return r


@pytest.mark.parametrize("k", range(20))
def test_drift_to_zero_gives_the_static_bradley_terry_estimate(k):
    rng = np.random.default_rng(40 + k)
    m = _inst(k, n=int(rng.integers(3, 6)), rounds=int(rng.integers(6, 12)), sigma=0.0)
    s0 = float(rng.uniform(0.8, 2.5))
    x = S.whr_fit(m, 1e-4, s0)[0]                                          # wird auf W_MIN = 0,01 begrenzt
    assert S.effective_w(1e-4) == C.W_MIN
    r = _static_bt(m, s0)
    assert np.ptp(x, axis=0).max() < 0.03                                  # fast konstant in der Zeit
    assert x.mean(axis=0) == pytest.approx(r, abs=0.04)


# --- 5. Kausalität der Vorschau -------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_lagged_estimate_uses_only_games_up_to_t0_plus_lag(k):
    m = _inst(k, rounds=16, p_rare=1.0)
    w, s0 = _params(k)
    rng = np.random.default_rng(600 + k)
    t0 = int(rng.integers(2, 9))
    lag = int(rng.choice([0, 2, 5]))
    edge = t0 + lag
    base = S.lagged_estimate(m, t0, lag, w, s0)
    later = m.rnd > edge
    assert later.any()
    flipped = _replace(m, win=np.where(later, 1 - m.win, m.win))
    assert S.lagged_estimate(flipped, t0, lag, w, s0) == pytest.approx(base, abs=1e-9)
    # Die Partien im Fenster zählen dagegen: Spiegeln der Partien der letzten Fensterrunde ändert die Schätzung (Spiegeln im Fenster ändert Ergebnisse).
    inside = m.rnd == edge
    assert inside.any()
    flipped_in = _replace(m, win=np.where(inside, 1 - m.win, m.win))
    assert np.abs(S.lagged_estimate(flipped_in, t0, lag, w, s0) - base).max() > 1e-4


def test_lagged_estimate_with_all_rounds_sees_the_last_game():
    m = _inst(3, n=4, rounds=12, sigma=0.1)
    w, s0 = 0.2, 1.0
    last = m.rnd == m.rounds - 1
    flipped = _replace(m, win=np.where(last, 1 - m.win, m.win))
    a = S.lagged_estimate(m, 3, C.LAG_ALL, w, s0)
    b = S.lagged_estimate(flipped, 3, C.LAG_ALL, w, s0)
    assert np.abs(a - b).max() > 1e-6                                      # klein, aber vorhanden: auch ein sehr später Ausgang wirkt zurück


# --- 6. Zeitumkehr ------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(N_INST))
def test_time_reversal_mirrors_the_path_when_there_is_no_prior_on_the_start(k):
    m = _inst(k)
    w = _params(k)[0]
    T = m.rounds
    order = np.argsort(T - 1 - m.rnd, kind="stable")
    m2 = _replace(m, i=m.i[order], j=m.j[order], win=m.win[order], rnd=(T - 1 - m.rnd)[order])
    x = S.whr_fit(m, w, None)[0]
    assert S.whr_fit(m2, w, None)[0][::-1] == pytest.approx(x, abs=1e-6)


def test_time_reversal_is_not_a_symmetry_with_a_prior_on_the_start():
    """Gegenprobe: Der Prior sitzt nur auf Runde 0; die Aussage der Zeitumkehr gilt deshalb ausdrücklich nur ohne Prior. Sonst fiele der Test oben nicht auf."""
    m = build(2, 6, [[(0, 1, True)], [(0, 1, True)], [], [], [], []])
    T = m.rounds
    m2 = build(2, 6, [[], [], [], [], [(0, 1, True)], [(0, 1, True)]])
    a = S.whr_fit(m, 0.3, 0.5)[0]
    b = S.whr_fit(m2, 0.3, 0.5)[0][::-1]
    assert T == 6 and np.abs(a - b).max() > 0.05


# --- 7. Spieler ohne Partien, Randfälle --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(10))
def test_players_without_games_stay_at_zero(k):
    rng = np.random.default_rng(80 + k)
    rounds = int(rng.integers(5, 10))
    games = [[(0, 1, bool(rng.random() < 0.6)), (2, 3, bool(rng.random() < 0.5))] for _ in range(rounds)]
    m = build(6, rounds, games)                                            # Spieler 4 und 5 spielen nie
    x = S.whr_fit(m, 0.2, 1.0)[0]
    assert np.abs(x[:, 4:]).max() < 1e-9
    assert np.abs(x[:, :4]).max() > 0.01


def test_edge_cases_one_round_and_no_games():
    one = build(2, 1, [[(0, 1, True)]])
    x, _ = S.whr_fit(one, 0.3, 1.0)
    assert x.shape == (1, 2)
    assert x[0, 0] == pytest.approx(-x[0, 1], abs=1e-9)
    assert x[0, 0] == pytest.approx((1.0 - float(S.sigmoid(2.0 * x[0, 0]))) * 1.0 ** 2, abs=1e-6)     # Von Hand: x₀ = (1 − σ(2 x₀)) s₀² (Stationarität bei einer Partie)
    none = build(3, 5, [[], [], [], [], []])
    assert np.abs(S.whr_fit(none, 0.3, 1.0)[0]).max() < 1e-9
    with pytest.raises(ValueError):
        S.whr_fit(none, 0.3, 1.0, T=0)


# --- 8. TrueSkill Through Time ------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", range(8))
def test_ttt_mirroring_negates_and_relabeling_permutes(k):
    m = _inst(k, n=int(3 + k % 4), rounds=8, sigma=0.05)
    gamma, sd0 = 0.1 + 0.02 * (k % 5), 1.5
    mu, sd, _ = S.ttt_fit(m, gamma, sd0=sd0, tol=1e-6)
    mu_m, sd_m, _ = S.ttt_fit(_replace(m, win=1 - m.win), gamma, sd0=sd0, tol=1e-6)
    assert mu_m == pytest.approx(-mu, abs=1e-7)
    assert sd_m == pytest.approx(sd, abs=1e-7)
    perm = np.random.default_rng(k).permutation(m.n)
    mu_p, sd_p, _ = S.ttt_fit(_replace(m, i=perm[m.i], j=perm[m.j]), gamma, sd0=sd0, tol=1e-6)
    assert mu_p[:, perm] == pytest.approx(mu, abs=1e-6)
    assert sd_p[:, perm] == pytest.approx(sd, abs=1e-6)


def test_the_filter_lets_the_uncertainty_grow_during_a_pause():
    """Regressionstest (Befund 2026-10-17): Der Glicko-Filter (Vergleichsverfahren) ließ φ in der Pause nicht wachsen. Handinstanz: Spieler 0
    spielt 2 Runden gegen 1, pausiert 4 Runden (1 gegen 2), kehrt zurück. Ohne Partien wächst φ je Runde um c (Deckel φ₀); das Ergebnis muss dasselbe
    sein, als hätte Spieler 0 vier Runden im Feld ohne Partie ausgesetzt."""
    import whr_filter as F
    pre = [[(0, 1, True)], [(0, 1, False)]]
    mid = [[(1, 2, True)] for _ in range(4)]
    back = [[(0, 1, True)]]
    games = pre + mid + back
    stay = build(3, 7, games)
    paused = build(3, 7, games, pause=(2, 6))
    c, phi0 = 0.2, 1.5
    rs, rp = F.run_glicko(stay, c, phi0), F.run_glicko(paused, c, phi0)
    assert rp.sd[6, 0] == pytest.approx(rs.sd[6, 0], rel=1e-9)
    assert rp.mu[6, 0] == pytest.approx(rs.mu[6, 0], rel=1e-9)
