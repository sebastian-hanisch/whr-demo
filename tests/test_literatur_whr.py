"""Literatur-Referenzwerte (ORACLE-PLAYBOOK §5.1): Werte und Formeln aus Quellen, die für diese Datei tatsächlich eingesehen wurden. Einsicht jeweils am 2026-10-17
(PDF per WebFetch geladen und mit pdftotext gelesen bzw. Seite per WebFetch gelesen). Kein Wert stammt aus dem Code der Demo.

1. Glickman, M. E. (1999): „The Glicko system“, http://www.glicko.net/glicko/glicko.pdf, Seite 4 („Example calculation“ zu Schritt 2) und Seite 5 (Erwartungswert).
   Eine Spielerin mit Wertung 1500 und RD 200 schlägt den Gegner 1400/RD 30 und verliert gegen 1550/RD 100 und 1700/RD 300 (alle in einer Periode):
   g(RD_j) = 0,9955 / 0,9531 / 0,7242; E = 0,639 / 0,432 / 0,303; d² = 53670,85 (d = 231,672); neue Wertung 1464 (aus 1500 − 36); neues RD 151,4.
   Dazu Seite 6: RD 50, c = 63,2, 30 Perioden ohne Spiel führen auf RD ≈ 350 (Gleichung „350 = 50² + c²(30)“, dort mit gerundetem c).
   Das Glicko des Demos (whr_filter.py) rechnet in θ-Einheiten mit θ = (Punkte − 1500)/SCALE und φ = RD/SCALE, SCALE = 400/ln 10 = 1/q (q = 0,0057565 im Papier).
2. Coulom, R. (2008): „Whole-History Rating“, LNCS 5131, http://remi.coulom.free.fr/WHR/WHR.pdf (PDF-Seiten, die LNCS-Zählung weicht ab):
   - Abschnitt 2.1/2.2 (S. 3): natürliche Wertung r = R ln 10/400, γ = 10^(R/400), P(i schlägt j) = γ_i/(γ_i + γ_j).
   - Anhang A.1 (S. 10): ∂ln P/∂r_i = |W(i)| − γ_i Σ_j C_ij/(C_ij γ_i + D_ij), ∂²ln P/∂r_i² = −γ_i Σ_j C_ij D_ij/(C_ij γ_i + D_ij)²; für das einfache Bradley-Terry-Modell
     (C_ij = 1, D_ij = γ_j) also Gradient = Siege − Σ_j p_ij und Krümmung = −Σ_j p_ij (1 − p_ij).
   - Anhang A.2 (S. 10): Wiener-Prior, σ² = |t₂ − t₁| w²: ∂ln p/∂r_i(t₁) = −(r_i(t₁) − r_i(t₂))/σ², ∂²ln p/∂r_i(t₁)² = −1/σ², ∂²ln p/∂r_i(t₁)∂r_i(t₂) = +1/σ².
   - Anhang C (S. 12): Zwischen zwei Zeitpunkten t₁, t₂ ist der Mittelwert des Wiener-Prozesses μ = (μ₁(t₂ − t) + μ₂(t − t₁))/(t₂ − t₁), also linear interpoliert.
3. Wikipedia „Elo rating system“, https://en.wikipedia.org/wiki/Elo_rating_system, Einleitung: 100 Punkte Vorsprung bedeuten einen erwarteten Anteil von 64 %, 200 Punkte 76 %;
   Bildunterschrift im Abschnitt „Ratings inflation and deflation“: 1800 gegen 1400, Gewinnwahrscheinlichkeit des Stärkeren 0,91 (Stand der Einsicht 2026-10-17; Wikipedia ist Sekundärquelle).
4. Dangauthier, P., Herbrich, R., Minka, T., Graepel, T. (2007): „TrueSkill Through Time: Revisiting the History of Chess“, NIPS 20,
   https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/NIPS2007_0931.pdf, Abschnitt 3.1 (Vanilla TrueSkill, S. 3), am 2026-10-17 im Volltext gelesen:
   c² = 2β² + σ_W² + σ_L², μ_W ← μ_W + σ_W²/c · v((μ_W − μ_L)/c, ε/c), σ_W² ← σ_W² (1 − σ_W²/c² · w(...)) und entsprechend für den Verlierer (mit − beim Mittel),
   v(t, ε) = N(t − ε)/Φ(t − ε), w(t, ε) = v (v + t − ε); mit Remis-Marge ε = 0 (die Demo kennt kein Remis).
   Zusätzlich das Modell (Abschnitt 3.1): Ein Sieg ist das Ereignis „Leistung des Siegers > Leistung des Verlierers“ mit Leistung ~ N(Stärke, β²). Folgerung (nicht Wortlaut der Quelle): Für eine
   einzelne Partie sind Mittel und Varianzen dieser Aktualisierung die exakten Momente der Posteriori (Leistungsdifferenz normalverteilt, bei 0 abgeschnitten); der Test rechnet sie mit scipy nach.

Anmerkung zu 1: Der Text „The Glicko system“ der Autorenseite ist nicht das Papier in Appl. Statist. 48 (3); das Papier selbst (Volltext glicko.net/research/glicko.pdf, am 2026-10-17 gelesen) enthält
dieselben Formeln für g, E und die Aktualisierung (Gleichungen (7), (9), (10)), aber weder das Rechenbeispiel noch die Deckelung bei 350.

Nicht eingesehen bzw. nicht verwertbar (kein Test, siehe Bericht „offen“): Zahlenwerte aus Dangauthier et al. (2007), TrueSkill Through Time: Das Papier (NIPS 2007, 8 Seiten, PDF eingesehen) enthält
nur Ergebnisse auf dem Schachdatensatz (Log-Evidenz, Parameter 480/60), kein kleines nachrechenbares Beispiel; Coulom (2008) nennt Vorhersageraten auf KGS-Daten (Tabelle 1), die ohne die Daten
nicht reproduzierbar sind (die eingesehene Fassung ist die Autorenfassung mit „Draft“ im Kopf, nicht die LNCS-Druckfassung).
"""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_filter as F
import whr_smoother as S
from conftest import build


# --- Glickman 1999 ------------------------------------------------------------------------------------------------------------------------------

def _th(points):
    return (points - 1500.0) / C.SCALE


def test_scale_is_the_q_of_the_glicko_paper():
    assert 1.0 / C.SCALE == pytest.approx(0.0057565, abs=5e-8)             # Seite 3: q = ln 10 / 400 = 0,0057565


def test_glicko_worked_example_of_the_paper():
    opp = [(1400, 30, 1.0), (1550, 100, 0.0), (1700, 300, 0.0)]
    g_ref, e_ref = (0.9955, 0.9531, 0.7242), (0.639, 0.432, 0.303)
    mu, phi = _th(1500.0), 200.0 / C.SCALE
    games = []
    for (r, rd, s), g_p, e_p in zip(opp, g_ref, e_ref):
        g = F.g_factor(rd / C.SCALE)
        e = 1.0 / (1.0 + math.exp(-g * (mu - _th(r))))
        assert g == pytest.approx(g_p, abs=5e-5)
        assert e == pytest.approx(e_p, abs=5e-4)
        games.append((_th(r), rd / C.SCALE, s))
    new_mu, new_phi = F.glicko_update(mu, phi, games)
    assert new_mu * C.SCALE + 1500.0 == pytest.approx(1464.0, abs=0.5)     # Papier: 1500 − 36 = 1464 (gerundet)
    assert new_phi * C.SCALE == pytest.approx(151.4, abs=0.05)
    d2 = 1.0 / (sum(g * g * e * (1 - e) for g, e in zip(g_ref, e_ref)) * (1.0 / C.SCALE) ** 2)
    assert d2 == pytest.approx(53670.85, rel=2e-3)                         # d² mit den im Papier gerundeten g und E


def test_glicko_uncertainty_growth_with_the_papers_c():
    """Seite 6: RD 50, c = 63,2, 30 Perioden ohne Spiel: RD ≈ 350 (Obergrenze 350)."""
    phi, c, cap = 50.0 / C.SCALE, 63.2 / C.SCALE, 350.0 / C.SCALE
    for _ in range(30):
        phi = F.inflate(phi, c, cap)
    assert phi * C.SCALE == pytest.approx(350.0, abs=0.5)
    assert F.inflate(349.0 / C.SCALE, c, cap) == pytest.approx(cap)         # nie über der RD einer Unbewerteten


# --- Elo-Skala (Coulom 2.1/2.2, Wikipedia) ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("diff,expected,tol", [(100.0, 0.64, 0.005), (200.0, 0.76, 0.005), (400.0, 10.0 / 11.0, 1e-9)])
def test_elo_expected_score_scale(diff, expected, tol):
    """Wikipedia: 100 Punkte → 64 %, 200 → 76 %; Coulom/Elo: 400 Punkte = Faktor 10 der Gewinnchancen (γ = 10^(R/400), P = γ_i/(γ_i + γ_j))."""
    p = F.expected_score(diff / C.SCALE, 0.0)
    assert p == pytest.approx(expected, abs=tol)
    assert float(S.sigmoid(diff / C.SCALE)) == pytest.approx(expected, abs=tol)
    gamma_i, gamma_j = 10.0 ** (diff / 400.0), 1.0
    assert p == pytest.approx(gamma_i / (gamma_i + gamma_j), abs=1e-12)


def test_elo_wikipedia_1800_against_1400():
    assert F.expected_score(_th(1800.0), _th(1400.0)) == pytest.approx(0.91, abs=0.005)


# --- Coulom 2008, Anhänge ------------------------------------------------------------------------------------------------------------------------

def test_coulom_appendix_a1_gradient_and_curvature_of_the_bradley_terry_terms():
    """Eine Runde, kein Prior, kein Ridge, Drift ohne Wirkung (T = 1): Gradient = |W(i)| − Σ_j γ_i/(γ_i + γ_j); Krümmung = −Σ_j γ_i γ_j/(γ_i + γ_j)² (Handrechnung der Formel)."""
    games = [[(0, 1, True), (0, 2, False), (0, 2, True), (1, 2, True), (3, 0, True)]]
    m = build(4, 1, games)
    x = np.array([[0.4, -0.3, 0.9, 0.1]])
    gam = np.exp(x[0])
    wins = np.zeros(4)
    sums = np.zeros(4)
    curv = np.zeros(4)
    for i, j, y in games[0]:
        wins[i if y else j] += 1
        for a, b in ((i, j), (j, i)):
            sums[a] += gam[a] / (gam[a] + gam[b])
            curv[a] -= gam[a] * gam[b] / (gam[a] + gam[b]) ** 2
    ref = wins - sums
    g = S.whr_gradient(m, x, 0.3, None, ridge=0.0)[0]
    assert g == pytest.approx(ref, abs=1e-12)
    for i in range(4):                                                     # Krümmung: zentrale Differenz des Gradienten
        h = 1e-5
        xp, xm = x.copy(), x.copy()
        xp[0, i] += h
        xm[0, i] -= h
        d2 = (S.whr_gradient(m, xp, 0.3, None, 0.0)[0, i] - S.whr_gradient(m, xm, 0.3, None, 0.0)[0, i]) / (2 * h)
        assert d2 == pytest.approx(curv[i], abs=1e-7)


def test_coulom_appendix_a2_gradient_and_curvature_of_the_wiener_prior():
    """Ohne Partien, ohne Prior, ohne Ridge bleibt nur der Wiener-Term; σ² = w² je Runde. Jedes Rundenpaar liefert −(r(t₁) − r(t₂))/σ² bzw. −(r(t₂) − r(t₁))/σ²."""
    m = build(2, 4, [[], [], [], []])
    w = 0.37
    sig2 = w ** 2
    x = np.array([[0.5, -1.0], [0.1, 0.4], [-0.7, 0.9], [0.3, -0.2]])
    g = S.whr_gradient(m, x, w, None, ridge=0.0)
    ref = np.zeros_like(x)
    for t in range(3):
        ref[t] += -(x[t] - x[t + 1]) / sig2
        ref[t + 1] += -(x[t + 1] - x[t]) / sig2
    assert g == pytest.approx(ref, abs=1e-10)
    h = 1e-4
    for t, diag in ((0, -1 / sig2), (1, -2 / sig2), (3, -1 / sig2)):       # Krümmung: −1/σ² je angrenzendem Rundenpaar
        xp, xm = x.copy(), x.copy()
        xp[t, 0] += h
        xm[t, 0] -= h
        assert (S.whr_gradient(m, xp, w, None, 0.0)[t, 0] - S.whr_gradient(m, xm, w, None, 0.0)[t, 0]) / (2 * h) == pytest.approx(diag, rel=1e-5)
    xp, xm = x.copy(), x.copy()
    xp[1, 0] += h
    xm[1, 0] -= h
    cross = (S.whr_gradient(m, xp, w, None, 0.0)[2, 0] - S.whr_gradient(m, xm, w, None, 0.0)[2, 0]) / (2 * h)
    assert cross == pytest.approx(1 / sig2, rel=1e-5)                      # Kreuzterm +1/σ²


@pytest.mark.parametrize("t1,t2", [(3, 9), (2, 6), (5, 12)])
def test_coulom_appendix_c_mean_is_linear_between_two_times_without_games(t1, t2):
    """Spieler, die zwischen t₁ und t₂ nicht spielen: Der geschätzte Verlauf liegt dort auf der Geraden zwischen den Werten bei t₁ und t₂ (Mittelwert der Interpolation, Anhang C)."""
    rounds = t2 + 4
    rng = np.random.default_rng(t1 * 31 + t2)
    games = [[] for _ in range(rounds)]
    for t in list(range(0, t1 + 1)) + list(range(t2, rounds)):
        games[t] = [(0, 1, bool(rng.random() < 0.7)), (2, 3, bool(rng.random() < 0.5))]
    m = build(4, rounds, games)
    x = S.whr_fit(m, 0.2, 1.0)[0]
    for t in range(t1 + 1, t2):
        lin = (x[t1] * (t2 - t) + x[t2] * (t - t1)) / (t2 - t1)
        assert x[t] == pytest.approx(lin, abs=1e-4)
    assert np.abs(x[t1] - x[t2]).max() > 0.05                              # die Probe ist nicht trivial (Verlauf nicht konstant)


# --- Dangauthier et al. 2007, Abschnitt 3.1 ---------------------------------------------------------------------------------------------------

def _phi_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _phi_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _paper_vanilla_trueskill(mu_w, var_w, mu_l, var_l, beta, eps=0.0):
    """Die Gleichungen des Abschnitts 3.1 wörtlich: c² = 2β² + σ_W² + σ_L², v(t, ε) = N(t − ε)/Φ(t − ε), w = v (v + t − ε)."""
    c2 = 2.0 * beta ** 2 + var_w + var_l
    c = math.sqrt(c2)
    t, e = (mu_w - mu_l) / c, eps / c
    v = _phi_pdf(t - e) / _phi_cdf(t - e)
    w = v * (v + t - e)
    return (mu_w + var_w / c * v, var_w * (1.0 - var_w / c2 * w), mu_l - var_l / c * v, var_l * (1.0 - var_l / c2 * w))


@pytest.mark.parametrize("mu_w,var_w,mu_l,var_l,beta", [(0.0, 1.0, 0.0, 1.0, 1.0), (0.8, 0.5, -0.3, 1.7, 0.6), (-1.2, 2.0, 1.1, 0.4, 1.2), (2.5, 0.3, 0.1, 0.9, 0.9)])
def test_dangauthier_vanilla_trueskill_update_equations(mu_w, var_w, mu_l, var_l, beta):
    got = S.probit_update(np.array([mu_w]), np.array([var_w]), np.array([mu_l]), np.array([var_l]), beta=beta)
    assert [float(g[0]) for g in got] == pytest.approx(list(_paper_vanilla_trueskill(mu_w, var_w, mu_l, var_l, beta)), rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("mu_w,var_w,mu_l,var_l,beta", [(0.0, 1.0, 0.0, 1.0, 1.0), (0.8, 0.5, -0.3, 1.7, 0.6), (-1.2, 2.0, 1.1, 0.4, 1.2)])
def test_one_game_update_gives_the_exact_posterior_moments_of_the_papers_model(mu_w, var_w, mu_l, var_l, beta):
    """Leistung p ~ N(Stärke, β²), Sieg = {p_W − p_L > 0}: Die Leistungsdifferenz D ~ N(m, c²) wird bei 0 abgeschnitten; die Stärken sind linear-gaußsch an D gekoppelt (Momente exakt)."""
    truncnorm = pytest.importorskip("scipy.stats").truncnorm
    m, c2 = mu_w - mu_l, 2.0 * beta ** 2 + var_w + var_l
    c = math.sqrt(c2)
    a = -m / c
    mean_d = m + c * float(truncnorm.mean(a, math.inf))
    var_d = c2 * float(truncnorm.var(a, math.inf))
    got = S.probit_update(np.array([mu_w]), np.array([var_w]), np.array([mu_l]), np.array([var_l]), beta=beta)
    for mu, var, sign, new_mu, new_var in ((mu_w, var_w, 1.0, got[0][0], got[1][0]), (mu_l, var_l, -1.0, got[2][0], got[3][0])):
        k = var / c2                                                           # s = µ + sign·k (D − m) + unabhängiges Rauschen der Varianz σ² − k² c²
        assert new_mu == pytest.approx(mu + sign * k * (mean_d - m), abs=1e-9)
        assert new_var == pytest.approx(var - k * k * c2 + k * k * var_d, abs=1e-9)
