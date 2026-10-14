"""Glicko: Gewicht, Unsicherheitswachstum, Aktualisierung (Beispiel aus Glickman 1999), Lauf über Partienfolgen."""

import math

import numpy as np
import pytest

import whr_constants as C
import whr_filter as G
from conftest import build

S = C.SCALE


def test_g_factor_matches_the_values_of_the_paper():
    # Glickman 1999, Tabelle des Beispiels: RD 30, 100, 300 -> g = 0,9955, 0,9531, 0,7242
    assert G.g_factor(30 / S) == pytest.approx(0.9955, abs=5e-5)
    assert G.g_factor(100 / S) == pytest.approx(0.9531, abs=5e-5)
    assert G.g_factor(300 / S) == pytest.approx(0.7242, abs=5e-5)
    assert G.g_factor(0.0) == 1.0


def test_the_worked_example_of_the_paper():
    """1500 mit RD 200 gegen 1400 (RD 30, Sieg), 1550 (RD 100, Niederlage), 1700 (RD 300, Niederlage): Wertung 1464, RD 151,4."""
    games = [((1400 - 1500) / S, 30 / S, 1.0), ((1550 - 1500) / S, 100 / S, 0.0), ((1700 - 1500) / S, 300 / S, 0.0)]
    mu, phi = G.glicko_update(0.0, 200 / S, games)
    assert 1500 + S * mu == pytest.approx(1464.1, abs=0.06)
    assert S * phi == pytest.approx(151.4, abs=0.06)


def test_expected_scores_of_the_example_of_the_paper():
    # E = 0,639, 0,432, 0,303 nach den Spalten der Tabelle
    for r_j, rd_j, e in ((1400, 30, 0.639), (1550, 100, 0.432), (1700, 300, 0.303)):
        g = G.g_factor(rd_j / S)
        assert 1.0 / (1.0 + math.exp(-g * (0.0 - (r_j - 1500) / S))) == pytest.approx(e, abs=5e-4)


def test_no_games_leaves_the_rating_and_the_uncertainty_alone():
    assert G.glicko_update(0.3, 0.7, []) == (0.3, 0.7)


def test_the_uncertainty_shrinks_with_every_game_and_a_win_raises_the_rating():
    mu, phi = G.glicko_update(0.0, 1.0, [(0.0, 1.0, 1.0)])
    assert phi < 1.0 and mu > 0.0
    mu2, phi2 = G.glicko_update(0.0, 1.0, [(0.0, 1.0, 0.0)])
    assert phi2 == pytest.approx(phi) and mu2 == pytest.approx(-mu)           # Niederlage ist das Spiegelbild


def test_two_equal_new_players_by_hand():
    phi0 = 350 / S
    g = 1.0 / math.sqrt(1.0 + 3.0 * phi0 ** 2 / math.pi ** 2)
    phi2 = 1.0 / (1.0 / phi0 ** 2 + g * g * 0.25)                             # E = 1/2
    mu, phi = G.glicko_update(0.0, phi0, [(0.0, phi0, 1.0)])
    assert phi == pytest.approx(math.sqrt(phi2)) and mu == pytest.approx(phi2 * g * 0.5)


def test_an_uncertain_opponent_counts_less():
    big, _ = G.glicko_update(0.0, 1.0, [(0.0, 0.1, 1.0)])
    small, _ = G.glicko_update(0.0, 1.0, [(0.0, 2.0, 1.0)])
    assert big > small


def test_inflation_grows_by_c_and_stops_at_the_cap():
    assert G.inflate(0.5, 0.1, 2.0) == pytest.approx(math.sqrt(0.26))
    assert G.inflate(1.99, 0.5, 2.0) == 2.0
    assert G.inflate(2.0, 0.0, 2.0) == 2.0


def test_run_grows_the_uncertainty_of_a_player_who_does_not_play_and_caps_it():
    # Spieler 2 spielt nie; 0 und 1 spielen jede Runde
    m = build(3, 30, [[(0, 1, True)] for _ in range(30)])
    r = G.run_glicko(m, c=0.1, phi0=1.0)
    assert r.sd[0, 2] == pytest.approx(1.0)                                    # φ₀ ist schon die Obergrenze
    assert (r.sd[:, 0] < 1.0).all()                                            # wer spielt, wird sicherer als ein Unbewerteter
    assert 0.6 < r.sd[-1, 0] < 0.8 and abs(r.sd[-1, 0] - r.sd[-2, 0]) < 0.005  # und nähert sich dem Gleichgewicht aus Messung und Wachstum
    r2 = G.run_glicko(m, c=0.1, phi0=1.0)
    assert np.array_equal(r.mu, r2.mu)


def test_run_uses_the_pre_period_values_of_both_players():
    m = build(2, 1, [[(0, 1, True)]])
    r = G.run_glicko(m, c=0.0, phi0=1.0)
    g = G.g_factor(1.0)
    phi2 = 1.0 / (1.0 + g * g * 0.25)
    assert r.mu[0, 0] == pytest.approx(phi2 * g * 0.5) and r.mu[0, 1] == pytest.approx(-phi2 * g * 0.5)
    assert r.sd[0, 0] == pytest.approx(math.sqrt(phi2)) and r.sd[0, 1] == pytest.approx(math.sqrt(phi2))


def test_a_newcomer_enters_at_the_field_mean_with_the_full_uncertainty():
    games = [[(1, 2, True), (3, 4, False)] for _ in range(5)] + [[(0, 1, True), (2, 3, True)]]
    m = build(5, 6, games, join_round=5)
    r = G.run_glicko(m, c=0.05, phi0=2.0)
    assert np.isnan(r.mu[:5, 0]).all() and np.isnan(r.sd[:5, 0]).all()
    assert not np.isnan(r.mu[5, 0])
    assert r.sd[5, 0] > r.sd[5, 1]                                             # der Neuling ist unsicherer als die Eingespielten
    assert r.mu[5, 0] > 0.0                                                    # hat gewonnen, Start beim Mittel des Feldes
