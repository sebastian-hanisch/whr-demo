"""Elo: Erwartung, Aktualisierung, Anlauf-K, Nullsumme, Neuling."""

import numpy as np
import pytest

import whr_constants as C
import whr_filter as E
from conftest import build


def test_expected_score_is_symmetric_and_overflow_safe():
    assert E.expected_score(0.0, 0.0) == 0.5
    assert E.expected_score(1.0, 0.0) + E.expected_score(0.0, 1.0) == pytest.approx(1.0)
    assert E.expected_score(1.0e4, 0.0) == pytest.approx(1.0) and E.expected_score(-1.0e4, 0.0) == pytest.approx(0.0, abs=1e-300)


def test_400_points_are_a_factor_ten_in_the_odds():
    e = E.expected_score(400.0 / C.SCALE, 0.0)
    assert e / (1.0 - e) == pytest.approx(10.0)


def test_effective_k_is_fixed_or_decays_towards_k():
    assert E.effective_k(20.0, None, 0) == 20.0
    assert E.effective_k(20.0, 60.0, 0) == pytest.approx(60.0)
    assert E.effective_k(20.0, 60.0, 10.0) == pytest.approx(40.0)         # nach g0 = 10 Partien ist der Aufschlag halbiert
    assert E.effective_k(20.0, 60.0, 1.0e6) == pytest.approx(20.0, abs=1e-3)


def test_one_game_by_hand():
    k = 20.0 / C.SCALE
    r = E.run_elo(build(2, 1, [[(0, 1, True)]]), 20.0)
    assert r.mu[0, 0] == pytest.approx(k / 2) and r.mu[0, 1] == pytest.approx(-k / 2)
    assert r.mean_k == 20.0 and r.sd is None


def test_the_sum_of_the_ratings_is_conserved_with_a_fixed_k(two):
    r = E.run_elo(two, 32.0)
    assert np.allclose(r.mu.sum(axis=1), 0.0)


def test_a_provisional_k_breaks_the_zero_sum(two):
    r = E.run_elo(two, 20.0, K0=80.0)
    assert abs(r.mu[0].sum()) < 1e-12                     # erste Runde: beide Neulinge, gleiches k
    games = [[(0, 1, True), (2, 3, True)], [(0, 2, True), (1, 3, True)]] + [[(0, 1, True), (2, 3, False)] for _ in range(20)]
    m = build(5, 22, games + [], join_round=0)
    r2 = E.run_elo(m, 20.0, K0=80.0)
    assert r2.mean_k > 20.0 and r2.mean_k < 80.0


def test_the_newcomer_starts_at_the_field_mean():
    games = [[(1, 2, True), (3, 4, False)] for _ in range(4)] + [[(0, 1, False), (2, 3, True)]]
    m = build(5, 5, games, join_round=4)
    r = E.run_elo(m, 20.0)
    assert np.isnan(r.mu[:4, 0]).all() and not np.isnan(r.mu[4, 0])
    k = 20.0 / C.SCALE
    start = r.mu[3, 1:].mean()                             # Feld nach der Runde vor dem Eintritt
    e = E.expected_score(start, r.mu[3, 1])
    assert r.mu[4, 0] == pytest.approx(start + k * (0.0 - e))             # der Neuling verliert gegen Spieler 1
