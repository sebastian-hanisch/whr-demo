"""Szenario: wirksame Werte der Regler (Begrenzungen, Szenario-Abhängigkeit) und die daraus erzeugten Partien."""

import numpy as np
import pytest

import whr_constants as C
from whr_scenario import Settings, make_matches


def test_rare_activity_applies_only_in_its_scenario():
    assert Settings(scenario=C.SCENARIOS[1], p_rare=0.3).p_rare_eff == 0.3
    for sc in (C.SCENARIOS[0], C.SCENARIOS[2], C.SCENARIOS[3]):
        assert Settings(scenario=sc, p_rare=0.3).p_rare_eff == 1.0


def test_join_round_is_zero_outside_the_newcomer_scenario():
    assert Settings(scenario=C.SCENARIOS[0], join=30).join_round == 0 and Settings(scenario=C.SCENARIOS[3], join=30).join_round == 0


def test_join_round_is_clamped_into_the_run():
    base = dict(scenario=C.SCENARIOS[2], rounds=100)
    assert Settings(join=30, **base).join_round == 30
    assert Settings(join=0, **base).join_round == C.JOIN_MIN
    assert Settings(join=1, **base).join_round == C.JOIN_MIN
    assert Settings(join=500, **base).join_round == 98


def test_pause_is_empty_outside_its_scenario():
    assert Settings(scenario=C.SCENARIOS[0], pause_start=40, pause_len=30).pause == (0, 0)


def test_pause_is_clamped_to_leave_room_before_and_after():
    base = dict(scenario=C.SCENARIOS[3], rounds=100)
    assert Settings(pause_start=40, pause_len=30, **base).pause == (40, 30)
    start, length = Settings(pause_start=1, pause_len=1, **base).pause
    assert (start, length) == (C.PAUSE_START_MIN, C.PAUSE_LEN_MIN)
    start, length = Settings(pause_start=500, pause_len=500, **base).pause
    assert start == C.pause_start_max(100) == 100 - C.PAUSE_LEN_MIN - 6
    assert length == C.pause_len_max(100, start) and start + length <= 100 - 5


def test_pause_length_is_capped_at_its_maximum():
    s = Settings(scenario=C.SCENARIOS[3], rounds=200, pause_start=10, pause_len=190)
    assert s.pause == (10, C.PAUSE_LEN_MAX)


def test_drift_assumption_is_a_multiple_of_the_true_drift_with_a_floor():
    assert Settings(sigma=0.06, w_factor=2.0).w == pytest.approx(0.12)
    assert Settings(sigma=0.06, w_factor=0.5).w == pytest.approx(0.03)
    assert Settings(sigma=0.06, w_factor=0.0).w == C.W_MIN
    assert Settings(sigma=0.0, w_factor=1.0).w == C.W_MIN


def test_c_is_converted_from_points_to_theta_units():
    assert Settings(c=C.SCALE).c_theta == pytest.approx(1.0)


def test_make_matches_is_deterministic_and_follows_the_settings():
    s = Settings(n=6, rounds=40, scenario=C.SCENARIOS[2], join=20, gap=1.5, seed=9)
    a, b = make_matches(s), make_matches(s)
    assert np.array_equal(a.rnd, b.rnd) and np.array_equal(a.win, b.win)
    assert a.join_round == 20 and a.n == 6 and a.rounds == 40
    assert make_matches(Settings(n=6, rounds=40, seed=10)).win.tolist() != make_matches(Settings(n=6, rounds=40, seed=11)).win.tolist()
    p = make_matches(Settings(n=6, rounds=40, scenario=C.SCENARIOS[3], pause_start=10, pause_len=15))
    assert p.pause == (10, 25)
