"""Partienerzeugung: Determinismus, Aktivität, Neuling, Pause, Anwesenheit, Eingabeprüfung."""

import numpy as np
import pytest

import whr_drift as D


def sim(**kw):
    args = dict(n=8, delta=0.3, rounds=60, sigma=0.06, p_rare=1.0, join_round=0, gap=0.0, pause_start=0, pause_len=0, seed=3)
    args.update(kw)
    return D.simulate_matches(args["n"], args["delta"], args["rounds"], args["sigma"], args["p_rare"], args["join_round"], args["gap"], args["pause_start"],
                              args["pause_len"], args["seed"])


def test_same_inputs_same_matches_and_other_seed_differs():
    a, b, c = sim(), sim(), sim(seed=4)
    assert np.array_equal(a.i, b.i) and np.array_equal(a.win, b.win) and np.array_equal(a.truth, b.truth)
    assert not (len(a.i) == len(c.i) and np.array_equal(a.win, c.win))


def test_everybody_plays_every_round_without_a_scenario():
    m = sim()
    assert len(m.i) == 60 * 4 and m.rare == ()
    for t in range(60):
        players = np.concatenate([m.i[m.rnd == t], m.j[m.rnd == t]])
        assert sorted(players.tolist()) == list(range(8))


def test_a_player_never_plays_himself_and_each_plays_at_most_once_per_round():
    m = sim(p_rare=0.3)
    assert (m.i != m.j).all()
    for t in range(60):
        players = np.concatenate([m.i[m.rnd == t], m.j[m.rnd == t]])
        assert len(players) == len(set(players.tolist()))


def test_the_rare_half_plays_less_often():
    m = sim(p_rare=0.2, rounds=300)
    games = np.zeros(8, dtype=int)
    np.add.at(games, m.i, 1)
    np.add.at(games, m.j, 1)
    assert m.rare == (1, 3, 5, 7)
    assert games[list(m.rare)].mean() < 0.5 * games[[0, 2, 4, 6]].mean()


def test_the_strengths_start_equidistant_and_drift_only_with_sigma():
    still = sim(sigma=0.0)
    assert np.allclose(still.truth, still.truth[0])
    assert np.allclose(np.diff(still.truth[0]), 0.3)
    moving = sim(sigma=0.1)
    assert not np.allclose(moving.truth[-1], moving.truth[0])


def test_the_newcomer_starts_at_the_field_mean_plus_the_gap_and_plays_only_after_joining():
    m = sim(join_round=20, gap=1.5, sigma=0.0)
    assert m.truth[0][0] == pytest.approx(m.truth[0][1:].mean() + 1.5)
    assert not ((m.i == 0) | (m.j == 0))[m.rnd < 20].any()
    assert ((m.i == 0) | (m.j == 0))[m.rnd >= 20].any()
    assert not m.present(0, 19) and m.present(0, 20) and m.present(3, 0)


def test_the_pause_removes_player_zero_for_exactly_the_given_rounds():
    m = sim(pause_start=10, pause_len=15)
    played = ((m.i == 0) | (m.j == 0))
    assert not played[(m.rnd >= 10) & (m.rnd < 25)].any()
    assert played[m.rnd < 10].any() and played[m.rnd >= 25].any()
    assert m.pause == (10, 25) and not m.presence[10:25, 0].any() and m.presence[:10, 0].all() and m.presence[25:, 0].all()


def test_invalid_scenarios_are_rejected():
    with pytest.raises(ValueError):
        sim(join_round=60)
    with pytest.raises(ValueError):
        sim(join_round=10, pause_start=20, pause_len=10)
    with pytest.raises(ValueError):
        sim(p_rare=0.0)
    with pytest.raises(ValueError):
        sim(p_rare=1.5)


def test_normals_are_standard_normal():
    z = D.normals(D.T.SplitMix64(5), 20000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1.0) < 0.03


def test_the_newcomer_may_join_in_the_second_round_but_not_before_or_after_the_run():
    m = sim(join_round=1, rounds=10, sigma=0.0)
    assert m.join_round == 1 and not m.present(0, 0) and m.present(0, 1)
    for bad in (-1, 60):
        with pytest.raises(ValueError):
            sim(join_round=bad)


def test_the_strengths_have_mean_zero_without_a_newcomer():
    m = sim(sigma=0.0)
    assert m.truth[0].mean() == pytest.approx(0.0, abs=1e-12)
