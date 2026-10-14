"""Einheitentests des gemeinsamen Generators (turnier_generator.py). Derselbe Satz Tests liegt in jedem Repo der Linie."""

import numpy as np
import pytest

import turnier_generator as T


# --- SplitMix64 ------------------------------------------------------------------------------------------------------------------------

def reference_splitmix64(seed, count):
    """Unabhängige Fassung mit Python-Ganzzahlen (Steele/Lea/Flood 2014), schrittweise."""
    mask = (1 << 64) - 1
    state = seed & mask
    out = []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        out.append(z ^ (z >> 31))
    return out


def test_splitmix64_matches_the_scalar_reference():
    for seed in (0, 1, 35, 123456789, (1 << 63) + 5, (1 << 64) - 1):
        ours = [int(v) for v in T.SplitMix64(seed).u64(500)]
        assert ours == reference_splitmix64(seed, 500), seed


def test_splitmix64_published_first_outputs_for_seed_zero():
    assert reference_splitmix64(0, 3) == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]
    assert [int(v) for v in T.SplitMix64(0).u64(3)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F]


def test_the_stream_is_continuous_across_calls():
    a = T.SplitMix64(9)
    first = np.concatenate([a.u64(3), a.u64(2)])
    assert [int(v) for v in first] == [int(v) for v in T.SplitMix64(9).u64(5)]


def test_uniform_is_in_unit_interval_and_roughly_uniform():
    u = T.SplitMix64(4).uniform(20000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01
    hist = np.histogram(u, bins=10, range=(0, 1))[0]
    assert hist.min() > 1700 and hist.max() < 2300


def test_permutation_is_a_permutation_and_varies_with_the_seed():
    perms = {tuple(T.SplitMix64(s).permutation(8)) for s in range(20)}
    assert all(sorted(p) == list(range(8)) for p in perms)
    assert len(perms) > 10


# --- Stärken, Stile, Wahrscheinlichkeiten ------------------------------------------------------------------------------------------------

def test_strengths_are_equidistant_ascending_and_centered():
    s = T.make_strengths(8, 0.3)
    assert s[0] == pytest.approx(-1.05) and s[-1] == pytest.approx(1.05)
    assert np.allclose(np.diff(s), 0.3)
    assert s.mean() == pytest.approx(0.0)


def test_style_matrix_is_antisymmetric_and_cyclic():
    K = T.style_matrix([0, 1, 2], 0.3)
    assert np.allclose(K, -K.T)
    assert K[1, 0] == 0.3 and K[2, 1] == 0.3 and K[0, 2] == 0.3      # 1 schlägt 0, 2 schlägt 1, 0 schlägt 2
    assert K[0, 0] == 0.0


def test_win_probabilities_sum_to_one_and_respect_the_clip():
    s = T.make_strengths(4, 1.2)                     # Stärkeabstand 3,6: sigmoid > 0,97
    K = T.style_matrix([0, 1, 2, 0], 0.3)
    P = T.win_probabilities(s, K)
    off = ~np.eye(4, dtype=bool)
    assert np.allclose((P + P.T)[off], 1.0)
    assert P[off].min() >= T.P_MIN and P[off].max() <= T.P_MAX
    assert P.max() == T.P_MAX                         # der Zuschlag treibt den Stärksten gegen den Schwächsten in die Begrenzung


def test_expected_rate_sums_to_n_over_two():
    for n in (3, 8, 12):
        P = T.win_probabilities(T.make_strengths(n, 0.3), T.style_matrix(np.arange(n) % 3, 0.2))
        e = T.expected_rate(P)
        assert e.sum() == pytest.approx(n / 2)


def test_style_assignment_modes():
    assert list(T.style_assignment(7, "ausgewogen", T.SplitMix64(1))) == [0, 1, 2, 0, 1, 2, 0]
    for mode, main, alt in (("konter", 1, 0), ("beute", 2, 1)):
        st = T.style_assignment(12, mode, T.SplitMix64(5))
        assert st[11] == 0                                          # der Stärkste hat immer Stil 0
        assert (st == main).sum() == 8 and (st[:11] == alt).sum() == 3
    with pytest.raises(ValueError):
        T.style_assignment(8, "wild", T.SplitMix64(1))


# --- Spielpläne --------------------------------------------------------------------------------------------------------------------------

def test_complete_schedule_is_symmetric_without_diagonal():
    G = T.schedule_complete(6, 3)
    assert (G == G.T).all() and (np.diag(G) == 0).all()
    assert G.sum() == 6 * 5 * 3


def test_staffeln_intra_complete_cross_count_and_k_scaling():
    n, x = 12, 2
    G = T.schedule_staffeln(n, x, 1, T.SplitMix64(8))
    weak, strong = np.arange(6), np.arange(6, 12)
    assert (G == G.T).all()
    for grp in (weak, strong):
        sub = G[np.ix_(grp, grp)]
        assert (sub[np.triu_indices(6, 1)] == 1).all()
    assert G[np.ix_(strong, weak)].sum(axis=1).tolist() == [x] * 6          # jeder Starke genau x Kreuzpartien
    assert (T.schedule_staffeln(n, x, 3, T.SplitMix64(8)) == 3 * G).all()
    assert T.components(T.schedule_staffeln(n, 0, 1, T.SplitMix64(8))) == 2
    assert T.schedule_staffeln(n, 99, 1, T.SplitMix64(8))[np.ix_(strong, weak)].sum() == 36   # x wird begrenzt: alle Kreuzpartien


def test_random_schedule_extremes_and_frequency():
    assert T.schedule_random(8, 1.0, 2, T.SplitMix64(1)).sum() == 8 * 7 * 2
    assert T.schedule_random(8, 0.0, 2, T.SplitMix64(1)).sum() == 0
    shares = [T.schedule_random(12, 0.4, 1, T.SplitMix64(s))[np.triu_indices(12, 1)].mean() for s in range(200)]
    assert np.mean(shares) == pytest.approx(0.4, abs=0.02)


# --- Partien und Turnier -----------------------------------------------------------------------------------------------------------------

def test_play_conserves_games_and_has_no_self_play():
    G = T.schedule_complete(6, 4)
    P = T.win_probabilities(T.make_strengths(6, 0.3))
    W = T.play(G, P, T.SplitMix64(3))
    assert ((W + W.T) == G).all()
    assert (np.diag(W) == 0).all() and (W >= 0).all()
    assert T.play(np.zeros((4, 4), dtype=int), np.full((4, 4), 0.5), T.SplitMix64(1)).sum() == 0


def test_play_frequencies_match_the_probabilities():
    """20 000 Partien je Paar: relative Häufigkeit im Vier-Sigma-Band um P_ij."""
    P = T.win_probabilities(T.make_strengths(4, 0.6), T.style_matrix([0, 1, 2, 1], 0.2))
    W = T.play(T.schedule_complete(4, 20000), P, T.SplitMix64(17))
    for i in range(4):
        for j in range(4):
            if i != j:
                sigma = (P[i, j] * (1 - P[i, j]) / 20000) ** 0.5
                assert abs(W[i, j] / 20000 - P[i, j]) < 4 * sigma, (i, j)


def test_build_turnier_is_deterministic_and_seed_sensitive():
    args = dict(n=10, delta=0.3, k=2, plan="staffeln", x=1, p=0.5, c=0.2, stil="konter")
    a = T.build_turnier(seed=5, **args)
    b = T.build_turnier(seed=5, **args)
    c = T.build_turnier(seed=6, **args)
    assert (a.wins == b.wins).all() and a.names == b.names
    assert not (a.wins == c.wins).all() or a.names != c.names


def test_build_turnier_names_are_distinct_labels_and_plan_is_validated():
    t = T.build_turnier(8, 0.3, 1, "vollstaendig", 1, 0.5, 0.0, "ausgewogen", 3)
    assert len(set(t.names)) == 8 and set(t.names) <= set(T.LABELS)
    assert t.n == 8
    with pytest.raises(ValueError):
        T.build_turnier(8, 0.3, 1, "chaos", 1, 0.5, 0.0, "ausgewogen", 3)
