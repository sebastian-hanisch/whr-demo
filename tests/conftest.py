"""Gemeinsame Fixtures: von Hand gebaute Partienfolgen, an denen jede Einheit nachgerechnet wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from whr_drift import Matches  # noqa: E402


def build(n, rounds, games, join_round=0, pause=(0, 0), truth=None, rare=()):
    """Partienfolge aus Hand. `games`: je Runde eine Liste (i, j, i_gewinnt); `truth`: (Runden, n) oder None (alle 0)."""
    rnd, ii, jj, ww = [], [], [], []
    for t, gl in enumerate(games):
        for i, j, w in gl:
            rnd.append(t)
            ii.append(i)
            jj.append(j)
            ww.append(int(w))
    tr = np.zeros((rounds, n)) if truth is None else np.asarray(truth, dtype=float)
    return Matches(n, tr, np.array(rnd, dtype=int), np.array(ii, dtype=int), np.array(jj, dtype=int), np.array(ww, dtype=int), join_round, tuple(pause), tuple(rare))


@pytest.fixture
def two():
    """Zwei Spieler, vier Runden: 0 gewinnt, 0 gewinnt, 1 gewinnt, 0 gewinnt."""
    return build(2, 4, [[(0, 1, True)], [(0, 1, True)], [(0, 1, False)], [(0, 1, True)]])
