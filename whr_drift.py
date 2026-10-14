"""Kopie von glt_drift.py (glicko-trueskill-demo), unverändert bis auf den Namen: dieselben Partien für dieselben Seeds wie im Vorgängerstück.

Partien mit wandernden Stärken, ungleicher Aktivität, Neuling und Pause (lokal; der gemeinsame Generator liefert nur die Anfangsstärken).

Die Stärken starten bei turnier_generator.make_strengths (Mittel 0) und folgen einer Irrfahrt mit Schritt σ je Runde. In jeder Runde spielen die aktiven Spieler in
zufälliger Paarung (bei ungerader Zahl setzt einer aus); P(i schlägt j) = σ(θ_i(t) − θ_j(t)). Aktiv heißt: Spieler mit gerader Nummer immer, mit ungerader mit
Wahrscheinlichkeit `p_rare` (nur im Szenario „Ungleiche Aktivität“). Spieler 0 fehlt vor `join_round` (Neuling, Startstärke Feldmittel + `gap`) bzw. in der Pause
[`pause_start`, `pause_start + pause_len`). Zufall nur über SplitMix64; Normalverteilung per Box-Muller."""

from dataclasses import dataclass
from functools import cached_property

import numpy as np

import turnier_generator as T

_SALT = 0x61C0


@dataclass(frozen=True)
class Matches:
    n: int
    truth: np.ndarray          # (Runden, n): wahre Stärken während der Runde
    rnd: np.ndarray            # Runde je Partie
    i: np.ndarray              # Spieler i
    j: np.ndarray              # Spieler j
    win: np.ndarray            # 1, wenn i gewinnt
    join_round: int            # 0 = kein Neuling; sonst kommt Spieler 0 in dieser Runde
    pause: tuple               # (Start, Ende) der Pause von Spieler 0 (Ende exklusiv), (0, 0) = keine
    rare: tuple                # Nummern der seltenen Spieler (leer, wenn alle gleich oft spielen)

    @property
    def rounds(self) -> int:
        return len(self.truth)

    @cached_property
    def presence(self) -> np.ndarray:
        """(Runden, n) bool: wer in welcher Runde zum Feld gehört (alle bis auf Spieler 0 vor dem Eintritt bzw. in der Pause)."""
        out = np.ones((self.rounds, self.n), dtype=bool)
        if self.join_round:
            out[:self.join_round, 0] = False
        if self.pause[1] > self.pause[0]:
            out[self.pause[0]:self.pause[1], 0] = False
        return out

    def present(self, player: int, t: int) -> bool:
        """Ob der Spieler in Runde t zum Feld gehört (nicht vor dem Eintritt, nicht in der Pause)."""
        if player != 0:
            return True
        if self.join_round and t < self.join_round:
            return False
        return not (self.pause[0] <= t < self.pause[1])


@dataclass(frozen=True)
class Run:
    """Ergebnis eines Verfahrens auf einer Partienfolge, in θ-Einheiten."""
    mu: np.ndarray             # (Runden, n): Wertung nach der Runde, NaN solange der Spieler nicht zum Feld gehört (vor dem Neuling-Eintritt)
    sd: object                 # (Runden, n): Unsicherheit (Glicko φ, TrueSkill σ) oder None bei Elo
    final: np.ndarray          # Wertung nach der letzten Runde
    mean_k: float              # mittlere effektive Verstärkung je Aktualisierung in Elo-Punkten (|Änderung| / |Ergebnis − Erwartung|)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


def normals(rng: T.SplitMix64, size: int) -> np.ndarray:
    """Standardnormalverteilte Zahlen per Box-Muller aus der Gleichverteilten des Generators."""
    u1 = 1.0 - rng.uniform(size)
    u2 = rng.uniform(size)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def simulate_matches(n: int, delta: float, rounds: int, sigma: float, p_rare: float, join_round: int, gap: float, pause_start: int, pause_len: int,
                     seed: int) -> Matches:
    """Deterministisch aus den Eingaben. `p_rare` = 1 und `join_round`, `pause_len` = 0 heißen: alle spielen jede Runde."""
    if join_round and not (0 < join_round < rounds):
        raise ValueError("Der Neuling muss vor dem Ende der Runden kommen")
    if join_round and pause_len:
        raise ValueError("Neuling und Pause schließen einander aus")
    if not 0.0 < p_rare <= 1.0:
        raise ValueError("p_rare muss in (0, 1] liegen")
    s = T.make_strengths(n, delta)
    theta = np.array(s, dtype=float)                                   # make_strengths liefert gleichabständige Stärken mit Mittel 0
    if join_round:
        theta[0] = theta[1:].mean() + gap
    rng = T.SplitMix64(seed ^ _SALT)
    p = np.ones(n)
    rare = tuple(range(1, n, 2)) if p_rare < 1.0 else ()
    for r in rare:
        p[r] = p_rare
    pause = (int(pause_start), int(pause_start) + int(pause_len)) if pause_len else (0, 0)
    mm = Matches(n, np.zeros((rounds, n)), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int),
                 int(join_round), pause, rare)
    truth = np.zeros((rounds, n))
    R, I, J, W = [], [], [], []
    for t in range(rounds):
        truth[t] = theta
        u_act = rng.uniform(n)
        active = np.array([q for q in range(n) if u_act[q] < p[q] and mm.present(q, t)], dtype=int)
        order = active[rng.permutation(len(active))] if len(active) else active
        m = len(order) // 2
        u_win = rng.uniform(max(m, 1))[:m]
        if m:
            a, b = order[:2 * m:2], order[1:2 * m:2]
            win = u_win < sigmoid(theta[a] - theta[b])
            R.extend([t] * m)
            I.extend(a.tolist())
            J.extend(b.tolist())
            W.extend(win.astype(int).tolist())
        if sigma > 0:
            theta = theta + sigma * normals(rng, n)
    return Matches(n, truth, np.array(R, dtype=int), np.array(I, dtype=int), np.array(J, dtype=int), np.array(W, dtype=int), int(join_round), pause, rare)
