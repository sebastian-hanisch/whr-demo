"""Gemeinsamer Turnier-Generator der Ranking-Linie (WORTGLEICHE Kopie in allen Repos der Linie, kein Modul-Präfix).

Version 1 (2026-10-11). Nur numpy, kein Streamlit. Änderungen an diesem Modul betreffen alle Repos der Linie: Schnittstelle, Namen und
Reihenfolge des Zufallsverbrauchs nur ändern, wenn alle Kopien nachgezogen werden (tools/portfolio_audit.py, SHARED_GROUPS).

Modell: n Teilnehmer mit latenter Stärke s_i (Index aufsteigend = stärker). Eine Partie i gegen j gewinnt i mit
P_ij = clip(sigmoid(s_i - s_j) + K_ij, 0,02; 0,98); K_ij ist eine zyklische Störung über drei Spielstile (Stil a schlägt Stil a-1).
Beobachtet werden nur Ausgänge (Sieg/Niederlage). Zwei Zielgrößen: die Generator-Stärke s und die erwartete Siegquote
e_i = Summe_j P_ij / (n - 1) gegen ein gleichverteiltes Feld.

Zufall nur über einen übergebenen `SplitMix64`; Verbrauch je Turnier in fester Reihenfolge: Stile, Namen, Spielplan, Partien.
"""

from dataclasses import dataclass

import numpy as np

GENERATOR_VERSION = 1
P_MIN, P_MAX = 0.02, 0.98
LABELS = "ABCDEFGHIJKLMNOP"
PLANS = ("vollstaendig", "staffeln", "luecken")
STILE = ("ausgewogen", "konter", "beute")

_GAMMA = 0x9E3779B97F4A7C15
_MASK = (1 << 64) - 1
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB


class SplitMix64:
    """SplitMix64-Strom (Steele, Lea, Flood 2014). Ausgabe k hängt nur von (Seed, k) ab: plattform- und versionsunabhängig."""

    def __init__(self, seed: int):
        self.state = int(seed) & _MASK

    def u64(self, size: int) -> np.ndarray:
        """Die nächsten `size` 64-Bit-Werte; der Strom rückt um `size` Schritte weiter."""
        idx = np.arange(1, size + 1, dtype=np.uint64)
        with np.errstate(over="ignore"):
            z = np.uint64(self.state) + idx * np.uint64(_GAMMA)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(_M1)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(_M2)
            z = z ^ (z >> np.uint64(31))
        self.state = (self.state + size * _GAMMA) & _MASK
        return z

    def uniform(self, size: int) -> np.ndarray:
        """Gleichverteilte Werte in [0, 1) mit 53 Bit."""
        return (self.u64(size) >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def permutation(self, n: int) -> np.ndarray:
        return np.argsort(self.uniform(n), kind="stable")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_strengths(n: int, delta: float) -> np.ndarray:
    """Äquidistante Stärken, aufsteigend, Mittelwert 0: s_i = delta * (i - (n-1)/2)."""
    return delta * (np.arange(n) - (n - 1) / 2.0)


def style_assignment(n: int, mode: str, rng: SplitMix64) -> np.ndarray:
    """Stil (0, 1, 2) je Teilnehmer. `ausgewogen`: reihum. `konter`/`beute`: der Stärkste (Index n-1) hat Stil 0; fast alle anderen haben
    den Stil, der ihn schlägt (1, Konter) bzw. den, den er schlägt (2, Beute); drei zufällige andere weichen davon ab."""
    if mode == "ausgewogen":
        return np.arange(n) % 3
    if mode not in ("konter", "beute"):
        raise ValueError(f"unbekannte Stilverteilung: {mode}")
    main, alt = (1, 0) if mode == "konter" else (2, 1)
    styles = np.full(n, main, dtype=int)
    best = n - 1
    styles[best] = 0
    others = [i for i in rng.permutation(n) if i != best]
    for i in others[: min(3, max(n - 2, 0))]:
        styles[i] = alt
    return styles


def style_matrix(styles, c: float) -> np.ndarray:
    """K_ij = +c, wenn Stil_i den Stil_j schlägt ((Stil_i - Stil_j) mod 3 == 1), -c in Gegenrichtung, sonst 0."""
    st = np.asarray(styles)
    d = (st[:, None] - st[None, :]) % 3
    return np.where(d == 1, c, np.where(d == 2, -c, 0.0))


def win_probabilities(strengths, K=None) -> np.ndarray:
    """P[i, j] = Wahrscheinlichkeit, dass i gegen j gewinnt; Diagonale 0. Mit P_ij + P_ji = 1."""
    s = np.asarray(strengths, dtype=float)
    P = sigmoid(s[:, None] - s[None, :])
    if K is not None:
        P = np.clip(P + K, P_MIN, P_MAX)
    np.fill_diagonal(P, 0.0)
    return P


def expected_rate(P) -> np.ndarray:
    """Erwartete Siegquote gegen ein gleichverteiltes Feld der übrigen n - 1: e_i = Summe_j P_ij / (n - 1)."""
    n = len(P)
    return P.sum(axis=1) / (n - 1)


def schedule_complete(n: int, k: int) -> np.ndarray:
    """Jedes Paar spielt k Partien."""
    G = np.full((n, n), int(k), dtype=int)
    np.fill_diagonal(G, 0)
    return G


def schedule_staffeln(n: int, x: int, k: int, rng: SplitMix64) -> np.ndarray:
    """Zwei Staffeln: die schwächere Hälfte (Index < n//2) und die stärkere. Innerhalb jeder Staffel spielt jedes Paar; dazu spielt jeder
    Teilnehmer der starken Staffel gegen x zufällige der schwachen (x wird auf die Größe der schwachen Staffel begrenzt). Alles k-fach."""
    n_weak = n // 2
    weak = np.arange(n_weak)
    strong = np.arange(n_weak, n)
    G = np.zeros((n, n), dtype=int)
    for group in (weak, strong):
        for a in group:
            for b in group:
                if a < b:
                    G[a, b] = G[b, a] = 1
    x = max(0, min(int(x), n_weak))
    for a in strong:
        pick = weak[np.argsort(rng.uniform(n_weak), kind="stable")[:x]]
        for b in pick:
            G[a, b] += 1
            G[b, a] += 1
    return G * int(k)


def schedule_random(n: int, p: float, k: int, rng: SplitMix64) -> np.ndarray:
    """Jedes Paar wird mit Wahrscheinlichkeit p gespielt (dann k-fach), sonst gar nicht."""
    iu = np.triu_indices(n, 1)
    played = rng.uniform(len(iu[0])) < p
    G = np.zeros((n, n), dtype=int)
    G[iu[0][played], iu[1][played]] = int(k)
    return G + G.T


def play(G, P, rng: SplitMix64) -> np.ndarray:
    """W[i, j] = Siege von i gegen j (W[j, i] = G[i, j] - W[i, j]). Eine Gleichverteilte je Partie, Paare in Zeilenordnung i < j."""
    n = len(G)
    iu = np.triu_indices(n, 1)
    games = G[iu]
    total = int(games.sum())
    W = np.zeros((n, n), dtype=int)
    if total == 0:
        return W
    u = rng.uniform(total)
    pair_of_game = np.repeat(np.arange(len(games)), games)
    hit = u < P[iu][pair_of_game]
    wins = np.bincount(pair_of_game, weights=hit, minlength=len(games)).astype(int)
    W[iu] = wins
    W[iu[1], iu[0]] = games - wins
    return W


def components(G) -> int:
    """Zahl der Zusammenhangskomponenten des Spielgraphen (Kante = mindestens eine Partie)."""
    n = len(G)
    seen = [False] * n
    count = 0
    for start in range(n):
        if seen[start]:
            continue
        count += 1
        stack = [start]
        while stack:
            u = stack.pop()
            if seen[u]:
                continue
            seen[u] = True
            stack.extend(v for v in range(n) if G[u][v] > 0 and not seen[v])
    return count


@dataclass(frozen=True)
class Turnier:
    strengths: np.ndarray   # s_i, aufsteigend
    styles: np.ndarray
    P: np.ndarray           # Siegwahrscheinlichkeiten
    expected: np.ndarray    # erwartete Siegquote e_i
    games: np.ndarray       # G[i, j] gespielte Partien
    wins: np.ndarray        # W[i, j] Siege von i gegen j
    names: tuple            # Anzeigename je Index (zufällig vergeben, damit der Index nicht die Stärke verrät)

    @property
    def n(self) -> int:
        return len(self.strengths)


def build_turnier(n: int, delta: float, k: int, plan: str, x: int, p: float, c: float, stil: str, seed: int) -> Turnier:
    """Ein komplettes Turnier aus den Reglerwerten; deterministisch in (alle Argumente)."""
    if plan not in PLANS:
        raise ValueError(f"unbekannter Spielplan: {plan}")
    rng = SplitMix64(seed)
    s = make_strengths(n, delta)
    styles = style_assignment(n, stil, rng)
    K = style_matrix(styles, c) if c > 0 else None
    perm = rng.permutation(n)
    names = tuple(LABELS[int(i)] for i in perm)
    if plan == "vollstaendig":
        G = schedule_complete(n, k)
    elif plan == "staffeln":
        G = schedule_staffeln(n, x, k, rng)
    else:
        G = schedule_random(n, p, k, rng)
    P = win_probabilities(s, K)
    W = play(G, P, rng)
    return Turnier(s, styles, P, expected_rate(P), G, W, names)
