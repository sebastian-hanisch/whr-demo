"""Die Glätter: Whole-History Rating (Coulom 2008) als Maximum a posteriori und TrueSkill Through Time (Dangauthier et al. 2007) als Erwartungsabgleich, beide in θ-Einheiten.

Modell (beide): Die Stärke x[t, i] von Spieler i in Runde t wandert als Wiener-Prozess, x[t+1, i] − x[t, i] ~ N(0, w²) (auch in Runden ohne Partie); die Anfangswertung
x[0, i] hat optional einen Normal-Prior N(0, s₀²). Jede Partie ist ein Paarvergleich: WHR mit der logistischen Wahrscheinlichkeit σ(x_i − x_j), TTT mit der Probit-Wahrscheinlichkeit
Φ((x_i − x_j)/(√2 β)).

WHR: x maximiert log L(x) − Σ (x[t+1] − x[t])²/(2w²) − Σ x[0, i]²/(2 s₀²) − ρ Σ x²/2 (ρ kleiner Ridge, fixiert das Niveau). Der Newton-Schritt löst ein linearen System mit
Blocktridiagonalmatrix (ein n×n-Block je Runde, außerhalb der Diagonale −q·I mit q = 1/w²), in O(T n³) statt O((T n)³).

TTT: Gauß-Nachrichten auf der Kette je Spieler; jede Partie ist ein Probit-Faktor, dessen Nachricht aus den Cavity-Verteilungen der beiden Spieler per Momentenabgleich folgt (die
TrueSkill-Formeln v und w). Vorwärts- und Rückwärtsdurchläufe wechseln mit der Aktualisierung der Partien, bis sich die Mittelwerte nicht mehr ändern."""

import math

import numpy as np

import whr_constants as C
from whr_drift import Matches, Run

SQ2 = math.sqrt(2.0)
SQ2PI = math.sqrt(2.0 * math.pi)


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=float)))


# --- WHR --------------------------------------------------------------------------------------------------------------------------------------

def effective_w(w: float) -> float:
    """Drift-Annahme mit Untergrenze: w = 0 (statisch) ist numerisch ein sehr steifes Problem."""
    return max(float(w), C.W_MIN)


def solve_block_tridiagonal(D: np.ndarray, q: float, r: np.ndarray) -> np.ndarray:
    """Löst A x = r mit A_tt = D[t] (n×n), A_{t,t+1} = A_{t+1,t} = −q·I. D: (T, n, n), r: (T, n). Blockweises Thomas-Verfahren."""
    T = D.shape[0]
    Dp = np.empty_like(D)
    rp = np.empty_like(r)
    Dp[0], rp[0] = D[0], r[0]
    for t in range(1, T):
        inv = np.linalg.inv(Dp[t - 1])
        Dp[t] = D[t] - q * q * inv
        rp[t] = r[t] + q * (inv @ rp[t - 1])
    x = np.empty_like(r)
    x[T - 1] = np.linalg.solve(Dp[T - 1], rp[T - 1])
    for t in range(T - 2, -1, -1):
        x[t] = np.linalg.solve(Dp[t], rp[t] + q * x[t + 1])
    return x


def whr_objective(m: Matches, x: np.ndarray, w: float, prior_sd=None, ridge: float = C.RIDGE) -> float:
    """log L(x) − Wiener-Prior − Prior auf die Anfangswertung − Ridge, für die Runden 0..len(x)−1."""
    T = len(x)
    sel = m.rnd < T
    R, I, J, Y = m.rnd[sel], m.i[sel], m.j[sel], m.win[sel].astype(float)
    d = x[R, I] - x[R, J]
    ll = float(np.sum(-np.logaddexp(0.0, -d) * Y + (-np.logaddexp(0.0, d)) * (1.0 - Y)))
    q = 1.0 / effective_w(w) ** 2
    pr = 0.5 * q * float(np.sum((x[1:] - x[:-1]) ** 2)) + 0.5 * ridge * float(np.sum(x ** 2))
    if prior_sd:
        pr += 0.5 * float(np.sum(x[0] ** 2)) / prior_sd ** 2
    return ll - pr


def whr_gradient(m: Matches, x: np.ndarray, w: float, prior_sd=None, ridge: float = C.RIDGE) -> np.ndarray:
    """Gradient von whr_objective nach x."""
    T = len(x)
    sel = m.rnd < T
    R, I, J, Y = m.rnd[sel], m.i[sel], m.j[sel], m.win[sel].astype(float)
    p = sigmoid(x[R, I] - x[R, J])
    g = np.zeros_like(x)
    np.add.at(g, (R, I), Y - p)
    np.add.at(g, (R, J), -(Y - p))
    q = 1.0 / effective_w(w) ** 2
    pg = ridge * x
    pg[1:] += q * (x[1:] - x[:-1])
    pg[:-1] += q * (x[:-1] - x[1:])
    if prior_sd:
        pg[0] += x[0] / prior_sd ** 2
    return g - pg


def whr_fit(m: Matches, w: float, prior_sd=None, T=None, ridge: float = C.RIDGE, iters: int = 60, tol: float = 1e-9) -> tuple:
    """MAP der Stärken für die Runden 0..T−1 (Standard: alle). Rückgabe: (x (T, n), Newton-Iterationen)."""
    n = m.n
    T = m.rounds if T is None else int(T)
    if T < 1:
        raise ValueError("Mindestens eine Runde")
    q = 1.0 / effective_w(w) ** 2
    sel = m.rnd < T
    R, I, J = m.rnd[sel], m.i[sel], m.j[sel]
    deg = np.full(T, 2.0)
    deg[0] -= 1.0
    deg[-1] -= 1.0                                                    # bei T = 1 ergibt das 0: keine Kopplung
    diag = np.arange(n)
    x = np.zeros((T, n))
    used = 0
    for used in range(1, iters + 1):
        p = sigmoid(x[R, I] - x[R, J])
        h = p * (1.0 - p)
        D = np.zeros((T, n, n))
        D[:, diag, diag] = (ridge + q * deg)[:, None]
        if prior_sd:
            D[0, diag, diag] += 1.0 / prior_sd ** 2
        np.add.at(D, (R, I, I), h)
        np.add.at(D, (R, J, J), h)
        np.add.at(D, (R, I, J), -h)
        np.add.at(D, (R, J, I), -h)
        step = solve_block_tridiagonal(D, q, whr_gradient(m, x, w, prior_sd, ridge))
        x += step
        if np.abs(step).max() < tol:
            break
    return x, used


def lagged_estimate(m: Matches, t0: int, lag: int, w: float, prior_sd=None) -> np.ndarray:
    """Schätzung der Stärken zum Zeitpunkt t0 aus den Partien bis t0 + lag (lag = LAG_ALL: alle)."""
    T = m.rounds if lag == C.LAG_ALL else min(t0 + int(lag) + 1, m.rounds)
    return whr_fit(m, w, prior_sd, T=T)[0][t0]


# --- TrueSkill Through Time -------------------------------------------------------------------------------------------------------------------

def _pdf(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def _cdf(x):
    return 0.5 * np.array([math.erfc(-v / SQ2) for v in np.atleast_1d(x)])


def probit_update(mw, vw, ml, vl, beta: float = C.BETA):
    """TrueSkill-Aktualisierung (ohne Remis) für Sieger (Mittel mw, Varianz vw) und Verlierer (ml, vl): neue Mittel und Varianzen. Vektorisiert."""
    c2 = 2.0 * beta * beta + vw + vl
    c = np.sqrt(c2)
    t = (mw - ml) / c
    v = _pdf(t) / np.maximum(_cdf(t), 1e-300)
    wf = v * (v + t)
    return mw + vw * v / c, vw * (1.0 - vw * wf / c2), ml - vl * v / c, vl * (1.0 - vl * wf / c2)


def ttt_fit(m: Matches, gamma: float, beta: float = C.BETA, sd0: float = 2.0, sweeps: int = 60, tol: float = 1e-4) -> tuple:
    """Rückgabe: (mu (T, n), sd (T, n), Durchläufe). `gamma`: Standardabweichung der Dynamik je Runde, `sd0`: Prior auf die Anfangswertung."""
    n, T = m.n, m.rounds
    g2 = effective_w(gamma) ** 2
    bounds = np.searchsorted(m.rnd, np.arange(T + 1))
    win_idx = np.where(m.win == 1, m.i, m.j)
    los_idx = np.where(m.win == 1, m.j, m.i)
    lp, lm = np.zeros((T, n)), np.zeros((T, n))
    fp, fm = np.zeros((T, n)), np.zeros((T, n))
    bp, bm = np.zeros((T, n)), np.zeros((T, n))
    msg = np.zeros((len(m.rnd), 4))                                   # je Partie: Praezision und Praezision·Mittel der Nachricht an Sieger und Verlierer
    fp[0] = 1.0 / (sd0 * sd0)
    prev = None
    used = 0
    for used in range(1, sweeps + 1):
        for t in range(1, T):
            P = fp[t - 1] + lp[t - 1]
            var = 1.0 / P + g2
            fp[t] = 1.0 / var
            fm[t] = (fm[t - 1] + lm[t - 1]) / P / var
        for t in range(T - 2, -1, -1):
            P = bp[t + 1] + lp[t + 1]
            M = bm[t + 1] + lm[t + 1]
            ok = P > 1e-12
            safe = np.where(ok, P, 1.0)
            var = 1.0 / safe + g2
            bp[t] = np.where(ok, 1.0 / var, 0.0)
            bm[t] = np.where(ok, (M / safe) / var, 0.0)
        for t in range(T):
            a, b = bounds[t], bounds[t + 1]
            if a == b:
                continue
            w_i, l_i = win_idx[a:b], los_idx[a:b]
            old = msg[a:b]
            Pw = fp[t, w_i] + bp[t, w_i] + lp[t, w_i] - old[:, 0]
            Mw = fm[t, w_i] + bm[t, w_i] + lm[t, w_i] - old[:, 1]
            Pl = fp[t, l_i] + bp[t, l_i] + lp[t, l_i] - old[:, 2]
            Ml = fm[t, l_i] + bm[t, l_i] + lm[t, l_i] - old[:, 3]
            good = (Pw > 1e-9) & (Pl > 1e-9)
            Pw, Pl = np.where(good, Pw, 1.0), np.where(good, Pl, 1.0)
            nmw, nvw, nml, nvl = probit_update(Mw / Pw, 1.0 / Pw, Ml / Pl, 1.0 / Pl, beta)
            new = np.stack([1.0 / nvw - Pw, nmw / nvw - Mw, 1.0 / nvl - Pl, nml / nvl - Ml], axis=1)
            new = np.where(good[:, None], new, old)
            lp[t, w_i] += new[:, 0] - old[:, 0]
            lm[t, w_i] += new[:, 1] - old[:, 1]
            lp[t, l_i] += new[:, 2] - old[:, 2]
            lm[t, l_i] += new[:, 3] - old[:, 3]
            msg[a:b] = new
        P = fp + bp + lp
        mu = (fm + bm + lm) / P
        cen = mu - mu.mean(axis=1, keepdims=True)                      # Das Niveau ist nur durch den Prior bestimmt und wandert langsam; geprüft wird die zentrierte Schätzung
        if prev is not None and np.abs(cen - prev).max() < tol:
            break
        prev = cen
    return mu, np.sqrt(1.0 / P), used


def as_run(m: Matches, mu: np.ndarray, sd=None) -> Run:
    """Verpackt eine Schätzung (Runden, n) als Run (NaN außerhalb des Feldes, also für den Neuling vor seinem Eintritt)."""
    out = np.where(m.presence, mu, np.nan)
    return Run(out, None if sd is None else np.where(m.presence, sd, np.nan), mu[-1].copy(), float("nan"))
