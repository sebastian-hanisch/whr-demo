"""Die Filter, gegen die der Glätter gemessen wird: Elo mit festem K und Glicko (Glickman 1999, feste Volatilität = Glicko-1), beide in θ-Einheiten.
Kopie der Verfahren aus glicko-trueskill-demo (glt_elo.py, glt_glicko.py), unverändert bis auf den Namen.

Glicko: Jeder Spieler hat Wertung µ und Unsicherheit φ (in θ-Einheiten; RD = SCALE · φ Punkte). Je Runde:
  Schritt 1: φ ← min(√(φ² + c²), φ₀)
  Schritt 2: g(φ_j) = 1/√(1 + 3φ_j²/π²),  E = σ(g(φ_j)(µ − µ_j)),  1/v = Σ g² E (1 − E),  φ' = 1/√(1/φ² + 1/v),  µ' = µ + φ'² Σ g (s − E)"""

import math

import numpy as np

import whr_constants as C
from whr_drift import Matches, Run

def expected_score(a: float, b: float) -> float:
    """E = σ(a − b) in Skalarrechnung, ohne Überlauf."""
    x = float(a) - float(b)
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def effective_k(K: float, K0, played: float, g0: float = 10.0) -> float:
    """K in Punkten: fest, oder mit Anlauf K0 für Neulinge, nach g0 Partien halbiert, strebt gegen K."""
    if K0 is None:
        return K
    return K + (K0 - K) / (1.0 + played / g0)


def run_elo(m: Matches, K: float, K0=None, g0: float = 10.0) -> Run:
    n, rounds = m.n, m.rounds
    R = [0.0] * n
    played = [0] * n
    hist = np.full((rounds, n), np.nan)
    gi, gj, gw, gr = m.i.tolist(), m.j.tolist(), m.win.tolist(), m.rnd.tolist()
    pres = m.presence.tolist()
    g, G = 0, len(gi)
    ks, cnt = 0.0, 0
    for t in range(rounds):
        if m.join_round and t == m.join_round:
            R[0] = sum(R[1:]) / (n - 1)                        # Startwertung des Neulings: Mittel des Feldes
        while g < G and gr[g] == t:
            i, j, w = gi[g], gj[g], float(gw[g])
            e = expected_score(R[i], R[j])
            ki, kj = effective_k(K, K0, played[i], g0), effective_k(K, K0, played[j], g0)
            R[i] += ki / C.SCALE * (w - e)
            R[j] -= kj / C.SCALE * (w - e)
            played[i] += 1
            played[j] += 1
            ks += ki + kj
            cnt += 2
            g += 1
        for q in range(n):
            if pres[t][q]:
                hist[t, q] = R[q]
    return Run(hist, None, np.array(R), ks / max(cnt, 1))


PI2 = math.pi ** 2


def g_factor(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / PI2)


def inflate(phi: float, c: float, cap: float) -> float:
    """Schritt 1: Wachstum der Unsicherheit um eine Periode, höchstens bis `cap`."""
    return min(math.sqrt(phi * phi + c * c), cap)


def glicko_update(mu: float, phi: float, games) -> tuple:
    """Schritt 2 für einen Spieler: `games` = Liste (µ_j, φ_j, s_j) der Partien der Periode (mehrere sind erlaubt). Ohne Partien: (µ, φ) unverändert."""
    if not games:
        return mu, phi
    inv_v = 0.0
    total = 0.0
    for mu_j, phi_j, s in games:
        g = g_factor(phi_j)
        e = 1.0 / (1.0 + math.exp(-g * (mu - mu_j)))
        inv_v += g * g * e * (1.0 - e)
        total += g * (s - e)
    new_phi2 = 1.0 / (1.0 / (phi * phi) + inv_v)
    return mu + new_phi2 * total, math.sqrt(new_phi2)


def run_glicko(m: Matches, c: float, phi0: float) -> Run:
    """`c` und `phi0` in θ-Einheiten. Unbewertete (der Neuling vor seinem Eintritt) haben φ = φ₀ ab dem Eintritt."""
    n, rounds = m.n, m.rounds
    mu = [0.0] * n
    phi = [phi0] * n
    mu_h = np.full((rounds, n), np.nan)
    sd_h = np.full((rounds, n), np.nan)
    gi, gj, gw, gr = m.i.tolist(), m.j.tolist(), m.win.tolist(), m.rnd.tolist()
    pres = m.presence.tolist()
    g, G = 0, len(gi)
    ks, cnt = 0.0, 0
    for t in range(rounds):
        if m.join_round and t == m.join_round:
            mu[0] = sum(mu[1:]) / (n - 1)
            phi[0] = phi0
        pre = [inflate(phi[q], c, phi0) if pres[t][q] else phi[q] for q in range(n)]      # Schritt 1 für alle im Feld
        opp = {}
        while g < G and gr[g] == t:
            i, j, w = gi[g], gj[g], float(gw[g])
            opp[i] = (j, w)
            opp[j] = (i, 1.0 - w)
            g += 1
        new_mu, new_phi = list(mu), list(pre)
        for q, (j, s) in opp.items():
            new_mu[q], new_phi[q] = glicko_update(mu[q], pre[q], [(mu[j], pre[j], s)])
            e = 1.0 / (1.0 + math.exp(-g_factor(pre[j]) * (mu[q] - mu[j])))
            if abs(s - e) > 1e-9:
                ks += C.SCALE * abs(new_mu[q] - mu[q]) / abs(s - e)
                cnt += 1
        mu, phi = new_mu, new_phi
        for q in range(n):
            if pres[t][q]:
                mu_h[t, q] = mu[q]
                sd_h[t, q] = phi[q]
    return Run(mu_h, sd_h, np.array(mu), ks / max(cnt, 1))
