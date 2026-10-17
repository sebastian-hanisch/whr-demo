"""Kennzahlen, Analyse des gezeigten Laufs, sechs Experimente über die festen Läufe (SWEEP_SEEDS) und die Verdikte.

Alle Verfahren sehen dieselben Partien. Fehlermaße in θ-Einheiten (1 Einheit = 173,7 Elo-Punkte); Verfolgungsfehler = RMSE der zentrierten Wertungen gegen die zentrierten wahren
Stärken ab Runde BURN, über die Spieler im Feld. Die Filter (Elo, Glicko) bekommen je Zelle den besten Parameter eines festen Rasters; der Glätter bekommt die Drift-Annahme w = σ
(bzw. das Vielfache aus Experiment 4) geschenkt. Der Fehler des Glätters ist ein Fehler im Rückblick: Er verwendet Partien nach dem bewerteten Zeitpunkt."""

import functools
import math
from dataclasses import dataclass

import numpy as np

import whr_constants as C
import whr_drift as D
import whr_filter as F
import whr_smoother as S
from whr_scenario import Settings, make_matches

NAN = float("nan")


def de(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}".replace(".", ",")


def pct(x: float, digits: int = 0) -> str:
    return de(100 * x, digits)


def _mean(values) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(v) / len(v)) if v else NAN


# --- Maße -------------------------------------------------------------------------------------------------------------------------------------

def _present(m: D.Matches, t: int) -> np.ndarray:
    return m.presence[t]


def _round_stats(run: D.Run, m: D.Matches, subset=None):
    """Je Runde: Zahl der Spieler im Feld, zentrierte Abweichung (Wertung − Stärke, beide über die Spieler im Feld zentriert) und die Maske der Spieler im Feld."""
    ok = m.presence.copy()
    if subset is not None:
        mask = np.zeros(m.n, dtype=bool)
        mask[list(subset)] = True
        ok &= mask[None, :]
    cnt = ok.sum(axis=1)
    safe = np.maximum(cnt, 1)
    mu = np.where(ok, run.mu, 0.0)
    tr = np.where(ok, m.truth, 0.0)
    d = np.where(ok, (mu - (mu.sum(axis=1) / safe)[:, None]) - (tr - (tr.sum(axis=1) / safe)[:, None]), 0.0)
    return cnt, d, ok


def round_error(run: D.Run, m: D.Matches, t: int, subset=None) -> float:
    """RMSE der zentrierten Wertungen gegen die zentrierten Stärken in Runde t über die Spieler im Feld (optional nur über `subset`, zentriert in der Teilmenge)."""
    cnt, d, _ = _round_stats(run, m, subset)
    return float(math.sqrt((d[t] ** 2).sum() / cnt[t])) if cnt[t] >= 2 else NAN


def tracking_error(run: D.Run, m: D.Matches, burn: int = C.BURN, subset=None) -> float:
    cnt, d, _ = _round_stats(run, m, subset)
    use = cnt[burn:] >= 2
    if not use.any():
        return NAN
    rmse = np.sqrt((d[burn:] ** 2).sum(axis=1) / np.maximum(cnt[burn:], 1))
    return float(rmse[use].mean())


def coverage(run: D.Run, m: D.Matches, burn: int = C.BURN) -> float:
    """Abdeckung des 95-%-Intervalls (nur Verfahren mit Unsicherheit); NaN bei Elo."""
    if run.sd is None:
        return NAN
    cnt, d, ok = _round_stats(run, m)
    use = (cnt >= 2)[:, None] & ok
    use[:burn] = False
    sd = np.where(ok, run.sd, 0.0)
    hit = (np.abs(d) <= 1.96 * sd) & use
    tot = int(use.sum())
    return float(hit.sum() / tot) if tot else NAN


def player_rounds(m: D.Matches, player: int, after: int = 0) -> list:
    """Runden ab `after`, in denen der Spieler eine Partie hatte (aufsteigend)."""
    seen = sorted({int(r) for r, i, j in zip(m.rnd, m.i, m.j) if (i == player or j == player) and r >= after})
    return seen


def player_error(run: D.Run, m: D.Matches, t: int, player: int = 0) -> float:
    """|zentrierte Wertung − zentrierte wahre Stärke| des Spielers in Runde t (zentriert über die Spieler im Feld)."""
    ok = _present(m, t)
    r, s = run.mu[t], m.truth[t]
    return float(abs((r[player] - r[ok].mean()) - (s[player] - s[ok].mean())))


def error_after_games(run: D.Run, m: D.Matches, games: int, after: int = 0) -> float:
    """Fehler von Spieler 0 in der Runde seiner `games`-ten Partie (gezählt ab Runde `after`); NaN, wenn es so viele Partien nicht gibt."""
    rounds = player_rounds(m, 0, after)
    if len(rounds) < games:
        return NAN
    return player_error(run, m, rounds[games - 1], 0)


def kendall_tau(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    iu = np.triu_indices(len(a), 1)
    sa, sb = np.sign(a[iu[0]] - a[iu[1]]), np.sign(b[iu[0]] - b[iu[1]])
    counted = (sa != 0) & (sb != 0)
    return float((sa[counted] * sb[counted]).mean()) if counted.any() else NAN


# --- Verfahren auf Partien --------------------------------------------------------------------------------------------------------------------

def run_method(m: D.Matches, method: str, param: float, phi0: float = C.PHI0_INFORMED, k0=None) -> D.Run:
    """Filter: `param` ist K in Punkten (Elo) bzw. c in θ-Einheiten (Glicko). `phi0` in θ-Einheiten, `k0` nur für Elo (Anlauf-K)."""
    if method == "Elo":
        return F.run_elo(m, param, k0)
    if method == "Glicko":
        return F.run_glicko(m, param, phi0)
    raise ValueError(f"unbekanntes Verfahren: {method}")


def smooth_run(m: D.Matches, w: float, prior_sd=None) -> D.Run:
    """WHR über die ganze Geschichte als Run."""
    x, _ = S.whr_fit(m, w, prior_sd)
    return S.as_run(m, x)


def ttt_run(m: D.Matches, gamma: float, sd0: float = 2.0) -> D.Run:
    mu, sd, _ = S.ttt_fit(m, gamma, sd0=sd0)
    return S.as_run(m, mu, sd)


def lagged_error(m: D.Matches, lag: int, w: float, prior_sd=None, checkpoints=None) -> float:
    """Fehler der Schätzung aus den Partien bis t + lag, gemittelt über die Zeitpunkte t (Standard: acht gleichmäßig verteilte ab Runde BURN, die so weit vom Ende entfernt sind,
    dass die Vorschau voll zur Verfügung steht). lag = LAG_ALL: die ganze Geschichte, dann der Verfolgungsfehler ab Runde BURN."""
    if lag == C.LAG_ALL:
        return tracking_error(smooth_run(m, w, prior_sd), m)
    last = m.rounds - 1 - lag
    cps = checkpoints if checkpoints is not None else [int(t) for t in np.linspace(C.BURN, last, 8)] if last > C.BURN else [last]
    errs = []
    for t in cps:
        x = S.lagged_estimate(m, t, lag, w, prior_sd)
        run = D.Run(np.where(m.presence, np.tile(x, (m.rounds, 1)), np.nan), None, x, float("nan"))
        errs.append(round_error(run, m, t))
    return _mean(errs)


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    matches: D.Matches
    runs: dict                 # Verfahren -> Run (Elo, Glicko, WHR, TTT)
    error: dict                # Verfahren -> Verfolgungsfehler (θ)
    error_rare: dict
    error_regular: dict
    lag_error: float           # Fehler des Glätters mit der gewählten Vorschau
    games: np.ndarray
    player0: dict              # Verfahren -> (Fehler nach 1, 5, 20 Partien nach Eintritt bzw. Rückkehr), nur bei Neuling und Pause
    ttt_sweeps: int
    tau: float                 # Kendall τ der Endwertungen WHR gegen TTT
    whr_ttt_gap: float         # mittlerer Abstand der zentrierten Schätzungen WHR und TTT in θ
    raw_whr: np.ndarray = None  # rohe Schätzung des Glätters (Runden, n), auch in der Pause


METHODS = ("Elo", "Glicko", "WHR", "TTT")


def analyse(s: Settings) -> Analysis:
    m = make_matches(s)
    x, _ = S.whr_fit(m, s.w, s.prior)
    mu, sd, sweeps = S.ttt_fit(m, s.w, sd0=s.prior)
    runs = {"Elo": F.run_elo(m, s.k), "Glicko": F.run_glicko(m, s.c_theta, C.PHI0_INFORMED), "WHR": S.as_run(m, x), "TTT": S.as_run(m, mu, sd)}
    regular = tuple(q for q in range(m.n) if q not in m.rare)
    after = m.join_round if m.join_round else (m.pause[1] if m.pause[1] else 0)
    games = np.zeros(m.n, dtype=int)
    np.add.at(games, m.i, 1)
    np.add.at(games, m.j, 1)
    error = {name: tracking_error(run, m) for name, run in runs.items()}
    cen = lambda a: a - a.mean(axis=1, keepdims=True)                     # noqa: E731
    gap = float(np.sqrt(np.mean((cen(x) - cen(mu)) ** 2)))
    out = dict(error=error, error_rare={}, error_regular={}, games=games, player0={})
    if m.rare:
        out["error_regular"] = {k: tracking_error(r, m, subset=regular) for k, r in runs.items()}
        out["error_rare"] = {k: tracking_error(r, m, subset=m.rare) for k, r in runs.items()}
    if after:
        out["player0"] = {k: tuple(error_after_games(r, m, g, after) for g in PLAYER_GAMES) for k, r in runs.items()}
    lag = lagged_error(m, s.lag, s.w, s.prior)
    return Analysis(s, m, runs, out["error"], out["error_rare"], out["error_regular"], lag, games, out["player0"], sweeps, kendall_tau(runs["WHR"].final, runs["TTT"].final), gap, x)


PLAYER_GAMES = (1, 5, 20)


# --- Feste Instanzen und Läufe der Experimente ------------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4096)
def _matches(kind: str, a: float, seed: int) -> D.Matches:
    """kind: "equal" (a = Drift), "active" (a = Drift, die Hälfte spielt mit P_RARE_EXP), "new" (a = Abstand, Drift 0), "pause" (a = Drift)."""
    n, d = C.EXP_N, C.EXP_DELTA
    if kind == "equal":
        return D.simulate_matches(n, d, C.EXP_ROUNDS, a, 1.0, 0, 0.0, 0, 0, seed)
    if kind == "active":
        return D.simulate_matches(n, d, C.EXP_ROUNDS, a, C.P_RARE_EXP, 0, 0.0, 0, 0, seed)
    if kind == "new":
        return D.simulate_matches(n, d, C.NEW_ROUNDS, 0.0, 1.0, C.NEW_JOIN, a, 0, 0, seed)
    if kind == "pause":
        return D.simulate_matches(n, d, C.PAUSE_ROUNDS, a, 1.0, 0, 0.0, C.PAUSE_START, C.PAUSE_LEN, seed)
    raise ValueError(kind)


@functools.lru_cache(maxsize=65536)
def _filter_error(kind: str, a: float, seed: int, method: str, param: float, phi0: float, part: str) -> float:
    """Verfolgungsfehler eines Filters; part: "total", "regular" oder "rare"."""
    m = _matches(kind, a, seed)
    run = run_method(m, method, param, phi0)
    if part == "total":
        return tracking_error(run, m)
    regular = tuple(q for q in range(m.n) if q not in m.rare)
    return tracking_error(run, m, subset=regular if part == "regular" else m.rare)


@functools.lru_cache(maxsize=65536)
def _whr_x(kind: str, a: float, seed: int, w: float, prior) -> np.ndarray:
    """Rohe Schätzung des Glätters (Runden, n), auch in den Runden, in denen ein Spieler nicht im Feld ist."""
    return S.whr_fit(_matches(kind, a, seed), w, prior)[0]


def _whr_run(kind: str, a: float, seed: int, w: float, prior) -> D.Run:
    return S.as_run(_matches(kind, a, seed), _whr_x(kind, a, seed, w, prior))


def _grid(method: str) -> tuple:
    return C.K_GRID if method == "Elo" else C.C_GRID


def best_filter(kind: str, a: float, method: str, phi0: float = C.PHI0_INFORMED, part: str = "total", seeds=C.SWEEP_SEEDS) -> tuple:
    """(bester Parameter, mittlerer Fehler) des Filters: bester Wert des festen Rasters im Mittel über die festen Läufe. Dem Filter wird damit die Drift geschenkt."""
    best = None
    for p in _grid(method):
        v = _mean([_filter_error(kind, a, sd, method, p, phi0, part) for sd in seeds])
        if best is None or v < best[0]:
            best = (v, p)
    return best[1], best[0]


# --- Experiment 1: Rückblick gegen Filter -----------------------------------------------------------------------------------------------------

def sweep_hindsight(sigmas=C.SIGMA_SWEEP) -> list:
    """Alle spielen jede Runde: bester Filter je Verfahren gegen den Glätter mit der richtigen Drift-Annahme w = σ."""
    rows = []
    for sg in sigmas:
        ke, ee = best_filter("equal", sg, "Elo")
        kg, eg = best_filter("equal", sg, "Glicko")
        ew = _mean([tracking_error(_whr_run("equal", sg, sd, sg, C.EXP_PRIOR), _matches("equal", sg, sd)) for sd in C.SWEEP_SEEDS])
        best = min(ee, eg)
        rows.append({"sigma": sg, "elo": ee, "elo_k": ke, "glicko": eg, "glicko_c": kg, "whr": ew, "gain": 1.0 - ew / best})
    return rows


# --- Experiment 2: Vorschau ---------------------------------------------------------------------------------------------------------------------

def sweep_lag(sigmas=C.LAG_SIGMAS, t0: int = C.LAG_T0) -> list:
    """Fehler der Schätzung zum Zeitpunkt t0 aus den Partien bis t0 + L, im Vergleich zu den Filtern an derselben Stelle."""
    rows = []
    for sg in sigmas:
        ke = best_filter("equal", sg, "Elo")[0]
        kg = best_filter("equal", sg, "Glicko")[0]
        elo = _mean([round_error(run_method(_matches("equal", sg, sd), "Elo", ke), _matches("equal", sg, sd), t0) for sd in C.SWEEP_SEEDS])
        gli = _mean([round_error(run_method(_matches("equal", sg, sd), "Glicko", kg), _matches("equal", sg, sd), t0) for sd in C.SWEEP_SEEDS])
        by_lag = {}
        for lag in C.LAG_CHOICES:
            vals = []
            for sd in C.SWEEP_SEEDS:
                m = _matches("equal", sg, sd)
                x = S.lagged_estimate(m, t0, lag, sg, C.EXP_PRIOR)
                run = D.Run(np.tile(x, (m.rounds, 1)), None, x, float("nan"))
                vals.append(round_error(run, m, t0))
            by_lag[lag] = _mean(vals)
        rows.append({"sigma": sg, "elo": elo, "glicko": gli, "lags": by_lag})
    return rows


# --- Experiment 3: Lage in der Zeit -------------------------------------------------------------------------------------------------------------

def sweep_position(sigma=C.POS_SIGMA, bins=C.POS_BINS) -> dict:
    """Fehler je Zehntel der Runden: Glätter gegen die besten Filter (σ = 0,06)."""
    ke = best_filter("equal", sigma, "Elo")[0]
    kg = best_filter("equal", sigma, "Glicko")[0]
    size = C.EXP_ROUNDS // bins
    out = {name: np.zeros(bins) for name in ("Elo", "Glicko", "WHR")}
    for sd in C.SWEEP_SEEDS:
        m = _matches("equal", sigma, sd)
        runs = {"Elo": run_method(m, "Elo", ke), "Glicko": run_method(m, "Glicko", kg), "WHR": _whr_run("equal", sigma, sd, sigma, C.EXP_PRIOR)}
        for name, run in runs.items():
            for b in range(bins):
                out[name][b] += _mean([round_error(run, m, t) for t in range(b * size, (b + 1) * size)])
    return {"sigma": sigma, "bins": bins, **{k: v / len(C.SWEEP_SEEDS) for k, v in out.items()}}


# --- Experiment 4: falsche Drift-Annahme --------------------------------------------------------------------------------------------------------

def sweep_wrong_w(sigmas=C.SIGMA_SWEEP) -> list:
    """Fehler des Glätters bei w = σ/2, σ, 2σ und statisch (w = 0), gegen den besten Filter."""
    rows = []
    for sg in sigmas:
        best = min(best_filter("equal", sg, "Elo")[1], best_filter("equal", sg, "Glicko")[1])
        row = {"sigma": sg, "filter": best}
        for f in C.W_FACTORS:
            row[f"w{f}"] = _mean([tracking_error(_whr_run("equal", sg, sd, f * sg, C.EXP_PRIOR), _matches("equal", sg, sd)) for sd in C.SWEEP_SEEDS])
        row["static"] = _mean([tracking_error(_whr_run("equal", sg, sd, 0.0, C.EXP_PRIOR), _matches("equal", sg, sd)) for sd in C.SWEEP_SEEDS])
        rows.append(row)
    return rows


# --- Experiment 5: ungleiche Aktivität, Neuling, Pause -------------------------------------------------------------------------------------------

def sweep_activity(sigma=C.POS_SIGMA) -> dict:
    """Die Hälfte der Spieler spielt nur mit Wahrscheinlichkeit 0,2 je Runde: Fehler gesamt, der oft und der selten spielenden Hälfte."""
    ke = best_filter("active", sigma, "Elo")[0]
    kg = best_filter("active", sigma, "Glicko")[0]
    out = {"sigma": sigma, "elo_k": ke, "glicko_c": kg}
    for part in ("total", "regular", "rare"):
        out[f"elo_{part}"] = _mean([_filter_error("active", sigma, sd, "Elo", ke, C.PHI0_INFORMED, part) for sd in C.SWEEP_SEEDS])
        out[f"glicko_{part}"] = _mean([_filter_error("active", sigma, sd, "Glicko", kg, C.PHI0_INFORMED, part) for sd in C.SWEEP_SEEDS])
        vals = []
        for sd in C.SWEEP_SEEDS:
            m = _matches("active", sigma, sd)
            run = _whr_run("active", sigma, sd, sigma, C.EXP_PRIOR)
            regular = tuple(q for q in range(m.n) if q not in m.rare)
            vals.append(tracking_error(run, m) if part == "total" else tracking_error(run, m, subset=regular if part == "regular" else m.rare))
        out[f"whr_{part}"] = _mean(vals)
    return out


def sweep_newcomer(gap=C.NEW_GAP) -> list:
    """Neuling (Abstand `gap`, σ = 0): mittlerer Betrag des Fehlers von Spieler 0 am Ende, Glätter mit verschiedenen Priors gegen die Filter nach 20 eigenen Partien."""
    rows = []
    for prior in C.NEW_PRIORS:
        vals = []
        for sd in C.SWEEP_SEEDS:
            m = _matches("new", gap, sd)
            run = _whr_run("new", gap, sd, 0.0, prior)
            vals.append(player_error(run, m, m.rounds - 1, 0))
        rows.append({"label": "Glätter ohne Prior" if prior is None else f"Glätter, Prior {de(prior, 1)}", "kind": "WHR", "prior": prior, "error": _mean(vals)})
    for label, method, param, k0, games in (("Elo mit Anlauf-K 60, nach 20 Partien", "Elo", 20.0, 60.0, 20), ("Glicko, nach 20 Partien", "Glicko", 0.02, None, 20),
                                            ("Glicko, nach 40 Partien (am Ende)", "Glicko", 0.02, None, 40)):
        vals = []
        for sd in C.SWEEP_SEEDS:
            m = _matches("new", gap, sd)
            run = run_method(m, method, param, C.PHI0_DEFAULT, k0)
            vals.append(error_after_games(run, m, games, C.NEW_JOIN))
        rows.append({"label": label, "kind": method, "prior": None, "error": _mean(vals)})
    return rows


def sweep_pause(sigmas=C.PAUSE_SIGMAS) -> list:
    """Pause von Spieler 0 (60 Runden): Fehler in der Mitte der Pause (der Glätter interpoliert, ein Filter hält den letzten Wert) und nach der Rückkehr."""
    rows = []
    for sg in sigmas:
        ke = best_filter("pause", sg, "Elo")[0]
        kg = best_filter("pause", sg, "Glicko")[0]
        mid = {"WHR": [], "Elo": [], "Glicko": []}
        ret = {"WHR": [], "Elo": [], "Glicko": []}
        for sd in C.SWEEP_SEEDS:
            m = _matches("pause", sg, sd)
            t_mid = (m.pause[0] + m.pause[1]) // 2
            others = list(range(1, m.n))
            truth0 = m.truth[t_mid, 0] - m.truth[t_mid, others].mean()
            x = _whr_x("pause", sg, sd, sg, C.EXP_PRIOR)
            el, gl = run_method(m, "Elo", ke), run_method(m, "Glicko", kg)
            mid["WHR"].append(((x[t_mid, 0] - x[t_mid, others].mean()) - truth0) ** 2)
            for name, run in (("Elo", el), ("Glicko", gl)):
                last = m.pause[0] - 1                              # ein Filter schreibt in der Pause nichts fort: der letzte Wert vor der Pause
                mid[name].append(((run.mu[last, 0] - np.nanmean(run.mu[last, others])) - truth0) ** 2)
            for name, run in (("WHR", S.as_run(m, x)), ("Elo", el), ("Glicko", gl)):
                ret[name].append([error_after_games(run, m, g, m.pause[1]) for g in PLAYER_GAMES])
        rows.append({"sigma": sg, "mid": {k: math.sqrt(_mean(v)) for k, v in mid.items()},
                     "after": {k: tuple(_mean([r[i] for r in v]) for i in range(len(PLAYER_GAMES))) for k, v in ret.items()}})
    return rows


# --- Experiment 6: WHR gegen TTT ----------------------------------------------------------------------------------------------------------------

def compare_whr_ttt(sigma=C.POS_SIGMA) -> dict:
    """Auf denselben Partien (alle spielen, w = γ = σ): Fehlerdifferenz TTT − WHR, mittlerer Abstand der zentrierten Schätzungen, Kendall τ der Endwertungen, Durchläufe."""
    diffs, gaps, taus, sweeps = [], [], [], []
    for sd in C.SWEEP_SEEDS:
        m = _matches("equal", sigma, sd)
        x, _ = S.whr_fit(m, sigma, C.EXP_PRIOR)
        mu, s_, used = S.ttt_fit(m, sigma, sd0=C.EXP_PRIOR)
        diffs.append(tracking_error(S.as_run(m, mu, s_), m) - tracking_error(S.as_run(m, x), m))
        cen = lambda a: a - a.mean(axis=1, keepdims=True)              # noqa: E731
        gaps.append(float(np.sqrt(np.mean((cen(x) - cen(mu)) ** 2))))
        taus.append(kendall_tau(x[-1], mu[-1]))
        sweeps.append(used)
    return {"diff_mean": _mean(diffs), "diff_sd": float(np.std(diffs)), "gap": _mean(gaps), "tau": _mean(taus), "sweeps": _mean(sweeps), "runs": len(diffs)}


# --- Verdikte ---------------------------------------------------------------------------------------------------------------------------------

def _best_filter_name(err: dict) -> str:
    return min(("Elo", "Glicko"), key=lambda k: err[k])


def verdict_analysis(a: Analysis):
    s, e = a.settings, a.error
    bf = _best_filter_name(e)
    gain = 1.0 - e["WHR"] / e[bf]
    text = (f"Verfolgungsfehler: Elo {de(e['Elo'], 3)}, Glicko {de(e['Glicko'], 3)}, **WHR {de(e['WHR'], 3)}**, TTT {de(e['TTT'], 3)} (θ-Einheiten, kleiner ist besser). "
            f"Der Glätter liegt {pct(abs(gain))} % {'unter' if gain >= 0 else 'über'} dem besseren Filter ({bf}).")
    if s.lag != C.LAG_ALL:
        text += f" Mit nur {s.lag} Runden Vorschau sind es {de(a.lag_error, 3)}."
    if gain < 0:
        text += " **Der Rückblick bringt hier nichts:** Entweder passt die Drift-Annahme nicht, oder das Feld ist zu klein."
    return ("info" if gain >= 0 else "warning"), text


def verdict_hindsight(rows):
    r = {x["sigma"]: x for x in rows}
    lo, mid, hi = (r[s] for s in C.SIGMA_SWEEP)
    return "info", (f"Im Rückblick ist der Fehler des Glätters bei Drift σ = {de(lo['sigma'])} / {de(mid['sigma'])} / {de(hi['sigma'])} **{de(lo['whr'], 3)} / {de(mid['whr'], 3)} / "
                    f"{de(hi['whr'], 3)}** gegen {de(min(lo['elo'], lo['glicko']), 3)} / {de(min(mid['elo'], mid['glicko']), 3)} / {de(min(hi['elo'], hi['glicko']), 3)} des besten Filters: "
                    f"**{pct(lo['gain'])} / {pct(mid['gain'])} / {pct(hi['gain'])} % weniger**. Der Glätter nutzt Partien nach dem bewerteten Zeitpunkt; der Vergleich ist einer des "
                    "Rückblicks, keiner der Vorhersage.")


def verdict_lag(rows):
    r = {x["sigma"]: x for x in rows}[C.POS_SIGMA]
    lags = r["lags"]
    total = r["glicko"] - lags[C.LAG_ALL]
    share = lambda L: (r["glicko"] - lags[L]) / total      # noqa: E731
    return "info", (f"Mit **0 Runden Vorschau ist der Glätter ein Filter und hier so genau wie Glicko** ({de(lags[0], 3)} gegen Glicko {de(r['glicko'], 3)} bei σ = {de(r['sigma'])}). 5 Runden holen {pct(share(5))} % des "
                    f"Gewinns, 20 Runden {pct(share(20))} %, die ganze Geschichte ({de(lags[C.LAG_ALL], 3)}) 100 %. Wer in Echtzeit bewerten muss, gewinnt von der Vorschau wenig; der "
                    "Glätter ist ein Werkzeug der nachträglichen Auswertung.")


def verdict_position(res):
    w, g = res["WHR"], res["Glicko"]
    return "info", (f"Der Gewinn hängt an der Lage: im ersten Zehntel {de(w[0], 3)} gegen {de(g[0], 3)} (Kaltstart der Filter), in der Mitte {de(w[len(w) // 2], 3)} gegen "
                    f"{de(g[len(g) // 2], 3)}, **im letzten Zehntel nur {de(w[-1], 3)} gegen {de(g[-1], 3)}:** Am Rand der Zeit gibt es kaum spätere Partien, aus denen der Glätter lernen könnte; der Vorsprung ist dort klein.")


def verdict_wrong_w(rows):
    r = {x["sigma"]: x for x in rows}
    hi, mid = r[C.SIGMA_SWEEP[-1]], r[C.POS_SIGMA]
    cost = lambda x, k: x[k] / x["w1.0"] - 1.0      # noqa: E731
    return "warning", (f"Die Drift-Annahme kostet Genauigkeit: bei σ = {de(mid['sigma'])} kostet w = σ/2 {pct(cost(mid, 'w0.5'))} % und w = 2σ {pct(cost(mid, 'w2.0'))} % Fehler. "
                       f"**Ein statisches Modell (w = 0) ist bei σ = {de(hi['sigma'])} mit {de(hi['static'], 3)} schlechter als der beste Filter ({de(hi['filter'], 3)})** und kein "
                       "Glätter mehr, sondern Bradley-Terry über alle Partien.")


def verdict_activity(res):
    return "info", (f"Bei ungleicher Aktivität (die Hälfte spielt mit p = {de(C.P_RARE_EXP, 1)}) bleibt der Gewinn etwa so groß wie bei gleicher Aktivität: gesamt {de(res['whr_total'], 3)} gegen Elo "
                    f"{de(res['elo_total'], 3)} und Glicko {de(res['glicko_total'], 3)}; die selten spielenden {de(res['whr_rare'], 3)} gegen {de(res['elo_rare'], 3)} bzw. "
                    f"{de(res['glicko_rare'], 3)}. Wer selten spielt, wird aus den späteren Partien der anderen mitgeschätzt.")


def verdict_newcomer(rows):
    w_none = next(x["error"] for x in rows if x["kind"] == "WHR" and x["prior"] is None)
    w_one = next(x["error"] for x in rows if x["kind"] == "WHR" and x["prior"] == 1.0)
    w_low = next(x["error"] for x in rows if x["kind"] == "WHR" and x["prior"] == 0.7)
    g20 = next(x["error"] for x in rows if x["label"].startswith("Glicko, nach 20"))
    g40 = next(x["error"] for x in rows if x["label"].startswith("Glicko, nach 40"))
    return "warning", (f"Ein Neuling mit Abstand {de(C.NEW_GAP, 0)}: Der Glätter liegt am Ende bei {de(w_none, 3)} ohne Prior, **{de(w_one, 3)} mit Prior 1,0** und {de(w_low, 3)} mit zu engem "
                       f"Prior 0,7; Glicko nach 20 Partien bei {de(g20, 3)}, nach 40 bei {de(g40, 3)}. **Am Ende der Aufzeichnung bringt der Rückblick dem Neuling fast nichts**, was nicht auch der Prior bringt; für die ersten Partien des Neulings bringt er dagegen viel.")


def verdict_pause(rows):
    r = {x["sigma"]: x for x in rows}[C.PAUSE_SIGMAS[-1]]
    return "info", (f"In der Mitte einer Pause von {C.PAUSE_LEN} Runden bei σ = {de(r['sigma'])} liegt der Glätter bei {de(r['mid']['WHR'], 3)}, ein Filter, der den letzten Wert hält, bei "
                    f"{de(r['mid']['Elo'], 3)} (Elo) bzw. {de(r['mid']['Glicko'], 3)} (Glicko): Der Glätter **interpoliert** zwischen vor und nach der Pause. Nach der Rückkehr "
                    f"(1 / 5 / 20 Partien): {de(r['after']['WHR'][0], 3)} / {de(r['after']['WHR'][1], 3)} / {de(r['after']['WHR'][2], 3)} gegen Elo {de(r['after']['Elo'][0], 3)} / "
                    f"{de(r['after']['Elo'][1], 3)} / {de(r['after']['Elo'][2], 3)}; nach 20 Partien holt der Filter auf.")


def verdict_compare(res):
    return "info", (f"Auf denselben {res['runs']} Läufen unterscheiden sich WHR und TrueSkill Through Time im Fehler um {de(res['diff_mean'], 3)} ± {de(res['diff_sd'], 3)} (Streuung über die Läufe); die zentrierten"
                    f"Schätzungen liegen im Mittel {de(res['gap'], 3)} θ-Einheiten ({de(res['gap'] * C.SCALE, 0)} Punkte) auseinander, die Endwertungen stimmen zu Kendall τ = {de(res['tau'], 2)} "
                    f"überein. Logistisch und Maximum a posteriori gegen Probit und Erwartungsabgleich ({de(res['sweeps'], 0)} Durchläufe): zwei Rechenwege mit hier praktisch gleichem Fehler.")
