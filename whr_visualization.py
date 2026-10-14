"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import whr_constants as C

COLORS = {"Elo": C.COLOR_ELO, "Glicko": C.COLOR_FILTER, "WHR": C.COLOR_WHR, "TTT": C.COLOR_TTT}


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def de(x, digits=2):
    return f"{x:.{digits}f}".replace(".", ",")


def to_points(theta):
    """θ (logistische Einheiten, relativ zum Feldmittel) in Elo-Punkte um 1500."""
    return C.BASE + C.SCALE * np.asarray(theta, dtype=float)


def _centered(values, ok):
    """Je Runde um das Mittel der Spieler im Feld zentrieren (NaN für Spieler außerhalb)."""
    v = np.where(ok, values, np.nan)
    return v - np.nanmean(v, axis=1)[:, None]


def build_trajectory(a, player: int, raw_whr=None):
    """Wertung des gewählten Spielers nach jeder Runde: Filter (Glicko), Glätter (WHR) und TTT; die wahre Stärke gepunktet. In einer Pause zeigt der Glätter seine Interpolation."""
    m = a.matches
    ok = m.presence
    rounds = np.arange(1, m.rounds + 1)
    fig = go.Figure()
    truth = to_points(_centered(m.truth, ok)[:, player])
    fig.add_trace(go.Scatter(x=rounds, y=truth, mode="lines", name="wahre Stärke", line=dict(color=C.COLOR_TRUTH, width=1.5, dash="dot"),
                             hovertemplate="wahr, Runde %{x}: %{y:.0f}<extra></extra>"))
    for name in ("Glicko", "WHR", "TTT"):
        mu = _centered(a.runs[name].mu, ok)[:, player]
        fig.add_trace(go.Scatter(x=rounds, y=to_points(mu), mode="lines", name=name, line=dict(color=COLORS[name], width=2.5 if name == "WHR" else 1.8), connectgaps=False,
                                 hovertemplate=name + ", Runde %{x}: %{y:.0f} Punkte<extra></extra>"))
    if raw_whr is not None and m.pause[1] > m.pause[0] and player == 0:
        others = np.arange(1, m.n)
        interp = raw_whr[:, 0] - raw_whr[:, others].mean(axis=1)
        sl = slice(m.pause[0], m.pause[1])
        fig.add_trace(go.Scatter(x=rounds[sl], y=to_points(interp[sl]), mode="lines", name="WHR in der Pause (interpoliert)", line=dict(color=C.COLOR_WHR, width=2.5, dash="dash"),
                                 hovertemplate="WHR, Pause, Runde %{x}: %{y:.0f}<extra></extra>"))
    if m.join_round:
        fig.add_vline(x=m.join_round + 1, line_dash="dash", line_color="rgba(150,150,150,0.7)", annotation_text="Neuling kommt", annotation_position="top left")
    if m.pause[1] > m.pause[0]:
        fig.add_vrect(x0=m.pause[0] + 1, x1=m.pause[1], fillcolor="rgba(150,150,150,0.15)", line_width=0, annotation_text="Pause", annotation_position="top left")
    fig.update_xaxes(title_text="Runde")
    fig.update_yaxes(title_text="Wertung (Elo-Punkte, zentriert)")
    return _base(fig, height=420)


def build_hindsight(rows):
    names = [f"σ = {de(r['sigma'])}" for r in rows]
    fig = go.Figure()
    for key, label in (("elo", "Elo (bester K)"), ("glicko", "Glicko (bestes c)"), ("whr", "WHR (w = σ)")):
        color = COLORS[{"elo": "Elo", "glicko": "Glicko", "whr": "WHR"}[key]]
        fig.add_trace(go.Bar(x=names, y=[r[key] for r in rows], name=label, marker_color=color, text=[de(r[key], 3) for r in rows], textposition="auto",
                             hovertemplate=label + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_lag(rows):
    labels = [("alle" if L == C.LAG_ALL else str(L)) for L in C.LAG_CHOICES]
    fig = go.Figure()
    for k, r in enumerate(rows):
        fig.add_trace(go.Scatter(x=labels, y=[r["lags"][L] for L in C.LAG_CHOICES], mode="lines+markers", name=f"Glätter, σ = {de(r['sigma'])}",
                                 line=dict(color=C.PALETTE[k], width=3), hovertemplate="Vorschau %{x}: %{y:.3f}<extra></extra>"))
        fig.add_hline(y=r["glicko"], line_dash="dash", line_color=C.PALETTE[k], annotation_text=f"Glicko, σ = {de(r['sigma'])}", annotation_position="top right")
    fig.update_xaxes(title_text="Vorschau: Runden nach dem bewerteten Zeitpunkt", type="category")
    fig.update_yaxes(title_text="Fehler zum Zeitpunkt 75 (θ-Einheiten)")
    return _base(fig)


def build_position(res):
    x = [f"{b + 1}" for b in range(res["bins"])]
    fig = go.Figure()
    for name in ("Elo", "Glicko", "WHR"):
        fig.add_trace(go.Scatter(x=x, y=res[name], mode="lines+markers", name=name, line=dict(color=COLORS[name], width=3),
                                 hovertemplate=name + ", Zehntel %{x}: %{y:.3f}<extra></extra>"))
    fig.update_xaxes(title_text="Zehntel der Runden (1 = Anfang, 10 = Ende)", type="category")
    fig.update_yaxes(title_text="Fehler (θ-Einheiten)")
    return _base(fig)


def build_wrong_w(rows):
    names = [f"σ = {de(r['sigma'])}" for r in rows]
    items = (("filter", "bester Filter", "#9aa0a6"), ("w0.5", "w = σ/2", "#9cc5bf"), ("w1.0", "w = σ", C.COLOR_WHR), ("w2.0", "w = 2σ", "#5f9d95"), ("static", "statisch (w = 0)", "#c0392b"))
    fig = go.Figure()
    for key, label, color in items:
        fig.add_trace(go.Bar(x=names, y=[r[key] for r in rows], name=label, marker_color=color, hovertemplate=label + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_activity(res):
    parts = (("total", "gesamt"), ("regular", "oft spielende Hälfte"), ("rare", "selten spielende Hälfte"))
    fig = go.Figure()
    for key, label in (("elo", "Elo"), ("glicko", "Glicko"), ("whr", "WHR")):
        fig.add_trace(go.Bar(x=[p[1] for p in parts], y=[res[f"{key}_{p[0]}"] for p in parts], name=label, marker_color=COLORS[label],
                             hovertemplate=label + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Verfolgungsfehler (θ-Einheiten)")
    return _base(fig)


def build_newcomer(rows):
    fig = go.Figure(go.Bar(x=[r["label"] for r in rows], y=[r["error"] for r in rows], marker_color=[COLORS[r["kind"]] for r in rows], text=[de(r["error"], 3) for r in rows],
                           textposition="auto", hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title_text="Fehler der Wertung des Neulings (θ-Einheiten)")
    return _base(fig)


def build_pause(rows, sigma):
    r = next(x for x in rows if x["sigma"] == sigma)
    cats = ["Mitte der Pause", "nach 1 Partie", "nach 5 Partien", "nach 20 Partien"]
    fig = go.Figure()
    for name in ("Elo", "Glicko", "WHR"):
        fig.add_trace(go.Bar(x=cats, y=[r["mid"][name], *r["after"][name]], name=name, marker_color=COLORS[name], hovertemplate=name + ", %{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Fehler von Spieler 0 (θ-Einheiten)")
    return _base(fig)
