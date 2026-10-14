"""PDF-Export der aktuellen Konfiguration und der Wertungen (fpdf2, Kernschrift: nur Zeichen aus Latin-1, daher „sigma“ statt σ)."""

from fpdf import FPDF

import whr_constants as C
import whr_evaluation as ev


def _latin(text: str) -> str:
    for old, new in (("–", "-"), ("—", "-"), ("τ", "Tau"), ("σ", "sigma"), ("φ", "phi"), ("µ", "mu"), ("θ", "theta"), ("≈", "ca."), ("≥", ">="), ("≤", "<="),
                     ("→", "->"), ("**", ""), ("€", "EUR"), ("−", "-")):
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: ev.Analysis) -> bytes:
    s, m = a.settings, a.matches
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _latin("WHR und TrueSkill Through Time: die Vergangenheit neu schätzen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)
    pdf.multi_cell(0, 5.5, _latin(f"Spieler: {s.n}, Runden: {s.rounds}, Drift sigma: {ev.de(s.sigma)}, Szenario: {s.scenario}, Seed: {s.seed}."), new_x="LMARGIN", new_y="NEXT")
    lag = "alle späteren Runden" if s.lag == C.LAG_ALL else f"{s.lag} Runden"
    pdf.multi_cell(0, 5.5, _latin(f"Filter: Elo K = {ev.de(s.k, 0)}, Glicko c = {ev.de(s.c, 0)} Punkte. Glätter: Drift-Annahme w = {ev.de(s.w, 3)} ({ev.de(s.w_factor)} sigma), "
                                  f"Prior {ev.de(s.prior, 1)}, Vorschau {lag}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    _level, text = ev.verdict_analysis(a)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Kernaussage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5.5, _latin(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    headers = ("Spieler", "Partien", "Elo", "Glicko", "WHR", "TTT", "wahr")
    widths = (24, 24, 26, 26, 26, 26, 26)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(headers, widths):
        pdf.cell(w, 6, h, border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    ok = m.presence[-1]
    last = {k: a.runs[k].final for k in ("Elo", "Glicko", "WHR", "TTT")}
    cen = {k: v - v[ok].mean() for k, v in last.items()}
    truth = m.truth[-1] - m.truth[-1][ok].mean()
    for p in range(m.n):
        row = (str(p + 1), str(int(a.games[p])), *(f"{C.BASE + C.SCALE * cen[k][p]:.0f}" for k in ("Elo", "Glicko", "WHR", "TTT")), f"{C.BASE + C.SCALE * truth[p]:.0f}")
        for value, w in zip(row, widths):
            pdf.cell(w, 6, _latin(value), border=1)
        pdf.ln()
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4.5, _latin(
        "Alles ist simuliert (Anfangsstärken aus dem Generator der Ranking-Linie, danach Irrfahrt der Stärken); die wahre Stärke ist nur in der Simulation bekannt. "
        "Der Glätter nutzt auch Partien nach dem bewerteten Zeitpunkt: sein Fehler ist ein Fehler im Rückblick. Elo-Punkte: 400 Punkte entsprechen dem Faktor 10 der "
        "Gewinnchancen. Quelle: Demo von Sebastian Hanisch, sebastianhanisch.net."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
