"""WHR und TrueSkill Through Time – die Vergangenheit neu schätzen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Neuntes Stück der Ranking-Linie: Elo, Glicko und TrueSkill schreiben die Wertung fort und korrigieren die Vergangenheit nie. Whole-History Rating und TrueSkill Through Time schätzen
jeden Zeitpunkt aus allen Partien, auch den späteren. Sechs Experimente messen, was der Rückblick bringt (Genauigkeit, Vorschau, Lage in der Zeit), was er kostet (die Annahme über die
Drift) und wo er hilft (ungleiche Aktivität, Pause). Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import whr_constants as C
import whr_evaluation as ev
from whr_pdf_export import build_pdf
from whr_presets import (
    apply_preset,
    bounds,
    clamp_kept,
    current_value,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    remember,
    seed_widget,
    sync_query_params,
)
from whr_scenario import Settings
from whr_visualization import (
    build_activity,
    build_hindsight,
    build_lag,
    build_newcomer,
    build_pause,
    build_position,
    build_trajectory,
    build_wrong_w,
)

st.set_page_config(page_title="WHR und TrueSkill Through Time – Sebastian Hanisch", layout="wide")

SHOW = {"success": st.success, "info": st.info, "warning": st.warning}


def verdict(level, text):
    SHOW[level](text)


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _hindsight():
    return ev.sweep_hindsight()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _lag():
    return ev.sweep_lag()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _position():
    return ev.sweep_position()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _wrong_w():
    return ev.sweep_wrong_w()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _activity():
    return ev.sweep_activity()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _newcomer():
    return ev.sweep_newcomer()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _pause():
    return ev.sweep_pause()


@st.cache_data(show_spinner="Experiment rechnet über 60 feste Läufe …")
def _compare():
    return ev.compare_whr_ttt()


def lag_label(value):
    return "alle späteren Runden" if value == C.LAG_ALL else f"{value} Runden"


st.title("⏪ WHR und TrueSkill Through Time: die Vergangenheit neu schätzen")
st.markdown(
    """
Elo, Glicko und TrueSkill sind **Filter**: Sie schreiben die Wertung nach jeder Partie fort und **korrigieren die Vergangenheit nie**. Diese Demo ist das **neunte Stück der
Ranking-Linie** und fragt, was ein **Glätter** bringt, der jeden Zeitpunkt aus **allen** Partien schätzt, auch den späteren: **Whole-History Rating** (WHR, Maximum a posteriori) und
**TrueSkill Through Time** (TTT, Erwartungsabgleich). Gemessen wird, was der **Rückblick** an Genauigkeit bringt, wie er von der **Vorschau** abhängt, was die **Annahme über die
Drift** kostet und wo er hilft (**ungleiche Aktivität**, **Pause**).

Weiter unten: sechs **🔬 Experimente** mit Verdikt, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Filter und Glätter", expanded=True):
    st.markdown(
        """
1. **Der Filter sieht die Vergangenheit:** Die Wertung nach Runde t hängt nur von den Partien bis t ab. Eine spätere Partie ändert nichts an ihr.
2. **Der Glätter sieht alles:** Die Stärke jedes Spielers in jeder Runde wird aus allen Partien geschätzt, gekoppelt durch eine Annahme über die Drift: Von Runde zu Runde
   wandert sie typisch um w (ein Wiener-Prozess, Schritte N(0, w²)). Eine Partie in Runde 100 sagt damit auch etwas über Runde 90.
3. **Die Vorschau entscheidet:** Mit 0 Runden Vorschau ist der Glätter ein Filter und hier so genau wie Glicko, mit allen späteren Runden ist er am genauesten. Seine Genauigkeit ist ein
   **Fehler im Rückblick**, keine Vorhersage.
4. **Zwei Gewänder:** WHR rechnet logistisch (Bradley-Terry) und sucht das wahrscheinlichste Gesamtbild; TTT rechnet mit Probit und gleicht Gauß-Nachrichten vorwärts und rückwärts ab.
   Beide kommen auf nahezu dieselbe Schätzung (ein paar Punkte Unterschied), so wie Glicko und TrueSkill im Vorgängerstück.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Spieler und Runden**")
    n = st.slider("Spieler n", *bounds("n_slider"), key="n_slider", help="Wie viele Spieler im Feld sind (jede Runde zufällig gepaart).")
    delta = st.slider("Stärkeabstand zwischen Nachbarn", *bounds("delta_slider"), step=C.DELTA_STEP, key="delta_slider",
                      help="Stärkedifferenz zweier benachbarter Spieler zu Beginn (logistische Einheiten).")
    rounds = st.slider("Runden", *bounds("rounds_slider"), key="rounds_slider", help="Jeder aktive Spieler spielt je Runde einmal (bei ungeradem Feld setzt einer aus).")
    sigma = st.slider("Drift σ je Runde", *bounds("sigma_slider"), step=C.SIGMA_STEP, key="sigma_slider",
                      help="Wie weit die wahren Stärken je Runde wandern (logistische Einheiten). 0 = konstant.")
    st.markdown("**Szenario**")
    scenario = st.selectbox("Wer spielt wann?", C.SCENARIOS, key="scenario_select")
    if scenario == C.SCENARIOS[1]:
        seed_widget("p_rare_slider")
        p_rare = st.slider("Spielwahrscheinlichkeit der seltenen Hälfte je Runde", *bounds("p_rare_slider"), step=C.P_RARE_STEP, key="p_rare_slider",
                           help="Die Hälfte der Spieler (ungerade Nummern) spielt nur mit dieser Wahrscheinlichkeit je Runde.")
        remember("p_rare_slider", p_rare)
    if scenario == C.SCENARIOS[2]:
        clamp_kept("join_slider", C.JOIN_MIN, int(rounds) - 2)
        seed_widget("join_slider")
        join = st.slider("Neuling kommt in Runde", C.JOIN_MIN, int(rounds) - 2, key="join_slider", help="Spieler 1 beginnt erst in dieser Runde; Startwertung der Filter ist das Feldmittel.")
        remember("join_slider", join)
        seed_widget("gap_slider")
        gap = st.slider("Abstand des Neulings vom Feld", *bounds("gap_slider"), step=C.GAP_STEP, key="gap_slider",
                        help="Wahre Stärke des Neulings minus Mittel des Feldes (logistische Einheiten).")
        remember("gap_slider", gap)
    if scenario == C.SCENARIOS[3]:
        clamp_kept("pause_start_slider", C.PAUSE_START_MIN, C.pause_start_max(int(rounds)))
        seed_widget("pause_start_slider")
        pause_start = st.slider("Pause beginnt in Runde", C.PAUSE_START_MIN, C.pause_start_max(int(rounds)), key="pause_start_slider",
                                help="Spieler 1 spielt in der Pause nicht, seine Stärke wandert weiter.")
        remember("pause_start_slider", pause_start)
        len_max = C.pause_len_max(int(rounds), int(pause_start))
        clamp_kept("pause_len_slider", C.PAUSE_LEN_MIN, len_max)
        seed_widget("pause_len_slider")
        pause_len = st.slider("Dauer der Pause in Runden", C.PAUSE_LEN_MIN, len_max, key="pause_len_slider")
        remember("pause_len_slider", pause_len)
    st.markdown("**Filter**")
    k = st.slider("Elo: K in Punkten", *bounds("k_slider"), key="k_slider", help="Fester Verstärkungsfaktor: Wertungsänderung je Partie ist K mal (Ergebnis minus Erwartung).")
    c = st.slider("Glicko: Zuwachs der Unsicherheit je Runde c in Punkten", *bounds("c_slider"), key="c_slider",
                  help="Annahme über die Drift beim Filter Glicko (Anfangsunsicherheit fest bei 0,7 θ-Einheiten = 122 Punkte). 8 Punkte ≈ 0,046 θ-Einheiten.")
    st.markdown("**Glätter (WHR und TTT)**")
    w_factor = st.slider("Drift-Annahme w als Vielfaches der wahren Drift σ", *bounds("w_factor_slider"), step=C.W_FACTOR_STEP, key="w_factor_slider",
                         help="1 = der Glätter kennt die Drift (geschenkt). 0 = statisch (Bradley-Terry über alle Partien). Zu klein oder zu groß kostet Genauigkeit (Experiment 4).")
    prior = st.slider("Prior auf die Anfangswertung (Standardabweichung)", *bounds("prior_slider"), step=C.PRIOR_STEP, key="prior_slider",
                      help="Wie weit die Anfangswertung eines Spielers vom Mittel abweichen darf (θ-Einheiten). Ein enger Prior bremst den Neuling; ohne Prior läuft er weg (Experiment 5).")
    lag = st.selectbox("Vorschau des Glätters", C.LAG_CHOICES, format_func=lag_label, key="lag_select",
                       help="So viele Runden nach dem bewerteten Zeitpunkt darf der Glätter sehen. 0 = Filter. Gezeigt wird der Fehler mit dieser Vorschau.")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf würfeln", width="stretch", on_click=randomize_seed)

p_rare_v = round(float(current_value("p_rare_slider")), 2)
join_v = int(current_value("join_slider"))
gap_v = round(float(current_value("gap_slider")), 1)
pause_start_v = int(current_value("pause_start_slider"))
pause_len_v = int(current_value("pause_len_slider"))
sync_query_params({"n_slider": int(n), "delta_slider": round(float(delta), 2), "rounds_slider": int(rounds), "sigma_slider": round(float(sigma), 2),
                   "scenario_select": scenario, "p_rare_slider": p_rare_v, "join_slider": join_v, "gap_slider": gap_v, "pause_start_slider": pause_start_v,
                   "pause_len_slider": pause_len_v, "k_slider": int(k), "c_slider": int(c), "w_factor_slider": round(float(w_factor), 2), "prior_slider": round(float(prior), 1),
                   "lag_select": int(lag), "seed_input": int(seed)})

settings = Settings(int(n), round(float(delta), 2), int(rounds), round(float(sigma), 2), scenario, p_rare_v, join_v, gap_v, pause_start_v, pause_len_v, float(k), float(c),
                    round(float(w_factor), 2), round(float(prior), 1), int(lag), int(seed))
a = _analysis(settings)
m = a.matches

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Die Wertungen")
best_filter = min(a.error["Elo"], a.error["Glicko"])
m1, m2, m3, m4 = st.columns(4)
m1.metric("Verfolgungsfehler: Elo", ev.de(a.error["Elo"], 3), help="RMSE der zentrierten Wertungen gegen die zentrierten wahren Stärken ab Runde 30, in θ-Einheiten. Kleiner ist besser.")
m2.metric("Verfolgungsfehler: Glicko", ev.de(a.error["Glicko"], 3))
m3.metric("Verfolgungsfehler: WHR", ev.de(a.error["WHR"], 3), delta=f"{ev.pct(1 - a.error['WHR'] / best_filter)} % unter dem besseren Filter", delta_color="off",
          help="Im Rückblick: Der Glätter nutzt auch Partien nach dem bewerteten Zeitpunkt.")
m4.metric("Verfolgungsfehler: TTT", ev.de(a.error["TTT"], 3), delta=f"{ev.de(a.whr_ttt_gap * C.SCALE, 0)} Punkte Abstand zu WHR", delta_color="off",
          help=f"Erwartungsabgleich mit Probit; {a.ttt_sweeps} Vorwärts-/Rückwärtsdurchläufe bis zur Konvergenz.")
if settings.lag != C.LAG_ALL:
    st.metric(f"Fehler des Glätters mit {lag_label(settings.lag)} Vorschau", ev.de(a.lag_error, 3),
              help="Mittel über acht Zeitpunkte; nur die Partien bis t + Vorschau stehen zur Verfügung. Bei 0 ist der Glätter ein Filter.")

st.session_state.setdefault("player_pick", 0)
st.session_state["player_pick"] = min(int(st.session_state["player_pick"]), settings.n - 1)
pick = st.selectbox("Spieler im Verlauf", list(range(settings.n)), format_func=lambda i: f"Spieler {i + 1}" + (" (selten)" if i in m.rare else ""), key="player_pick")
st.plotly_chart(build_trajectory(a, int(pick), a.raw_whr), width="stretch", key="trajectory_chart")
st.caption("Wertung nach jeder Runde bei Glicko (Filter), WHR und TTT (Glätter), um das Feldmittel zentriert, in Elo-Punkten um 1500; gepunktet die wahre Stärke, die nur in der "
           "Simulation bekannt ist. Der Filter hängt der Wahrheit nach, der Glätter zieht jede Schätzung aus der ganzen Geschichte.")

ok = m.presence[-1]
cen = {name: a.runs[name].final - a.runs[name].final[ok].mean() for name in ev.METHODS}
truth = m.truth[-1] - m.truth[-1][ok].mean()
rows = []
for p in range(settings.n):
    rows.append({"Spieler": p + 1, "Partien": int(a.games[p]), "Elo": round(C.BASE + C.SCALE * float(cen["Elo"][p])), "Glicko": round(C.BASE + C.SCALE * float(cen["Glicko"][p])),
                 "WHR": round(C.BASE + C.SCALE * float(cen["WHR"][p])), "TTT": round(C.BASE + C.SCALE * float(cen["TTT"][p])), "wahre Stärke": round(C.BASE + C.SCALE * float(truth[p]))})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.download_button("📄 PDF der aktuellen Wertungen", data=build_pdf(a), file_name="whr-ttt.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was bringt der Blick in die Zukunft?")
verdict(*ev.verdict_analysis(a))

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Rückblick gegen Filter")
st.caption("8 Spieler (feste Instanz, unabhängig vom Regler), 150 Runden, alle spielen jede Runde; je Filter der beste Parameter (K, c) nach dem Verfolgungsfehler, der Glätter mit w = σ und "
           "Prior 1,0 geschenkt; Mittel über 60 feste Läufe.")
rows1 = _hindsight()
st.plotly_chart(build_hindsight(rows1), width="stretch", key="exp1_chart")
verdict(*ev.verdict_hindsight(rows1))
st.dataframe(pd.DataFrame([{"Drift σ": ev.de(r["sigma"]), "Elo": round(r["elo"], 3), "bester K": round(r["elo_k"]), "Glicko": round(r["glicko"], 3), "bestes c": ev.de(r["glicko_c"], 3),
                            "WHR": round(r["whr"], 3), "Gewinn": f"{ev.pct(r['gain'])} %"} for r in rows1]), hide_index=True, width="stretch")

st.subheader("🔬 Experiment 2: Wie viel Vorschau braucht der Glätter?")
st.caption("Fehler der Schätzung zum Zeitpunkt 75, wenn der Glätter nur die Partien bis 75 + L sieht (L = 0 bis alle), gegen die besten Filter an derselben Stelle; 60 feste Läufe.")
rows2 = _lag()
st.plotly_chart(build_lag(rows2), width="stretch", key="exp2_chart")
verdict(*ev.verdict_lag(rows2))

st.subheader("🔬 Experiment 3: Lage in der Zeit")
st.caption("Fehler je Zehntel der 150 Runden (Drift σ = 0,06), Mittel über 60 feste Läufe.")
res3 = _position()
st.plotly_chart(build_position(res3), width="stretch", key="exp3_chart")
verdict(*ev.verdict_position(res3))

st.subheader("🔬 Experiment 4: Die falsche Drift-Annahme")
st.caption("Der Glätter mit w = σ/2, σ, 2σ und statisch (w = 0) gegen den besten Filter; alle spielen jede Runde, 60 feste Läufe.")
rows4 = _wrong_w()
st.plotly_chart(build_wrong_w(rows4), width="stretch", key="exp4_chart")
verdict(*ev.verdict_wrong_w(rows4))

st.subheader("🔬 Experiment 5: Ungleiche Aktivität, Neuling und Pause")
st.markdown("**Ungleiche Aktivität** (die Hälfte spielt mit p = 0,2, Drift σ = 0,06):")
res5a = _activity()
st.plotly_chart(build_activity(res5a), width="stretch", key="exp5a_chart")
verdict(*ev.verdict_activity(res5a))
st.markdown("**Neuling** (kommt nach 60 von 100 Runden, Abstand 2, Drift 0; mittlerer Betrag des Fehlers seiner Wertung am Ende):")
rows5b = _newcomer()
st.plotly_chart(build_newcomer(rows5b), width="stretch", key="exp5b_chart")
verdict(*ev.verdict_newcomer(rows5b))
st.markdown("**Pause** (Spieler 0 pausiert 60 Runden, 125 Runden insgesamt):")
rows5c = _pause()
sig_choice = st.radio("Drift σ", C.PAUSE_SIGMAS, format_func=ev.de, horizontal=True, key="pause_sigma_choice")
st.plotly_chart(build_pause(rows5c, sig_choice), width="stretch", key="exp5c_chart")
verdict(*ev.verdict_pause(rows5c))

st.subheader("🔬 Experiment 6: WHR gegen TrueSkill Through Time")
st.caption("Alle spielen jede Runde, Drift σ = 0,06, w = γ = σ, Prior 1,0; dieselben 60 festen Läufe.")
res6 = _compare()
verdict(*ev.verdict_compare(res6))

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Drift ist bekannt** | Dem Glätter ist w hier geschenkt. Zu klein oder zu groß kostet Genauigkeit, ein statisches Modell ist bei starker Drift schlechter als Elo und Glicko mit gutem Parameter (Experiment 4). | – |
| **Rückblick ist gefragt** | Der Fehler des Glätters verwendet spätere Partien; für die Vorhersage der nächsten Partie gilt nur die Vorschau 0, und dort ist er ein Filter (Experiment 2). | – |
| **Der Rand zählt** | Am Ende der Zeit gibt es keine späteren Partien: Der Gewinn ist dort klein (Experiment 3). | – |
| **Der Prior ist passend** | Ohne Prior läuft die Schätzung eines Neulings weg, ein zu enger bremst ihn (Experiment 5); die Voreinstellung 1,0 ist gewählt, nicht begründet. | – |
| **Kleine Felder, wenige Runden** | Dichtes Rechnen mit n ≤ 12 und 200 Runden; für große Felder rechnet das Verfahren von Coulom blockweise und dünn. | – |
| **Ein Spiel ist Sieg oder Niederlage** | Remis, Tore und Heimvorteil haben ein eigenes Modell; Teams brauchen eine eigene Zerlegung. | Von der Mannschaft zum Einzelnen |
| **Die Dynamik ist eine Irrfahrt** | Sprünge (Verletzung, Wechsel) und Rückkehr zum Mittel sind nicht Teil des Modells. | – |
| **Alles ist simuliert** | Stärken, Irrfahrt, Paarung und Aktivität sind Konstruktionen des Generators; keine Aussage über einen Sport. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Skala.** Stärken $x_{t,i}$ in logistischen Einheiten; $1$ Einheit $= 400/\ln 10 \approx 173{,}7$ Elo-Punkte. Eine Partie $i$ gegen $j$ in Runde $t$ hat die Ausgänge $y \in \{0, 1\}$
mit $P(y = 1) = \sigma(x_{t,i} - x_{t,j})$.

**Dynamik.** $x_{t+1,i} - x_{t,i} \sim \mathcal N(0, w^2)$ für jede Runde, auch ohne Partie; $x_{0,i} \sim \mathcal N(0, s_0^2)$.

**WHR** (Maximum a posteriori). $\hat x = \arg\max_x \sum_{\text{Partien}} \ln P(y) - \sum_{t,i} \frac{(x_{t+1,i} - x_{t,i})^2}{2 w^2} - \sum_i \frac{x_{0,i}^2}{2 s_0^2}$. Der
Newton-Schritt löst $(H + L)\,\Delta = g$ mit $L$ der Kettenmatrix (Einträge $q = 1/w^2$) und $H$ der Hesse-Matrix der Partien; sie ist blocktridiagonal (ein $n \times n$-Block je Runde),
das System wird in $O(T n^3)$ gelöst. Ein kleiner Ridge fixiert das Niveau.

**TTT** (Erwartungsabgleich). Probit-Faktor je Partie: $P(y = 1) = \Phi\bigl((x_i - x_j)/(\sqrt 2\,\beta)\bigr)$ mit $\beta = 1{,}702/\sqrt 2$, damit er zum logistischen Modell passt.
Mit $c^2 = 2\beta^2 + \sigma_w^2 + \sigma_l^2$, $t = (\mu_w - \mu_l)/c$, $v = \phi(t)/\Phi(t)$ und $w = v\,(v + t)$:
$\mu_w \leftarrow \mu_w + \sigma_w^2 v/c$, $\mu_l \leftarrow \mu_l - \sigma_l^2 v/c$, $\sigma^2 \leftarrow \sigma^2\,(1 - \sigma^2 w/c^2)$ aus den Cavity-Verteilungen der beiden Spieler;
die Nachricht der Partie ist der Quotient aus neuer und Cavity-Verteilung. Vorwärts- und Rückwärtsnachrichten längs der Kette addieren $w^2$ zur Varianz.

**Vorschau.** Die Schätzung zum Zeitpunkt $t$ mit Vorschau $L$ ist $\hat x_t$ aus den Partien der Runden $0, \dots, t + L$. $L = 0$ ist der Filter.

**Verfolgungsfehler.** $\sqrt{\tfrac1n\sum_i\bigl((\hat x_i - \bar{\hat x}) - (s_i - \bar s)\bigr)^2}$, gemittelt über die Runden ab 30; die Filter sind Elo und Glicko (Glickman 1999, feste
Volatilität), wie im Vorgängerstück.

**Zufall.** Irrfahrt mit Normalverteilung per Box-Muller aus der Gleichverteilten des Generators (SplitMix64); Paarung durch eine Zufallspermutation je Runde.

**Literatur.**
- Coulom, R. (2008): Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength. *Computers and Games*, 113–124 (in der Autorenfassung remi.coulom.free.fr/WHR/WHR.pdf eingesehen, im Kopf als „Draft“ gekennzeichnet, nicht die LNCS-Druckfassung: dynamisches
  Bradley-Terry-Modell mit Wiener-Prozess, Newton-Verfahren je Spieler mit tridiagonaler Hesse-Matrix, Gradient und Hesse-Matrix, Interpolation).
- Dangauthier, P., Herbrich, R., Minka, T., Graepel, T. (2007): TrueSkill Through Time: Revisiting the History of Chess. *Advances in Neural Information Processing Systems 20*
  (im Volltext eingesehen: Modell mit Probit-Leistung und Gaußscher Drift, Vorwärts- und Rückwärtsdurchläufe des Erwartungsabgleichs).
- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C* 48 (3), 377–394 (Filter Glicko, wie im
  Vorgängerstück).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html)."
)
