# WHR und TrueSkill Through Time: die Vergangenheit neu schätzen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-whr-demo.streamlit.app/)**

Neuntes Stück der **Ranking-Linie** (Konzepte-Demo), Nachfolger von `glicko-trueskill-demo`: Elo, Glicko und TrueSkill sind **Filter**, sie schreiben die Wertung fort und
korrigieren die Vergangenheit nie. **Whole-History Rating** (WHR, Maximum a posteriori, logistisch) und **TrueSkill Through Time** (TTT, Erwartungsabgleich, Probit) sind **Glätter**:
Sie schätzen jeden Zeitpunkt aus allen Partien, auch den späteren. Gemessen wird, was der Rückblick bringt, wie er von der Vorschau abhängt, was die Annahme über die Drift kostet
und wo er am meisten hilft. Das Vehikel ist das neutrale, simulierte Turnier der Wurzel (Anfangsstärken aus dem Generator Version 1, wortgleich; danach eine Irrfahrt der Stärken,
dieselben Partien für dieselben Seeds wie im Vorgängerstück); alles ist simuliert, es gibt keinen echten Datensatz.

## Kernfrage

Was bringt der Blick in die Zukunft? **Im Rückblick rund ein Viertel:** Verfolgungsfehler **0,184 / 0,242 / 0,359** gegen 0,250 / 0,329 / 0,481 des besten Filters bei Drift
σ = 0,03 / 0,06 / 0,12 (26 / 26 / 25 % weniger). **Der Gewinn hängt an der Vorschau:** mit 0 Runden ist der Glätter der Filter (0,328 gegen Glicko 0,329), mit 5 Runden holt er
16 % des Gewinns, mit 20 Runden 63 %, mit der ganzen Geschichte alles (0,224); am Rand der Zeit ist er kaum besser (letztes Zehntel 0,290 gegen 0,312). **Die Annahme über die Drift
ist seine Schwachstelle:** w = σ/2 kostet 8 %, w = 2σ 11 %, ein statisches Modell ist bei starker Drift mit 0,565 schlechter als der beste Filter (0,481). Am meisten hilft er bei
**ungleicher Aktivität** (0,345 gegen 0,474 bei Elo und 0,445 bei Glicko) und in der **Pause**, die er interpoliert (Mitte der Pause 0,627 gegen 0,850 bei σ = 0,12). Dem **Neuling**
bringt der Rückblick **nichts, was nicht auch der Prior bringt** (0,363 mit Prior 1,0, ohne Prior 0,454, Glicko am Ende 0,349). Und **WHR und TTT sind dasselbe Verfahren in zwei
Gewändern:** Fehlerdifferenz −0,001 ± 0,009, die Schätzungen liegen im Mittel 0,036 θ-Einheiten (6 Punkte) auseinander.

## Modell

- **Dynamik (beide Glätter):** Die Stärke x[t, i] von Spieler i in Runde t wandert als Wiener-Prozess, x[t+1, i] − x[t, i] ~ N(0, w²) (auch in Runden ohne Partie); die Anfangswertung
  hat einen Normal-Prior N(0, s₀²). Die Annahme w ist wie c bei Glicko eine Annahme über die Drift, kein Messwert.
- **WHR:** x maximiert Σ ln P(Ausgang) − Σ (x[t+1] − x[t])²/(2w²) − Σ x[0]²/(2s₀²) mit P(i schlägt j) = σ(x_i − x_j) (Bradley-Terry). Newton-Verfahren; die Hesse-Matrix ist
  blocktridiagonal (ein n×n-Block je Runde, außerhalb der Diagonale −1/w²), das System wird blockweise in O(T n³) gelöst; ein kleiner Ridge fixiert das Niveau.
- **TTT:** Gauß-Nachrichten auf der Kette je Spieler, Probit-Faktor je Partie (β = 1,702/√2 in θ-Einheiten, damit er zum logistischen Modell passt); die Nachricht jeder Partie folgt aus
  den Cavity-Verteilungen der beiden Spieler per Momentenabgleich (die TrueSkill-Formeln v und w); Vorwärts- und Rückwärtsdurchläufe wechseln mit der Aktualisierung der Partien bis zur
  Konvergenz (6 bis 15 Durchläufe bei vollem Feld, rund 40 bei ungleicher Aktivität und Neuling).
- **Vorschau L:** Die Schätzung zum Zeitpunkt t aus den Partien der Runden 0 … t + L; L = 0 ist der Filter, „alle“ die ganze Geschichte.
- **Filter:** Elo (festes K) und Glicko (Glicko-1, feste Volatilität, Anfangsunsicherheit 0,7 θ-Einheiten), unverändert aus dem Vorgängerstück kopiert.
- **Partien:** Anfangsstärken `make_strengths` (n ≤ 12, Abstand δ); Irrfahrt mit Schritt σ je Runde; in jeder Runde spielen die aktiven Spieler in zufälliger Paarung. **Szenarien:**
  alle spielen; die Hälfte spielt nur mit Wahrscheinlichkeit p; ein **Neuling** (Spieler 1) kommt in einer Runde mit einem Abstand vom Feld; eine **Pause** von Spieler 1.
- **Maße:** *Verfolgungsfehler* (RMSE der zentrierten Schätzungen gegen die zentrierten wahren Stärken ab Runde 30, über die Spieler im Feld, in θ-Einheiten; 1 Einheit = 173,7
  Punkte), *Fehler mit Vorschau L* (acht gleichmäßig verteilte Zeitpunkte), *Fehler nach g eigenen Partien* (Neuling, Rückkehrer).
- Alle Formeln stehen im 📐-Expander der App.

## Methodik

- **Experimente** über 60 feste Läufe (`SWEEP_SEEDS`, nie ein einzelner Seed), je mit einem live berechneten Verdikt; feste Instanzen mit 8 Spielern, Abstand 0,3 und 150 Runden
  (Neuling 100, Pause 125), unabhängig vom Regler. Den **Filtern** wird je Zelle der beste Parameter eines festen Rasters gegeben (K, c); dem **Glätter** wird die Drift geschenkt
  (w = σ, Prior 1,0). Das ist für den Glätter optimistisch; Experiment 4 misst die falsche Annahme.
- **Der Glätter ist keine Vorhersage:** Sein Fehler zum Zeitpunkt t verwendet Partien nach t. Die Demo zeigt deshalb die Vorschau L als Regler und als Experiment; bei L = 0 ist er
  der Filter.
- **Zufall:** `turnier_generator.py` mit SplitMix64-Strom (wortgleich mit den Repos der Linie).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: 60 feste Läufe, 8 Spieler, Abstand 0,3, 150 Runden, Prior 1,0, wenn nicht anders genannt. Fehler in θ-Einheiten.

| Frage | Befund | Test |
|---|---|---|
| Alle spielen; bester Filter gegen Glätter (σ = 0,03 / 0,06 / 0,12) | **Elo 0,269 / 0,334 / 0,486; Glicko 0,250 / 0,329 / 0,481; WHR 0,184 / 0,242 / 0,359** (−26 / −26 / −25 %). | `test_the_smoother_is_a_quarter_more_accurate_than_the_best_filter` |
| Beste Filterparameter | K = 20 / 28 / 56 Punkte, c = 0,03 / 0,06 / 0,12. | `test_the_best_filter_parameters_follow_the_drift` |
| Vorschau (Fehler zum Zeitpunkt 75, σ = 0,06) | Glicko 0,329, **Glätter bei 0 / 2 / 5 / 10 / 20 / alle Runden: 0,328 / 0,323 / 0,313 / 0,295 / 0,263 / 0,224**; bei σ = 0,12: 0,482 / – / 0,441 / – / 0,372 / 0,338. 5 Runden holen 16 %, 20 Runden 63 % des Gewinns. | `test_without_lookahead_the_smoother_is_the_filter`, `test_lookahead_brings_the_gain_step_by_step` |
| Lage in der Zeit (σ = 0,06, je Zehntel) | WHR 0,277 im ersten, 0,221 im fünften, **0,290 im letzten**; Glicko 0,502 / 0,324 / 0,312; Elo 0,551 im ersten. | `test_the_gain_is_small_at_the_end_of_the_record` |
| Falsche Drift-Annahme (σ = 0,06) | w = σ/2 / σ / 2σ / statisch: **0,262 / 0,242 / 0,268 / 0,292** (+8 %, +11 %) gegen den besten Filter 0,329. | `test_a_wrong_drift_assumption_costs_accuracy` |
| Statisches Modell bei σ = 0,12 | **0,565 gegen 0,481** des besten Filters; bei σ = 0,03 noch besser als der Filter (0,192 gegen 0,250). | `test_a_static_model_is_worse_than_the_filter_under_strong_drift` |
| Ungleiche Aktivität (Hälfte spielt mit p = 0,2), σ = 0,06 | gesamt / oft / selten: **Elo (K 40) 0,474 / 0,311 / 0,518; Glicko 0,445 / 0,293 / 0,476; WHR 0,345 / 0,220 / 0,370.** | `test_unequal_activity_gains_most` |
| Neuling (Abstand 2, σ = 0), mittlerer Betrag des Fehlers | Glätter ohne Prior 0,454; Prior 2,0 0,390; **Prior 1,0 0,363**; Prior 0,7 0,467; Glicko nach 20 Partien 0,376, **am Ende 0,349**; Elo mit Anlauf-K nach 20 Partien 0,716. | `test_the_newcomer_gains_little_and_only_with_a_fitting_prior` |
| Pause (60 Runden), Mitte der Pause, σ = 0,06 / 0,12 | **WHR 0,421 / 0,627** gegen eingefrorenes Elo 0,586 / 0,850 und Glicko 0,555 / 0,854. | `test_the_smoother_interpolates_a_pause` |
| Nach der Rückkehr (1 / 5 / 20 Partien), σ = 0,12 | WHR 0,417 / 0,400 / 0,435; Elo 0,802 / 0,659 / 0,463; der Filter holt nach 20 Partien auf. | `test_the_smoother_interpolates_a_pause` |
| WHR gegen TTT (σ = 0,06, w = γ = σ) | Fehlerdifferenz −0,001 ± 0,009; zentrierte Schätzungen im Mittel 0,036 auseinander; Kendall τ der Endwertungen 0,99; 9 Durchläufe. | `test_whr_and_ttt_are_the_same_method_in_two_garments` |

## Befunde und Korrekturen gegenüber dem Plan

- **Korrektur 2026-10-17 (Eigenschaftstest in `glicko-trueskill-demo`, übernommen):** Der Glicko-Filter, der hier als Vergleich läuft, ließ die Unsicherheit φ in der Pause nicht wachsen (Schritt 1 lief nur für
  Spieler im Feld; nach Glickman wächst φ auch ohne Partien). Behoben, Regressionstest `test_the_filter_lets_the_uncertainty_grow_during_a_pause`. **Alle Zahlen dieser README blieben innerhalb der Toleranz der Claims-Tests unverändert.**

- **Die Vorab-Zahlen verschieben sich um wenige Tausendstel** (WHR 0,190 / 0,244 / 0,356 gegen **0,184 / 0,242 / 0,359**; Gewinn 29 / 26 / 26 % gegen **26 / 26 / 25 %**): 60 statt 30
  Läufe, Prior 1,0 im Glätter, Glicko mit dem informierten Prior 0,7 statt 350 Punkten (0,250 / 0,329 / 0,481 statt 0,277 / 0,341 / 0,481).
- **Der Neuling im Rückblick bringt nichts:** Der Plan hatte „nur mit passendem Prior etwas“; mit der Vergleichsgröße am Ende (Glicko nach 40 Partien: 0,349) ist selbst der beste
  Prior nicht besser als der Filter (0,363). Ohne Prior läuft das Maximum Likelihood weg (0,454, mit Ausreißern).
- **Blockweise statt dicht:** Der Plan sah ein dichtes Newton-Verfahren vor (n·T im Quadrat). Die Hesse-Matrix ist blocktridiagonal; der Blocklöser rechnet 20 bis 30 ms je Anpassung bis
  n = 12 und 200 Runden und ist im Orakel gegen die dichte Lösung (identisch auf 1e-8) und gegen einen unabhängigen Maximierer geprüft. Das Kostenargument entfällt für die Demo;
  das Verfahren bleibt auf kleine Felder begrenzt, weil die Aufgabe selbst (alle Spieler, alle Runden) nicht skaliert wird.
- **TTT-Konvergenz:** Das Niveau der EP-Schätzung wandert langsam (0,002 je Durchlauf); geprüft wird deshalb die um das Feldmittel zentrierte Schätzung, die nach rund 10 Durchläufen
  steht.
- **Die Messung der Vorschau liegt bei 16 % für 5 Runden** (Plan: 17 %), der Anteil für 20 Runden bei 63 %.

## Abgrenzung

- [RTS-Glätter](https://sebastianhanisch-rts-smoother-demo.streamlit.app/) (Zustandsschätzung und Filter): derselbe Gedanke im Gauß-Modell mit Messungen eines Zustands (dort holt ein
  Lag von 5 s 21 % des Gewinns). Hier die Ranking-Fassung: relative Stärken aus Paarvergleichen, logistische oder Probit-Likelihood, ungleiche Information (Neuling, Pause, seltene
  Spieler) und ein Prior auf das Niveau.
- **Glicko und TrueSkill** (Stück 8): die Filter, gegen die gemessen wird; Partien und Maße kommen von dort.
- **Bradley-Terry** (Stück 5): der statische Grenzfall (w → 0); WHR ist Bradley-Terry mit Zeitachse.
- **Nachfolger (nur beim Namen):** Von der Mannschaft zum Einzelnen (Teams).

## Ehrliche Grenzen

- **Dem Glätter wird die Drift geschenkt** (w = σ); in der Praxis ist w eine Annahme, und Experiment 4 zeigt, was sie kostet. Sie aus den Daten zu schätzen, ist hier nicht gemessen.
- **Der Glätter ist keine Vorhersage** und im Betrieb nur zeitversetzt verfügbar (Experiment 2).
- **Der Rand der Zeit gewinnt wenig:** Die jüngsten Wertungen bleiben die unsichersten.
- **Der Prior ist gewählt, nicht begründet:** Standard 1,0, Glicko arbeitet mit 2,0; der Neuling hängt an ihm.
- **n ≤ 12, 200 Runden, nur Sieg und Niederlage,** kein Remis, keine Teams, Irrfahrt ohne Sprünge und ohne Rückkehr zum Mittel; ein Generator, 60 Läufe je Zelle; Unterschiede unter
  etwa 0,005 sind Rauschen.
- **TTT ist hier die 1-gegen-1-Fassung ohne Remis;** das Paket der Originalautoren wurde nicht verwendet.
- **Alles simuliert.**

## Quellen

- Coulom, R. (2008): Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength. *Computers and Games* (CG 2008), Lecture Notes in Computer Science 5131,
  113–124 (Verfahren nach der üblichen Darstellung; das Original wurde nicht eingesehen).
- Dangauthier, P., Herbrich, R., Minka, T., Graepel, T. (2007): TrueSkill Through Time: Revisiting the History of Chess. *Advances in Neural Information Processing Systems 20*
  (das Original wurde nicht eingesehen; die Einzelfaktor-Aktualisierung ist gegen das Paket `trueskill` geprüft).
- Glickman, M. E. (1999): Parameter estimation in large dynamic paired comparison experiments. *Journal of the Royal Statistical Society Series C (Applied Statistics)* 48 (3),
  377–394 (die Filter, wie im Vorgängerstück).
- Steele, G. L., Lea, D., Flood, C. H. (2014): Fast splittable pseudorandom number generators. *OOPSLA 2014*, 453–472 (Zufallsstrom des Generators).

## Tests

144 Tests, rund drei Minuten: Einheiten gegen Handwerte (`test_smoother.py`: Blocklöser, Gradient gegen finite Differenzen, Newton-Verfahren, Vorschau, Probit-Aktualisierung,
TTT-Durchlauf), Szenario (`test_scenario.py`), Partien und Filter (`test_drift.py`, `test_elo.py`, `test_glicko.py`, Kopien aus dem Vorgängerstück), Generator
(`test_generator.py`, wortgleich), **Orakel** (`test_oracle_whr.py`: WHR gegen den Maximierer von scipy; jeder der ersten Newton-Schritte gegen eine unabhängige dichte Rechnung in
Python-Schleifen; die Grenzfälle w → 0 (statisches Bradley-Terry) und w → ∞; die abgeschnittene Geschichte gegen die Schätzung auf den abgeschnittenen Partien; die
Probit-Aktualisierung gegen das Paket `trueskill`; **TTT gegen die exakte Posteriori** auf einem Baum (eine Partie) und auf einem Pfad mit zwei Partien durch Vorwärts-Rückwärts-Rechnung
auf einem feinen Gitter; auf einer Schleife im Faktorgraphen bis auf die EP-Näherung), Maße und Verdikte (`test_evaluation.py`), alle Zahlen dieses README (`test_claims.py`), Presets,
ausblendbare Regler (`test_hidden_widget_state.py`), PDF, Laufzeit-Importe und AppTest-Rauchtests.

**Mutationstest** (`tools/oracle_mutation.py` der Website, eingeschränkt auf `whr_smoother.py` und `whr_scenario.py`; die Filter und Partien sind Kopien aus dem Vorgängerstück und dort
geprüft): Im ersten Lauf überlebten 34 von 147 Stellen: Zähler der Iterationen und Durchläufe, die Grenzen der Szenario-Regler, die Vorzeichen der Cavity im Erwartungsabgleich und die
Hesse-Matrix auf einer zu gutartigen Instanz (Wertungen nahe null). Dazu kamen Handwert-Tests, eine neue Testdatei für das Szenario, der exakte Vergleich auf Baum und Pfad und
Newton-Schritte auf einer starken Instanz; eine überflüssige Verzweigung (T = 1) wurde entfernt. Zweiter Lauf: **144 Stellen, 136 getötet, 8 überlebt;** von den 8 sind 6 auf einer Kopie
einzeln mit den neuen Tests getötet (die vier Cavity-Vorzeichen, das Vorzeichen im Hesse-Gewicht, das Vorzeichen der Rückwärtsvarianz), **2 sind äquivalent** (die Niveauverschiebung in
der Konvergenzprüfung von TTT ändert nur die Zahl der Durchläufe, eine zusätzliche Intervallgrenze von `searchsorted` bleibt ungenutzt). **Nicht Teil dieses Laufs:** die neuen
Funktionen in `whr_evaluation.py` (Vorschau, Experimente, Verdikte); die Maße darin sind Kopien aus dem Vorgängerstück, die Experimente werden durch `test_claims.py` gedeckt.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `turnier_generator.py` | gemeinsamer Generator (wortgleich in allen Repos der Linie; liefert die Anfangsstärken) |
| `whr_drift.py` | Irrfahrt der Stärken, Paarung, Aktivität, Neuling, Pause; `Matches`, `Run` (Kopie aus dem Vorgängerstück) |
| `whr_filter.py` | die Filter Elo und Glicko (Kopie aus dem Vorgängerstück) |
| `whr_smoother.py` | WHR (Newton, Blocklöser, Vorschau) und TTT (Erwartungsabgleich) |
| `whr_scenario.py` | Reglerwerte (`Settings`) und die Partien |
| `whr_evaluation.py` | Maße, Analyse, sechs Experimente über die festen Läufe, Verdikte |
| `whr_presets.py`, `whr_constants.py` | Permalink, Presets, ausblendbare Regler, Grenzen |
| `whr_visualization.py`, `whr_pdf_export.py` | Plotly-Abbildungen, PDF-Export |
| `tests/` | Einheiten-, Orakel-, Claims- und App-Tests |

## Bewusst nicht umgesetzt

- **Drift-Annahme aus den Daten schätzen** (Maximum Likelihood in w): eigenes Thema.
- **Sprünge und Rückkehr zum Mittel** in der Dynamik; **Remis, Tore, Teams** (Stück „Von der Mannschaft zum Einzelnen“).
- **Dünn besetzte Löser für große Felder** (Coulom rechnet je Spieler und Zeitraum getrennt).

## Lokal ausführen

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` (enthält scipy und das Paket `trueskill` für das Orakel) und `python -m pytest tests -q`.

Gebaut mit Streamlit, Plotly, numpy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Ranking: von der Siegtabelle bis Learning to Rank](https://sebastianhanisch.net/konzepte-ranking.html).
