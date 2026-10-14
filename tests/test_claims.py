"""JEDE Zahl und Aussage aus App-Texten und README wird hier nachgerechnet (über die festen Instanzen SWEEP_SEEDS, 60 Läufe; nie ein einzelner Seed).
Der Generator hängt nur von Seed und Position ab (SplitMix64), die Zahlen sind plattformunabhängig; Schwellen trotzdem mit Abstand zur Messung."""

import pytest

import whr_constants as C
import whr_evaluation as ev


@pytest.fixture(scope="module")
def hindsight():
    return {r["sigma"]: r for r in ev.sweep_hindsight()}


@pytest.fixture(scope="module")
def lag():
    return {r["sigma"]: r for r in ev.sweep_lag()}


@pytest.fixture(scope="module")
def position():
    return ev.sweep_position()


@pytest.fixture(scope="module")
def wrong_w():
    return {r["sigma"]: r for r in ev.sweep_wrong_w()}


@pytest.fixture(scope="module")
def activity():
    return ev.sweep_activity()


@pytest.fixture(scope="module")
def newcomer():
    return {r["label"]: r["error"] for r in ev.sweep_newcomer()}


@pytest.fixture(scope="module")
def pause():
    return {r["sigma"]: r for r in ev.sweep_pause()}


@pytest.fixture(scope="module")
def compare():
    return ev.compare_whr_ttt()


# --- Experiment 1: Rückblick gegen Filter -----------------------------------------------------------------------------------------------------

def test_the_smoother_is_a_quarter_more_accurate_than_the_best_filter(hindsight):
    expected = {0.03: (0.269, 0.250, 0.184), 0.06: (0.334, 0.329, 0.242), 0.12: (0.486, 0.481, 0.359)}
    for sg, (elo, gl, whr) in expected.items():
        r = hindsight[sg]
        assert r["elo"] == pytest.approx(elo, abs=0.002) and r["glicko"] == pytest.approx(gl, abs=0.002) and r["whr"] == pytest.approx(whr, abs=0.002), sg
        assert 0.24 < r["gain"] < 0.28, sg


def test_the_best_filter_parameters_follow_the_drift(hindsight):
    assert [hindsight[s]["elo_k"] for s in C.SIGMA_SWEEP] == [20.0, 28.0, 56.0]
    assert [hindsight[s]["glicko_c"] for s in C.SIGMA_SWEEP] == [0.03, 0.06, 0.12]


# --- Experiment 2: Vorschau -------------------------------------------------------------------------------------------------------------------

def test_without_lookahead_the_smoother_is_the_filter(lag):
    r = lag[0.06]
    assert r["lags"][0] == pytest.approx(0.328, abs=0.003) and r["glicko"] == pytest.approx(0.329, abs=0.003) and abs(r["lags"][0] - r["glicko"]) < 0.01
    assert lag[0.12]["lags"][0] == pytest.approx(0.482, abs=0.004) and lag[0.12]["glicko"] == pytest.approx(0.482, abs=0.004)


def test_lookahead_brings_the_gain_step_by_step(lag):
    r = lag[0.06]["lags"]
    assert [round(r[L], 3) for L in (2, 5, 10, 20, C.LAG_ALL)] == pytest.approx([0.323, 0.313, 0.295, 0.263, 0.224], abs=0.004)
    assert r[0] > r[2] > r[5] > r[10] > r[20] > r[C.LAG_ALL]
    total = lag[0.06]["glicko"] - r[C.LAG_ALL]
    assert (lag[0.06]["glicko"] - r[5]) / total == pytest.approx(0.16, abs=0.03)
    assert (lag[0.06]["glicko"] - r[20]) / total == pytest.approx(0.63, abs=0.05)
    assert [round(lag[0.12]["lags"][L], 3) for L in (5, 20, C.LAG_ALL)] == pytest.approx([0.441, 0.372, 0.338], abs=0.005)


# --- Experiment 3: Lage in der Zeit -----------------------------------------------------------------------------------------------------------

def test_the_gain_is_small_at_the_end_of_the_record(position):
    w, g, e = position["WHR"], position["Glicko"], position["Elo"]
    assert w[0] == pytest.approx(0.277, abs=0.004) and g[0] == pytest.approx(0.502, abs=0.006) and e[0] == pytest.approx(0.551, abs=0.006)
    assert w[-1] == pytest.approx(0.290, abs=0.004) and g[-1] == pytest.approx(0.312, abs=0.004)
    assert w[4] == pytest.approx(0.221, abs=0.004) and g[4] == pytest.approx(0.324, abs=0.004)
    assert (g[-1] - w[-1]) < 0.35 * (g[4] - w[4])


# --- Experiment 4: falsche Drift-Annahme ------------------------------------------------------------------------------------------------------

def test_a_wrong_drift_assumption_costs_accuracy(wrong_w):
    r = wrong_w[0.06]
    assert (r["w0.5"], r["w1.0"], r["w2.0"], r["static"], r["filter"]) == pytest.approx((0.262, 0.242, 0.268, 0.292, 0.329), abs=0.003)
    assert r["w0.5"] / r["w1.0"] - 1 == pytest.approx(0.08, abs=0.02) and r["w2.0"] / r["w1.0"] - 1 == pytest.approx(0.11, abs=0.02)
    assert wrong_w[0.12]["w0.5"] == pytest.approx(0.419, abs=0.004) and wrong_w[0.12]["w2.0"] == pytest.approx(0.406, abs=0.004)


def test_a_static_model_is_worse_than_the_filter_under_strong_drift(wrong_w):
    r = wrong_w[0.12]
    assert r["static"] == pytest.approx(0.565, abs=0.006) and r["filter"] == pytest.approx(0.481, abs=0.003) and r["static"] > r["filter"]
    assert wrong_w[0.03]["static"] == pytest.approx(0.192, abs=0.004) and wrong_w[0.03]["filter"] == pytest.approx(0.250, abs=0.003)
    assert wrong_w[0.03]["static"] < wrong_w[0.03]["filter"]


# --- Experiment 5: ungleiche Aktivität, Neuling, Pause ----------------------------------------------------------------------------------------

def test_unequal_activity_gains_most(activity):
    assert (activity["elo_total"], activity["glicko_total"], activity["whr_total"]) == pytest.approx((0.474, 0.445, 0.345), abs=0.004)
    assert (activity["elo_rare"], activity["glicko_rare"], activity["whr_rare"]) == pytest.approx((0.518, 0.476, 0.370), abs=0.005)
    assert (activity["elo_regular"], activity["glicko_regular"], activity["whr_regular"]) == pytest.approx((0.311, 0.293, 0.220), abs=0.004)
    assert 1 - activity["whr_total"] / activity["glicko_total"] > 0.2


def test_the_newcomer_gains_little_and_only_with_a_fitting_prior(newcomer):
    assert newcomer["Glätter ohne Prior"] == pytest.approx(0.454, abs=0.01) and newcomer["Glätter, Prior 2,0"] == pytest.approx(0.390, abs=0.01)
    assert newcomer["Glätter, Prior 1,0"] == pytest.approx(0.363, abs=0.01) and newcomer["Glätter, Prior 0,7"] == pytest.approx(0.467, abs=0.01)
    assert newcomer["Glicko, nach 20 Partien"] == pytest.approx(0.376, abs=0.01) and newcomer["Elo mit Anlauf-K 60, nach 20 Partien"] == pytest.approx(0.716, abs=0.01)
    assert abs(newcomer["Glätter, Prior 1,0"] - newcomer["Glicko, nach 20 Partien"]) < 0.03
    assert newcomer["Glicko, nach 40 Partien (am Ende)"] == pytest.approx(0.349, abs=0.01) and newcomer["Glicko, nach 40 Partien (am Ende)"] <= newcomer["Glätter, Prior 1,0"] + 0.005
    assert newcomer["Glätter ohne Prior"] > newcomer["Glätter, Prior 1,0"] and newcomer["Glätter, Prior 0,7"] > newcomer["Glätter, Prior 1,0"]


def test_the_smoother_interpolates_a_pause(pause):
    a, b = pause[0.06], pause[0.12]
    assert (a["mid"]["WHR"], a["mid"]["Elo"], a["mid"]["Glicko"]) == pytest.approx((0.421, 0.586, 0.555), abs=0.01)
    assert (b["mid"]["WHR"], b["mid"]["Elo"], b["mid"]["Glicko"]) == pytest.approx((0.627, 0.850, 0.854), abs=0.012)
    assert b["after"]["WHR"] == pytest.approx((0.417, 0.400, 0.435), abs=0.012)
    assert b["after"]["Elo"] == pytest.approx((0.802, 0.659, 0.463), abs=0.015)
    assert a["after"]["WHR"] == pytest.approx((0.298, 0.270, 0.264), abs=0.012)


# --- Experiment 6: WHR gegen TTT --------------------------------------------------------------------------------------------------------------

def test_whr_and_ttt_are_the_same_method_in_two_garments(compare):
    assert abs(compare["diff_mean"]) < 0.003 and compare["diff_sd"] == pytest.approx(0.009, abs=0.003)
    assert compare["gap"] == pytest.approx(0.036, abs=0.006) and compare["tau"] > 0.98 and 6 <= compare["sweeps"] <= 15


# --- Verdikte nennen die Zahlen aus den Daten -------------------------------------------------------------------------------------------------

def test_verdicts_quote_the_measured_numbers(hindsight, lag, position, wrong_w, activity, newcomer, pause, compare):
    assert "**0,184 / 0,242 / 0,359**" in ev.verdict_hindsight(list(hindsight.values()))[1]
    assert "**26 / 26 / 25 % weniger**" in ev.verdict_hindsight(list(hindsight.values()))[1]
    assert "5 Runden holen 16 % des Gewinns, 20 Runden 63 %" in ev.verdict_lag(list(lag.values()))[1]
    assert "**im letzten Zehntel nur 0,290 gegen 0,312:**" in ev.verdict_position(position)[1]
    assert "w = σ/2 8 % und w = 2σ 11 %" in ev.verdict_wrong_w(list(wrong_w.values()))[1]
    assert "0,345 gegen Elo 0,474 und Glicko 0,445" in ev.verdict_activity(activity)[1]
    assert "**0,363 mit Prior 1,0**" in ev.verdict_newcomer([{"label": k, "kind": "WHR" if k.startswith("Glätter") else "Elo", "prior": _prior(k), "error": v}
                                                          for k, v in newcomer.items()])[1]
    assert "0,627" in ev.verdict_pause(list(pause.values()))[1]
    assert "0,993" not in ev.verdict_compare(compare)[1] and "0,99" in ev.verdict_compare(compare)[1]


def _prior(label):
    return {"Glätter ohne Prior": None, "Glätter, Prior 2,0": 2.0, "Glätter, Prior 1,0": 1.0, "Glätter, Prior 0,7": 0.7}.get(label)
