"""Invariante der ausblendbaren Regler (DEMO-PLAYBOOK §3, KEPT-Muster): ein nicht gezeichneter Regler bekommt nie einen Wert in seinem Widget-Zustand;
sein zuletzt gewählter Wert wird gemerkt und kommt beim nächsten Zeichnen zurück. Die Obergrenzen der Neuling-Runde und der Pause hängen von den Runden ab."""

import os

from streamlit.testing.v1 import AppTest

import whr_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 600
HIDDEN = {"p_rare_slider", "join_slider", "gap_slider", "pause_start_slider", "pause_len_slider"}


def fresh():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def widget_keys(at):
    return {s.key for s in at.slider}


def test_widgets_are_only_drawn_in_their_scenario():
    at = fresh()
    assert not HIDDEN & widget_keys(at)
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[1]).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"p_rare_slider"}
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[2]).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"join_slider", "gap_slider"}
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[3]).run(timeout=TIMEOUT)
    assert widget_keys(at) & HIDDEN == {"pause_start_slider", "pause_len_slider"}
    assert not at.exception


def test_hidden_values_survive_and_are_not_written_into_hidden_widgets():
    at = fresh()
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[1]).run(timeout=TIMEOUT)
    at.slider(key="p_rare_slider").set_value(0.55).run(timeout=TIMEOUT)
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[0]).run(timeout=TIMEOUT)
    assert "p_rare_slider" not in at.session_state
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[1]).run(timeout=TIMEOUT)
    assert abs(at.slider(key="p_rare_slider").value - 0.55) < 1e-9
    assert not at.exception


def test_join_round_follows_a_shrinking_number_of_rounds():
    at = fresh()
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[2]).run(timeout=TIMEOUT)
    at.slider(key="join_slider").set_value(120).run(timeout=TIMEOUT)
    at.slider(key="rounds_slider").set_value(40).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="join_slider").value <= 38


def test_pause_follows_a_shrinking_number_of_rounds():
    at = fresh()
    at.selectbox(key="scenario_select").set_value(C.SCENARIOS[3]).run(timeout=TIMEOUT)
    at.slider(key="pause_start_slider").set_value(100).run(timeout=TIMEOUT)
    at.slider(key="rounds_slider").set_value(40).run(timeout=TIMEOUT)
    assert not at.exception
    assert at.slider(key="pause_start_slider").value <= C.pause_start_max(40)
    assert at.slider(key="pause_len_slider").value <= C.pause_len_max(40, at.slider(key="pause_start_slider").value)


def test_preset_sets_values_and_hidden_widgets_for_the_next_drawing():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Neuling mit Abstand 2"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="scenario_select").value == C.SCENARIOS[2]
    assert at.slider(key="join_slider").value == 60 and at.slider(key="rounds_slider").value == 100 and at.slider(key="sigma_slider").value == 0.0
    assert abs(at.slider(key="gap_slider").value - 2.0) < 1e-9
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Eingespieltes Feld"]].click().run(timeout=TIMEOUT)
    assert at.selectbox(key="scenario_select").value == C.SCENARIOS[0] and "join_slider" not in at.session_state
    assert not at.exception


def test_a_rare_player_is_pre_selected_by_the_activity_preset():
    at = fresh()
    labels = {b.label: i for i, b in enumerate(at.button)}
    at.button[labels["Ungleiche Aktivität"]].click().run(timeout=TIMEOUT)
    assert at.session_state["player_pick"] == 1 and not at.exception
