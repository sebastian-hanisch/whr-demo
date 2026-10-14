"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button. Ausblendbare Regler (Aktivität, Neuling, Pause je nach Szenario) folgen dem KEPT-Muster
(DEMO-PLAYBOOK §3): der Wert eines nicht gezeichneten Reglers wird nie in sein Widget geschrieben, sondern in KEPT abgelegt und unmittelbar vor dem Zeichnen per
`seed_widget` eingesetzt. Grenzen, die von anderen Reglern abhängen (Runde des Neulings, Beginn und Länge der Pause), begrenzt `clamp_kept` vorher."""

import random
from dataclasses import dataclass

import streamlit as st

import whr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "delta_slider": SettingSpec("d", float, C.DEFAULT_DELTA, C.DELTA_MIN, C.DELTA_MAX),
    "rounds_slider": SettingSpec("r", int, C.DEFAULT_ROUNDS, C.ROUNDS_MIN, C.ROUNDS_MAX),
    "sigma_slider": SettingSpec("s", float, C.DEFAULT_SIGMA, C.SIGMA_MIN, C.SIGMA_MAX),
    "scenario_select": SettingSpec("sc", str, C.DEFAULT_SCENARIO, choices=C.SCENARIOS),
    "p_rare_slider": SettingSpec("pr", float, C.DEFAULT_P_RARE, C.P_RARE_MIN, C.P_RARE_MAX),
    "join_slider": SettingSpec("j", int, C.DEFAULT_JOIN, C.JOIN_MIN, C.ROUNDS_MAX),
    "gap_slider": SettingSpec("g", float, C.DEFAULT_GAP, C.GAP_MIN, C.GAP_MAX),
    "pause_start_slider": SettingSpec("ps", int, C.DEFAULT_PAUSE_START, C.PAUSE_START_MIN, C.ROUNDS_MAX),
    "pause_len_slider": SettingSpec("pl", int, C.DEFAULT_PAUSE_LEN, C.PAUSE_LEN_MIN, C.PAUSE_LEN_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "c_slider": SettingSpec("c", int, C.DEFAULT_C, C.C_MIN, C.C_MAX),
    "w_factor_slider": SettingSpec("wf", float, C.DEFAULT_W_FACTOR, C.W_FACTOR_MIN, C.W_FACTOR_MAX),
    "prior_slider": SettingSpec("p0", float, C.DEFAULT_PRIOR, C.PRIOR_MIN, C.PRIOR_MAX),
    "lag_select": SettingSpec("lg", int, C.DEFAULT_LAG, choices=C.LAG_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "delta": "delta_slider", "rounds": "rounds_slider", "sigma": "sigma_slider", "scenario": "scenario_select", "p_rare": "p_rare_slider",
               "join": "join_slider", "gap": "gap_slider", "pause_start": "pause_start_slider", "pause_len": "pause_len_slider", "k": "k_slider",
               "c": "c_slider", "w_factor": "w_factor_slider", "prior": "prior_slider", "lag": "lag_select",
               "seed": "seed_input"}
KEPT_KEYS = ("p_rare_slider", "join_slider", "gap_slider", "pause_start_slider", "pause_len_slider")


def _kept() -> dict:
    return st.session_state.setdefault("kept_values", {})


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key in KEPT_KEYS:
            _kept().setdefault(state_key, spec.default)
        elif state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def seed_widget(state_key):
    """Unmittelbar vor dem Zeichnen eines ausblendbaren Reglers: seinen zuletzt gewählten Wert einsetzen."""
    if state_key not in st.session_state:
        st.session_state[state_key] = _kept().get(state_key, SETTING_SPECS[state_key].default)


def clamp_kept(state_key, lo, hi):
    """Begrenzt den gemerkten (und, falls gezeichnet, den Widget-)Wert auf [lo, hi]; nötig, wenn die Grenzen von anderen Reglern abhängen."""
    kept = _kept()
    kept[state_key] = min(max(kept.get(state_key, SETTING_SPECS[state_key].default), lo), hi)
    if state_key in st.session_state:
        st.session_state[state_key] = min(max(st.session_state[state_key], lo), hi)


def remember(state_key, value):
    _kept()[state_key] = value


def current_value(state_key):
    """Aktueller Wert, auch wenn der Regler in diesem Lauf nicht gezeichnet wird (dann der gemerkte)."""
    if state_key in KEPT_KEYS:
        return st.session_state.get(state_key, _kept().get(state_key, SETTING_SPECS[state_key].default))
    return st.session_state.get(state_key, SETTING_SPECS[state_key].default)


def _store(state_key, value):
    if state_key in KEPT_KEYS:
        _kept()[state_key] = value
        st.session_state.pop(state_key, None)    # beim nächsten Zeichnen aus KEPT einsetzen
    else:
        st.session_state[state_key] = value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        _store(state_key, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            _store(state_key, C.PRESETS[name][key])
    st.session_state["player_pick"] = 1 if C.PRESETS[name]["scenario"] == C.SCENARIOS[1] else 0         # bei ungleicher Aktivität zuerst ein seltener Spieler


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
    st.session_state["player_pick"] = 0
