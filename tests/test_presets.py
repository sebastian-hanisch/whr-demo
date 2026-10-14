"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Adressparameter."""

import whr_constants as C
import whr_presets as P
from whr_scenario import Settings


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds():
    for name, p in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
            if spec.choices is not None:
                assert p[key] in spec.choices, (name, key)


def test_default_preset_equals_the_default_settings():
    p = C.PRESETS["Eingespieltes Feld"]
    s = Settings(p["n"], p["delta"], p["rounds"], p["sigma"], p["scenario"], p["p_rare"], p["join"], p["gap"], p["pause_start"], p["pause_len"], p["k"], p["c"], p["w_factor"],
                 p["prior"], p["lag"], p["seed"])
    assert s == Settings()


def test_every_scenario_has_a_preset():
    assert {p["scenario"] for p in C.PRESETS.values()} == set(C.SCENARIOS)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_permalink_parameters_are_unique():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_hidden_widgets_are_exactly_the_scenario_dependent_ones():
    assert set(P.KEPT_KEYS) == {"p_rare_slider", "join_slider", "gap_slider", "pause_start_slider", "pause_len_slider"}


def test_sweep_instances_are_fixed_and_distinct():
    assert len(C.SWEEP_SEEDS) == len(set(C.SWEEP_SEEDS)) == 60
