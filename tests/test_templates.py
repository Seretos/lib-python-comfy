"""Tests for the workflow-template subsystem (ticket #10).

All imports go through the public ``lib_python_comfy`` namespace to verify
that re-exports and ``__all__`` are wired correctly.
"""
from __future__ import annotations

import pytest

from lib_python_comfy import (
    MissingParameterError,
    TemplateParam,
    discover_params,
    list_builtin_templates,
    load_builtin_template,
    render,
)


# ---------------------------------------------------------------------------
# discover_params
# ---------------------------------------------------------------------------


def test_discover_params_finds_required():
    """PARAM_INT_STEPS is discovered as a required TemplateParam with type INT."""
    template = {"1": {"class_type": "KSampler", "inputs": {"steps": "PARAM_INT_STEPS"}}}
    params = discover_params(template)
    assert len(params) == 1
    p = params[0]
    assert p.name == "STEPS"
    assert p.type == "INT"
    assert p.required is True


def test_discover_params_finds_optional():
    """PARAM_OPT_STR_NEG|default: is discovered as optional (required=False)."""
    template = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "PARAM_OPT_STR_NEG|default:"}}}
    params = discover_params(template)
    assert len(params) == 1
    p = params[0]
    assert p.name == "NEG"
    assert p.type == "STR"
    assert p.required is False


def test_discover_params_seed_is_optional():
    """PARAM_SEED_SEED is always treated as optional (required=False)."""
    template = {"1": {"class_type": "KSampler", "inputs": {"seed": "PARAM_SEED_SEED"}}}
    params = discover_params(template)
    assert len(params) == 1
    p = params[0]
    assert p.name == "SEED"
    assert p.type == "SEED"
    assert p.required is False


# ---------------------------------------------------------------------------
# render — type coercion
# ---------------------------------------------------------------------------


def test_render_substitutes_int():
    """INT placeholder: caller value '25' is coerced to Python int 25."""
    template = {"1": {"class_type": "KSampler", "inputs": {"steps": "PARAM_INT_STEPS"}}}
    result = render(template, {"STEPS": "25"})
    assert result["1"]["inputs"]["steps"] == 25
    assert isinstance(result["1"]["inputs"]["steps"], int)


def test_render_substitutes_float():
    """FLOAT placeholder: caller value '1.5' is coerced to Python float 1.5."""
    template = {"1": {"class_type": "KSampler", "inputs": {"cfg": "PARAM_FLOAT_CFG"}}}
    result = render(template, {"CFG": "1.5"})
    assert result["1"]["inputs"]["cfg"] == 1.5
    assert isinstance(result["1"]["inputs"]["cfg"], float)


def test_render_substitutes_bool_false():
    """BOOL placeholder: caller value 'false' is coerced to Python bool False."""
    template = {"1": {"class_type": "SomeNode", "inputs": {"flag": "PARAM_BOOL_FLAG"}}}
    result = render(template, {"FLAG": "false"})
    assert result["1"]["inputs"]["flag"] is False


def test_render_substitutes_bool_true():
    """BOOL placeholder: caller value 'true' is coerced to Python bool True."""
    template = {"1": {"class_type": "SomeNode", "inputs": {"flag": "PARAM_BOOL_FLAG"}}}
    result = render(template, {"FLAG": "true"})
    assert result["1"]["inputs"]["flag"] is True


def test_render_substitutes_bool_actual_bool():
    """BOOL placeholder: actual Python bool passes through unchanged."""
    template = {"1": {"class_type": "SomeNode", "inputs": {"flag": "PARAM_BOOL_FLAG"}}}
    assert render(template, {"FLAG": False})["1"]["inputs"]["flag"] is False
    assert render(template, {"FLAG": True})["1"]["inputs"]["flag"] is True


def test_render_substitutes_str():
    """STR placeholder: caller value passes through as a string unchanged."""
    template = {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": "PARAM_STR_PROMPT"}}}
    result = render(template, {"PROMPT": "a cat"})
    assert result["1"]["inputs"]["text"] == "a cat"
    assert isinstance(result["1"]["inputs"]["text"], str)


# ---------------------------------------------------------------------------
# render — required / optional / seed
# ---------------------------------------------------------------------------


def test_render_missing_required_raises():
    """Omitting a required parameter raises MissingParameterError."""
    template = {"1": {"class_type": "KSampler", "inputs": {"steps": "PARAM_INT_STEPS"}}}
    with pytest.raises(MissingParameterError):
        render(template, {})


def test_render_missing_optional_uses_default():
    """Omitting an optional parameter uses the embedded default value."""
    template = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "PARAM_OPT_STR_NEGATIVE_PROMPT|default:ugly"},
        }
    }
    result = render(template, {})
    assert result["1"]["inputs"]["text"] == "ugly"


def test_render_missing_optional_empty_default():
    """Optional with empty default: omitting param yields empty string."""
    template = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "PARAM_OPT_STR_NEGATIVE_PROMPT|default:"},
        }
    }
    result = render(template, {})
    assert result["1"]["inputs"]["text"] == ""


def test_render_seed_auto_randomizes():
    """SEED placeholder: rendering without seed yields an int in [0, 2**32-1]."""
    template = {"1": {"class_type": "KSampler", "inputs": {"seed": "PARAM_SEED_SEED"}}}
    result = render(template, {})
    seed_val = result["1"]["inputs"]["seed"]
    assert isinstance(seed_val, int)
    assert 0 <= seed_val <= 2**32 - 1


def test_render_seed_can_be_fixed():
    """SEED placeholder: supplying a seed value overrides auto-randomization."""
    template = {"1": {"class_type": "KSampler", "inputs": {"seed": "PARAM_SEED_SEED"}}}
    result = render(template, {"SEED": 42})
    assert result["1"]["inputs"]["seed"] == 42


# ---------------------------------------------------------------------------
# render — output shape
# ---------------------------------------------------------------------------


def test_render_output_shape_matches_api_format():
    """render(txt2img_basic, required_params) produces a dict where every
    top-level value has 'class_type' and 'inputs' keys — same shape as to_api().
    """
    template = load_builtin_template("txt2img_basic")
    result = render(template, {"POSITIVE_PROMPT": "a cat", "STEPS": 20})
    assert isinstance(result, dict)
    assert len(result) > 0
    for node_val in result.values():
        assert "class_type" in node_val, f"Missing 'class_type' in {node_val!r}"
        assert "inputs" in node_val, f"Missing 'inputs' in {node_val!r}"


def test_render_does_not_mutate_template():
    """render() must not modify the original template dict (deep-copy contract)."""
    template = {"1": {"class_type": "KSampler", "inputs": {"steps": "PARAM_INT_STEPS"}}}
    original_steps = template["1"]["inputs"]["steps"]
    render(template, {"STEPS": 20})
    assert template["1"]["inputs"]["steps"] == original_steps


# ---------------------------------------------------------------------------
# Built-in template helpers
# ---------------------------------------------------------------------------


def test_list_builtin_templates_returns_names():
    """list_builtin_templates() returns a non-empty list containing 'txt2img_basic'."""
    names = list_builtin_templates()
    assert isinstance(names, list)
    assert len(names) > 0
    assert "txt2img_basic" in names


def test_load_builtin_template_returns_dict():
    """load_builtin_template('txt2img_basic') returns a non-empty dict."""
    tmpl = load_builtin_template("txt2img_basic")
    assert isinstance(tmpl, dict)
    assert len(tmpl) > 0


def test_load_builtin_template_missing_raises():
    """load_builtin_template with unknown name raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_builtin_template("nonexistent_template")
