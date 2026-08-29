"""Tests for the workflow-template subsystem (ticket #10, #50).

All imports go through the public ``lib_python_comfy`` namespace to verify
that re-exports and ``__all__`` are wired correctly.
"""
from __future__ import annotations

import json

import pytest

from lib_python_comfy import (
    MissingParameterError,
    TemplateParam,
    discover_params,
    list_builtin_templates,
    load_builtin_template,
    load_template,
    list_templates,
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
    result = render(template, {"POSITIVE_PROMPT": "a cat", "STEPS": 20, "MODEL": "some.ckpt"})
    assert isinstance(result, dict)
    assert len(result) > 0
    for node_val in result.values():
        assert "class_type" in node_val, f"Missing 'class_type' in {node_val!r}"
        assert "inputs" in node_val, f"Missing 'inputs' in {node_val!r}"


def test_txt2img_basic_model_param_is_required():
    """Omitting MODEL from txt2img_basic template raises MissingParameterError."""
    template = load_builtin_template("txt2img_basic")
    with pytest.raises(MissingParameterError):
        render(template, {"POSITIVE_PROMPT": "a cat", "STEPS": 20})


def test_txt2img_basic_model_param_is_substituted():
    """The supplied MODEL value lands in CheckpointLoaderSimple ckpt_name."""
    template = load_builtin_template("txt2img_basic")
    result = render(template, {"POSITIVE_PROMPT": "a cat", "STEPS": 20, "MODEL": "my-model.safetensors"})
    ckpt_node = next(
        node for node in result.values() if node["class_type"] == "CheckpointLoaderSimple"
    )
    assert ckpt_node["inputs"]["ckpt_name"] == "my-model.safetensors"


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


# ---------------------------------------------------------------------------
# External template directories (ticket #50)
# ---------------------------------------------------------------------------


def _write_json(path, data):
    """Write *data* as JSON to *path*, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


# --- R1: external template discoverable and loadable alongside packaged ---


def test_list_templates_includes_external(tmp_path):
    """list_templates(extra_dirs=[tmp_path]) includes both the packaged
    txt2img_basic entry (origin='packaged') and the external one written
    into tmp_path (origin='external')."""
    external_data = {"1": {"class_type": "Foo", "inputs": {"x": "PARAM_STR_X"}}}
    _write_json(tmp_path / "my_project_template.json", external_data)

    infos = list_templates(extra_dirs=[tmp_path])
    names = {info.name: info for info in infos}

    assert "txt2img_basic" in names
    assert names["txt2img_basic"].origin == "packaged"
    assert "my_project_template" in names
    assert names["my_project_template"].origin == "external"


def test_load_template_loads_external(tmp_path):
    """load_template('<external_name>', extra_dirs=[tmp_path]).data equals
    the JSON written into tmp_path."""
    external_data = {"1": {"class_type": "Foo", "inputs": {"x": "PARAM_STR_X"}}}
    _write_json(tmp_path / "my_project_template.json", external_data)

    loaded = load_template("my_project_template", extra_dirs=[tmp_path])
    assert loaded.data == external_data
    assert loaded.info.origin == "external"
    assert loaded.info.name == "my_project_template"


def test_list_templates_no_args_returns_only_packaged():
    """list_templates() with no extra_dirs returns only packaged entries,
    including txt2img_basic, sorted by name."""
    infos = list_templates()
    names = [info.name for info in infos]
    assert "txt2img_basic" in names
    assert all(info.origin == "packaged" for info in infos)
    assert names == sorted(names)


def test_list_templates_ignores_non_json_files(tmp_path):
    """A non-.json file in the external dir is ignored."""
    (tmp_path / "notes.txt").write_text("not a template", encoding="utf-8")
    infos = list_templates(extra_dirs=[tmp_path])
    names = [info.name for info in infos]
    assert "notes" not in names


# --- R2: collision — external wins, caller can tell origin ---


def test_external_template_overrides_packaged(tmp_path):
    """An external dir containing txt2img_basic.json overrides the packaged
    entry in list_templates/load_template, while load_builtin_template still
    returns the packaged content directly."""
    external_data = {"1": {"class_type": "Overridden", "inputs": {}}}
    _write_json(tmp_path / "txt2img_basic.json", external_data)

    infos = list_templates([tmp_path])
    matches = [info for info in infos if info.name == "txt2img_basic"]
    assert len(matches) == 1
    assert matches[0].origin == "external"

    loaded = load_template("txt2img_basic", [tmp_path])
    assert loaded.data == external_data
    assert loaded.info.origin == "external"

    # The packaged loader is unaffected by the external override.
    packaged = load_builtin_template("txt2img_basic")
    assert packaged != external_data


def test_external_dir_precedence_earlier_wins(tmp_path):
    """With two external dirs both containing a template of the same stem,
    the earlier dir in extra_dirs wins (PATH-like precedence)."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_data = {"1": {"class_type": "First", "inputs": {}}}
    second_data = {"1": {"class_type": "Second", "inputs": {}}}
    _write_json(first_dir / "shared.json", first_data)
    _write_json(second_dir / "shared.json", second_data)

    loaded = load_template("shared", extra_dirs=[first_dir, second_dir])
    assert loaded.data == first_data
    assert str(first_dir) in loaded.info.path


# --- R3: missing/unreadable external directory is skipped, not an error ---


def test_missing_external_dir_is_skipped(tmp_path):
    """A non-existent external dir does not raise; packaged entries are
    still returned."""
    missing = tmp_path / "does_not_exist"
    infos = list_templates([missing])
    names = [info.name for info in infos]
    assert "txt2img_basic" in names


def test_external_dir_that_is_a_file_is_skipped(tmp_path):
    """An extra_dirs entry that is a file (not a directory) does not raise;
    packaged entries are still returned."""
    a_file = tmp_path / "not_a_dir.txt"
    a_file.write_text("hello", encoding="utf-8")
    infos = list_templates([a_file])
    names = [info.name for info in infos]
    assert "txt2img_basic" in names


def test_valid_dir_after_missing_dir_still_contributes(tmp_path):
    """A valid external dir listed after a missing one still contributes its
    templates."""
    missing = tmp_path / "does_not_exist"
    valid_dir = tmp_path / "valid"
    _write_json(valid_dir / "after_missing.json", {"1": {"class_type": "X", "inputs": {}}})

    infos = list_templates([missing, valid_dir])
    names = [info.name for info in infos]
    assert "after_missing" in names


# --- R4: no validation; malformed files fail only at load time ---


def test_malformed_external_file_is_listed_but_fails_on_load(tmp_path):
    """A broken.json with invalid JSON content is listed by list_templates
    but raises json.JSONDecodeError when load_template parses it."""
    (tmp_path / "broken.json").write_text("not json at all", encoding="utf-8")

    infos = list_templates([tmp_path])
    names = [info.name for info in infos]
    assert "broken" in names

    with pytest.raises(json.JSONDecodeError):
        load_template("broken", [tmp_path])


def test_non_api_format_external_file_loads_unchecked(tmp_path):
    """A syntactically valid but non-API-format file is listed and loaded
    without complaint (no schema validation)."""
    _write_json(tmp_path / "not_api_format.json", {"foo": 1})

    infos = list_templates([tmp_path])
    names = [info.name for info in infos]
    assert "not_api_format" in names

    loaded = load_template("not_api_format", [tmp_path])
    assert loaded.data == {"foo": 1}


def test_load_template_unknown_name_raises_file_not_found(tmp_path):
    """load_template with a name present in neither packaged nor external
    sources raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_template("nonexistent_template_xyz", [tmp_path])
