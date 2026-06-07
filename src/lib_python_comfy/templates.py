"""Workflow template loader and renderer for ComfyUI API-format prompts.

Templates are API-format JSON files (same shape as :func:`to_api` produces)
containing typed ``PARAM_<TYPE>_<NAME>`` placeholder strings in their ``inputs``
values.  :func:`render` substitutes caller-supplied values with type coercion,
required/optional enforcement, and automatic seed randomisation.

Placeholder syntax::

    PARAM_<TYPE>_<NAME>               # required
    PARAM_OPT_<TYPE>_<NAME>           # optional (uses embedded default when absent)
    PARAM_OPT_<TYPE>_<NAME>|default:<value>|  # optional with explicit default

Supported types: ``STR``, ``INT``, ``FLOAT``, ``BOOL``, ``SEED``.
``SEED``-typed placeholders are always optional and auto-randomised when absent.

Built-in templates live in :mod:`lib_python_comfy.template_data` and are
accessible via :func:`list_builtin_templates` / :func:`load_builtin_template`.
"""
from __future__ import annotations

import copy
import importlib.resources
import json
import random
import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------


class MissingParameterError(Exception):
    """Raised when a required template parameter is not supplied by the caller."""


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class TemplateParam:
    """Description of a single parameter declared in a template.

    Attributes:
        name: The parameter name (the ``<NAME>`` segment, uppercased).
        type: The type tag string (``STR``, ``INT``, ``FLOAT``, ``BOOL``, ``SEED``).
        required: ``False`` for ``PARAM_OPT_*`` and all ``PARAM_SEED_*`` placeholders.
    """

    name: str
    type: str
    required: bool


# ---------------------------------------------------------------------------
# Internal regex
# ---------------------------------------------------------------------------

# Matches the canonical placeholder form, with an optional |default:...| suffix.
# Groups:
#   1 — "OPT_" or "" (presence means optional)
#   2 — TYPE tag  (STR | INT | FLOAT | BOOL | SEED)
#   3 — NAME segment (everything up to end-of-string or first "|")
_PLACEHOLDER_RE = re.compile(
    r"^PARAM_(OPT_)?(STR|INT|FLOAT|BOOL|SEED)_([^|]+?)(?:\|default:([^|]*)\|?)?$"
)


def _parse_placeholder(value: str) -> tuple[str, str, bool, str] | None:
    """Parse a placeholder string.

    Returns:
        ``(name, type_tag, required, default_str)`` or ``None`` if *value* is
        not a placeholder.
    """
    m = _PLACEHOLDER_RE.match(value)
    if m is None:
        return None
    opt_prefix, type_tag, name, default_str = m.groups()
    is_optional = bool(opt_prefix) or type_tag == "SEED"
    required = not is_optional
    return name.upper(), type_tag, required, default_str or ""


# ---------------------------------------------------------------------------
# discover_params
# ---------------------------------------------------------------------------


def _collect_strings(obj: Any, found: set[str]) -> None:
    """Recursively collect every string value in *obj*."""
    if isinstance(obj, str):
        found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_strings(v, found)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_strings(item, found)


def discover_params(template: dict) -> list[TemplateParam]:
    """Discover all ``PARAM_*`` placeholders declared in *template*.

    Args:
        template: An API-format template dict (as returned by
            :func:`load_builtin_template`).

    Returns:
        A list of :class:`TemplateParam`, one per unique placeholder found.
        Order is not guaranteed.
    """
    strings: set[str] = set()
    _collect_strings(template, strings)

    seen_names: dict[str, TemplateParam] = {}
    for s in strings:
        parsed = _parse_placeholder(s)
        if parsed is None:
            continue
        name, type_tag, required, _default = parsed
        if name not in seen_names:
            seen_names[name] = TemplateParam(name=name, type=type_tag, required=required)

    return list(seen_names.values())


# ---------------------------------------------------------------------------
# Type coercion
# ---------------------------------------------------------------------------


def _coerce(value: Any, type_tag: str) -> Any:
    """Coerce *value* to the Python type indicated by *type_tag*."""
    if type_tag == "STR":
        return str(value)
    if type_tag in ("INT", "SEED"):
        return int(value)
    if type_tag == "FLOAT":
        return float(value)
    if type_tag == "BOOL":
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes"}
    raise ValueError(f"Unknown type tag: {type_tag!r}")


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def _render_value(raw: Any, params: dict[str, Any]) -> Any:
    """Substitute placeholders in a single value (recursive for containers).

    When a string is *entirely* a placeholder, the substituted value preserves
    the coerced Python type (int, float, bool, str).  When a placeholder appears
    as a substring within a larger string, its string form is spliced in.
    """
    if isinstance(raw, str):
        parsed = _parse_placeholder(raw)
        if parsed is not None:
            # Whole-string placeholder → return coerced Python type.
            name, type_tag, required, default_str = parsed
            if name in params:
                return _coerce(params[name], type_tag)
            if type_tag == "SEED":
                return random.randint(0, 2**32 - 1)
            if not required:
                # Optional: use embedded default (coerce if non-empty).
                if default_str:
                    return _coerce(default_str, type_tag)
                return default_str  # empty string default for STR
            raise MissingParameterError(
                f"Required template parameter {name!r} was not supplied."
            )
        # Not a standalone placeholder — check for embedded placeholders as substrings.
        # We do a simple scan so callers can embed placeholders in larger strings.
        def _sub(m: re.Match) -> str:  # noqa: ANN001
            full_match = m.group(0)
            parsed_sub = _parse_placeholder(full_match)
            if parsed_sub is None:
                return full_match
            sub_name, sub_type, sub_required, sub_default = parsed_sub
            if sub_name in params:
                return str(_coerce(params[sub_name], sub_type))
            if sub_type == "SEED":
                return str(random.randint(0, 2**32 - 1))
            if not sub_required:
                if sub_default:
                    return str(_coerce(sub_default, sub_type))
                return sub_default
            raise MissingParameterError(
                f"Required template parameter {sub_name!r} was not supplied."
            )

        embedded_re = re.compile(
            r"PARAM_(?:OPT_)?(?:STR|INT|FLOAT|BOOL|SEED)_[^|\s]+(?:\|default:[^|]*\|?)?"
        )
        return embedded_re.sub(_sub, raw)

    if isinstance(raw, dict):
        return {k: _render_value(v, params) for k, v in raw.items()}
    if isinstance(raw, list):
        return [_render_value(item, params) for item in raw]
    if isinstance(raw, tuple):
        return tuple(_render_value(item, params) for item in raw)
    return raw


def render(template: dict, params: dict[str, Any]) -> dict:
    """Render *template* by substituting ``PARAM_*`` placeholders.

    Args:
        template: An API-format template dict (as returned by
            :func:`load_builtin_template`).
        params: Caller-supplied parameter values keyed by ``<NAME>``
            (the uppercased name segment of the placeholder).

    Returns:
        A concrete API-format prompt dict with the same top-level structure as
        :func:`to_api` produces: every top-level value has ``"class_type"`` and
        ``"inputs"`` keys, and all placeholders have been replaced with
        coerced Python values.

    Raises:
        MissingParameterError: If a required parameter is absent from *params*.
    """
    working = copy.deepcopy(template)
    return _render_value(working, params)


# ---------------------------------------------------------------------------
# Built-in template helpers
# ---------------------------------------------------------------------------


def list_builtin_templates() -> list[str]:
    """Return the stem names of all built-in templates.

    Returns:
        A list of template stem names (filename without ``.json`` extension),
        e.g. ``["txt2img_basic"]``.
    """
    pkg = importlib.resources.files("lib_python_comfy.template_data")
    stems: list[str] = []
    for resource in pkg.iterdir():
        name = resource.name
        if name.endswith(".json"):
            stems.append(name[: -len(".json")])
    return stems


def load_builtin_template(name: str) -> dict:
    """Load and parse a built-in template by stem name.

    Args:
        name: The template stem name (e.g. ``"txt2img_basic"``).

    Returns:
        The parsed JSON dict for the requested template.

    Raises:
        FileNotFoundError: If no built-in template with *name* exists.
    """
    pkg = importlib.resources.files("lib_python_comfy.template_data")
    resource = pkg.joinpath(f"{name}.json")
    try:
        text = resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"No built-in template named {name!r}. "
            f"Available: {list_builtin_templates()}"
        )
    return json.loads(text)
