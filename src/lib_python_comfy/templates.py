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
Project-specific templates in caller-supplied external directories are
discoverable and loadable alongside the packaged set via :func:`list_templates`
/ :func:`load_template`.
"""
from __future__ import annotations

import copy
import importlib.resources
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


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


@dataclass(frozen=True)
class TemplateInfo:
    """Identifies one discoverable template, packaged or external.

    Attributes:
        name: The template stem name (filename without ``.json`` extension).
        origin: ``"packaged"`` for a template shipped in
            :mod:`lib_python_comfy.template_data`, ``"external"`` for one found
            in a caller-supplied ``extra_dirs`` entry.
        path: The filesystem path the template was found at (as a string).
    """

    name: str
    origin: str
    path: str


@dataclass(frozen=True)
class LoadedTemplate:
    """A loaded template: its identity plus the parsed JSON content.

    Attributes:
        info: The :class:`TemplateInfo` this template was resolved from.
        data: The parsed API-format template dict.
    """

    info: TemplateInfo
    data: dict


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


def _packaged_infos() -> list[TemplateInfo]:
    """Enumerate the packaged templates in :mod:`lib_python_comfy.template_data`.

    Returns:
        A list of :class:`TemplateInfo` with ``origin="packaged"``, one per
        ``*.json`` resource, in ``iterdir()`` order.
    """
    pkg = importlib.resources.files("lib_python_comfy.template_data")
    infos: list[TemplateInfo] = []
    for resource in pkg.iterdir():
        name = resource.name
        if name.endswith(".json"):
            infos.append(
                TemplateInfo(
                    name=name[: -len(".json")],
                    origin="packaged",
                    path=str(resource),
                )
            )
    return infos


def list_builtin_templates() -> list[str]:
    """Return the stem names of all built-in templates.

    Returns:
        A list of template stem names (filename without ``.json`` extension),
        e.g. ``["txt2img_basic"]``.
    """
    return [info.name for info in _packaged_infos()]


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


# ---------------------------------------------------------------------------
# External template directories (ticket #50)
# ---------------------------------------------------------------------------


def _external_infos(directory: str | Path) -> list[TemplateInfo]:
    """Enumerate the ``*.json`` templates directly inside *directory*.

    Non-recursive, top-level ``*.json`` files only — the same rule as the
    packaged directory. If *directory* does not exist, is not a directory, or
    raises ``OSError`` while being scanned, returns an empty list silently (no
    logging, no error, no flag).
    """
    path = Path(directory)
    if not path.is_dir():
        return []
    infos: list[TemplateInfo] = []
    try:
        for entry in path.iterdir():
            if entry.is_file() and entry.name.endswith(".json"):
                infos.append(
                    TemplateInfo(
                        name=entry.name[: -len(".json")],
                        origin="external",
                        path=str(entry),
                    )
                )
    except OSError:
        return []
    return infos


def _resolve_templates(extra_dirs: Sequence[str | Path] = ()) -> dict[str, TemplateInfo]:
    """Resolve the full set of discoverable templates by name.

    Seeds the result with the packaged entries, then applies *extra_dirs* on
    top so that external entries always override packaged ones. Precedence
    among multiple *extra_dirs* is PATH-like — earlier entries win — which is
    implemented by applying the directories in reverse order with plain
    overwrite (a later, lower-precedence write can never clobber an earlier,
    higher-precedence one).

    Returns:
        A dict mapping template stem name to its resolved :class:`TemplateInfo`.
    """
    resolved: dict[str, TemplateInfo] = {info.name: info for info in _packaged_infos()}
    for directory in reversed(list(extra_dirs)):
        for info in _external_infos(directory):
            resolved[info.name] = info
    return resolved


def list_templates(extra_dirs: Sequence[str | Path] = ()) -> list[TemplateInfo]:
    """List every discoverable template, packaged and external, sorted by name.

    External files must already be API-format, ``PARAM_*``-placeholder JSON —
    this function performs no validation and does not resolve directories
    itself (the caller supplies them).

    Args:
        extra_dirs: Additional directories to search for project-specific
            templates, in precedence order (PATH-like: earlier directories
            win when the same stem name appears in more than one). Each
            directory is scanned non-recursively for top-level ``*.json``
            files. A directory that is missing, is not a directory, or
            raises ``OSError`` on scan is skipped silently. An external
            entry always overrides a packaged entry of the same name.

    Returns:
        A list of :class:`TemplateInfo`, sorted by ``name``.
    """
    return sorted(_resolve_templates(extra_dirs).values(), key=lambda info: info.name)


def load_template(name: str, extra_dirs: Sequence[str | Path] = ()) -> LoadedTemplate:
    """Load a template by stem name from the packaged set or *extra_dirs*.

    External files must already be API-format, ``PARAM_*``-placeholder JSON —
    this function performs no validation; a malformed file only fails when its
    JSON is parsed (``json.JSONDecodeError`` propagates) or later at ComfyUI.

    Args:
        name: The template stem name.
        extra_dirs: Additional directories to search, in precedence order
            (PATH-like: earlier directories win). See :func:`list_templates`
            for the discovery rules.

    Returns:
        A :class:`LoadedTemplate` combining the resolved :class:`TemplateInfo`
        (with its actual ``origin``, ``"packaged"`` or ``"external"``) and the
        parsed JSON content.

    Raises:
        FileNotFoundError: If *name* is present in neither the packaged set
            nor any of *extra_dirs*. The message lists the available names.
        json.JSONDecodeError: If the resolved file's content is not valid JSON.
    """
    resolved = _resolve_templates(extra_dirs)
    info = resolved.get(name)
    if info is None:
        raise FileNotFoundError(
            f"No template named {name!r}. Available: {sorted(resolved)}"
        )
    if info.origin == "packaged":
        data = load_builtin_template(info.name)
    else:
        data = json.loads(Path(info.path).read_text(encoding="utf-8"))
    return LoadedTemplate(info=info, data=data)
