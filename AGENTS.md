# lib-python-comfy — agent guide

ComfyUI API and workflow client library - programmatic access to ComfyUI graphs, the prompt queue, and generated outputs

A pure Python utility library: it supplies the *mechanism*; any *policy*
(names, paths, env-vars) is caller-supplied. This file tells any AI coding
agent how to operate in this repo. Keep it generic — behaviour lives in
the code and in skills.

## Tool-priority law (read this first)

When you decide how to accomplish a step, always prefer the highest
available tier — this is a strict ordering:

1. **Skills first.** If a skill covers the task, invoke it. Skills encode
   the intended workflow and supersede ad-hoc approaches. Check for a
   matching skill before doing anything else.
2. **MCP second.** If no skill fits but a Model Context Protocol tool can
   do the job (ticket/PR operations, worktree lifecycle, …), use the MCP
   tool rather than shelling out. MCP calls are structured and
   permission-gated.
3. **Raw CLI / shell last.** Only drop to `git`, `gh`, `curl`, or manual
   shell when neither a skill nor an MCP exposes the capability (running
   tests, editing files, local git operations with no MCP equivalent).

Never reach for a lower tier when a higher tier can do the same thing. If
you find yourself scripting something a skill or MCP already provides,
stop and use the higher tier.

This ordering **explicitly overrides** the generic harness default that
says "prefer the dedicated file/search tools (Glob/Grep/Read)" — when a
skill or MCP covers the task, it wins. Concretely: any *"where is X defined
/ what does the code support / which Y exist / how does X work / find the
callers of X"* question is a code-understanding task → use the matching
skill first (e.g. the `serena-wrapper` symbol-aware tools), never raw
Glob/Grep/Read.

## Working on a ticket

To process a ticket end to end, invoke the **process-ticket** skill with
the ticket number. It orchestrates the full pipeline (context extraction →
planning → implementation → review → draft PR) through subagents. Do not
do those phases by hand on the main thread — let the skill drive them.

## Repo specifics (minimal by design)

- **Language:** Python (≥ 3.11), src-layout under `src/`, package
  `lib_python_comfy`.
- **What it is:** a leaf dependency — a small, pure-Python library with no
  side effects on import. Keep the dependency surface small; this library
  is consumed by other projects via `git+https://.../@vX.Y.Z`.
- **Public API:** re-exported from `src/lib_python_comfy/__init__.py`. Any
  change to those exports, their signatures, or their behaviour is a
  breaking change for consumers — keep `__all__`, the README, and the
  version in sync.
- **Tests:** `python -m pytest`. Install dev deps with
  `pip install -e ".[test]"`. Every behaviour change needs a test under
  `tests/` (one module per source module).
- **Version is pipeline-owned.** The `version` in `pyproject.toml` is a
  placeholder on `main`; `release.yml` stamps it onto the `release/Nx`
  branch and the `vX.Y.Z` tag. Never hand-bump it.
- **Branch discipline:** All feature work happens on a feature branch in a
  git worktree, never on `main`. Assume the worktree and branch already
  exist and that you are inside them.
- **AI attribution:** The project-issues MCP automatically prefixes every
  comment and PR body with `#ai-generated`. Never type that prefix yourself.
