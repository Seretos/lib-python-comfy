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

## Downstream dependency notifications

When a release is published, `release.yml` automatically opens a
"bump dependency" ticket in the downstream consumer `Seretos/agent-comfy`
so it can update its pin to the new `vX.Y.Z`.

- **Trigger:** every `release.yml` run, immediately after the GitHub
  Release is created.
- **Prerequisite — `COMFY_TICKET_TOKEN`:** a fine-grained PAT scoped to
  `Seretos/agent-comfy` with **Issues: write**, stored as a repo secret
  on `lib-python-comfy`. The built-in `GITHUB_TOKEN` cannot create issues
  in a foreign repo. A human must create this secret before the first
  release; until then the step is silently skipped.
- **Non-blocking:** the step is `continue-on-error: true`, so a missing or
  invalid token never fails the release.
- **Idempotent:** if an open issue with the same title already exists in
  the consumer, the step skips creating a duplicate.
- **Manual fallback:** if the automatic step was skipped or failed, re-file
  via the `open-dep-ticket` workflow:
  `gh workflow run open-dep-ticket --field version=X.Y.Z`
- **Emergency one-liner:**
  `gh issue create --repo Seretos/agent-comfy --title "chore(deps): bump lib-python-comfy to vX.Y.Z" --body "Update the pin in pyproject.toml to lib-python-comfy @ git+https://github.com/Seretos/lib-python-comfy@vX.Y.Z, run tests, open a PR."`
