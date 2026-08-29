"""Driving tests for ticket #51: `.github/scripts/prev-release-tag.sh` must
print the immediately-preceding released `v*` tag for a new version — by
explicit semver comparison, never by assuming the new tag is already (or is
not yet) present in `git tag --list` — so `release.yml`'s "Create GitHub
Release" step can pass `--notes-start-tag` and get a correct merge-base
compare instead of GitHub's failed ancestor walk (release commits are
orphaned by design: each run rebuilds the release branch fresh off `main`
HEAD).

Each test builds a throwaway git repo under `tmp_path`, tags it, and invokes
the script via `bash <repo>/.github/scripts/prev-release-tag.sh <new-tag>`
with `cwd=tmp_path` — so the script's own `git tag --list` sees only the
throwaway repo's tags, never this project's real release tags.

Expected RED reason (this phase): the script does not exist yet
(`.github/scripts/` is new). Each `subprocess.run` invocation fails to
start / bash reports "No such file or directory" and returns exit code 127
— not a pass-by-accident failure, and not a assertion mismatch.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or shutil.which("git") is None,
    reason="requires bash and git on PATH",
)

SCRIPT = (
    Path(__file__).resolve().parent.parent
    / ".github"
    / "scripts"
    / "prev-release-tag.sh"
)


def _init_repo(repo_path):
    """Create a git repo with a single empty commit at repo_path."""
    subprocess.run(["git", "init", "-q"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo_path, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-q", "-m", "init"],
        cwd=repo_path,
        check=True,
    )


def _tag(repo_path, name):
    subprocess.run(["git", "tag", name], cwd=repo_path, check=True)


def _run_script(repo_path, new_tag):
    return subprocess.run(
        ["bash", str(SCRIPT), new_tag],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )


# --- R1 case 1: basic sequential selection, new tag already exists locally
# (the realistic in-workflow ordering: `git tag -a "$TAG"` runs before this
# script does) ---------------------------------------------------------


def test_basic_sequential_selection(tmp_path):
    _init_repo(tmp_path)
    for t in ["v0.0.1", "v0.0.2", "v0.0.3", "v0.0.4", "v0.0.5"]:
        _tag(tmp_path, t)
    _tag(tmp_path, "v0.0.6")

    result = _run_script(tmp_path, "v0.0.6")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.0.5"


# --- R1 case 2: new tag absent from `git tag --list` — the case a naive
# positional-lookup design (e.g. `git tag --list | grep -B1 "$NEW_TAG"`)
# gets wrong, since it has nothing to anchor on ------------------------


def test_new_tag_absent_from_local_tags(tmp_path):
    _init_repo(tmp_path)
    for t in ["v0.0.1", "v0.0.2", "v0.0.3", "v0.0.4", "v0.0.5"]:
        _tag(tmp_path, t)
    # v0.0.6 is deliberately never created locally.

    result = _run_script(tmp_path, "v0.0.6")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.0.5"


# --- R1 case 3: present-vs-absent parity — same repo, same new tag, the
# only difference is whether the new tag itself was created locally first.
# Correctness must not depend on that, and the new tag must never be
# reported as its own predecessor. -------------------------------------


def test_present_vs_absent_parity(tmp_path):
    repo_absent = tmp_path / "absent"
    repo_present = tmp_path / "present"
    repo_absent.mkdir()
    repo_present.mkdir()

    for repo in (repo_absent, repo_present):
        _init_repo(repo)
        for t in ["v0.0.1", "v0.0.2", "v0.0.3", "v0.0.4", "v0.0.5"]:
            _tag(repo, t)
    _tag(repo_present, "v0.0.6")

    result_absent = _run_script(repo_absent, "v0.0.6")
    result_present = _run_script(repo_present, "v0.0.6")

    assert result_absent.returncode == 0, result_absent.stderr
    assert result_present.returncode == 0, result_present.stderr
    assert result_absent.stdout.strip() == "v0.0.5"
    assert result_present.stdout.strip() == "v0.0.5"


# --- R1 case 4: genuinely first release, zero tags in the repo ---------


def test_first_release_ever_no_tags(tmp_path):
    _init_repo(tmp_path)

    result = _run_script(tmp_path, "v0.1.0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --- R1 case 5: only tag present is the new tag itself ------------------


def test_first_release_only_new_tag_present(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.1.0")

    result = _run_script(tmp_path, "v0.1.0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --- R1 case 6: backport — a patch release for an older minor line must
# not pick up a newer minor's tag as its predecessor --------------------


def test_backport_picks_older_minor_not_newer(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.1.0")
    _tag(tmp_path, "v0.2.0")

    result = _run_script(tmp_path, "v0.1.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.1.0"


# --- R1 case 7: a prerelease sorts below its final release --------------


def test_prerelease_sorts_below_final_release(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.1.0-rc1")

    result = _run_script(tmp_path, "v0.1.0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.1.0-rc1"


# --- R1 case 8: a prerelease of a new minor still sorts above the prior
# minor's final release --------------------------------------------------


def test_prerelease_of_new_minor_sorts_above_prior_final(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.0.9")

    result = _run_script(tmp_path, "v0.1.0-rc1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.0.9"


# --- Generation 2, case (a): rc-suffix ordering is now correct per true
# SemVer §11.4, REPLACING the deleted test_prerelease_numeric_ordering_
# not_lexicographic above, with the OPPOSITE expected result. "rc2" and
# "rc10" are each a single, non-dot-separated alphanumeric identifier
# (SemVer §11.4.3: an identifier containing any non-digit character is
# compared as a whole ASCII string, never split into digit runs). Compared
# byte-for-byte, "rc10" < "rc2" (first differing byte: '1' 0x31 < '2'
# 0x32), so "rc2" is the HIGHER-precedence (later) prerelease and is the
# correct predecessor of a subsequent release — not "rc10", which
# generation 1's bespoke numeric-rc-ordering heuristic incorrectly
# preferred by treating "10" and "2" as digit runs to compare by value.
#
# Verified against the CURRENT (generation-1, unmodified) script: FAILS —
# it prints "v0.1.0-rc10" (the old, now-wrong answer), confirming this is
# a genuine RED (wrong value, not a crash/missing-file error).


def test_rc_suffix_ordering_is_semver_correct(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.1.0-rc2")
    _tag(tmp_path, "v0.1.0-rc10")

    result = _run_script(tmp_path, "v0.2.0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.1.0-rc2"


# --- Generation 2, case (c) — the DISCRIMINATING case that actually
# exercises the new class-byte mechanism (the one piece of new machinery
# this generation adds). "2" is a purely numeric identifier; "1x" is
# alphanumeric (it contains a non-digit). SemVer §11.4.3 requires a
# numeric identifier to ALWAYS have LOWER precedence than an alphanumeric
# one, regardless of which specific digits/letters follow — so "1x" must
# outrank "2".
#
# Verified by hand-tracing generation 1's key-building awk block, then
# confirmed empirically by running the CURRENT script directly: it prints
# "v1.0.0-2", the WRONG answer — because generation 1 (and any naive
# digit-byte-only comparison) has no class distinction between numeric and
# alphanumeric identifiers, so it compares "2" and "1x" by their leading
# digit byte and ranks "1x" below "2" (since '1' 0x31 < '2' 0x32 as raw
# bytes) — the exact inverse of the required SemVer outcome. This is the
# test that would fail if the new class byte were removed or implemented
# wrong.


def test_numeric_identifier_outranked_by_alphanumeric_identifier(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-2")
    _tag(tmp_path, "v1.0.0-1x")

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0-1x"


# --- Generation 2, test-critic note: the case above only demonstrates the
# class-byte fix for an alphanumeric identifier ("1x") that happens to
# start with a digit whose length-prefix encoding sorts favorably above
# the numeric identifier "2" even WITHOUT a class byte in play (compare
# encoded numeric "2" vs raw "1x": "1" < "2" as the very first byte, so a
# naive fix that merely tweaked the digit-run encoding, without any real
# numeric/alphanumeric class distinction, could conceivably pass the case
# above by accident of layout). This case closes that gap with an
# alphanumeric identifier that starts with "-" (a hyphen — legal per
# SemVer section 9's alphanumeric-identifier grammar, since the identifier
# just needs at least one non-digit character) instead of a digit: "-x"
# vs the purely numeric "2". Under any naive byte-first comparison without
# a class byte, "-" (0x2D) loses to "2" (0x32), which is backwards — an
# alphanumeric identifier must ALWAYS outrank a numeric one regardless of
# its leading byte. This only passes if the class byte is actually being
# applied, not merely coincidentally implied by the digit encoding.


def test_alphanumeric_identifier_starting_with_hyphen_outranks_numeric(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-2")
    _tag(tmp_path, "v1.0.0--x")

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0--x"


# --- Generation 2, case (b): dot-separated numeric fields compare by
# value, not lexically. "alpha" matches on both sides first (a per-field,
# not whole-string, comparison), then the numeric field "10" > "2" by
# value (SemVer §11.4.2).
#
# This is expected to already PASS under the CURRENT (generation-1)
# script too — generation 1's length-prefix digit encoding already orders
# same-position numeric digit runs correctly by value. Verified
# empirically: the current script already prints "v1.0.0-alpha.10" here.
# Written anyway as regression-pinning coverage per the plan, since this
# per-field numeric comparison must keep working under the new
# per-identifier key scheme too.


def test_dot_separated_numeric_field_compares_by_value(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-alpha.2")
    _tag(tmp_path, "v1.0.0-alpha.10")

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0-alpha.10"


# --- Generation 2, case (d): a shorter identifier list has LOWER
# precedence than an extension of it that shares the same prefix (SemVer
# §11.4.4) — "alpha.1" (two identifiers) must outrank "alpha" (one
# identifier).
#
# DISCREPANCY FROM THE PLAN'S OWN PREDICTION: the plan states this case
# "is expected to already PASS" under generation 1. Empirically, it does
# NOT — the CURRENT (generation-1, unmodified) script prints
# "v1.0.0-alpha", not "v1.0.0-alpha.1". Root cause traced by hand and
# confirmed with a standalone `sort` repro: generation 1's final sort
# operates on the whole "key<TAB>tag" line, not the key alone. The key for
# "alpha" ends right after "...-alpha"; the key for "alpha.1" continues
# "...-alpha" + 0x01 (the "." sentinel) + the encoded "1". Immediately
# after the shared "-alpha" prefix, the "alpha" line has the 0x09 TAB
# byte (the key/tag field separator) while the "alpha.1" line has the
# 0x01 sentinel byte — and since 0x01 < 0x09, "alpha.1"'s whole line sorts
# BEFORE "alpha"'s, which is backwards: it makes "alpha" look like the
# later/higher-precedence tag when it selects it as the predecessor of
# v1.0.1's own line. So this test is a genuine RED against the current
# script, not an already-passing regression-pin as the plan assumed —
# reported here rather than silently reclassified.


def test_shorter_identifier_list_outranked_by_its_extension(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-alpha")
    _tag(tmp_path, "v1.0.0-alpha.1")

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0-alpha.1"


# --- Generation 2, case (e): a release outranks any prerelease of its own
# core (SemVer §11.3) — sanity-check that this UNCHANGED "~" sentinel
# behavior still holds under the new per-identifier key scheme.
#
# Expected to already PASS under the CURRENT (generation-1) script too —
# verified empirically: the current script already prints "v1.0.0" here.
# Written anyway as regression-pinning coverage per the plan.


def test_release_outranks_prerelease_of_same_core(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-rc1")
    _tag(tmp_path, "v1.0.0")

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0"


# --- R1 case 10: tags that are not strict `vMAJOR.MINOR.PATCH[-PRE]` semver
# are excluded from consideration entirely. `v0.0.1.1` is the trap: it has
# an extra fourth numeric component, so it is not valid semver, but under a
# naive version-sort (e.g. `sort -V`) with no semver regex filter applied,
# it sorts strictly between `v0.0.1` and `v0.0.2` and would be picked as the
# (wrong) predecessor of `v0.0.2`. `nightly` is kept as a second junk tag
# that sorts low regardless and never threatens the answer either way. Only
# an implementation that actually filters to strict semver before sorting
# can still land on the correct `v0.0.1` here. ---------------------------


def test_non_semver_tags_are_ignored(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "nightly")
    _tag(tmp_path, "v0.0.1.1")
    _tag(tmp_path, "v0.0.1")

    result = _run_script(tmp_path, "v0.0.2")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.0.1"


# --- R2 finding (fixed round 3): a numeric core component with a leading
# zero (e.g. "v0.01.0") is not valid strict semver ("01" is neither "0" nor
# a non-zero digit followed by digits) and must be excluded from
# consideration entirely.
#
# This must be tested in isolation, with NO valid sibling tag present (e.g.
# no "v0.1.0"), and must assert on an EMPTY result — not "predecessor is
# v0.1.0". A two-tag version (tagging both "v0.01.0" and "v0.1.0", then
# asserting the predecessor of "v0.2.0" is "v0.1.0") is tautological and
# passes identically whether or not the leading-zero rejection is applied:
# under the old permissive regex, "v0.01.0"'s zero-padded sort key is
# byte-identical to "v0.1.0"'s, so `sort -u` deduplicates them down to a
# single key, and the tie-break on the full "key<TAB>tag" line still
# happens to land on "v0.1.0" either way — the test cannot distinguish
# fixed from unfixed code. Tagging ONLY "v0.01.0" (no valid sibling) removes
# that escape hatch: with the old permissive regex the malformed tag is
# accepted and printed (wrong — a malformed tag reported as a real previous
# release); with the current strict-core regex it is rejected, leaving no
# valid candidate, so the correct output is empty. ------------------------


def test_leading_zero_core_component_is_ignored(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.01.0")
    # No valid sibling tag on purpose — see comment above for why that
    # matters.

    result = _run_script(tmp_path, "v0.2.0")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --- Round-4 fix (closes the Codex "21 vs 22 digit runs" recurring
# finding): the old sort key zero-padded every digit run to a FIXED width
# of 20 via `sprintf("%020d", numstr)`. That padding goes through awk's
# internal double conversion, which only has ~15-17 significant decimal
# digits of precision — so once a digit run exceeds ~17 significant digits
# (let alone the 20-character pad width itself), the padded value silently
# loses precision and can misorder two runs of DIFFERENT digit-length
# relative to each other. This is not a theoretical concern: it is
# empirically reproducible with the concrete pair below.
#
# 1000000000000000000000 (22 digits: "1" followed by 21 zeros, i.e.
# 10**21) is unambiguously the numerically larger prerelease vs
# 999999999999999999999 (21 nines, i.e. 10**21 - 1) — a plain
# "one more than the biggest N-digit number" relationship, no ambiguity
# about which is bigger. Under the OLD fixed-width-20 scheme, converting
# the 21-nines run through `sprintf("%020d", ...)` rounds it (via double
# precision loss) up to "1000000000000000000000" as well — an exact
# collision with the 22-digit run's own padded key — and the resulting tie
# is broken by the OLD script by picking the numerically SMALLER
# (21-digit) tag as "later", producing the wrong predecessor.
#
# The NEW length-prefixed scheme (see prev-release-tag.sh) never converts
# through a double at all — it uses `length()` and string ops only — so a
# 22-digit run always sorts strictly above a 21-digit run via the 4-digit
# length prefix, with no precision bound to exceed.
#
# (This replaces the previous test_prerelease_long_numeric_suffix_sorts_
# correctly, which used rc2 vs rc99999999999 (11 digits) — a pair well
# under the old 20-character pad width and under double's ~15-17 digit
# precision, so it passed identically under both the old and new schemes
# and did not actually exercise the overflow it was named for.)
#
# Generation-2 update: this fixture used to be `rc`-prefixed
# (rc999999999999999999999 / rc1000000000000000000000). Under generation
# 2's now-correct SemVer §11.4.3 precedence, an `rc`-prefixed digit run is
# a single ALPHANUMERIC identifier ("has a non-digit character") and is
# therefore compared byte-verbatim as a whole ASCII string, never numerically
# re-encoded at all — so with that fixture shape "rc1000..." (starts with
# byte "1") would correctly sort BELOW "rc999..." (starts with byte "9"),
# which is the opposite of what this test originally asserted and is not a
# bug; it is just testing the wrong code path for what this test's name and
# intent (proving the length-prefix numeric encoding has no double-precision
# loss for very long digit runs) requires. The digit-run-length-prefix
# encoding (`encode_num`, see above) is only ever applied to PURELY numeric
# identifiers, so this test now uses purely numeric identifiers (no "rc"
# prefix, no letters) to actually exercise that path — which still needs to
# be proven double-precision-safe.


def test_prerelease_numeric_suffix_beyond_double_precision_sorts_correctly(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v0.1.0-999999999999999999999")  # 21 nines: 10**21 - 1
    _tag(tmp_path, "v0.1.0-1000000000000000000000")  # 22 digits: 10**21

    result = _run_script(tmp_path, "v0.2.0")

    # Both identifiers are purely numeric, so per SemVer §11.4.3 they are
    # compared BY VALUE (via encode_num's length-prefix scheme), not as
    # ASCII strings — and the 22-digit number is genuinely the larger value.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v0.1.0-1000000000000000000000"


# --- Round-4 fix (closes the Codex "alpha.1 vs alpha-1" recurring
# finding): the sort key now replaces every "." with a sentinel byte
# (0x01, sorting below every real tag character) before running the
# digit-padding loop, while the ORIGINAL tag (unchanged) is still what
# gets filtered/printed — see prev-release-tag.sh next to `keysrc`/`gsub`.
#
# A dot boundary now sorts lower than any actual following character, so
# "alpha.1" (with the low sentinel standing in for its dot) sorts below
# "alpha-1" (unchanged "-" byte) — matching true SemVer §11.4 precedence
# for this pair (a two-field identifier list "alpha", "1" ranks below the
# single field "alpha-1"). This is the opposite result from before this
# fix, when raw byte comparison put "." (0x2E) above "-" (0x2D) and got it
# backwards.
#
# This is still not a full implementation of SemVer §11.4's recursive
# per-identifier numeric/alphanumeric precedence rules for every
# conceivable prerelease shape — it is a targeted fix for the concrete,
# common "." vs "-" byte-order mismatch. This project's release pipeline
# only ever produces plain `vX.Y.Z` and `-rcN` tags in practice (see
# release.yml's `--prerelease` path); this test exercises the general dot-
# separated-identifier case for correctness, not a shape this pipeline
# itself generates. ------------------------------------------------------


def test_prerelease_dot_separated_identifier_sorts_below_hyphenated(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-alpha.1")
    _tag(tmp_path, "v1.0.0-alpha-1")

    result = _run_script(tmp_path, "v1.0.1")

    # Fixed behavior: the dot-substitution sentinel makes "alpha.1" sort
    # below "alpha-1", so "v1.0.0-alpha-1" (the numerically/precedence-
    # later of the two per true SemVer) is correctly reported as the
    # predecessor of v1.0.1 — matching true SemVer §11.4 for this pair.
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "v1.0.0-alpha-1"


# --- Round-6 fix (Codex finding R2): the prerelease portion of SEMVER_RE
# was maximally permissive (`-[0-9A-Za-z.-]+`), so it accepted malformed
# prerelease identifiers that real SemVer forbids — a leading-zero numeric
# identifier (e.g. "01") and an empty identifier between two dots (e.g.
# "alpha..1"). "v1.0.0-01" is the trap: it is NOT valid semver (a purely
# numeric prerelease identifier with a leading zero is disallowed by
# SemVer §9), but the old permissive regex accepted it as a real semver
# tag and would report it as a genuine previous release. Tagging ONLY the
# malformed tag (no valid sibling) removes the escape hatch a two-tag
# version would have: under the old regex the malformed tag is accepted
# and printed (wrong); under the fixed regex it is rejected as non-semver,
# leaving no valid candidate, so the correct output is empty — mirroring
# the isolation technique used by test_leading_zero_core_component_is_
# ignored above for the analogous core-component case. ------------------


def test_malformed_prerelease_identifier_is_ignored(tmp_path):
    _init_repo(tmp_path)
    _tag(tmp_path, "v1.0.0-01")
    # No valid sibling tag on purpose — see comment above for why that
    # matters.

    result = _run_script(tmp_path, "v1.0.1")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# --- Fix round (Codex finding, blocking): `encode_num`'s length prefix is
# `sprintf("%04d", length(stripped))`, fixed-width only as long as
# `length(stripped)` itself stays a 1-4 digit number. Once a purely-numeric
# prerelease identifier (or core field) reaches 10,000+ digits,
# `length(stripped)` is a 5-digit number and `%04d` prints it at its
# natural (wider) width instead of truncating, so the length-prefix field
# silently stops being fixed-width — a 9999-digit run would then sort ABOVE
# a 10000-digit run, inverting correct by-value ordering. Rather than just
# re-documenting this as an accepted limit (the pattern that caused
# generation 1 to hit its review hard cap on a different recurring
# finding), `encode_num` now enforces the bound as an invariant: it fails
# loudly (non-zero exit, clear stderr message) instead of silently
# producing a wrong ordering.
#
# This test proves the enforcement fires: a purely-numeric prerelease
# identifier of 10,006 digits (well past the 9999-digit bound, no leading
# zero so it is still a syntactically valid SemVer numeric identifier)
# must make the script fail rather than return any answer at all — right
# or wrong. No sibling tag is needed; invoking the script directly with
# the crafted tag is the cheapest way to exercise `encode_num` on an
# oversized digit run, without fabricating 10,000+ real git tags.


def test_oversized_numeric_identifier_is_rejected_not_silently_misordered(tmp_path):
    _init_repo(tmp_path)
    big_tag = "v1.0.0-1" + ("0" * 10005)  # 10,006-digit numeric identifier

    result = _run_script(tmp_path, big_tag)

    assert result.returncode != 0
    assert result.stdout.strip() == ""
    assert "9999" in result.stderr
