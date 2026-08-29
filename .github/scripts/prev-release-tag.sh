#!/usr/bin/env bash
# Print the released `v*` tag that immediately precedes <new-tag>, by
# explicit semver comparison — never by assuming <new-tag> is (or is not)
# already present in `git tag --list`. Prints nothing (and still exits 0)
# when no tag precedes <new-tag> (first release, or first release of a
# new major/minor line). Never blocks a release: always exits 0.
#
# Why this exists (ticket #51): release.yml rebuilds the release branch
# fresh off main HEAD on every run (`git checkout -B "$BRANCH"`), so
# consecutive release commits are never ancestors of one another. That
# breaks `gh release create --generate-notes`'s default ancestor walk,
# which silently falls back to "notes from the beginning of history".
# Passing this script's output as `--notes-start-tag` gives GitHub an
# explicit start point so it uses a merge-base compare instead.
#
# Usage: prev-release-tag.sh <new-tag>
#   e.g. prev-release-tag.sh v1.2.3
#
# Requires the repo to have been checked out with full history
# (fetch-depth: 0) so `git tag --list` sees every prior release tag.
set -euo pipefail

NEW_TAG="${1:?usage: prev-release-tag.sh <new-tag>}"

# Strict semver: vMAJOR.MINOR.PATCH with an optional -PRERELEASE suffix.
# Anchored full-string match — this excludes malformed tags such as
# v0.0.1.1 (an extra fourth numeric component) that a looser filter (or
# no filter, relying on `sort -V`) would otherwise let through and sort
# between v0.0.1 and v0.0.2.
#
# Each of major/minor/patch is `(0|[1-9][0-9]*)`, not a bare `[0-9]+`:
# strict SemVer forbids a leading zero in a numeric core component (e.g.
# `v01.0.0` is not valid semver), so a component must be exactly "0" or a
# non-zero digit followed by any digits.
#
# The prerelease part after `-` validates the actual SemVer prerelease
# grammar (SemVer §9): a dot-separated list of identifiers, each of which
# is either a numeric identifier with no leading zero (`0` or
# `[1-9][0-9]*`) or an alphanumeric identifier (at least one non-digit
# character, letters/digits/hyphens allowed — MAY have leading zeros,
# since the "no leading zero" rule applies only to identifiers that are
# entirely numeric, e.g. "rc01" is a single alphanumeric identifier and
# stays valid, but a bare "01" is a leading-zero numeric identifier and is
# rejected). An empty identifier between two dots (e.g. "alpha..1") is
# rejected too, since neither alternative can match zero characters. This
# is still not the byte-for-byte ERE the SemVer BNF would produce, but it
# rejects the two concrete malformed shapes ("01", "alpha..1") that the
# previous maximally-permissive `[0-9A-Za-z.-]+` let through — see
# test_malformed_prerelease_identifier_is_ignored.
ONE_ID='(0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)'
SEMVER_RE="^v(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)\\.(0|[1-9][0-9]*)(-${ONE_ID}(\\.${ONE_ID})*)?\$"
# No build-metadata (`+...`) component: this is intentional, not a gap
# against this pipeline's actual inputs. release.yml's "Validate version is
# semver" step gates every `version` input this repo will ever turn into a
# tag with `^[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$` — no `+` alternative
# anywhere in that pattern — so a `+build` suffix can never reach the
# "Stamp...tag..." step that creates `v$V`, and therefore never reach
# `git tag --list` here either. Full SemVer does allow build metadata, but
# no tag with a `+` in it can exist in this repo's tag history, so nothing
# this filter excludes was ever a reachable candidate.

{
  # Known, accepted limitation (pre-existing in the tag/release pipeline
  # design, not introduced by this ticket): candidates come from *local git
  # tags*, not confirmed GitHub Releases. release.yml's "Stamp...tag...
  # force-push" step creates and pushes the tag before the later "Create
  # GitHub Release" step runs; if that later step ever fails, the tag is
  # already pushed with no Release published, and a subsequent run of this
  # script would still treat that orphaned tag as a legitimate predecessor.
  # Hardening this (e.g. cross-checking candidates against the GitHub API's
  # published releases instead of `git tag --list`) would be a separate,
  # larger change — a new dependency on `gh`/the API, new error handling,
  # new failure modes — and is out of scope here unless this failure mode is
  # actually observed in practice.
  #
  # Existing local tags, filtered to strict semver. `|| true`: grep exits
  # 1 when no tag matches (e.g. very first release ever), which must
  # never be treated as a script failure under `set -e`/`pipefail`.
  git tag --list 'v*' | grep -E "$SEMVER_RE" || true
  # Explicitly append the new tag — correctness must not depend on it
  # already existing locally (release.yml tags locally before this
  # script runs, but nothing here should assume that ordering).
  echo "$NEW_TAG"
} | awk '
  # Encode a digit run (already leading-zero-stripped per SEMVER_RE, but
  # stripped defensively here too) as a 4-digit length prefix followed by
  # the digits themselves, so numeric fields compare correctly under a
  # plain byte sort (e.g. "2" sorts below "10") with no fixed bound on how
  # many digits a run can have in practice (4 digits of length-prefix
  # supports runs up to 9999 digits long — unbounded for any real-world
  # version number). Two numbers of different canonical length always
  # differ in the length-prefix, which itself always sorts correctly since
  # it is fixed-width and small; two numbers of the same canonical length
  # compare correctly via plain byte comparison of their (leading-zero-
  # free) digits. `length()` and string ops only — never routed through a
  # double — so precision never degrades no matter how many digits long
  # the run is (see test_prerelease_numeric_suffix_beyond_double_precision_
  # sorts_correctly, which uses 21- and 22-digit purely-numeric prerelease
  # identifiers).
  #
  # The "unbounded in practice" claim above only holds while the digit run
  # itself stays within 9999 digits: `%04d` is fixed-width only up to that
  # bound — a run of 10000+ digits makes `length(stripped)` a 5-digit
  # number, which `%04d` prints at its natural (wider) width instead of
  # truncating, so the length-prefix field silently stops being
  # fixed-width and a 9999-digit run would sort ABOVE a 10000-digit run
  # (inverting correct by-value ordering). Rather than let that produce a
  # silently wrong answer, enforce the bound as an invariant: a numeric
  # field this large is not a real version number, and something is badly
  # wrong upstream — fail loudly instead of guessing, mirroring how the
  # script already rejects a missing `$1` via `${1:?...}` above.
  function encode_num(numstr,    stripped) {
    stripped = numstr
    sub(/^0+/, "", stripped)
    if (stripped == "") stripped = "0"
    if (length(stripped) > 9999) {
      print "prev-release-tag.sh: numeric field has " length(stripped) \
        " digits, exceeding the 9999-digit bound the length-prefix " \
        "encoding can represent correctly; refusing to produce a " \
        "possibly-wrong ordering" > "/dev/stderr"
      exit 1
    }
    return sprintf("%04d", length(stripped)) stripped
  }
  BEGIN {
    # Structural joiner between key fields/identifiers. Deliberately NOT
    # "\001": the final pipeline sorts whole "key<TAB>tag" lines, not the
    # key alone, so whenever one candidates key is a strict prefix of
    # anothers (e.g. identifier list ["alpha"] vs ["alpha","1"]), the
    # shorter ones line continues with the 0x09 TAB field separator right
    # where the longer ones continues with this joiner. A prior version of
    # this script used "\001" (0x01) there, and since 0x01 < 0x09 that made
    # the LONGER key (more identifiers, i.e. higher SemVer precedence) sort
    # BEFORE the shorter one — backwards. SEP must sort above TAB (0x09) so
    # that "ends here" (followed by TAB) always sorts below "continues"
    # (followed by SEP + more content) — see
    # test_shorter_identifier_list_outranked_by_its_extension. 0x1F (unit
    # separator) also sorts below every real tag byte we key on (digits,
    # letters, "-" 0x2D, the class bytes "0"/"1", and the "~" 0x7E
    # no-prerelease sentinel), so it never collides with real content.
    SEP = sprintf("%c", 31)
  }
  {
    tag = $0

    # Core vs prerelease split: strip the leading "v", then the FIRST "-"
    # (if any) separates the numeric core from the prerelease suffix — safe
    # because SEMVER_RE guarantees the core itself contains no "-", so the
    # first "-" in the body is always the core/prerelease boundary, even
    # when the prerelease itself starts with "-" (a legal alphanumeric
    # identifier per SemVer §9, e.g. "-x").
    body = substr(tag, 2)
    dash = index(body, "-")
    if (dash > 0) {
      core = substr(body, 1, dash - 1)
      prerelease = substr(body, dash + 1)
    } else {
      core = body
      prerelease = ""
    }

    # Core: major/minor/patch, each length-prefix-encoded, joined by SEP.
    n = split(core, cf, ".")
    key = ""
    for (i = 1; i <= n; i++) {
      if (i > 1) key = key SEP
      key = key encode_num(cf[i])
    }

    if (prerelease == "") {
      # No prerelease: "~" (0x7E) sentinel so this sorts above any
      # prerelease of the same core (e.g. v1.0.0 sorts above v1.0.0-rc1)
      # under LC_ALL=C byte order — unchanged from the previous scheme.
      key = key "~"
    } else {
      # Has a prerelease: split into dot-separated identifiers (SemVer
      # §9) and key each one with a leading class byte — "0" for a purely
      # numeric identifier, "1" for an alphanumeric one — since "0" < "1"
      # this makes a numeric identifier ALWAYS sort below an alphanumeric
      # one regardless of what digits/letters follow (SemVer §11.4.3),
      # which a byte-first comparison without this class distinction gets
      # backwards for pairs like numeric "2" vs alphanumeric "1x" — see
      # test_numeric_identifier_outranked_by_alphanumeric_identifier.
      #
      # A numeric identifier is then length-prefix-encoded like a core
      # field, so same-position numeric prerelease fields compare by
      # value, not lexically (e.g. "alpha.10" outranks "alpha.2") — see
      # test_dot_separated_numeric_field_compares_by_value.
      #
      # An alphanumeric identifier is kept VERBATIM — its digits are never
      # touched/re-encoded — so it compares as a plain ASCII string, per
      # SemVer section 11.4.3 (identifiers with letters or hyphens are
      # compared lexically in ASCII sort order). This is what makes "rc2"
      # outrank "rc10" (first differing byte, digit 2 at 0x32 above digit
      # 1 at 0x31) — the true SemVer-correct answer, replacing generation
      # 1s deliberate (non-spec) numeric-value heuristic for such
      # suffixes — see test_rc_suffix_ordering_is_semver_correct.
      m = split(prerelease, pf, ".")
      for (i = 1; i <= m; i++) {
        key = key SEP
        if (pf[i] ~ /^[0-9]+$/) {
          key = key "0" encode_num(pf[i])
        } else {
          key = key "1" pf[i]
        }
      }
    }

    print key "\t" tag
  }
' | LC_ALL=C sort -u | awk -F'\t' -v new="$NEW_TAG" '
  $2 == new { print prev; found = 1; exit }
  { prev = $2 }
  END { if (!found) exit 0 }
'
