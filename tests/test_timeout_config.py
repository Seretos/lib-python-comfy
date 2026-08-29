"""Driving tests for R5 (package #49 / epic #42): a hanging test must be
killed by pytest-timeout's per-test bound instead of running until the CI
job's 10-minute cap.

Both tests here depend on config added to ``pyproject.toml`` by this
package's production change:

- ``[tool.pytest.ini_options]`` gains ``timeout = 60`` and
  ``timeout_method = "thread"`` (thread method is Windows-safe).
- ``addopts`` gains ``-p pytester`` so the ``pytester`` fixture is
  available for the driving test below, without adding a
  ``tests/conftest.py`` (this repo's rootdir/testpaths layout would hit
  pytest's "non-top-level conftest" restriction for a ``pytest_plugins``
  declaration there).

Before that change:

- ``test_hanging_test_is_killed_by_timeout`` fails at fixture setup with
  "fixture 'pytester' not found", because ``-p pytester`` is not yet in
  ``addopts``.
- ``test_real_session_timeout_configuration`` fails with a ``ValueError``
  from ``float(pytestconfig.getini("timeout"))``: the ``pytest-timeout``
  plugin happens to already be installed in this dev environment, so it
  auto-registers its ini options, but ``pyproject.toml`` does not yet set
  a value for them, so ``getini("timeout")`` returns the plugin's default
  empty string rather than ``"60"``.

``timeout_method = "thread"`` kills the whole worker process on expiry, so
outcomes/reports are never written for a timed-out run. The driving test
therefore asserts on the subprocess exit code and on ``"Timeout"`` appearing
in stdout, not on ``result.assert_outcomes(...)``.
"""


def test_real_session_timeout_configuration(pytestconfig):
    """The live pytest session enforces a 60s, thread-based per-test timeout."""
    assert float(pytestconfig.getini("timeout")) == 60
    assert pytestconfig.getini("timeout_method") == "thread"


def test_hanging_test_is_killed_by_timeout(pytester, pytestconfig):
    """A test whose await never resolves fails within the configured bound
    instead of hanging until the job's 10-minute cap."""
    # Real session's configured method (thread, per pyproject.toml) — the
    # inner throwaway suite must be killed the same way the real suite is.
    timeout_method = pytestconfig.getini("timeout_method")

    pytester.makepyfile(
        test_hangs_forever="""
        import asyncio

        def test_blocks_on_an_await_that_never_resolves():
            asyncio.run(asyncio.Event().wait())
        """
    )
    # Deliberately short (1-2s, not the real 60s) so this driving test
    # itself stays fast — noted as an acceptable deviation in the plan.
    pytester.makeini(
        f"""
        [pytest]
        timeout = 2
        timeout_method = {timeout_method}
        """
    )

    result = pytester.runpytest_subprocess()

    assert result.ret != 0
    assert "Timeout" in result.stdout.str()
