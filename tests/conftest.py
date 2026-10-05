"""tests/conftest.py -- stop the UTF-8 stdout guard from closing pytest's capture.

THE PROBLEM
-----------
Every experiment script in this project opens with:

    if not getattr(sys.stdout, "_nvp_utf8", False):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                      errors="replace")
        sys.stdout._nvp_utf8 = True

which exists because a cp1252 console cannot print the result tables. Under
pytest, `sys.stdout` is a capture object, so the wrapper is built around
`capture.buffer` -- and when that wrapper is later garbage-collected it CLOSES
the underlying buffer. pytest then writes its end-of-session summary to a closed
file:

    ValueError: I/O operation on closed file.

and the process exits 1 with no summary, even when every test passed. A suite
that fails while passing is worse than one that simply fails: the exit code
stops meaning anything, and a real failure later hides inside the same error.

THE FIX
-------
Set the module's own sentinel before any test imports an experiment module. The
guard is `if not getattr(sys.stdout, "_nvp_utf8", False)`, so marking the
capture object satisfies it and the rewrap never happens. No production code
changes, and the guard still does its job outside pytest.

`autouse` and function scope because pytest swaps the capture object per test;
a session-scoped hook would mark one object and miss the rest.
"""
import sys

import pytest


def _mark(stream):
    try:
        if stream is not None and not getattr(stream, "_nvp_utf8", False):
            stream._nvp_utf8 = True
    except (AttributeError, TypeError):
        # A stream that refuses attributes cannot be marked; the rewrap will
        # then happen as it does in production, which is the status quo.
        pass


@pytest.fixture(autouse=True)
def _no_stdout_rewrap():
    _mark(sys.stdout)
    _mark(sys.stderr)
    yield


def pytest_configure(config):
    _mark(sys.stdout)
    _mark(sys.stderr)
