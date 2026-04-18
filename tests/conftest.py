"""Shared pytest fixtures.

Detects whether the original pytmatrix is importable; parity tests use this
to skip gracefully when it isn't available.
"""

from __future__ import annotations

import pytest


def _has_pytmatrix() -> bool:
    try:
        import pytmatrix  # noqa: F401
        from pytmatrix.tmatrix import Scatterer  # noqa: F401
    except Exception:
        return False
    return True


HAS_PYTMATRIX = _has_pytmatrix()


@pytest.fixture
def skip_if_no_pytmatrix():
    if not HAS_PYTMATRIX:
        pytest.skip("pytmatrix not available — parity test skipped.")
