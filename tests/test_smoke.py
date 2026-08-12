"""Phase 0 smoke test: package imports and runtime baseline."""

import bayan_slm_engine


def test_package_version() -> None:
    assert bayan_slm_engine.__version__ == "0.1.0"


def test_python_runtime() -> None:
    import sys

    assert sys.version_info >= (3, 12)
