# Register asyncio mark so pytest does not warn about unknown marks.
# Live async tests (test_openai_client_live, etc.) require pytest-asyncio.
import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-colt-integration",
        action="store_true",
        default=False,
        help="Run COLT integration tests that require the cloned repo + a "
             "trained checkpoint + a GPU. Otherwise skipped by default.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "asyncio: mark test as an asyncio coroutine (requires pytest-asyncio)",
    )
    config.addinivalue_line(
        "markers",
        "colt_integration: heavy integration test for baselines/colt.py — "
        "requires clone + checkpoint + GPU. Run with --run-colt-integration.",
    )


def pytest_collection_modifyitems(config, items):
    """Skip COLT integration tests unless explicitly requested."""
    if config.getoption("--run-colt-integration") or os.environ.get("COLT_RUN_INTEGRATION") == "1":
        return
    skip_colt = pytest.mark.skip(
        reason="Pass --run-colt-integration or set COLT_RUN_INTEGRATION=1 to run."
    )
    for item in items:
        if "colt_integration" in item.keywords:
            item.add_marker(skip_colt)
