"""Keep mocked reconstruction and its assertions in the test interpreter.

Process isolation itself is covered by test_print_worker.py.
"""

import pytest

from server import print_api


@pytest.fixture(autouse=True)
def inline_print_worker(monkeypatch):
    monkeypatch.setattr(print_api, "execute_job", print_api._make_stl)
