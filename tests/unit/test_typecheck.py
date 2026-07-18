import pytest
from mypy import api


def test_mypy_strict_passes(pytestconfig: pytest.Config) -> None:
    src = str(pytestconfig.rootpath / "src")
    stdout, stderr, exit_code = api.run([src])
    assert exit_code == 0, f"mypy failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
