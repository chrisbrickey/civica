from pathlib import Path

from mypy import api

_SRC = str(Path(__file__).parent.parent / "src")


def test_mypy_strict_passes() -> None:
    stdout, stderr, exit_code = api.run([_SRC])
    assert exit_code == 0, f"mypy failed:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
