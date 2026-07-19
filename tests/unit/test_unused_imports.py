import subprocess

import pytest

_TARGETS = ["src", "tests"]


def test_no_unused_imports_or_variables(pytestconfig: pytest.Config) -> None:
    result = subprocess.run(
        [
            "uv", "run", "autoflake",
            "--check-diff",
            "--recursive",
            "--remove-all-unused-imports",
            "--remove-unused-variables",
            "--ignore-init-module-imports",
            *_TARGETS,
        ],
        cwd=pytestconfig.rootpath,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "autoflake found unused imports/variables. "
        "Run the fix command in CLAUDE.md to auto-remove.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
