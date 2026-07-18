import subprocess
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_TARGETS = ["src", "tests"]


def test_no_unused_imports_or_variables() -> None:
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
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "autoflake found unused imports/variables. "
        "Run the fix command in CLAUDE.md to auto-remove.\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
