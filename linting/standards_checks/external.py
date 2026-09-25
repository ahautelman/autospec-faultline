"""Delegation to the linters that own the rest of the mechanical rules.

The native checks cover line thresholds and docstring sections; everything else
the standard can check mechanically — naming, docstring presence, param counts,
magic numbers, broad excepts — is enforced by ruff and eslint. Both are pointed
at the configs this package ships, never at whatever config happens to sit in
the repository being checked, so the house rules apply even in a foreign tree."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .models import PYTHON_EXTS, TS_EXTS

# Shipped beside the code that applies it, so it travels with the install.
RUFF_CONFIG = Path(__file__).resolve().with_name("ruff.toml")

# The eslint half is a separate npm package: it wraps eslint with the matching
# config, resolved inside that package so the config's plugin imports work.
ESLINT_COMMAND = "code-check-eslint"

RUFF_INSTALL_HINT = "uv tool install ./linting"
ESLINT_INSTALL_HINT = "npm install --global ./linting/eslint"


def _ruff_executable() -> str:
    """Locate the ruff that belongs to this install.

    How:
        ruff is a dependency of this package, so it sits beside the interpreter
        running it; that copy is preferred over any ruff further up PATH, which
        could be a different version configured for a different project.

    Returns:
        str: Path to the bundled ruff, or the bare name to look up on PATH.
    """
    bundled = Path(sys.executable).parent / "ruff"
    return str(bundled) if bundled.exists() else "ruff"


def run_external(executable: str, command: list[str]) -> tuple[str | None, bool]:
    """Run an external linter, returning its output and whether it found anything.

    Args:
        executable (str): Executable to probe, by name on PATH or absolute path.
        command (list[str]): Full command to run when the tool exists.

    How:
        A missing tool returns None rather than raising, so one half of the
        toolchain being unavailable still lets the other half report. The
        verdict comes from the exit code, not from whether anything was
        printed: a clean ruff still writes "All checks passed!" to stdout.

    Returns:
        tuple[str | None, bool]: The tool's combined output (None if it isn't
            installed), and True when it reported violations.
    """
    if shutil.which(executable) is None:
        return None, False
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as err:
        return f"{executable} could not run: {err}", True
    return (result.stdout + result.stderr).strip(), result.returncode != 0


def echo_tool(name: str, output: str | None, install_hint: str = "") -> None:
    """Print an external linter's output, or note how to install it if absent.

    Args:
        name (str): Display name of the tool.
        output (str | None): Its output, or None when the tool isn't installed.
        install_hint (str): Command that installs the tool, shown on skip so the
            user can fix the gap instead of silently losing that check's coverage.

    How:
        None means "not installed" — the native checks still ran, so this is a
        heads-up, not an error; empty output means the tool ran and said nothing.
        Whether the run counts as a failure is the exit code's business, not
        this function's.
    """
    if output is None:
        suffix = f" — install with: {install_hint}" if install_hint else ""
        print(f"{name}: not installed (skipped){suffix}")
    elif output:
        print(f"{name}:\n{output}")
    else:
        print(f"{name}: no violations.")


def _with_extensions(paths: list[str], extensions: set[str]) -> list[str]:
    """Keep only the paths a given linter can actually read.

    Args:
        paths (list[str]): Files gathered for checking.
        extensions (set[str]): Suffixes the linter handles.

    How:
        Filters by suffix so ruff is never handed a .ts file (and eslint never a
        .py one), which would report a parse error instead of a real violation.

    Returns:
        list[str]: The subset of paths belonging to that linter.
    """
    return [path for path in paths if Path(path).suffix in extensions]


def run_optional_linters(paths: list[str]) -> bool:
    """Run ruff and eslint over the paths they own and echo their reports.

    Args:
        paths (list[str]): Files to hand to the linters.

    How:
        Splits the paths by language and runs each tool independently, so one
        missing tool doesn't suppress the other. A language with no files present
        is skipped silently rather than reported as clean. Both tools always run
        before the verdict is returned, so one failing does not hide the other's
        report.

    Returns:
        bool: True when either tool reported violations. A tool that isn't
            installed reports nothing and so cannot fail the run.
    """
    violations = False
    python_paths = _with_extensions(paths, PYTHON_EXTS)
    if python_paths:
        ruff = _ruff_executable()
        # --no-cache keeps ruff from dropping a .ruff_cache into the repository
        # being checked, where it could be committed or fail on a read-only tree.
        ruff_out, ruff_failed = run_external(
            ruff,
            [ruff, "check", "--no-cache", "--config", str(RUFF_CONFIG), *python_paths],
        )
        echo_tool("ruff", ruff_out, RUFF_INSTALL_HINT)
        violations |= ruff_failed

    typescript_paths = _with_extensions(paths, TS_EXTS)
    if typescript_paths:
        eslint_out, eslint_failed = run_external(
            ESLINT_COMMAND, [ESLINT_COMMAND, *typescript_paths]
        )
        echo_tool("eslint", eslint_out, ESLINT_INSTALL_HINT)
        violations |= eslint_failed

    return violations
