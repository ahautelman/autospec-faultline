"""The `code-check` command: the entry point for the coding-standards rules.

Wires discovery, the native checks (50 lines/function, 250 lines/file, the
required docstring sections), reporting, and the external linters together. Run
it on the files you changed, or on a directory:

    code-check faultline/spectrl/parser.py    # specific files
    code-check faultline/                     # a directory, recursively

Exit code is non-zero when anything is reported, a native violation or one
from ruff/eslint, so CI and pre-commit can fail on it. The heavy lifting lives
in the standards_checks package."""

import sys

from .discovery import gather_paths
from .external import run_optional_linters
from .line_checks import check_file
from .report import print_report


def main() -> int:
    """Run the native checks, then the external linters.

    How:
        Reads the files and directories to check from the command line, collects
        the native findings first (they need nothing installed) and prints them,
        then hands the same files to ruff and eslint. Both halves always run, so
        one report is never hidden by the other, and either failing makes the
        exit code non-zero so CI and pre-commit can fail the commit.

    Returns:
        int: Process exit code (0 clean, 1 violations found).
    """
    files = gather_paths(sys.argv[1:])
    findings = [finding for path in files for finding in check_file(path)]
    print_report(findings)
    external_violations = run_optional_linters([str(path) for path in files])
    return 1 if findings or external_violations else 0


if __name__ == "__main__":
    sys.exit(main())
