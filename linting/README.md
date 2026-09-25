# code-check

Custom linter for the FaultLine project, the companion to
`../.agents/skills/coding-standards/SKILL.md`. The skill states the rules; this
folder holds the mechanical subset of them that a machine can check.

Built as **two packages, not a folder of scripts.** Packages give us commands on
`PATH`, so any coding agent (and CI) can call them like any other CLI command.

---

## What it checks

`code-check <paths>` reports two things:

1. **Native checks**, walking the Python AST: the 50-line function and 250-line
   file limits (docstring lines excluded) and rule 6's required
   `Args:`/`How:`/`Returns:` sections.
2. **ruff** (Python) and **eslint** (TypeScript/JS), which own naming casing,
   docstring presence, parameter counts, magic values and broad excepts.

Both halves always run, and either one reporting makes the exit code non-zero: a
violation is a violation whichever half caught it. A linter that is not
installed reports nothing and cannot fail the run.

FaultLine is a Python-only project, so in practice only the native checks and
ruff run; the eslint half is kept so the tooling stays usable if TypeScript or
JavaScript is added later.

**Exclusions.** Vendored and generated trees are skipped (`.venv`, `node_modules`,
`dist`, `build`, `__pycache__`, ...). `autospec-original-code/` is the
unmodified AutoSpec reference implementation kept for comparison, so it is
skipped as well. Test files (`tests/`) keep the docstring-presence and
magic-value rules off, because tests assert on literal, independently derived
expected values and the native checker waives their docstring content.

Explicit paths are (very) helpful when working in a repo that is not fully
formatted according to the coding standards.

---

## Layout

```text
pyproject.toml                   dist "code-check", command: code-check, depends on ruff
standards_checks/
  cli.py                         the command's entry point
  discovery.py                   turns arguments into a list of source files
  line_checks.py                 the 50/250 limits
  docstring_checks.py            the Args:/How:/Returns: sections
  models.py  report.py           shared types, terminal output
  external.py                    hands the files to ruff and eslint
  ruff.toml                      the Python ruleset, shipped inside the package
eslint/
  package.json                   dist "code-check-eslint", depends on eslint + plugins
  eslint.config.mjs              the TypeScript/JS ruleset
  bin/code-check-eslint.mjs      runs eslint with that config
```

---

## Install

The linting is treated like any other package, from the project root.

```bash
uv tool install ./linting
npm install --global --install-links ./linting/eslint
```

That puts `code-check` and `code-check-eslint` on `PATH`. Re-run the install
after editing anything under `linting/` (the `ruff.toml` and `eslint.config.mjs`
rulesets ship inside the packages, so an installed copy is a snapshot).

---

## Errors Encountered

**`npm install --global <dir>` symlinks; it does not install.** npm treats a
local path like `npm link` — it linked the source folder and installed none of
the dependencies, so `code-check-eslint` failed with `Cannot find package
'eslint'`. `--install-links` makes it pack and install properly.

**ruff cache.** `external.py` passes `--no-cache` so linting does not leave
`.ruff_cache/` in the repository being checked. eslint caches only when asked.

**`D209` contradicts the standard it enforces.** It wants a multi-line
docstring's closing quotes on their own line, which rule 7's own file-docstring
example does not do. It is in the ignore list for that reason, not by accident.

**`D403` over-reaches on openers.** It wants a docstring's first word
capitalised, but some docstrings legitimately open with a lowercase code
identifier or fixed syntax (for example `` ``achieve b`` ``). It is ignored for
that reason.

**Method docstrings are not required.** `D102` (public methods), `D105` (magic
methods) and `D107` (`__init__`) are ignored: an override is documented by the
base method, protocol dunders are self-evident, and a class docstring already
documents construction. Requiring duplicate per-method docstrings would push
reviewers to write boilerplate rather than meaning. The standard in
`.agents/skills/coding-standards/SKILL.md` states the same carve-out.

---

## Modifying the Linter

- **A Python rule** → `standards_checks/ruff.toml`, or `line_checks.py` / `docstring_checks.py` for the native ones.
- **A TypeScript rule** → `eslint/eslint.config.mjs`.
- **A tool version** → `pyproject.toml` for ruff, `eslint/package.json` for the Node side. Both are pinned exactly.

Whatever you change, update `.agents/skills/coding-standards/SKILL.md` in the
same edit if it changes what the agent is being asked to do. **A linter and a
standard that disagree is a recipe for disaster!**
