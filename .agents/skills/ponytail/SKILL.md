---
name: ponytail
description: >-
  Lazy senior dev mode for code changes: understand the problem, reuse what already exists, 
  keep the diff as small as possible, and leave one runnable check behind.
---

# Ponytail, Lazy Senior Dev Mode

Use this skill when you are about to change code and want the smallest correct diff, not the fanciest one.

## The ladder

Before writing code, climb this ladder and stop at the first rung that holds:

1. Does this need to be built at all? If not, do nothing.
2. Does it already exist in this codebase? Reuse it.
3. Does the standard library already do this? Use it.
4. Does a native platform feature cover it? Use it.
5. Does an already-installed dependency solve it? Use it.
6. Can it be one line? Make it one line.
7. Only then write the minimum code that works.

The ladder runs **after** you understand the problem, not instead of it. Read the task, trace the real flow end to end, then climb.

## Non-negotiables

- Fix the root cause, not the symptom.
- If a bug report names one failing path, grep every caller of the function you touch and fix the shared function once when possible.
- No abstractions that were not explicitly requested.
- No new dependency if it can be avoided.
- No boilerplate nobody asked for.
- Prefer deletion over addition.
- Prefer boring over clever.
- Use the fewest files possible.
- The smallest change in the wrong place is not lazy; it is a second bug.

## When two options are close

Pick the edge-case-correct option.

If two stdlib approaches are the same size, choose the one that is safer or more correct for boundary cases.

If a deliberate simplification is worth keeping, mark it with a `ponytail:` comment that names:

- the ceiling you are accepting, and
- the upgrade path when that ceiling stops being acceptable.

Example:

```python
# ponytail: linear scan is fine for the current N; switch to indexed lookup if this grows.
```

## What still deserves care

Do **not** be lazy about:

- understanding the problem before changing code,
- input validation at trust boundaries,
- error handling that prevents data loss,
- security,
- accessibility,
- platform or hardware calibration,
- anything explicitly requested.

A small diff you do not understand is not efficiency.

## Implementation habits

- Reuse existing helpers, utilities, and patterns before writing new ones.
- If one shared guard fixes all callers, do that instead of patching each caller separately.
- Keep the fix local and targeted.
- Avoid broad refactors unless the task truly needs them.
- If you can remove code instead of adding code, remove it.

## Minimum proof

Non-trivial logic leaves behind one runnable check:

- an assert-based demo,
- a tiny self-check script, or
- one small test file.

Keep it as small as possible. No framework ceremony unless the project already requires it. Trivial one-line changes do not need a test.

## Before you finish

- Re-read the touched flow end to end.
- Confirm no sibling caller is still broken.
- Make sure the deliberate tradeoff, if any, is documented with a `ponytail:` comment.
- Keep the final diff as small as it can honestly be.

