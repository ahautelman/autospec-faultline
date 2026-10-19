# FaultLine implementation plan

This plan builds the FaultLine library from scratch. The goal is a solid,
tested implementation of the AutoSpec pipeline that fits the research questions
in [`README.md`](../README.md:1). The original AutoSpec code in
[`autospec-original-code/`](../autospec-original-code) stays read-only and is
used only for comparison.

Work proceeds in phases. Each item is one deliverable. Items are ordered so
that each one can be tested before the next starts.

Source documents:

- [`AutoSpec_research_paper.md`](../AutoSpec_research_paper.md:1) — how AutoSpec
  works, with pseudocode and reference results.
- [`README.md`](../README.md:1) — project goal, research questions, and metrics.
- [`.agents/skills/coding-standards/SKILL.md`](../.agents/skills/coding-standards/SKILL.md:1)
  — code standards.
- [`.agents/skills/writing-tests/SKILL.md`](../.agents/skills/writing-tests/SKILL.md:1)
  — test standards.

## Phase 0 — Setup

- [x] Document the 9-rooms layout and add the validation target table to
  [`AutoSpec_research_paper.md`](../AutoSpec_research_paper.md:1).
- [x] Rewrite [`README.md`](../README.md:1) as project documentation.
- [x] Scaffold the `faultline/` package with file-level docstrings, re-exports,
  and a module-to-paper-section map.
- [x] Add dependencies to [`pyproject.toml`](../pyproject.toml:1) and create
  `tests/`. Confirm `code-check` runs clean.

## Phase 1 — SpectRL front end

- [x] Implement the predicate layer: `evaluate(state) -> bool`, geometric atoms
  (box, disc, half-plane), and `And`/`Or`/`Not`.
- [x] Implement the SpectRL AST: `Achieve`, `Ensuring`, `Seq`, `Choice`.
- [x] Implement the recursive-descent parser for the grammar in section 2.2.
- [x] Implement the canonical printer for round-tripping.
- [x] Tests: predicate boundaries, parser round-trip, precedence, and errors.

## Phase 2 — Abstract graph and compiler

- [x] Implement `AbstractGraph`, `Vertex`, and `Edge`, with edits, queries,
  deep copy, and acyclicity.
- [x] Implement the compiler for `Achieve` and the source-region label.
- [x] Extend the compiler for `Seq` and `Choice`.
- [x] Extend the compiler for `Ensuring`.
- [x] Tests: structural assertions and the exhaustive equivalence test
  `zeta models phi` if and only if `zeta models G`.

## Phase 3 — Environment and learners

- [x] Implement the 9-rooms layout: a 3x3 room grid with walls and doors, plus
  rectangular START, MID, and GOAL regions.
- [x] Implement the `Rooms9` gymnasium environment: 4-D state
  `(x, y, theta, d)`, 2-D action `(v, theta)`, point-mass dynamics, wall
  collision, `set_abstract_states`, a dense reward, and `info['is_success']`.
- [x] Define the dimension-agnostic `Learner` protocol: `train_edge`,
  `estimate_reach`, and an explicit cannot-refine path.
- [x] Implement the fast deterministic tabular learner.
- [x] Implement the Stable-Baselines3 PPO learner with the section 6.1 recipe.
- [x] Tests: dynamics, collision, region membership, and the learner contract
  for both learners.

## Phase 4 — Refinement operators

- [x] Implement geometry helpers: convex hull, set difference and intersection,
  and a half-plane partition.
- [x] Implement `ReachRefine` and `AvoidRefine`, with empty-sample guards.
- [x] Implement `SeqRefine`.
- [x] Implement `AddRefine`.
- [x] Implement `PastRefine`.
- [x] Implement `OrRefine`.
- [x] Tests per operator: exact behaviour on synthetic trajectories, trigger and
  no-trigger cases, and the guard on zero successful trajectories.
- [x] Tests: soundness for all four operators. For every trajectory,
  `zeta models G_r` implies `zeta models G`.

## Phase 5 — Algorithm 1 loop

- [x] Implement `AutoSpecConfig` and a structured run record.
- [x] Implement the loop: train, scan edges in topological order, estimate
  success, try operators in fixed order, commit the first candidate that clears
  the threshold, and relearn affected edges.
- [x] Tests: operator order, first-clearing commit, no-refinement fallback,
  topological scan order, relearn, best-path product, and determinism.

## Phase 6 — 9-rooms validation

- [x] Encode the five 9-rooms scenarios: trap goal, narrow door, long
  corridor, mixed start, and blocked two-path.
- [x] Add a scenario runner for both learners, with per-edge before and after
  success and the chosen operator.
- [x] Acceptance tests against the section 11 table. Keep PPO runs in a separate
  slow eval suite.
- [x] Run the fast tabular demo suite over 5 seeds and confirm stable operator
  selection.

## Phase 7 — FaultLine metrics

- [ ] Implement region IoU, visitation-weighted IoU, and normalized graph edit
  distance.
- [ ] Add AST-level fault-injection helpers and a ground-truth repair encoder.
- [ ] Emit the run record as JSON for RQ1 and RQ2 analysis.

## Status

Phases 0 to 6 are complete. The default test suite has 171 tests and all tests
pass with the `dev` extra; 5 further PPO acceptance tests run under the `slow`
marker when the `interop` extra is installed. `code-check` reports no violations.

Next: Phase 7, the FaultLine metrics. The scenarios in
[`faultline/scenarios/`](../faultline/scenarios) run the loop in
[`faultline/refinement/loop.py`](../faultline/refinement/loop.py:1) end-to-end
against the section 11 table; the metrics phase then scores how close each repair
comes to the ground-truth repair.
