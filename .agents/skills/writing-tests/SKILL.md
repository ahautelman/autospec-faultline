---
name: writing-tests
description: Best practices for writing unit and integration tests. Use whenever writing new tests, fixing failing tests, doing test-driven development, reviewing test code, or testing code that involves LLM calls, randomness, time, or external services. Also use when asked to "add coverage", "write specs", or make tests less flaky.
---

# Writing Tests

**Core principle:** a test is only valuable if it can fail for a real defect, and only fails for real defects. All ten guidelines below aims to serve this purpose.

---

## I. Test behavior, not implementation

* Assert on observable outcomes through the public API: return values, emitted events, persisted state, calls to external boundaries. Never assert on private fields, internal call order, or helper-method invocations.
* A refactor that preserves behavior must not break tests. For example, if renaming a private method or variable breaks a test, the test is wrong.
* Don't test private functions directly. If a private function seems to need its own tests, that's a signal to extract it into its own unit.

## II. Exercise the real code, never re-implement it

* Import and call the actual production code. Never copy logic into the test "to check it" — a test that duplicates the implementation proves both share the same bugs.
* Expected values must be independently derived: hand-computed, taken from a spec, or from a known-good reference. Never compute the expected value in the test using the same algorithm under test (`expect(add(2,3)).toBe(2+3)` is fine; `expect(tax(x)).toBe(taxFormula(x))` where `taxFormula` mirrors the implementation is not).
* Over-mocking is a form of re-implementation: if every dependency is mocked, the test only verifies your mocks. Mock only at true boundaries (network, filesystem, clock, LLM APIs); use real objects for everything in-process and cheap.

## III. Never encode success

* A test must be able to fail. Before trusting a test, confirm it fails when the behavior it guards is broken (mutate the code or revert the fix and watch it go red).
* Forbidden patterns: assertions inside conditionals that may never execute, `try/catch` that swallows the assertion, asserting on a mock's return value you set yourself, tests with no assertions, `assert true`, catching the expected exception without failing when it *doesn't* occur.
* When a test fails, fix the code — never weaken the assertion, widen a tolerance, or delete the test to get green, unless the test itself is verifiably wrong. State explicitly which one it was.
* Never mark tests skip/xfail/`.todo` to make a suite pass. Skipping requires a documented reason.

## IV. Red first, then green (TDD discipline)

* When fixing a bug: write a test that reproduces it, run it, and confirm it fails for the expected reason (right error, right assertion — not an import error). Only then change the code. The red run is the proof the test measures anything.
* When adding a feature: write the failing test from the spec, implement until green, then refactor with the test as a safety net.
* A test that passes on first run against unchanged code is suspect: it either tests existing behavior (fine, label it a regression/characterization test) or it tests nothing.

## V. Isolation and independence

* Each test must pass alone, in any order, and in parallel. No test may depend on another test's side effects, execution order, or shared mutable state.
* Fresh fixtures per test: build state in setup/arrange, tear it down (or use transactions/temp dirs that auto-clean). Never share a mutated object across tests.
* No dependence on the environment: real network, wall-clock time, locale, timezone, machine-specific paths, or pre-existing database rows. Inject these; fake them at the boundary.
* Verify by running the suite shuffled (`pytest -p random-order`, `--shuffle`, etc.) if flakiness is suspected.

## VI. Determinism, no randomness

* Seed every RNG explicitly, or inject the RNG so the test controls it. Freeze time (`freezegun`, `jest.useFakeTimers`, injected clock). Sort before asserting on anything with unspecified order (dict/set/query results).
* Retry-until-pass is banned. A flaky test is a failing test; find the nondeterminism and remove it.
* Floating point: compare with explicit tolerances (`approx`, `closeTo`), never exact equality — but keep tolerances tight enough to catch real errors.

### VII. When LLMs are in the picture

* Separate deterministic logic from model calls. Prompt construction, response parsing, validation, retries, tool dispatch — all pure functions, unit-tested exhaustively with fixed strings. This should be the bulk of your tests.
* For code *around* the LLM: stub the API client and return canned responses (including malformed ones, refusals, empty content, tool-call shapes, streaming chunks, rate-limit errors).
* Never assert exact model output in unit tests, even temperature 0 is not guaranteed deterministic. If a test must hit a real model, assert on *properties*: valid JSON, schema conformance, contains/lacks required fields, length bounds, passes a validator.
* Quality-of-output testing ("is the answer good?") is an **eval**, not a unit test: separate suite, scored (possibly LLM-graded) with thresholds, run on its own cadence, never gating CI determinism. Don't mix the two.

## VIII. One concern per test, no duplicate coverage

* Each test verifies one behavior and its name states it: `rejects_expired_token`, not `test_auth_2`. A reader should diagnose a failure from the name alone.
* Multiple assertions are fine if they describe one outcome; not if they smuggle in a second scenario.
* Don't write two tests that fail for the same reason. Before adding a test, check what would newly break it. Redundant tests add maintenance cost and noise, not safety.
* Use parametrization/table-driven tests for the same behavior over many inputs; use separate named tests for distinct behaviors.

## IX. Structure and readability

* Arrange–Act–Assert (Given/When/Then), visually separated. One act per test.
* No logic in tests: no `if`, loops, or computation deciding what to assert. Branching in a test means it's two tests. (Loops over parametrized data are fine — the framework's parametrize is better.)
* Keep the whole story visible in the test body. Extract noisy setup into well-named builders/factories, but never hide the input that determines the assertion.
* Test code is production code: no dead tests, no commented-out tests, no copy-paste drift.

## X. What to cover

* Happy path, boundaries (empty, one, many, max, off-by-one), invalid input, and error paths (the right error, not just "an error").
* Every bug fix gets a regression test that failed before the fix.
* Coverage percentage is a diagnostic, not a target. 100% coverage with assertion-free tests is worth nothing; a missed branch is a prompt to ask "what behavior lives there?"
* Prefer many fast unit tests, fewer integration tests at real boundaries, fewest end-to-end tests (testing pyramid). Slow suites stop being run.

---

## Checklist before committing a test

1. Does it call the real code and assert an independently derived outcome?
2. Have I seen it fail for the right reason?
3. Is anything nondeterministic (time, RNG, network, LLM, ordering) controlled?
4. Does the name state the behavior, and does it cover something no other test covers?
5. Would a pure refactor keep it green?
