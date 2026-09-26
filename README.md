# FaultLine

FaultLine is a benchmark for specification repair. It checks whether AutoSpec
finds the **correct** repair, not only a repair that raises the success rate.

## Summary

- **What it is.** A from-scratch, tested implementation of the AutoSpec
  pipeline, plus a benchmark that measures repair quality.
- **Why it matters.** AutoSpec reports only end-to-end task success. An agent
  can raise success by shrinking a target region until the task becomes
  trivial. A high success rate does not prove a correct repair.
- **How FaultLine tests this.** FaultLine injects a known fault into a SpectRL
  specification. It records the ground-truth repair. Then it measures how close
  AutoSpec gets to that repair.
- **Status.** The SpectRL front end, the abstract-graph compiler, the 9-rooms
  environment, the learners, the four refinement operators, the Algorithm 1 loop,
  and the five-scenario 9-rooms validation are complete and tested. The FaultLine
  metrics are in progress. See [Roadmap](#roadmap).

## Goal and direction

The project studies the gap between two things:

1. A statistical success estimate from rollouts.
2. True correctness of a specification.

AutoSpec keeps soundness. If a trajectory satisfies the refined specification,
it also satisfies the original specification. FaultLine measures whether the
repair also recovers the **intended** specification. This is the core
contribution.

The code builds on the AutoSpec paper and the SpectRL language. The goal is a
solid, tested implementation that fits the research questions. We write the
code from scratch. The original authors' code stays in
[`autospec-original-code/`](autospec-original-code) as a read-only reference.

## Research questions

- **RQ1 (core).** Given a specification with a known injected flaw, does
  AutoSpec recover the correct repair?
  *Metrics:* recovery closeness per fault type, and operator selection
  accuracy.
- **RQ2 (extension).** Can failure statistics from rollout trajectories predict
  which repair operator will succeed? This would remove the sequential operator
  search.
  *Features:* failure locality (start, mid-path, goal) and failure
  concentration (trajectory variance, state entropy).
- **RQ3 (extension).** Can a lower confidence bound replace the point estimate
  `p`? The aim is fewer rollouts with a strict false-acceptance ceiling.

## What works now

| Part | State | Location |
| --- | --- | --- |
| SpectRL predicates and AST | Done | [`faultline/spectrl/`](faultline/spectrl) |
| SpectRL parser and printer | Done | [`parser.py`](faultline/spectrl/parser.py:1), [`printer.py`](faultline/spectrl/printer.py:1) |
| Reference specification semantics | Done | [`semantics.py`](faultline/spectrl/semantics.py:1) |
| Abstract graph and compiler | Done | [`faultline/graph/`](faultline/graph) |
| Graph semantics and best-path product | Done | [`graph/semantics.py`](faultline/graph/semantics.py:1) |
| 9-rooms layout and environment | Done | [`faultline/envs/`](faultline/envs) |
| Learner protocol, tabular and PPO learners | Done | [`faultline/learners/`](faultline/learners) |
| The four refinement operators | Done | [`faultline/refinement/`](faultline/refinement), [`faultline/geometry/`](faultline/geometry) |
| Algorithm 1 loop | Done | [`refinement/loop.py`](faultline/refinement/loop.py:1) |
| 9-rooms validation scenarios and runner | Done | [`faultline/scenarios/`](faultline/scenarios) |
| 9-rooms PPO reporting run | Done | [`faultline/experiments/`](faultline/experiments) |
| FaultLine metrics | In progress | planned |

The default test suite has 185 tests and all pass. Five further PPO acceptance
tests run under the `slow` marker; see [Run the tests](#run-the-tests).

## Requirements

- Python 3.13 or newer. The file [`.python-version`](.python-version:1) pins the
  version.
- [uv](https://docs.astral.sh/uv/) for the environment and the commands.
- Node.js only if you want the TypeScript/JavaScript half of the linter.

Core dependencies add NumPy and Gymnasium. The optional `interop` extra adds
Stable-Baselines3 and scikit-learn for the real PPO learner, and the optional
`reports` extra adds matplotlib for the 9-rooms report figures. The test suite
runs without either extra; the PPO learner test and the figure test then skip.

## Install

Run this command from the project root:

```bash
uv sync --extra dev
```

Add the `interop` extra when you work on the environment or the PPO learner, and
the `reports` extra when you run the 9-rooms report:

```bash
uv sync --extra dev --extra interop --extra reports
```

## Run the tests

Run the full test suite:

```bash
uv run --extra dev pytest
```

Run one test file:

```bash
uv run --extra dev pytest tests/test_compiler.py
```

Run one test by name:

```bash
uv run --extra dev pytest tests/test_parser.py -k round_trip
```

The 9-rooms PPO acceptance tests train a policy per edge, so they are marked
`slow` and excluded from the default run. Add the `interop` extra and select them
explicitly:

```bash
uv run --extra dev --extra interop pytest -m slow
```

The centrepiece test is
[`test_compiled_graph_agrees_with_spec_semantics`](tests/test_compiler.py:47).
It checks every trajectory in a small state space. For each trajectory, the
compiled graph must accept the trajectory if and only if the specification
semantics accept it. This is the paper property `zeta models phi` if and only
if `zeta models G`.

## Run the 9-rooms report

[`faultline/experiments/rooms9_report.py`](faultline/experiments/rooms9_report.py:1)
is a runnable pipeline that trains the five 9-rooms scenarios with PPO and writes
comparison figures against the paper's numbers:

```bash
uv run --extra interop --extra reports python -m faultline.experiments.rooms9_report
```

Defaults match the paper: a 200k-step budget for every scenario, 1000 evaluation
rollouts, checkpoints every 10k steps, and seeds 0-4. Training is expensive, so
`--budget`, `--rollouts`, `--interval`, `--intermediate-rollouts`, `--seeds`,
`--workers`, `--steps`, and `--output` are flags, and `--plot-only` redraws the
figures from the persisted data without retraining.

The independent `(scenario, seed)` jobs run concurrently by default. Steps 1 and 2
each dispatch their jobs to a process pool sized to `--workers`, which defaults to
`FAULTLINE_WORKERS` or `os.cpu_count() - 1` (one core stays free for the OS and the
orchestration loop); pass `--workers 1` for a serial run. Each worker is pinned to a
single compute thread (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, and
`torch.set_num_threads(1)`), so N workers run N distinct jobs instead of
oversubscribing the cores with internal threads. Jobs share no state, and a
concurrent run produces byte-identical records to a serial one. The pool prints one
`[done] scenario=<name> seed=<n> (k/N)` line per finished job, which is the progress
report while the workers run; a serial run also shows the per-edge tqdm bars.

Profiling showed the run is dominated by the per-checkpoint rollout evaluation,
not PPO itself. The two checkpoints the paper is compared at, the 80k reference
step and the final budget, keep the full `--rollouts` (1000) count; every other
checkpoint uses `--intermediate-rollouts` (100 by default), which only shapes the
curve. Evaluation also drops the redundant per-step `set_training_mode` call
Stable-Baselines3's `predict` makes (about a quarter of one evaluation): the policy
is switched to evaluation mode once, then rolled out through the same prediction
path ([`greedy_action_fn`](faultline/learners/ppo.py:78)), which a bit-for-bit test
pins to `predict`. The run prints a stage banner per scenario, the edge regions it
inspects, and one line per checkpoint with its success and rollout count.

The run writes raw records to `reports/rooms9/data/` (`step1_<scenario>.json`
checkpoint series and `step2_<scenario>.json` per-seed summaries) and figures to
`reports/rooms9/`: a reference comparison at the 80k checkpoint, per-scenario
learning curves over 0..200k, a geometric overlay of each operator's inputs and
its refined region, and per-seed spread bars, each per scenario plus a combined
figure.

## Run the linter

The linter is `code-check`. Install it once as a tool:

```bash
uv tool install ./linting
```

Then check the files you changed:

```bash
code-check faultline tests
```

Re-install the tool after you edit anything under [`linting/`](linting). See
[`linting/README.md`](linting/README.md:1) for details.

## Use the library

The example below parses a specification, compiles it to an abstract graph, and
checks a trajectory.

```python
from faultline.spectrl import parse, to_spectrl
from faultline.spectrl.predicates import Box
from faultline.graph import compile_spec, models_trajectory

# Name each region. A predicate is a boolean function over a state vector.
regions = {
    "start": Box((1.0, 1.0), (6.0, 6.0)),
    "goal": Box((17.0, 17.0), (23.0, 23.0)),
}

# Parse SpectRL text into an AST.
spec = parse("achieve start ; achieve goal", regions)
print(to_spectrl(spec))

# Compile the AST into an abstract graph.
graph = compile_spec(spec, source_region=regions["start"])
print(graph)

# Check one trajectory against the graph.
trajectory = [[3.0, 3.0], [10.0, 10.0], [20.0, 20.0]]
print(models_trajectory(graph, trajectory))
```

Notes on the syntax:

- The operators `;`, `ensuring`, and `or` apply to specifications.
- The operators `∧`, `∨`, and `¬` apply to predicates. You can also write `&`,
  `|`, and `!`.
- A bare name resolves through the region mapping. For example, `start`.
- You can also write a region inline. For example, `Rect(0, 0, 6, 6)`.

## Project structure

```text
faultline/
  spectrl/            SpectRL language: predicates, AST, parser, printer, semantics
  graph/              abstract graph, compiler, graph semantics
  geometry/           convex hulls, region algebra, and half-plane partitioning
  envs/               9-rooms layout and the gymnasium environment
  learners/           learner protocol, tabular learner, PPO learner
  refinement/         the Algorithm 1 loop and the four refinement operators
  scenarios/          the five 9-rooms validation scenarios and the runner
  experiments/        the runnable 9-rooms report: checkpointed training and figures
tests/                unit tests for the packages above
plans/                the implementation plan
reports/              generated report artifacts (figures and raw series)
linting/              code-check, the custom linter
autospec-original-code/   the original AutoSpec code, read only
AutoSpec_research_paper.md   implementation notes on the AutoSpec paper
```

Each module maps to a section of
[`AutoSpec_research_paper.md`](AutoSpec_research_paper.md:1):

| Module | Paper section | Content |
| --- | --- | --- |
| [`faultline/spectrl/predicates.py`](faultline/spectrl/predicates.py:1) | 2.2 | Predicates: region atoms and boolean combinators |
| [`faultline/spectrl/ast.py`](faultline/spectrl/ast.py:1) | 2.2 | Specification formulas: `achieve`, `ensuring`, sequence, choice |
| [`faultline/spectrl/parser.py`](faultline/spectrl/parser.py:1) | 2.2 | Text to AST parser |
| [`faultline/spectrl/semantics.py`](faultline/spectrl/semantics.py:1) | 2.2 | Reference semantics for a specification |
| [`faultline/graph/abstract_graph.py`](faultline/graph/abstract_graph.py:1) | 2.3 | The graph `G = (V, E, beta, s, t)` |
| [`faultline/graph/compiler.py`](faultline/graph/compiler.py:1) | 2.3 | Compiler from AST to graph |
| [`faultline/graph/semantics.py`](faultline/graph/semantics.py:1) | 2.3, 6.2 | Trajectory satisfaction and best-path success |
| [`faultline/envs/layout.py`](faultline/envs/layout.py:1) | 8 | The 3x3 room grid, walls, doors, and regions |
| [`faultline/envs/rooms9.py`](faultline/envs/rooms9.py:1) | 8 | The point-mass environment with reach-avoid subgoals |
| [`faultline/learners/tabular.py`](faultline/learners/tabular.py:1) | 3, 4 | The fast deterministic planner learner for tests |
| [`faultline/learners/ppo.py`](faultline/learners/ppo.py:1) | 6.1 | The Stable-Baselines3 PPO edge learner |
| [`faultline/refinement/operators.py`](faultline/refinement/operators.py:1) | 4 | The four refinement operators |
| [`faultline/refinement/loop.py`](faultline/refinement/loop.py:1) | 3 | The Algorithm 1 refinement loop, config, and run record |
| [`faultline/scenarios/catalogue.py`](faultline/scenarios/catalogue.py:1) | 8, 11 | The five 9-rooms validation scenarios |
| [`faultline/scenarios/runner.py`](faultline/scenarios/runner.py:1) | 3, 11 | The scenario runner and its per-edge report |
| [`faultline/experiments/pipeline.py`](faultline/experiments/pipeline.py:1) | 3, 11 | The single-seed and multi-seed checkpointed pipeline |
| [`faultline/experiments/figures.py`](faultline/experiments/figures.py:1) | 11 | Comparison, learning-curve, and spread figures |

## How the pipeline works

```mermaid
flowchart LR
    A["Spec text"] --> B["Parser"]
    B --> C["AST"]
    C --> D["Compiler"]
    D --> E["Abstract graph"]
    E --> F["Refinement loop"]
    F --> G["Repaired graph"]
```

One step follows the next.

1. The parser reads SpectRL text. It returns an AST.
2. The compiler walks the AST. It builds an abstract graph.
3. Each graph edge is a reach-avoid subgoal. A separate policy learns each
   edge.
4. The refinement loop finds a weak edge. It applies one operator. Then it
   repeats.
5. The graph semantics checks a trajectory. The best-path product gives the
   whole-task success.

## Fault taxonomy

Each fault targets one primitive. The ground-truth repair is known by
construction.

| Fault | Target primitive | Expected operator | Metric |
| --- | --- | --- | --- |
| Trap in the goal region | Predicate | `SeqRefine`, `ReachRefine` | Region IoU, wIoU |
| Over-wide safety region | Predicate | `SeqRefine`, `AvoidRefine` | Region IoU, wIoU |
| Unreachable start states | Predicate | `PastRefine` | Region IoU, wIoU |
| Missing waypoint | Graph topology | `AddRefine` | Normalized graph edit distance |
| Missing alternative path | Graph topology | `OrRefine` | Normalized graph edit distance |

Faults come in several severity levels. Control specifications have no fault.
They check that AutoSpec does not repair a valid specification.

## Metrics

- **IoU.** The overlap between the refined region and the ground-truth region,
  divided by their union.
- **wIoU.** The same overlap, weighted by how often the agent visits each
  state.
- **Normalized graph edit distance.** The number of vertex and edge changes
  that map the repaired graph to the ground-truth graph, divided by the size of
  the ground-truth graph.
- **Operator accuracy.** The fraction of runs that select the ground-truth
  operator.
- **Soundness.** The check that a refined region stays inside the original
  region and respects the safety envelope.

The benchmark compares AutoSpec with three baselines: the unrefined
specification, a uniform-random operator selector, and an oracle selector that
applies the intended repair.

## Roadmap

Work continues in this order.

1. FaultLine metrics: IoU, wIoU, and graph edit distance.
2. Fault-injection helpers and the ground-truth repair encoder.

If time runs short, drop RQ3 first, then RQ2. RQ1 is the core and stands alone.

## Related work

- **AutoSpec** refines SpectRL specifications over an abstract graph. It uses
  four sound operators: `SeqRefine`, `PastRefine`, `AddRefine`, and
  `OrRefine`. It evaluates only rollout success.
- **Specification gaming** shows how a proxy objective leads to unintended
  solutions.
- **Fault injection** uses a database of known faults for controlled studies.
  For example, Defects4J.
- **Verified AI and confidence intervals** motivate statistical guarantees.
  For example, the Clopper-Pearson interval.

## References

- Ambadkar, T., Žikić, Đ., and Verma, A. (2026). *Automating the Refinement of
  Reinforcement Learning Specifications.* ICLR. arXiv:2512.01047.
- Jothimurugan, K., Alur, R., and Bastani, O. (2021). *A Composable
  Specification Language for Reinforcement Learning Tasks.* NeurIPS.
  arXiv:2008.09293.
- Clopper, C. J., and Pearson, E. S. (1934). *The use of confidence or fiducial
  limits illustrated in the case of the binomial.* Biometrika, 26(4), 404-413.
- Just, R., Jalali, D., and Ernst, M. D. (2014). *Defects4J.* ISSTA, 437-440.
- Krakovna, V., et al. (2020). *Specification gaming: the flip side of AI
  ingenuity.* DeepMind Blog.
- Seshia, S. A., Sadigh, D., and Sastry, S. S. (2022). *Toward Verified
  Artificial Intelligence.* Communications of the ACM, 65(7), 46-55.

## Authors

- Alexander van den Arend Schmidt, TU Delft,
  `A.F.vandenArendSchmidt@student.tudelft.nl`
- Petre-Alexandru Hautelman, TU Delft,
  `p.a.hautelman-1@student.tudelft.nl`
