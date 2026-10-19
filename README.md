# FaultLine

FaultLine is a benchmark for specification repair in reinforcement learning. It injects a known fault into a
SpectRL specification, runs AutoSpec (Ambadkar et al., ICLR 2026) on it, and measures whether AutoSpec recovers the
**intended** repair or only a repair that raises the success rate.

This README is about the code: what it does, how to run it and how to extend it. The research write-up (introduction,
background, method, preliminary results) is in [`docs/paper/`](docs/paper):

| Document | Content |
|---|---|
| [01 Introduction](docs/paper/01-introduction.md) | Problem, research questions, contributions |
| [02 Background](docs/paper/02-background.md) | SpectRL, abstract graphs, DiRL, AutoSpec operators |
| [03 Method](docs/paper/03-method.md) | Environments, gold certification, fault families, gates, metrics, cache, RQ2 data |
| [04 Results](docs/paper/04-results.md) | Certification, funnel, preliminary RQ1 numbers, runtime |
| [Appendix: RL review](docs/paper/appendix-rl-review.md) | Correctness review of the env, PPO and DiRL |
| [Appendix: contracts](docs/paper/appendix-contracts.md) | What the code guarantees (success, traps, operator witnesses) |

Detailed notes on the AutoSpec paper: [`AutoSpec_research_paper.md`](AutoSpec_research_paper.md). The original
authors' code is kept read-only in [`autospec-original-code/`](autospec-original-code).

## Research questions

- **RQ1.** Given a specification with a known injected fault, does AutoSpec recover the intended repair?
- **RQ2.** Can failure statistics of an edge's rollouts predict which refinement operator fixes it? The benchmark logs
  the data set; training the model (XGBoost) is future work.

## Install

Python 3.13 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev --extra interop --extra reports
```

`interop` adds Stable-Baselines3 (the PPO learner), `reports` adds matplotlib (figures). The core library and most
tests need neither.

## Run the tests

```bash
uv run pytest                 # ~40 s; PPO acceptance tests are marked slow and skipped
uv run pytest -m slow         # the 9-rooms PPO acceptance suite
```

## Run the benchmark

All commands write under `--out` (default `reports/faultline`); the certified rooms9 golds live in
`reports/faultline_v1`. Every command is resumable and parallel (`--workers N`, one PPO training per core).

```bash
B="uv run python -m faultline.benchmark"
$B certify --out reports/faultline_v1 --workers 6      # Tier 2 + 3 for every gold -> certificates.json (hours)
$B funnel  --out reports/faultline_v1                  # generate + validate instances, no learner -> funnel.json
$B screen  --out reports/faultline_v1 --workers 12     # effectiveness / observability gates -> screen_funnel.json
$B run     --out reports/faultline_v1 --workers 12 --seeds 0 1 2 --instance-seeds 0 --only "|mid|" "|null|"
$B analyze --out reports/faultline_v1                  # tables + figures -> analysis/
uv run python scripts/rq2_dataset.py reports/faultline_v1   # RQ2 table -> rq2/dataset.csv
```

Useful flags: `--only <substring> ...` keeps instance ids containing any of the substrings (for example `goal_inflation` or `"|mid|" "|null|"`), `--limit N`,
`--instance-seeds`, `--seeds` (evaluation seeds), `--uncertified` (pilots on golds without a certificate),
`--cache-dir none` (disable the estimate cache), `--no-score-all` (skip scoring the operators the loop did not try,
which RQ2 uses), `--learner tabular` (fast planner, for plumbing tests only).

What each command produces:

| Path | Content |
|---|---|
| `certificates.json` | per gold: tiers passed, problems, per-edge Clopper-Pearson lower bounds, protocol hash |
| `funnel.json` | per family: generated / not applicable / invalid / valid, with every rejection reason |
| `screen/*.json`, `screen_funnel.json` | gate results per instance; per family: ineffective / unobservable / accepted |
| `<instance>.json` | one record per instance: gates, AutoSpec and random-operator runs per seed, metrics, baselines, git and protocol hash |
| `rq2/<instance>.jsonl`, `rq2/rollouts/` | RQ2 rows and their full rollouts ([format](docs/paper/03-method.md#37-rq2-data)) |
| `analysis/summary.md`, `runs.csv`, `*.pdf/png` | the RQ1 tables and figures |

### Timing and the estimate cache

One PPO edge takes about 15 s on one core. Estimates are cached on disk (`reports/cache/`, shared by all workers and
runs) by an exact fingerprint of everything that determines them, so a cache hit is bit-identical to retraining
([method 3.6](docs/paper/03-method.md#36-exact-edge-estimate-cache)). Measured on one instance (gold baselines, gates
and one evaluation seed): see [results](docs/paper/04-results.md#42-runtime). Bump `CACHE_VERSION` in
[`faultline/learners/cache.py`](faultline/learners/cache.py) whenever the environment, reward or learner code
changes; delete `reports/cache/` to start afresh.

`scripts/time_instance.py` reproduces the timing on any instance.

## Other entry points

- `python -m faultline.experiments.rooms9_autospec`: Algorithm 1 on the five hand-built 9-rooms validation scenarios
  (`faultline/scenarios/`), against the paper's numbers. Output in `reports/rooms9_autospec/`.
- `python -m faultline.experiments.rooms9_report`: single-operator 9-rooms report with learning curves.
- `python -m faultline.experiments.rooms100_autospec`: AutoSpec on the 100-rooms worlds.
- `python -m faultline.viz <data dir>`: redraw figures from saved data (room overlays per refinement step, learning
  curves, per-seed and paper-comparison figures).

## Project structure

```text
faultline/
  spectrl/       SpectRL predicates, AST, parser, printer, reference semantics
  graph/         abstract graph, compiler, graph semantics, best-path product
  geometry/      convex hulls, half-planes, region bounds and sampling
  envs/          RoomsEnv; rooms9 layouts; rooms/ = n-rooms generator, frozen worlds, room-graph analysis
  learners/      PPO and tabular edge learners, DiRL and LSTS graph learners, exact estimate cache
  refinement/    the four operators and Algorithm 1 (loop.py, with an observer hook for RQ2)
  golden/        gold specs: builder (Tier 1), certification (Tiers 2-3), cell-grid raster, shared env
  faults/        mutations (fault families), ground-truth repairs, validity gate, effectiveness/observability gates
  metrics/       region scores, GED, ceiling, trajectory audit, inference (bootstrap, Wilcoxon, Holm)
  baselines.py   unrefined, oracle, trivializer, random-operator
  benchmark/     golds, instance generation, runner, workers, CLI, analysis
  rq2/           RQ2 log format and offline features
  scenarios/, experiments/, viz/   9-rooms validation scenarios, reporting runs, figures
configs/protocol.toml   frozen benchmark protocol (hashed into every result: do not edit, version it)
docs/paper/      research write-up
plans/           design history (methodology plan, MVP plan)
reports/         generated output (git-ignored except certificates)
tests/           unit and integration tests
```

### The n-rooms worlds

`faultline/envs/rooms/generator.py` reproduces the paper's procedural 100-rooms generator (Start and Goal in opposite
corners, five intermediate 2x2-room regions placed uniformly, each door open with probability 0.5, redrawn until the
specification is solvable). The five generated worlds and the paper's Fig. 3 and Fig. 6 worlds are frozen in
`faultline/envs/rooms/data/*.json` (the JSON is the source of truth: results do not move if the generator changes, and
a test checks the generator still reproduces the files). Their specification, with the paper's region names:

```text
achieve Start ; (achieve M1 or achieve M2) ; achieve M3 ; (achieve M4 or achieve M5) ; achieve Goal
graph: Start -> M1 | M2 -> M3 -> M4 | M5 -> Goal   (8 edges, no safety clause)
```

`M1`/`M2` and `M4`/`M5` are the two alternative branches of the two `or`s, `M3` is where they join. These region names
are unrelated to the fault families. The FaultLine golds on rooms100 (`benchmark/golds.py::world_gold`) ignore these
regions and use a one-room-hop chain along the shortest door path instead (the paper's regions include sealed rooms).

## How to extend

- **A new gold.** Add a layout to `ROOMS9_LAYOUTS` in `benchmark/golds.py` (or a world to the rooms catalogue and
  pass `--worlds`), run `certify`. Only certified golds enter `run`. If a gold fails, change the world, not the
  threshold, and record why.
- **A new fault family.** Add a `FaultKind` and its expected operator in `faults/ground_truth.py`, a pure
  `(gold, params, rng) -> GroundTruthRepair` function in `faults/mutations.py` registered in `MUTATIONS`, and
  its severity grid in a new protocol version. The validity gate and the gates apply unchanged; add a test in
  `tests/test_mutations.py` that the mutant is valid and deterministic.
- **A new learner.** Implement `train_edge` and `estimate_reach` (`learners/base.py`); deterministic given the seed if
  it is to be cached. Wire it in `Settings.edge_learner` (`benchmark/runner.py`) and bump `CACHE_VERSION`.
- **A new metric.** Add it to `metrics/` and to `evaluate_repair`; the analysis reads records, so rerunning
  `analyze` is enough when the metric can be computed from the stored graphs.
- **A protocol change.** Copy `configs/protocol.toml` to a new version: the hash changes, existing certificates no
  longer match and golds must be recertified.

## Remaining work

1. Full evaluation grid (5 instance seeds x 5 evaluation seeds, all severities) on the certified golds; the
   preliminary results use a subset.
2. RQ2 model: XGBoost on `rq2/dataset.csv` (`operators_clearing` on decision rows), grouped cross-validation by gold.
3. rooms100 golds: Tier 1 passes on all seven worlds, Tier 2 fails because PPO stalls on edges whose straight line
   crosses a wall. Suspects: Euclidean progress shaping through walls and unnormalised observations
   ([appendix](docs/paper/appendix-rl-review.md), findings 4-5). Any learner change means recertifying rooms9.
4. Instrumentation gaps: oracle-valid operators per instance (needs a learner run with oracle witnesses), no region
   ceiling for structural families, the language audit has few fault-only trajectories, `min_semantic_class_count`
   in the protocol is unused.
5. AntMaze and higher-dimensional states are out of scope.

## Lint

```bash
uv tool install ./linting && code-check faultline tests
```

## References

- Ambadkar, T., Žikić, Đ., and Verma, A. (2026). *Automating the Refinement of Reinforcement Learning
  Specifications.* ICLR. arXiv:2512.01047.
- Jothimurugan, K., Alur, R., and Bastani, O. (2019). *A Composable Specification Language for Reinforcement
  Learning Tasks.* NeurIPS. arXiv:2008.09293.
- Jothimurugan, K., Bansal, S., Bastani, O., and Alur, R. (2021). *Compositional Reinforcement Learning from Logical
  Specifications.* NeurIPS.
- Clopper, C. J., and Pearson, E. S. (1934). *The use of confidence or fiducial limits illustrated in the case of the
  binomial.* Biometrika, 26(4), 404-413.
- Just, R., Jalali, D., and Ernst, M. D. (2014). *Defects4J.* ISSTA, 437-440.
- Krakovna, V., et al. (2020). *Specification gaming: the flip side of AI ingenuity.* DeepMind Blog.

## Authors

- Alexander van den Arend Schmidt, TU Delft, `A.F.vandenArendSchmidt@student.tudelft.nl`
- Petre-Alexandru Hautelman, TU Delft, `p.a.hautelman-1@student.tudelft.nl`
