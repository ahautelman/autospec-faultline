# FaultLine methodology plan (aligned with FaultLine-implementation-spec.md)

Scope: RQ1 on `rooms9` and `rooms100`. AntMaze is out of scope (future work).
Goal: certified gold specs plus parametric, gated fault injection, so that
AutoSpec's fault recovery can be tested with academic rigour.

## Decisions (from the project owner)

| # | Decision | Consequence |
|---|---|---|
| D1 | Hazards are **absorbing traps** (`RoomsEnv.unsafe_region`). Gold labels the whole trap unsafe. A fault **shrinks the labelled unsafe area**, so AutoSpec must infer the full trap. Standard across the whole project. | Retire `HiddenHazardRooms9`. Revert the `not step_safe` early stop in `run_episode` (a trap already terminates). The loop and learners stay unchanged (principle 5). `RoomsWorld.hazards` (spec-only) is redefined as the physical trap set, and gold safe = `Not(traps)`. |
| D2 | AntMaze skipped. | Drop `mazes/antmaze`, the pilot and the fallback ladder. |
| D3 | DiRL (PPO edge learner) is the default learner. Tabular is for fast tests only. | Size the grid from measured per-run wall-clock (step 1). |
| D4 | AutoSpec refinement threshold p = **0.8**. No 0.99 sweep. | Gold certification: per-edge Clopper-Pearson lower bound >= `p_gold_min`. Proposed `p_gold_min = 0.85`, so gold sits above the threshold and Tier 3 (fixpoint) is stable. Confirm. Drop the derived `p_gold - delta`. |
| D5 | Loop runs unchanged. `reuse_candidate_cache` stays off in the benchmark and exists only as an ablation. | Remove `reuse_candidate_cache=True` from `faultline_benchmark.py`. |
| D6 | Old `faultline_full_*` results are not cited. | Free to change instances and settings. Nothing to preserve. |

Trap semantics for M2 (AvoidRefine): physical trap H. Gold safe set = free space minus H. Faulty safe set
= free space minus H' with H' a strict subset of H, with the labelled core and an unlabelled approach part.
The env is identical for gold and faulty. Training sees the real violation, so the
**effectiveness gate** (not the env) must show that the faulty baseline fails.

## Steps

1. **Contracts and protocol.** `docs/contracts.md` (Phase 0). Verified already: success = within
   `SUCCESS_RADIUS = 1.0` of a sampled goal point and safe. AvoidRefine consumes `final_safe`.
   Still to read: operator witnesses, loop record, region algebra. `configs/protocol.toml` freezes p = 0.8,
   rollouts, `avoid_tail`, severity grid, grid resolution, seeds (cert 100-104, eval 0-4) and git-hash
   stamping. Time one rooms100 DiRL run and size the grid from that.
2. **Maze analysis on `RoomGrid`.** `shortest_path`, `cut_vertices`, `viability`, each cross-checked against brute force (T1).
3. **Gold builder and certifier.** Includes a rasterised cell-grid analysis (0.1 unit) for sub-room traps (M2); room-level `RoomGrid` is used for door topology. Gold from a world: waypoints at cut vertices, viability-shrunk labels.
   Tier 1 by BFS. Tier 2 = DiRL on cert seeds with CP bound. Tier 3 = Algorithm 1 on gold, held-out seeds, zero refinements.
   The five current hand cases are re-expressed as worlds and must certify.
4. **Parametric mutations M1-M5.** `(gold, params, rng)`, deterministic, severity changes geometry
   (pocket mass 0.10/0.25/0.50, unlabelled trap fraction 0.25/0.50/0.75, ...). Record recoverability factors.
   Validity gate: R* subset R_f and L(phi*) subset L(phi_f) checked mechanically (T5, T6).
5. **Effectiveness and observability gates, funnel.** Replace per-case `max_steps` tuning and the post-hoc classifier with
   pre-run gates and a reported funnel (generated, valid, effective, observable; unobservable stratum).
6. **Metrics.** Fault removal vs collateral retention, S* and proxy gap, semantic precision/recall, region-aware GED,
   operator ceiling and attainment ratio, soundness audit, CP/bootstrap/Wilcoxon-Holm (T3, T4).
7. **Baselines.** Unrefined, random-operator, oracle, trivializer (T7). Instrumentation regression T8.
8. **Runner.** rooms9 + rooms100 grid, composites, >= 20% nulls, result JSON with funnel and git hash.
   Reproduce today's five cases as instances (T9).

## Order and checkpoints
Steps 1-2 first (no learner needed). Checkpoint after step 3: do the gold specs certify at p = 0.8?
If a world fails Tier 2 or 3, fix the world, never the threshold. Checkpoint after step 5: funnel counts per family,
before any evaluation run. Report every rejection.

## Step 3 status (gold builder and certifier)
Built: `faultline/golden/{raster,spec,builder,certify}.py`, `faultline/stats.py` (exact Clopper-Pearson, no SciPy),
`certification_holdout` seeds and `[gold]` section in the protocol. Tier 1 is exact on a 0.1-unit cell grid and includes a
horizon lower bound. Tiers 2 and 3 run DiRL on cert seeds and Algorithm 1 on hold-out seeds.

Pilot on today's five hand-built correct graphs (PPO + DiRL, 1 cert seed, 1 hold-out seed, p_gold_min 0.85, p = 0.8):
`goal_trap`, `overwide_safety`, `unreachable_start`, `missing_waypoint` certify on all three tiers (38-88 s each).
`missing_alternative` fails Tier 1: gold edge `a->goal` needs at least 7 steps and the horizon is 6. It is dead by design,
and AutoSpec refines it on gold (Tier 3 fails). **Open design issue for M5**, see the discussion with the project owner.

### M5 decision: weak-by-design gold edges
Gold may contain edges marked weak by design (`build_gold(..., weak_edges=("a->goal",))`) when an alternative route of
non-weak edges exists. Tier 1: weak edges skip the horizon check; each needs an alternative route. Tier 2: non-weak
edges need a lower bound >= `p_gold_min` (0.85); each weak edge needs one alternative route whose edges all reach
`p_alt_min` (0.95). Tier 3: zero refinements on non-weak edges; refinements on weak edges are recorded as notes.
Pilot: `missing_alternative` certifies with `a->goal` weak (lower bounds 0.997 on the alternative route, 343 s).
**Consequence for later steps:** AutoSpec refines the weak edge on gold itself (OrRefine on hold-out seed 105), so the false-repair
rate on null runs, and the M5 effectiveness and recovery metrics, must treat designated weak edges separately.

## Step 4 status (parametric mutations and validity gate)
Built: `faults/{ground_truth,graph_ops,placement,mutations,sampling,validity}.py` and `golden/env.py` (one environment
for gold and faulty, traps installed as the absorbing `unsafe_region`). Mutations M1-M5 plus the null are pure
`(gold, params, rng)` functions, deterministic given the seed. The validity gate checks region containment on the cell
grid and L(phi*) in L(phi_f) on sampled gold-satisfying trajectories; it raises on a violation and on a vacuous check.
Notes: M3 at an inner vertex needs a spec-only mouth hazard on the outgoing edge (reachable-but-not-viable cannot arise from
physical traps with symmetric doors); M5 has no numeric severity (measured by the effectiveness gate later); the language
check is sampling-based evidence, not a proof. The old `faults/{catalogue,injectors}.py` stay until step 8 (T9).

## Step 5 status (gates and funnel)
Built `faults/gates.py`: `learn_baseline` (first phase of the loop, no refinement), `check_effectiveness` (faulty best-path
success trails gold by `effectiveness_min_gap` and some edge is below p, on `effectiveness_seed_fraction` of seeds),
`check_observability` (locus-edge rollouts hold `observability_min_witnesses` of the samples the operator consumes; route pruning
and the null need none), `screen` / `screen_with_learning`, and `Funnel` (per-family counts and reasons). Unobservable mutants
are their own stratum, not dropped. The gates are tested on scripted baselines only; their thresholds meet real PPO output in
step 8, where the funnel is the checkpoint before any evaluation run.

## Step 6 status (metrics)
Built `faultline/metrics/`: `regions` (fault removal, collateral retention = semantic recall, precision, IoU on the cell grid;
removal alone cannot be gamed because a wiped region scores retention 0), `ged` (region-aware normalised edit distance),
`ceiling` (oracle-witness repair per operator geometry, attainment ratio), `audit` (trajectory-level soundness audit and the
S* vs proxy gap), `inference` (bootstrap CI, exact/normal Wilcoxon signed-rank, Holm). Clopper-Pearson is in `stats.py`.
Open: `oracle_valid_operators` (which operators pass the threshold with oracle witnesses) needs a learner and is left for the
runner (step 8); M4/M5 have no region ceiling (structural repair, scored by GED and the audit); S* needs end-to-end
trajectories of the repaired routes, supplied by the runner via `experiments/satisfaction`. `min_semantic_class_count` in the
protocol is not used yet. The old `experiments/faultline_metrics.py` stays until T9.

## Step 7 status (baselines and instrumentation regression)
Built `faultline/baselines.py` (`unrefined`, `oracle`, `trivializer`, `random_operator`, `graph_baselines`) and
`faultline/metrics/evaluate.py` (`evaluate_repair`, `f1` of removal and retention). T7: on the M1 fixture the baselines order as
unrefined (F1 0) < trivializer (removal 1, retention < 0.2, F1 < 0.3) < oracle (F1 1, GED 0). T8: `random_operator` patches
only the loop's operator tuple, restores it, is deterministic in its rng and shuffles the order of tries; `evaluate_repair` is
deterministic. The loop and learners are untouched. Limits: `random_operator` patches a module global, so run one per
process; `oracle` for M4/M5 is the gold graph, not an operator-limited repair.

## Step 8 status (runner, in progress)
Cleanups done: `HiddenHazardRooms9` retired (overwide case uses the physical trap), `run_episode` early stop reverted,
`reuse_candidate_cache=True` removed from the benchmark. Built `faultline/benchmark/{golds,instances,runner,__main__}.py`:
rooms9 golds (default, mixed_star, blocked_two_path, each with and without a trap; `long_corridor` excluded, its fixed chain is
infeasible), deterministic instance generation with a funnel, the per-instance runner (gates, AutoSpec, random-operator baseline,
graph baselines, metrics, JSON with git and protocol hash) and a CLI (`funnel`, `run`). T9 passes: the five old cases are valid
coarsenings of feasible golds and same-family mutations apply. Funnel on rooms9, one instance seed: 57 valid; route pruning has
no site (no gold with a weak-by-design edge). Not done: rooms100 golds, M5 golds, PPO pilot timings, grid sizing, process pool.

### Step 8 finding: the first rooms9 golds were not learnable
A PPO pilot on the original `start -> mid -> goal` golds measured gold edges of 0.73-0.85 (and 0.75-0.99 as the horizon varied),
so gold itself would be refined; Tier 2 had never been run on them. Fix applied to the world, not the threshold: the intended
graph now has one waypoint per intermediate room on the door path (one-room hops) and the rooms9 horizon is 30.
Single-seed certification (cert seed 100, hold-out 105) at horizon 30: `default`, `default_trap`, `mixed_star`, `mixed_star_trap`,
`blocked_two_path` pass all three tiers (weakest edge lower bound 0.881 on `default`); `blocked_two_path_trap` has no feasible
trap. The full protocol certification (5 + 5 seeds) is still to be run before any evaluation. Margins on `default` are thin.

### Step 8 pilot on certified golds (PPO + DiRL, evaluation seed 0, gate seeds 0-2)
`rooms9_mixed_star`, severity mid, instance seed 0. M1 goal inflation: accepted (gaps 0.23-0.26, 742+ witnesses); AutoSpec commits
`seq` on `w3->goal`, removal 1.0, retention 0.96, F1 0.98, attainment 1.0. M3 dead start: accepted; AutoSpec commits `past` on
`start->w1`, F1 1.0. The random-operator baseline recovers both too (only one operator clears the threshold in these cases), so
operator choice is not what separates them here. M4 waypoint contraction: ineffective (gap 0.0, one-room-hop golds still learn
after contraction). M2 hazard unlabeled: ineffective (gap 0.0 on all seeds): with an absorbing trap identical for gold and
faulty, PPO is penalised by the physical trap either way, so M2 cannot bite. **Decision (project owner): keep the absorbing
standard, run M2 instances regardless and report them as the `ineffective_by_construction` stratum; no AvoidRefine recovery
claim is made from them.** Timing (single process): gate baselines about 115 s, one evaluation seed (AutoSpec plus random-operator)
335-390 s.

### Step 8: rooms100 deferred (project owner decision)
Rooms100 golds built as one-room-hop waypoint chains over each fixed world's layout (`world_gold`, route chosen by shortest door
path, the paper's own regions are not gold because they contain sealed rooms) pass Tier 1 on all 7 worlds (13-22 edges). Tier 2
fails on `random_seed0` with one certification seed: 8 of 16 edges below 0.85, 7 at 0.0. Diagnosis so far: the policy aims correctly
(heading 3.14 toward the goal) but its speed collapses to about 0.1, so the agent crawls; not a direction effect (open-map left,
right, up and down edges learn at 0.88-1.0); passing and failing edges sit in the same world, so it is not only the 0-120
observation scale. Unresolved. Decision: ship the benchmark on the certified rooms9 golds and treat rooms100 as an open follow-up
(candidate causes: observation scale, budget, seed sensitivity of the PPO recipe). Nothing was changed in the learner.
The CLI now has `certify` (protocol seeds, one gold per worker, writes `certificates.json`), `--workers`, and `run` uses only
certified golds unless `--uncertified` is given.

### Step 8: full protocol certification of the rooms9 golds (5 certification + 5 hold-out seeds, PPO + DiRL)
Certified: `mixed_star` (weakest edge lower bound 0.997), `mixed_star_trap` (0.972), `blocked_two_path` (0.997),
`blocked_two_path_trap` (0.992). Failed: `default` (`w3->goal` 0.686) and `default_trap` (`w3->goal` 0.764, and refined on hold-out
seeds 107 and 109). The single-seed pass at 0.88 earlier was a lucky seed. Fix applied to the world, not the threshold: on
`default` only, the start and goal boxes shrink from 0.75 to 0.5 of the room side (`ROOMS9_END_SIDE`). This is a design choice
made after seeing certification results; it is reported here and the two golds are recertified under the unchanged protocol.

### Step 8: `default` excluded; route pruning gets a site; M4 two-waypoint variant
Recertification of `default` and `default_trap` with 0.5-side end boxes under the full protocol also failed (`w3->goal` 0.744;
`w1->w2` 0.670 and `w3->goal` 0.825 plus a hold-out refinement). The layout is excluded (`EXCLUDED_ROOMS9`), not tuned further.
Route pruning had no site: `add_shortcut` adds a weak-by-design edge that skips one waypoint across a wall (the chain through
the skipped waypoint is its reliable alternative), giving `rooms9_<layout>_shortcut` golds; M5 applies and passes the validity gate
on both. M4 now also runs a two-waypoint-drop variant (instance id suffix `|n2`) because dropping one waypoint was ineffective.

## Status at hand-off and remaining work

### Done and committed (steps 1-7 complete, step 8 partly)
Steps 1-7 are implemented and tested. Step 8 has the cleanups, the rooms9 golds (one-room-hop waypoint chains, horizon 30), instance
generation with a funnel, the runner (gates, AutoSpec, random-operator baseline, graph baselines, metrics, JSON with git and
protocol hash), a CLI (`certify`, `funnel`, `screen`, `run`; `--workers` pool; resumable), and T9.

### State of the rooms9 benchmark (output dir `reports/faultline_v1`)
Full-protocol certification (5 + 5 seeds, PPO + DiRL) passed for `mixed_star`, `mixed_star_trap`, `blocked_two_path`,
`blocked_two_path_trap`, `blocked_two_path_shortcut`. Failed: `default`, `default_trap` (excluded, `EXCLUDED_ROOMS9`) and
`mixed_star_shortcut` (DiRL never trained `w3->goal` after reaching the goal through the weak shortcut; to be excluded in code,
it is currently skipped only because `certificates.json` marks it uncertified). `certificates.json` there is a merge of three
certification runs. Funnel (generation only, 5 instance seeds, 5 certified golds): M1 30 valid of 75, M2 30 of 75, M3 30 of 75,
M4 150 of 150, M5 5 of 25 (one gold), nulls 25. The null count is below the required 20%; the code now adds replicates (uncommitted
before this hand-off, committed with it), so the funnel must be re-run.

### Remaining work, in order
1. **Gate screening (the checkpoint before any evaluation run).** Run `python -m faultline.benchmark screen --out reports/faultline_v1
   --workers 12` (about 2 min per instance plus generation, roughly an hour) and read `screen_funnel.json`: counts per family and
   stage (ineffective, unobservable, accepted, `ineffective_by_construction` for M2). A killed run left no screen records. Decide
   from the counts whether the grid is worth running; M4 is expected to be mostly ineffective (one-room hops stay learnable),
   so check the `|n2` variant. If a family has no accepted instances, fix the mutation or golds, never the gates.
2. **Size the evaluation grid** from measured times (gate about 2 min per instance, one evaluation seed 335-390 s on one core):
   about 30 min per instance for 5 seeds, so a full grid is of the order of 150 CPU-hours before screening removes instances.
   Consider fewer instance seeds or evaluation seeds only with the project owner.
3. **Run `run`** on accepted and unobservable instances (M2 stays in its own stratum, no AvoidRefine recovery claim), 5 evaluation
   seeds, nulls included. Records are JSON per instance; the run is resumable.
4. **Analysis script** over the records: per family and severity, fault removal, collateral retention, F1, attainment, GED, audit,
   S* and the proxy gap (needs end-to-end trajectories of the repaired routes through `experiments/satisfaction`, not wired yet),
   Clopper-Pearson for rates, bootstrap CIs, Wilcoxon signed-rank with Holm across families (`metrics/inference.py`), the
   baselines table (unrefined, random-operator, oracle, trivializer), and the false-repair rate on nulls with designated weak
   edges excluded. Use the funnel counts as the denominator of every claim.
5. **Gaps in the instrumentation:** `oracle_valid_operators` per instance (needs a learner run with oracle witnesses, not done);
   PastRefine's fresh entry vertex at an inner vertex is not handled by `locus_region`; M4/M5 have no region ceiling; the language
   audit is low powered (few fault-only trajectories); `min_semantic_class_count` in the protocol is unused.
6. **Rooms100 (deferred by the project owner).** Tier 1 passes on all 7 worlds as one-room-hop chains (`world_gold`); Tier 2 fails
   on `random_seed0` (8 of 16 edges below 0.85, 7 at 0.0) because PPO speed collapses on some edges. Needs a diagnosis (observation
   scale 0-120, budget, seed sensitivity) before rooms100 golds can be certified; any learner change means re-certifying rooms9.
7. **Cleanup after the grid:** remove the old `faults/{catalogue,injectors}.py` and `experiments/faultline_metrics.py` once nothing
   depends on them (T9 still uses the catalogue); fix or update the pre-existing failure
   `tests/test_learners.py::test_ppo_default_recipe_matches_the_paper_recipe` (ent_coef 0.05 vs 0.02), failing on main as well.
8. **Write-up notes:** certification is "under the protocol", never proof; golds were chosen after pilots (one-room hops, horizon 30,
   `default` excluded, `default` end boxes tried then dropped): report this selection and the certification table; the old
   `faultline_full_*` results are not cited.
