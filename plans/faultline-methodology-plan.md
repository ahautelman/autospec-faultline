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
