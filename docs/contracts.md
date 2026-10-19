# FaultLine contracts (Phase 0 audit)

What the existing code guarantees, read from the source on 2026-10-19 (main at merge `d45b988`).
Items marked **open** were not verified and block a later step. AntMaze is out of scope, so spec item 6
(projection for high-dimensional states) is dropped; predicates already take a `dims` argument that defaults
to `(x, y)` of the 4-D room observation.

## 1. Success criterion (`envs/rooms/env.py`, `RoomsEnv.step`)
An edge episode succeeds when the agent is within `SUCCESS_RADIUS = 1.0` of the **sampled goal point** and the
position is safe (inside the spec's safe region and outside the trap). It does not succeed on merely entering the
target region. Consequence: goal inflation (M1) bites through goal sampling. Goals sampled in a sealed pocket
are unreachable, so the edge fails on that fraction of episodes. One criterion is used for rooms9 and rooms100.

## 2. Start and goal sampling
`reset` draws the goal from the target region and the start from the source region with `_sample_point`:
uniform rejection sampling inside the region's axis-aligned bounds (`region_bounds`), seeded by `reset(seed=...)`.
`reset(options={"start": (x, y)})` overrides the start (used to chain edges). Sampling is uniform over the
*region*, not over rooms, so a union region is sampled in proportion to area.

## 3. Trap semantics (project standard, decision D1)
`RoomsEnv(unsafe_region=T)` is an absorbing trap: entering it sets `_violated`, the episode terminates with the
violation reward, and `info["is_safe"]` is `False`. This holds in training and evaluation, so PPO sees the true
penalty whether or not the spec labels the trap. A fault therefore shrinks the **labelled** unsafe area while the
trap stays physical. Whether the faulty baseline still fails is decided by the effectiveness gate, not by the env.
`HiddenHazardRooms9` (evaluation-only hazard) is superseded and will be removed.

## 4. Operator witness conditions (`refinement/operators.py`)
| Operator | Consumes | Notes |
|---|---|---|
| ReachRefine | states inside the target region visited by **successful** episodes; hull ∩ region | follows the paper's prose, not Algorithm 2's pseudocode |
| AvoidRefine | last `k` states of every episode that **ended unsafe**; region minus their hull | unsafe = final state outside the safe set **or** `final_safe is False` (env trap). So a trap the spec does not label still triggers AvoidRefine, which M2 relies on. |
| SeqRefine | both of the above, each skipped if it has no data | |
| AddRefine | `states[len // 2]` of successful episodes; hull ∩ safe | temporal midpoint, so waypoint position matters (`witness_aligned`) |
| PastRefine | first states of successes vs failures; half-plane fit | needs at least one success and one failure; single half-plane only |
| OrRefine | no samples; adds parallel edges via other parents of the target | needs a sibling parent not already reachable from the source |

Every operator raises `CannotRefineError` when starved, and refuses regions with no samplable area.
Defaults: `AVOID_TAIL_LENGTH = 2` (the paper uses 10; documented deviation), `DEFAULT_BUDGET = 80_000`,
`DEFAULT_ROLLOUTS = 1000` (`learners/ppo.py`).

## 5. Loop run record (`refinement/loop.py`)
`AutoSpecRun` exposes the final graph, `initial_edge_success`, `edge_success`, `best_path_success`, and one
`RefinementStep` per examined weak edge: `operators_tried`, committed `operator` (or `None`), `before`, `after`,
and a `StepTrace` (graph diff of the committed edit plus its witness samples). **Gap:** the candidate graph of each
*rejected* operator is not recorded. This is enough for operator accuracy and locus checks. The oracle-valid operator
set (spec 6.5) is computed separately with oracle witnesses, so no loop change is needed. `reuse_candidate_cache`
defaults to off and the benchmark keeps it off (decision D5).

## 6. Region algebra (`geometry/`)
Available: `intersect`, `difference` (symbolic predicates), `convex_hull`, `half_plane_partition`,
`region_bounds`, `sample_within`, serialisation; predicates evaluate pointwise. **Not available:** area,
containment, union area, IoU. These are computed by deterministic grid rasterisation
(`faultline_metrics._predicate_counts`, currently 121 cells per axis). The protocol fixes 240 cells per axis (0.1 unit)
for the whole benchmark and the discretisation error bound will be reported.

## 7. Room-grid analysis (`envs/rooms/grid.py`)
`RoomGrid` answers `shortest_path`, `cut_rooms`, `components`, `sealed_rooms`, `viability` (backward, through safe
rooms) and, added in step 2, `reachable` (forward, through safe rooms). All are cross-checked against naive
implementations on random grids (`tests/test_room_grid.py`, `tests/test_room_grid_analysis.py`).
Endpoint convention: sources (for `reachable`) and targets (for `viability`) are always included even when not
in the safe set; every other room on the route must be safe.
**Limit (resolved):** the analysis is room-granular and rejects partial doors. A trap *inside* a room
(the old `overwide_safety` strip) is invisible to it. **Decision:** M2 uses a rasterised cell grid (the protocol's
0.1-unit resolution) for sub-room traps, built in step 3 alongside the gold builder.

## 8. Scenario catalogue (`scenarios/catalogue.py`)
Five 9-rooms scenarios (`trap_goal`, `narrow_door`, `long_corridor`, `mixed_start`, `blocked_two_path`) are
hand-built faulty specs, some with a physical trap the spec does not label (`narrow_door`). **Open:** whether they are
reproducible by injection into a derived gold spec (test T9). Not yet checked.

## 9. Compute reference
Recorded rooms100 runs (`reports/rooms100_autospec`: threshold 0.99, budget 100k, 1000 rollouts, 14-core pool):
about 1,000–1,230 s wall-clock per run. A rough grid of 3 layouts × 5 families × 3 severities × 5 seeds is 225 runs,
plus at least 20% nulls and one baseline run per instance for the gates: on the order of 500 runs, so roughly
half a day at those speeds. Fault instances have fewer edges than fig3/fig6, so this is an upper bound. Re-measure on
a real fault instance in step 5.
