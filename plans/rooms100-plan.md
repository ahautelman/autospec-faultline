# n-Rooms / 100-Rooms environment: implementation plan

Goal: a generic n-room environment that reproduces the AutoSpec paper's 100-rooms
experiments (paper §4.2 Fig. 3, App. A.3.1–A.3.5 Fig. 6), so the full AutoSpec loop can be
run and measured on it, and so a later fault-injection layer
([`FaultLine-implementation-spec.md`](../FaultLine-implementation-spec.md)) can plug in.

Source of truth: `AutoSpec.pdf`. [`AutoSpec_research_paper.md`](../AutoSpec_research_paper.md)
§7 is a reconstruction. Where the paper is silent, this plan marks the choice as **[decision]**
(made with the user) or **[assumption]** (to confirm during review).

## 0. Decisions already made

| Topic | Decision |
| --- | --- |
| Door shape | **Full-edge opening.** A door removes the whole 8-unit wall between two rooms, so joined rooms merge into one open area. This matches the existing `open_doors_except_sealed_rooms` convention and the red wall segments in Fig. 6, which are whole room-edges with some missing. |
| Solvability | **Spec-DAG path exists** in the room/door graph. Per-edge reachability and region-internal connectivity are recorded as diagnostics only. Generator rejection-samples until it passes. |
| Stochastic verification | **Protocol + placeholder only.** `DiRLVerifier` raises `NotImplementedError`. No tabular stand-in. |
| Refactor | New `faultline/envs/rooms/` package. `Rooms9` and the 9-specific door builders remain importable as thin wrappers. |
| Scenarios | Fig. 3 replica **and** a Fig. 6 instance, plus frozen random seeds (§5). |
| Fault-injection scope | **Hooks only.** No mutations, gates, or certification in this iteration. |
| Scale | Keep the repo dynamics (speed 2.0, current reward). `max_steps` becomes an env parameter, derived from the room graph. |
| Report | Reuse the AutoSpec runner, records, summary and viz. Report AutoSpec performance only: before/after overlay, learning curve, per seed. No reference run, because the paper gives no 100-rooms reference scores. |
| Deliverable | This file. |

### Does the paper put unsafe regions in the n-rooms layouts?

**No.** The 100-rooms spec is pure reach: `φ = φ_start; (φ_m1 or φ_m2); φ_m3; (φ_m4 or φ_m5); φ_goal`.
It has no `ensuring` clause and no avoid region. Avoid regions and traps appear only in the
9-rooms isolated tests (Figs. 1, 7, 8, 9). In Fig. 6 the red tint marks the Goal and the walls,
not a hazard. So the paper-faithful n-rooms scenarios have **no unsafe regions**.

Optional extension, off by default: the world model carries a `hazards` list (spec-only labels
that are physically passable, FaultLine spec §4.1). With the default empty list the layouts match
the paper. Later, fault injection M2 (`HAZARD_UNLABELED`) uses the same slot. See §3.4.

## 1. What the paper specifies (and what it doesn't)

Facts, from `AutoSpec.pdf`:

- Grid 10×10 rooms, 8×8 units per room, world `[0, 80]²` (A.3.2).
- Regions `R = {Start, Goal, M1, M2, MG, MG1, MG2}`, each a 2×2 block of contiguous rooms. Same
  DAG as Fig. 3, where `MG = M3`, `MG1 = M4`, `MG2 = M5` (A.3.1–A.3.2).
- **Start/Goal placement (A.3.3, eq. 2):** `(Pos_start, Pos_goal) ~ Uniform({(C_TL, C_BR), (C_TR, C_BL)})`.
  Opposite corners only.
- **Intermediate placement (A.3.3, eq. 3):** each remaining region is sampled uniformly from the
  valid 2×2 block positions that do not overlap any already-placed region. Adjacent (touching)
  regions are allowed; only overlap is excluded.
- **Doors (A.3.4, eq. 4):** for every pair of orthogonally adjacent rooms, a door exists with
  probability 0.5, independently. A 10×10 grid has 180 such pairs.
- Fig. 6a (one generated instance, see §5.2) shows Start at the bottom-right corner. That
  orientation cannot come from eq. 2 read literally (it gives `TL=Start, BR=Goal` or
  `TR=Start, BL=Goal`). It also shows the Goal **not in a corner**: it sits on the left edge,
  two rooms below the top. See §5.2 for what this means for the generator.
- Evaluation: 5 procedurally generated environments, one per seed, 80k/100k/120k-step budgets,
  1000 rollouts per edge, p = 0.99, PPO 2×64, lr 3e-4. Success = product of edge successes along
  the best path.
- Reported behaviour: AutoSpec ≈ 50–60% vs DiRL ≈ 20% on random layouts. "Bridge" bottleneck
  `MG → MG1` in 80% of seeds (10–15% → 50–70% via AddRefine/ReachRefine). Corridor switching to
  `MG → MG2` (>78%).
- Fixed Fig. 3: expected refinements ReachRefine on `M1`, PastRefine on `M3`, OrRefine via `M2`;
  success ≈ 0% → ≈ 60%. 8 baseline edges trained, 4–7 extra refinement edges per seed.

Not specified, so a choice is needed:

| Gap | Choice |
| --- | --- |
| Start/Goal orientation | **[assumption]** Sample uniformly over the **4 ordered** pairs `{(TL,BR),(BR,TL),(TR,BL),(BL,TR)}`. Fig. 6a shows a swapped orientation, which eq. 2 omits. Records the orientation in the layout metadata. |
| Order of intermediate placement | **[assumption]** `M1, M2, MG, MG1, MG2` as listed in A.3.3 (mirrors the notes). Fixed order matters, because the distribution depends on it. |
| Whether Start/Goal occupy the corner exactly | Yes: corner 2×2 blocks, e.g. rooms `[0..1]×[0..1]`. |
| Door width | Full edge **[decision]**. |
| Fixed Fig. 3 door map | Not given. Designed here (§5.1). |
| Whether generated layouts are re-sampled until solvable | Paper says "while maintaining a solvable structure". Implemented as rejection sampling, with the rejection count logged. |
| Speed bounds | Paper notes say `v∈[0,1]`, repo uses 2.0. Kept at the repo value **[decision]**. |

Note on percolation: P(door) = 0.5 is the bond-percolation threshold of the square lattice, so a
fair share of raw samples has no Start→Goal path. Expect a nontrivial rejection rate. §4.3 measures
it and the plan reports it instead of guessing.

## 2. Package refactor

### 2.1 Target structure

```text
faultline/envs/
  __init__.py            re-exports; keeps Rooms9, RoomsLayout, door builders
  layout.py              kept as thin shim re-exporting from rooms/ (back-compat imports)
  rooms9.py              Rooms9 = RoomsEnv subclass with 9-rooms defaults + the five 9-rooms door sets
  rooms/
    __init__.py
    layout.py            RoomsLayout (generic), WallKey, door-map helpers, layout_to_dict/from_dict
    env.py               RoomsEnv (generic dynamics, from Rooms9)
    grid.py              RoomGrid: room graph, components, shortest paths, cut rooms, sealed rooms
    regions.py           RegionBlock (2×2 block on room indices) and block → Box
    world.py             RoomsWorld: layout + named regions + hazards + spec DAG (the fault hook)
    generator.py         procedural n-room generator (paper A.3.3–A.3.4)
    verify.py            heuristic verifier + StochasticVerifier protocol + DiRL placeholder
    catalogue.py         fixed layouts (Fig. 3, Fig. 6, frozen seeds)
```

[assumption] The 9-rooms catalogue (`faultline/scenarios/catalogue.py`) keeps its builders in
`envs/rooms9.py` or imports them from `layout.py` unchanged. Only the module they live in changes.

### 2.2 What changes, file by file

| File | Change |
| --- | --- |
| `envs/layout.py` | Move `RoomsLayout`, `_room_index`, `layout_to_dict/from_dict` to `rooms/layout.py`. Remove the module-level `ROOMS_PER_SIDE`, `ROOM_SIZE`, `WORLD_SIZE` from the generic code path. `RoomsLayout.doors` default becomes `{}`, and no longer silently equals the 9-rooms corridor. Keep the 9-specific constants and builders (`default_doors`, `long_corridor_doors`, …) in `envs/rooms9.py`, and re-export them from `envs/layout.py` so existing imports still work. |
| `envs/rooms9.py` | Generic dynamics move to `rooms/env.py` as `RoomsEnv`. `Rooms9(RoomsEnv)` sets `layout` default to `RoomsLayout(default_doors(), 8.0, 3)` and `max_steps=200`. |
| `layout_from_dict` | Already stores `room_size`/`rooms_per_side`, so it already round-trips a 10×10 grid. Add a regression test. |
| `RoomsEnv` | Generalise `max_steps` default to `None → derived`, accepting an explicit value. Keep `MAX_SPEED`, rewards, `SUCCESS_RADIUS` unchanged. |
| `experiments/training.py:231` | `make_eval_env` rebuilds `Rooms9(layout, unsafe_region)` and drops `max_steps`. Change to copy `max_steps` (and use the env's own class). **Without this fix 100-rooms episodes would silently be capped at 200 steps.** |
| `experiments/training.py`, `satisfaction.py` | Type hints `Rooms9` → `RoomsEnv`. |
| `faultline/viz/*` | Already uses `layout.world_size` throughout; verified by test on a 10×10 layout (§6). |
| Tests | Existing 9-rooms tests must pass unchanged (the compatibility gate). |

### 2.3 Back-compat gate

Before any new feature: run the full default suite (`pytest`) on the refactor alone, with no
behaviour change, and check that `reports/rooms9_autospec` regenerates identical summary numbers for a
fixed seed with the tabular learner (the loop is deterministic). This isolates refactor bugs from
feature bugs.

## 3. Generic environment and world model

### 3.1 `RoomsEnv`

Dynamics, observation `(x, y, θ, d)`, action `(v, θ)`, reward, and success rule are those of
`Rooms9` today. `world_size = room_size × rooms_per_side` already comes from the layout, so the
only changes are naming and `max_steps`.

**`max_steps` for n rooms [decision]:** a function
`default_max_steps(grid, graph, speed=MAX_SPEED, slack=4.0)`:
longest shortest-path (in rooms) among the spec's edges × `room_size / speed` × `slack`, rounded
up, never below 200. This is computed from the `RoomGrid` and passed to the env at construction.
Reason: one fixed 200-step horizon on a winding 100-rooms edge could make a bottleneck come from
the horizon instead of the spec, which would corrupt the experiment.

`SUCCESS_RADIUS = 1.0` stays below one room, so a goal behind a wall is not reached through it.

### 3.2 `RoomGrid` (room-level analysis)

Pure functions over `(rooms_per_side, doors)`, no continuous geometry needed because full-edge doors
make room adjacency exact: two adjacent rooms are connected iff the wall key exists in the door map
with the full-edge interval.

- `neighbours(room)`, `components()` (union-find), `component_of(room)`.
- `shortest_path(src_rooms, dst_rooms)` (BFS), returns path and length.
- `cut_rooms(src_rooms, dst_rooms)`: rooms whose removal disconnects src from dst (matches the
  FaultLine spec §4.1 `cut_vertices`). Brute-force cross-check test.
- `sealed_rooms(reference_rooms)`: rooms not in the component of the reference.
- `region_rooms(block)`: the four rooms of a 2×2 block, and `region_internal_components(block)`.

Doors are validated at construction: a door map may hold only full-edge intervals on a `RoomGrid`
(partial gaps remain valid for the generic `RoomsLayout`, for the 9-rooms narrow-door scenario,
but the grid analysis treats a partial gap as an error, not as "closed"). **[assumption]** The
verifier refuses a layout with partial gaps instead of guessing.

This module is the future home of the spec's `mazes/analysis.py`; it keeps no dependency on the env,
so it can move there.

### 3.3 Regions

`RegionBlock(col, row)` = bottom-left room of a 2×2 block. `to_box()` returns
`Box((8c, 8r), (8c+16, 8r+16))` for `room_size=8`. The paper's regions are the whole 16×16 block
(no inset), unlike the 9-rooms regions, which are inset by one unit. **[assumption]** use the whole
block for 100-rooms, per A.3.2 and the notes (§7.1: "16.0 × 16.0 units"). Make the inset a
parameter (default 0) so it can be changed with no code edits.

`valid_blocks(n)` = all `(n−1)²` positions (81 for n=10). Overlap test on room sets.

### 3.4 `RoomsWorld` (the fault-injection hook)

One immutable record that separates what a fault injector must be able to change independently
(FaultLine spec principle 3: same environment, different spec):

```python
@dataclass(frozen=True)
class RoomsWorld:
    name: str
    layout: RoomsLayout                       # walls and doors (physical)
    regions: Mapping[str, RegionBlock]        # named 2×2 blocks (spec labels)
    topology: SpecTopology                    # which regions form which DAG
    hazards: tuple[Box, ...] = ()             # spec-only labels, physically passable; default empty
    max_steps: int | None = None
    meta: Mapping[str, object] = field(default_factory=dict)  # seed, orientation, rejections, etc.

    def build_graph(self) -> AbstractGraph: ...
    def make_env(self) -> RoomsEnv: ...
    def grid(self) -> RoomGrid: ...
```

`SpecTopology` is data: a list of vertex names and edges. `PAPER_TOPOLOGY` is the 8-edge DAG
`Start→M1, Start→M2, M1→M3, M2→M3, M3→M4, M3→M5, M4→Goal, M5→Goal`. Fault injection later swaps
`topology`, a region's block, or `hazards`, and leaves `layout` alone. This also makes the existing
9-rooms scenarios expressible as a `RoomsWorld`, which is spec test T9's precondition.

`hazards` in this iteration: **plumbed through, empty by default, no generator knob.** If hazards
are wanted for the paper-faithful runs, say so. See open questions.

Serialisation: `world_to_dict/from_dict` (layout via `layout_to_dict`, regions as `[col,row]`,
topology as lists). Needed for fixed-layout files, for reports, and for rebuilding a random world
from `(seed)` in worker processes (§7.3).

## 4. Layout generation and verification

### 4.1 Generator (`generator.py`)

`generate_world(seed, *, door_probability=0.5, n=10, regions=PAPER_REGION_ORDER, max_attempts=…) -> RoomsWorld`

All randomness comes from one `numpy.random.Generator(seed)`, consumed in a fixed order so a seed
fully determines the layout:

1. **Start/Goal.** Pick uniformly among the 4 ordered opposite-corner pairs (§1 assumption). Corner
   blocks are `(0,0), (n−2,0), (0,n−2), (n−2,n−2)`.
2. **Intermediates** in the order `M1, M2, M3(MG), M4(MG1), M5(MG2)`: valid blocks minus blocks that
   overlap anything placed so far; sample uniformly from the remainder.
3. **Doors.** Iterate the `2·n·(n−1)` adjacent pairs in a fixed order (vertical then horizontal,
   row-major) and open each with `P = door_probability`, as a full-edge interval.
4. **Verify** (§4.2). If it fails, resample from step 1 with the next draw from the same generator
   (so the result stays a pure function of the seed), up to `max_attempts`, then raise
   `LayoutGenerationError`. `meta` stores `attempts` (the rejection count), the corner orientation,
   and the verification report.

Open question for review: should steps 1–2 be redrawn with the doors, or only the doors? Paper
A.3.3 says placement is fixed first and connectivity afterwards, but says nothing about
re-sampling. **[assumption]** redraw everything, since that keeps the placement distribution
unconditioned on door luck and the layout still determined by the seed alone. The alternative
(redraw doors only) conditions the doors on the placement. Both are cheap to switch (a flag).

### 4.2 Heuristic verifier (`verify.py`)

`verify_heuristic(world) -> VerificationReport`

Exact for existence, since a full-edge door graph makes continuous reachability equal to room
connectivity. "Heuristic" refers to the spec-level approximation: it checks that the *DAG can be
traversed*, not that PPO can learn each edge.

Checks, in order:

1. **Room-graph components.** `RoomGrid.components()`.
2. **Region presence.** For each region, the set of its rooms that lie in the main component
   (the component holding Start's rooms). Empty set → that vertex is unreachable.
3. **DAG route.** For the topology, a route exists iff there is a path of vertices from `Start` to
   `Goal` along DAG edges where every vertex on it has a room in the Start component. Since
   reachability is undirected, this reduces to: each vertex on the route has ≥ 1 room in the Start
   component. The report lists every feasible route (`Start-M1-M3-M4-Goal`, …).
4. **Horizon check.** For each edge on the best route, shortest room-path length × `room_size / speed`
   must fit within `max_steps` (otherwise the layout is unsolvable *within the env budget*).
5. **Diagnostics** (never reject): per-edge reachability; fraction of each region's rooms in the main
   component (a predicted ReachRefine symptom: a low fraction means the target box is partly
   unreachable); per-region internal connectivity; the number of cut rooms between Start and Goal
   (a predicted AddRefine symptom); the set of feasible routes and which branches are dead (a
   predicted OrRefine / corridor-switching symptom).

`VerificationReport` is a frozen dataclass holding `solvable: bool`, the reasons for failure, and the
diagnostics above, JSON-serialisable.

**Pass rule [decision]:** checks 1–4 pass, meaning at least one full route exists. Dead branches
are allowed on purpose, because they model the paper's corridor-switching case.

### 4.3 Stochastic verification placeholder

```python
class StochasticVerifier(Protocol):
    def verify(self, world: RoomsWorld, *, seed: int) -> StochasticReport: ...

class DiRLVerifier:
    """Trains plain DiRL (no AutoSpec) on the unrefined graph. Not implemented yet."""
    def verify(self, world, *, seed):
        raise NotImplementedError("DiRL baseline is not implemented yet; see plans/rooms100-plan.md §4.3")
```

`StochasticReport` fields (placeholder contract, so callers can be written now): per-edge success,
best-path product, learner id, budget, seed, `passed` (best path ≥ a threshold the caller passes).
The generator takes an optional `stochastic: StochasticVerifier | None = None`; when given, a world
must pass both stages. Default `None`. A test checks the placeholder raises with the documented
message, so it can't be silently swapped for something that returns `True`.

### 4.4 Rejection-rate measurement

A script/test (marked `slow` if it exceeds ~10 s) generates N=1000 seeds with verification off and
records the pass rate, the rejection count distribution, and the share of solvable layouts with a
dead branch or a low region reachability fraction. The numbers go in the run report, not into code
constants.

## 5. Fixed layouts (`catalogue.py`)

All fixed layouts are stored as data (door lists + region blocks + topology), serialised with
`world_to_dict`, and built through `RoomsWorld`. Each has a unit test that pins its door count,
region blocks, and verification report.

### 5.1 `fig3` — hand-designed Fig. 3 replica

Spec and regions from the paper, in 0-indexed rooms (`col,row` from the bottom-left):

| Region | Rooms | Box |
| --- | --- | --- |
| Start | `[0..1]×[0..1]` | `[0,16]×[0,16]` |
| M1 | `[3..4]×[0..1]` | `[24,40]×[0,16]` |
| M2 | `[0..1]×[3..4]` | `[0,16]×[24,40]` |
| M3 | `[3..4]×[3..4]` | `[24,40]×[24,40]` |
| M4 | `[6..7]×[4..5]` | `[48,64]×[32,48]` |
| M5 | `[3..4]×[6..7]` | `[24,40]×[48,64]` |
| Goal | `[6..7]×[7..8]` | `[48,64]×[56,72]` |

These come from the notes (§7.3). **Verify against the Fig. 3 image first**: a zoomed render of
page 8 at high DPI, checked box by box, before they are frozen. The image is a tiny bitmap; if
a box disagrees with the notes, the image wins and the notes get corrected.

**Door map: designed, not transcribed.** The paper does not give it. Fig. 3 shows thick walls with
black dots/arrows joining some region centres. Design rule: carve open corridors so the paper's
reported symptoms occur, and record each intended symptom next to the door that causes it:

- **ReachRefine on M1:** one room of M1's 2×2 block is sealed off from the rest (the M1 box is
  partly unreachable) → the learner reaches only part of the sampled target.
- **PastRefine on M3:** M3's block is internally split, so only some start points sampled in M3
  have a route to M4/M5.
- **OrRefine:** the `M1→M3` leg is walled off, or very long, so the route goes through `M2`.
- Keep ≥ 1 full route so the layout is solvable (check 3), and keep the Start→Goal corridor
  winding enough that M4 or M5 is a real bottleneck.

A table in the file lists, for each of the 8 edges: expected operator, symptom, the doors
responsible. This replaces the notes' claim of what the figure's doors are with an explicit design,
which is honest about what is invented. **[assumption]** The paper's Fig. 3 numbers (≈0% → ≈60%)
are a *reference for order of magnitude only*; the replica's door map is not the paper's, so
exact agreement is not expected. Report it that way.

### 5.2 `fig6` — Figure 6a instance

#### What the figure shows

Figure 6a (paper p. 18) is one procedurally generated instance, drawn on the 10×10 grid. The caption
says: Start green, Goal red, intermediate refinement regions blue. Red line segments are walls; a
room edge with no red segment is an open door. Each region is a 2×2 block of rooms with its name
printed at its centre (`START`, `GOAL`, `M1`, `M2`, `MG`, `MG1`, `MG2`). The instance is the
paper's single published sample of A.3.3–A.3.4, and the only one with a door pattern.

Names map to the plan's names as `MG = M3`, `MG1 = M4`, `MG2 = M5`.

#### Region placement (read off a 400-dpi render)

Coordinates are `(col, row)` of the block's bottom-left room, row 0 at the bottom, as in `RegionBlock`.

| Figure label | Plan name | Block | Rooms | Box (x × y) |
| --- | --- | --- | --- | --- |
| START | Start | `(8, 0)` | cols 8–9, rows 0–1 | `[64, 80] × [0, 16]` |
| M1 | M1 | `(5, 2)` | cols 5–6, rows 2–3 | `[40, 56] × [16, 32]` |
| M2 | M2 | `(4, 0)` | cols 4–5, rows 0–1 | `[32, 48] × [0, 16]` |
| MG | M3 | `(8, 4)` | cols 8–9, rows 4–5 | `[64, 80] × [32, 48]` |
| MG1 | M4 | `(2, 3)` | cols 2–3, rows 3–4 | `[16, 32] × [24, 40]` |
| MG2 | M5 | `(3, 8)` | cols 3–4, rows 8–9 | `[24, 40] × [64, 80]` |
| GOAL | Goal | `(0, 6)` | cols 0–1, rows 6–7 | `[0, 16] × [48, 64]` |

The seven blocks are pairwise disjoint. `M1` and `M2` touch along a wall (rows 1/2), which the
paper's placement rule allows (it only forbids overlap).

#### Two ways the figure departs from the generator text

1. **Goal is not in a corner.** A.3.3 puts Start and Goal in opposite corner blocks, but Goal here is
   at `(0, 6)`: the left edge, with two free rows above it. Only Start is in a corner (bottom-right).
   So the figure was not produced by the procedure as written, or the text omits a step.
   Consequence for the generator: it stays as written (corner pairs, §4.1), because that is the
   stated method. The `fig6` world is a fixed layout and does not have to be a generator output.
2. **Door density is about 68%, not 50%.** Pixel detection finds 57 solid walls and 123 open
   edges out of 180 (a binomial with `p = 0.5` has mean 90, sd 6.7, so 123 is about 5 sd away).
   The stated `P(Door) = 0.5` does not describe this instance. It is either from a different
   setting or unrelated to the stated probability. The plan keeps `0.5` for the generator
   (the paper's number) and records this as an open observation, not a fix.

#### Wall transcription (pixel-detected, then checked)

Instead of reading 180 edges by eye, every internal room edge was tested for red pixels along its
middle half (so the corner dots do not count). The result is 57 walls, with one ambiguous edge
(`v(6,3)`, a clear vertical stroke at weak contrast, kept as a wall). Wall keys follow `RoomsLayout`:
`("v", col, row)` is the wall between `(col, row)` and `(col+1, row)`, `("h", col, row)` the wall
between `(col, row)` and `(col, row+1)`.

```text
v (27): (0,5) (0,7) (0,8) (0,9) (1,0) (1,3) (1,7) (1,8) (2,2) (2,4) (2,7) (3,0) (3,1)
        (3,8) (3,9) (4,0) (4,1) (4,4) (4,8) (5,4) (5,5) (5,8) (6,3) (6,7) (7,5) (7,6) (7,9)
h (30): (0,1) (0,3) (0,6) (0,7) (1,3) (1,4) (1,5) (2,0) (2,4) (2,8) (3,8) (4,0) (4,2)
        (4,3) (4,4) (4,5) (4,7) (4,8) (5,1) (5,2) (6,4) (6,6) (7,3) (7,5) (7,7) (8,5)
        (8,7) (9,1) (9,5) (9,8)
```

Doors are all other internal edges, each opening the full 8-unit edge (the figure draws whole-edge
segments, which supports the full-edge decision in §0). ASCII view (top row first; letters mark region
rooms: `S` Start, `1`/`2` M1/M2, `3`/`4`/`5` MG/MG1/MG2, `G` Goal; `|` and `+--` are walls):

```text
+--+--+--+--+--+--+--+--+--+--+
|  |      5 |5          |     |
+  +  +--+--+--+  +  +  +  +--+
|  |  |   5 |5 |  |           |
+--+  +  +  +--+  +  +--+--+  +
|G |G |  |           |        |
+--+  +  +  +  +  +--+  +  +  +
|G  G                   |     |
+  +--+  +  +--+  +  +--+--+--+
|  |              |     |3  3 |
+  +--+--+  +--+  +--+  +  +  +
|      4 |4    |  |      3  3 |
+--+--+  +  +--+  +  +--+  +  +
|     |4  4     1  1 |        |
+  +  +  +  +--+--+  +  +  +  +
|        |      1  1          |
+--+  +  +  +  +--+  +  +  +--+
|           |2 |2        S  S |
+  +  +--+  +--+  +  +  +  +  +
|     |     |2 |2        S  S |
+--+--+--+--+--+--+--+--+--+--+
```

#### Verification already done on the transcription

- Re-rendering the 57 walls with the repo's `draw_room` reproduces the paper's wall pattern edge for
  edge (compared side by side).
- `verify_heuristic` reports it solvable, with all four DAG routes alive and no dead branch and no cut
  room between Start and Goal.
- Region reachability: Start, M1, M3 and M4 are fully reachable; M2, M5 and Goal are 0.75 (one room of
  each is walled off from the rest). So the instance carries the paper's ReachRefine-type symptom on the
  M2, M5 and Goal targets.
- Shortest room paths per edge: `Start→M1` 3, `Start→M2` 3, `M1→M3` 3, `M2→M3` 6, `M3→M4` 6,
  `M3→M5` 9, `M4→Goal` 4, `M5→Goal` 4. The `M3→M4` and `M3→M5` legs are the longest, matching the paper's
  statement that the `MG → MG1` exit is the weak transition, but this instance does not isolate it
  (all branches are reachable, so no corridor switching is forced).
- The derived horizon is 200 steps (the floor), so the horizon is not the limiting factor.

#### How to freeze it

Store the walls and blocks above as `data/fig6.json` through `world_to_dict` (door map = all internal
edges minus the 57 walls). Add a test that pins the block table, the wall count (57), the verifier
report listed above, and the round trip. Mark the metadata: `source = "paper Fig. 6a"`,
`transcription = "pixel-detected from the PDF, one ambiguous edge v(6,3)"`, and the two departures
from the text. The detection method is a one-off (not shipped code); the wall list above is the record.

### 5.3 `random_seed0` … `random_seed4` — frozen generator outputs

Five generator outputs, one per paper seed, stored as data files (not recomputed at test time), so
results survive a generator change. A test checks that `generate_world(seed)` still reproduces the
stored file; if the generator is deliberately changed, the files are regenerated in the same commit,
with a diff.

### 5.4 Optional: per-operator 100-rooms variants

Not in scope (user chose Fig. 3 and Fig. 6). The `RoomsWorld` design makes them cheap later.

## 6. Tests

All pytest, following `.agents/skills/writing-tests/SKILL.md` (behavioural tests, hand-picked
coordinates). The default suite stays fast. Anything that trains a policy is marked `slow`.

| ID | Test | Where |
| --- | --- | --- |
| R1 | 9-rooms suite unchanged after the refactor; same deterministic tabular result as before for one seed. | existing tests |
| R2 | `RoomsEnv` on a 10×10 layout: wall block, full-edge door pass, `world_size == 80`, `max_steps` honoured, `make_eval_env` preserves `max_steps`. | `tests/test_rooms_env.py` |
| R3 | `layout_to_dict/from_dict` round-trips a 10×10 layout. | `test_layout.py` |
| R4 | `RoomGrid`: components, BFS, cut rooms vs brute force on random small grids; sealed rooms. | `tests/test_room_grid.py` |
| R5 | Generator determinism (same seed → identical world), corner pairs only, regions in bounds and non-overlapping, door set size distribution, `door_probability` 0 and 1 extremes. | `tests/test_generator.py` |
| R6 | Generator never returns a world that fails `verify_heuristic`; the rejection count is recorded. | `tests/test_generator.py` |
| R7 | `verify_heuristic` on hand-built grids: solvable, disconnected, one branch dead, region partly sealed, horizon too long. | `tests/test_verify.py` |
| R8 | `DiRLVerifier.verify` raises `NotImplementedError` with the documented message. | `tests/test_verify.py` |
| R9 | Fixed layouts: `fig3`, `fig6`, `random_seed0..4` build, verify as solvable, pin their door count and regions, and round-trip through `world_to_dict`. | `tests/test_rooms_catalogue.py` |
| R10 | Frozen seeds match `generate_world(seed)`. | same |
| R11 | **Visualisation tests (fixed layouts):** render each fixed layout to PNG under a test temp dir; assert the file exists, has the expected size, and that the wall segments drawn equal the layout's solid wall pieces (compute the expected segments, don't compare pixels). A manual-review mode (`--save-figures`) writes the images to `reports/rooms100/layouts/` for the user to look at. | `tests/test_viz_rooms100.py` |
| R12 | **Random layout test:** generate K seeds (default 20), verify each, render each into one contact-sheet PNG with the verification verdict in the title (pass/fail, attempts, routes), and a summary table. The test asserts the generator's contract (every returned world passes) and the report content, and also runs a **forced-fail** case (door probability too low) so the failing branch of the verdict display is exercised. The sheet is for the user to look at. | `tests/test_random_rooms.py` |
| R13 | (slow) tabular learner end to end on `fig3` with a small threshold, to check the loop runs on 100 rooms and produces a report. | `tests/test_rooms100_autospec.py` |

The viz tests need a `viz/` addition: `draw_room` already takes any `RoomsLayout` and a graph snapshot
and uses `world_size` throughout (checked), so the new code is a thin `draw_world(ax, world)` that
adds a door-count/verdict caption. 10×10 label sizes and the legend are the likely touch-ups.

## 7. AutoSpec run script

Command (new, mirrors `rooms9_autospec`):

```text
uv run --extra interop --extra reports python -m faultline.experiments.rooms100_autospec \
    --worlds fig3 fig6 random_seed0 ... --seeds 0 1 2 3 4 --curves --budget 100000 --workers N
```

### 7.1 Generalising `Scenario`

`Scenario` (`scenarios/catalogue.py`) is tied to `Rooms9` and to the 9-rooms table fields. Changes:

- `make_env()` returns a `RoomsEnv` built from a `RoomsWorld` (with `max_steps`).
- `reference_before` / `reference_after` become `float | None` (**no reference for 100 rooms**).
  The records and viz must tolerate `None`: skip the reference marker and legend entry.
  `SCHEMA_VERSION` is bumped, with a loader that still reads old files.
- `expected_operator` is `str | None` already; for 100 rooms it is a set/tuple of operators per
  edge (from §5 design), recorded as metadata. The summary shows it but does not score against it.
- `RoomsWorld.to_scenario(name)` builds a `Scenario` from a world. The 9-rooms catalogue is untouched.

### 7.2 Run pipeline

Reuses `autospec.runner.run_jobs`, `PooledCheckpointLearner`, `records`, `summary`, `curves`,
and `viz.render_directory`. New code is the entry module and a world registry. Outputs under
`reports/rooms100_autospec/{data,figures,summary.md}`:

- per `(world, seed)` JSON report (the existing schema);
- `summary.md` table: per world, mean ± std best-path success before/after, operators committed,
  refinement count, elapsed seconds;
- figures: **overlay before/after**, **learning curve**, **per-seed** comparisons. These are the
  existing viz figures, with 10×10 sizing. No reference line.

The 100-rooms graph has 8 edges, so the pooled executor gets 8 parallel trainings per `(world, seed)`.

### 7.3 Worker process lookup (a structural constraint)

`executor.py:91` resolves `SCENARIOS[request.scenario]` **by name in the worker process**. Fixed
layouts register by name, which works. Random worlds must too: register the name `rooms100_random_<seed>`
and have a resolver `resolve_scenario(name)` that builds it from `generate_world(seed)` (pure
function of the seed), falling back to `SCENARIOS` for the 9-rooms names. This replaces direct
`SCENARIOS[...]` lookups in `executor.py`, `viz/{run,render,load,cli}.py` with the resolver.
Alternative (pass the serialised world in the request) works but changes the request schema.
**[assumption]** the resolver, since it keeps requests small.

### 7.4 Compute

Paper: 5 seeds, ~100k steps/edge, ≈390 s per seed on a laptop for the loop. Here 100 rooms × N fixed
worlds × 5 seeds ≈ 8–15 edge trainings each. The runner prints the estimated number of trainings
before it starts, and supports `--resume` as 9-rooms does. `--smoke` runs one world, one seed, a
small budget, to check the pipeline without waiting. **Tabular learner:** `CELL_SIZE = 1.0` gives an
80×80 = 6400-cell planning grid per edge (BFS is cheap), but `MAX_WAYPOINTS = 500` may be too low
for a winding 100-rooms edge. Review during implementation and parameterise from `max_steps`.

## 8. Fault-injection hooks, summary

What this iteration delivers for the later FaultLine layers, with no mutation code:

- `RoomsWorld` separates layout (physical), regions and topology (spec), and `hazards` (spec-only).
  A fault injector builds a new world by replacing one of those.
- `RoomGrid` gives the analysis the spec's §4.1 needs: shortest paths, cut vertices, sealed cells.
  Viability (backward BFS) is the one piece still missing; add it here if cheap, since it is a small
  function over the same graph. **[assumption]** include `viability(target_rooms, safe_rooms)` now,
  with a brute-force cross-check.
- `VerificationReport` already computes the predicted symptoms (low region reachability → ReachRefine,
  cut rooms → AddRefine, dead branch → OrRefine) that a later gate can reuse.
- `world_to_dict` makes fault-injected variants storable and reproducible.

## 9. Implementation order

1. Refactor to `envs/rooms/` with wrappers; run the back-compat gate (§2.3). Fix `make_eval_env`.
2. `RoomGrid` + R4, `RegionBlock`, `RoomsWorld`, topology + serialisation + R3.
3. `RoomsEnv` `max_steps` derivation + R2.
4. Verifier + stochastic protocol/placeholder + R7, R8.
5. Generator + R5, R6, and the rejection-rate measurement (§4.4).
6. Fixed layouts (transcribe Fig. 3 boxes and Fig. 6a; design the Fig. 3 doors) + R9, R10.
7. Viz: `draw_world`, then R11 and R12 (look at the images with the user before going further).
8. `Scenario` generalisation, resolver, schema bump (§7.1, §7.3).
9. `rooms100_autospec` script + smoke run + R13.
10. Full run on the fixed layouts; write up the numbers beside the paper's.

## 10. Risks

- **Fig. 3 is a small bitmap and its doors are not given.** Mitigated by the explicit design table in §5.1.
  Fig. 6a was transcribed by pixel detection and checked by re-rendering (§5.2); one edge is ambiguous.
- **The Fig. 3 door map is invented.** Results are compared with the paper only in direction and
  order of magnitude, never as a reproduction claim.
- **Rejection rate at the percolation threshold** may be high. Measured in §4.4 before the generator is
  frozen. If it is unworkable, the options are a cap on attempts or a smarter door sampler (e.g. force
  one random route, then Bernoulli the rest). That would deviate from the paper, so it needs your call.
- **PPO on 100 rooms** may not clear thresholds inside the paper's 80–120k-step budgets with the repo's
  reward. The plan reports what happens and does not tune to hit ≈60%.
- **Schema change** (`reference_*` becoming optional) touches the records and viz. Covered by the loader
  fallback and by tests on old files.
- **Orientation and placement-order assumptions** (§1) change the layout distribution. Both are recorded
  in the layout metadata so a later change can be audited.

## 11. Open questions for review

1. Hazards: leave `RoomsWorld.hazards` plumbed but unused (as written), or also add an optional
   generator knob that places a hazard box on the Start→Goal route, for later fault experiments?
   The paper has none, so the default stays empty either way.
2. On a failed verification, redraw everything (as written) or only the doors?
3. Region inset: whole 16×16 block (as written, paper) or inset by 1 unit like the 9-rooms regions?
4. Should the Fig. 3 replica's design table (§5.1) be reviewed by you before I write the door map?
5. Fig. 6a puts Goal outside the corner and shows about 68% doors (§5.2). Should the generator stay as
   the text states (corner Start/Goal, `p = 0.5`), or should it be loosened to match the figure
   (Goal anywhere on the border, a higher door probability)? The plan keeps the text; `fig6` is a fixed
   layout either way.
