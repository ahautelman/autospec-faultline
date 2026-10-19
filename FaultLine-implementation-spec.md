# FaultLine: Golden Specs, Fault Injection, Metrics

**Keywords:** MUST / SHOULD / MAY as in RFC 2119.

**Notation:** φ* = gold spec, φ_f = faulty spec, φ_r = repaired spec (AutoSpec output). R*, R_f, R_r are the matching regions at one *locus* (a vertex label β(u) or an edge safe set β(e)). L(φ) is the set of trajectories that satisfy φ.

---

## 0. Goal and scope

**Goal.** Build the benchmark layer that answers RQ1: *given a spec with a known injected fault, does AutoSpec recover the intended repair, or only a repair that raises the success rate?*

**Deliverables**

| ID | Deliverable | Package (subject to change, depending on project's file structure) |
| --- | --- |--------------------------------------------------------------------|
| D1 | Shared maze model and AntMaze adapter | `faultline/mazes/`, `faultline/envs/antmaze.py`                    |
| D2 | Gold-spec builder and 3-tier certifier | `faultline/golden/`                                                |
| D3 | Fault mutations, gates, and ground-truth repair encoder | `faultline/faults/`                                                |
| D4 | Metrics and baselines | `faultline/metrics/`, `faultline/benchmark/baselines.py`           |
| D5 | Benchmark runner and statistics | `faultline/benchmark/`                                             |
| D6 | Tests for all of the above | `tests/`                                                           |

**Environments:** `rooms9` (existing), `rooms100` (procedural, per paper A.3.1–A.3.4), `antmaze` (new).

**Out of scope:** RQ2, RQ3, panda-gym, and any change to the refinement operators or Algorithm 1 behavior.

---

## 1. Non-negotiable principles

1. **Gold first, mutate second.** Never search for a gold spec. Construct it from environment ground truth, then derive faulty specs by mutation.
2. **Coarsening only.** Every mutation MUST satisfy L(φ*) ⊆ L(φ_f). AutoSpec is sound (φ_r ⟹ φ_f), so it can only tighten. A stricter-than-gold mutant cannot be recovered by construction. Verify the relation mechanically, never by assumption.
3. **Same environment, different spec.** Dynamics, layout, horizon, and success criterion are identical for φ* and φ_f. Only labels and graph structure differ.
4. **Evaluate against gold.** Success and correctness are measured against φ* and R*, never against φ_r. This is the whole point of the benchmark.
5. **AutoSpec is the system under test.** Use the existing operators and loop unchanged, with paper-default hyperparameters (k for AvoidRefine, PPO settings from paper A.3.6: lr 3e-4, n_steps 2048, batch 64, 10 epochs, γ 0.99, 2×64 network). Instrumentation MAY be added, but a regression test MUST show that the run output with and without it is identical for the same seed.
6. **Seed discipline.** Certification seeds and evaluation seeds are disjoint (default: certification 100–104, evaluation 0–4). All thresholds and the severity grid are frozen in `configs/protocol.toml` before any evaluation run. Record the git hash in every result file.
7. **Report the full funnel.** Report counts of generated, valid, effective, and observable mutants, plus unobservable strata. Never drop results after seeing them.
8. **No "proven" claims.** Gold is *certified under protocol*. Refinement is undecidable in general (AutoSpec §3.2, A.4), and neither DiRL nor LSTS is complete.

---

## 2. Phase 0: contract audit (do this before writing any new code)

Read the existing code and write `docs/contracts.md` with the answers below. Later steps depend on them. If an answer conflicts with an assumption in this document, stop and report it instead of working around it.

1. **Success criterion** in `envs/rooms9.py`: does an edge succeed on entering any state of β(target), or on reaching the sampled goal point within ε? Goal inflation (6.1, M1) only hurts under the second criterion or through reward shaping. Use one criterion for all three environments and record it.
2. **Start and goal sampling:** how start states are drawn from β(source) and goals from β(target).
3. **Operator witness conditions** in `refinement/operators.py`: exactly which trajectories and states each operator consumes (for example, whether AvoidRefine requires the final state to lie outside c).
4. **Loop run record** in `refinement/loop.py`: whether it exposes, per edge, the operator tried, the candidate graph, the measured success, and which candidate was accepted. If not, add these fields additively (principle 5).
5. **Region algebra** in `geometry/`: which exact operations exist (intersection, difference, convex hull, area, containment, half-plane fit). Anything missing is implemented by deterministic grid rasterization (see 7.1).
6. **Predicate projection:** whether predicates can act on the (x, y) coordinates of a larger state vector. AntMaze observations are high-dimensional, so add a projection wrapper if missing.
7. **Scenario catalogue:** whether the five existing 9-rooms scenarios are faulty specs. If so, they SHOULD be reproducible by injecting faults into a derived gold spec (see 9, test T9).

---

## 3. Architecture (subject to change, depending on the project's file structure)

```text
faultline/
  mazes/
    layout.py        MazeLayout, CellGraph, cell<->xy transforms
    analysis.py      shortest paths, cut vertices, viability (backward BFS)
    catalogue.py     ASCII maps: umaze, medium, large; procedural rooms100
  envs/
    antmaze.py       Gymnasium-Robotics adapter + EdgeTask wrapper
  golden/
    builder.py       gold graph from layout + analysis
    certify.py       Tier 1/2/3 certification, Certificate record
  faults/
    mutations.py     the five mutation families + composite + null
    gates.py         validity / effectiveness / observability gates
    ground_truth.py  GroundTruthRepair record, operator ceiling (oracle witnesses)
  metrics/
    regions.py       region measure, fault removal, collateral, IoU/F1, wIoU
    success.py       S_prod, S* (gold-evaluated), proxy gap
    semantic.py      trajectory sampler, precision/recall/F1 of φ_r vs φ*
    structure.py     region-aware graph edit distance
    selection.py     operator accuracy, confusion matrix, false-repair rate
    stats.py         Clopper-Pearson, bootstrap, paired tests
  benchmark/
    baselines.py     unrefined, random-operator, oracle, trivializer
    runner.py        instance grid -> runs -> result JSON
configs/protocol.toml
```

**Key types** (frozen dataclasses, fully type-hinted):

```python
@dataclass(frozen=True)
class Certificate:
    tier1_ok: bool; tier2_ok: bool; tier3_ok: bool
    cert_seeds: tuple[int, ...]
    edge_cp_lower: Mapping[EdgeId, tuple[float, ...]]   # per seed, one-sided 95%
    p_gold: float                                        # threshold derived in 5.3
    protocol_hash: str

@dataclass(frozen=True)
class GoldSpec:
    id: str; env_id: str
    layout: MazeLayout
    graph: AbstractGraph
    viability: Mapping[EdgeId, Region]    # Viab_e, union of cells
    certificate: Certificate

class FaultKind(Enum):
    GOAL_INFLATION; HAZARD_UNLABELED; DEAD_START
    WAYPOINT_CONTRACTION; ROUTE_PRUNING; NULL

@dataclass(frozen=True)
class GroundTruthRepair:
    gold: AbstractGraph; faulty: AbstractGraph
    kind: FaultKind; locus: Locus                  # vertex or edge + attribute
    severity_level: str; severity: float
    factors: Mapping[str, object]                  # e.g. hull_recoverable, halfplane_separable
    target: Mapping[Locus, Region]                 # R*
    ceiling: Mapping[Locus, Region] | None         # R_op, see 6.5
    expected_operators: frozenset[str]             # a priori, from taxonomy
    oracle_valid_operators: frozenset[str] | None  # empirical, see 6.5
```

---

## 4. Environments

### 4.1 Shared maze model (`mazes/`)

All three environments are grid mazes. Model them with one abstraction.

- `MazeLayout`: cell grid (FREE/WALL), `cell_size`, and an xy transform. For `rooms9`/`rooms100` a cell is a room and adjacency is a door. For AntMaze a cell is a maze cell and adjacency is 4-neighborhood of free cells. Read `envs/layout.py` and assume nothing about room interiors beyond what it guarantees (rooms are obstacle-free).
- `CellGraph`: nodes are free cells, edges are adjacencies. `cell_rect(c) -> Box` returns the continuous cell region, shrunk by `margin` (default 0.1·cell_size; for AntMaze at least 0.25·cell_size so the ant torso can reach region interiors).
- **Hazard cells** are spec-only labels (physically passable). They exist in the layout so gold and faulty share one environment (principle 3).
- **Sealed cells** are free cells with no path to the rest of the maze. They are computed, not stored.
- `analysis.py` MUST provide, each with a brute-force cross-check test:
  - `shortest_path(layout, src_cells, dst_cells, allowed)`
  - `cut_vertices(layout, src_cells, dst_cells, allowed)`: cells whose removal disconnects src from dst
  - `viability(layout, target_cells, safe_cells)`: backward BFS from the target through safe cells. This is Viab_e, the set of source states from which edge e can succeed while staying safe.

### 4.2 rooms9, rooms100

Reuse `envs/rooms9.py`. For `rooms100` use the procedural generator of paper A.3.1–A.3.4: 10×10 rooms, 8×8 units each, regions as 2×2 room blocks, opposite-corner start/goal, door probability 0.5, per-seed layout. Layouts that fail Tier 1 are re-sampled and the rejection count is logged.

### 4.3 AntMaze adapter (`envs/antmaze.py`)

Wrap the Gymnasium-Robotics AntMaze family with a custom `maze_map`. **The API details below come from memory. Verify each against the installed version before relying on it, and pin the version in `pyproject.toml`.**

- Build with a custom `maze_map` (walls 1, free 0). Use the env's own maze helpers for cell↔xy conversion (`env.unwrapped.maze`). A round-trip test MUST pass for every cell; never hand-derive the transform.
- Observation for the edge policy: ant proprioception + torso xy (`achieved_goal`) + (goal_xy − xy). Regions are predicates on the xy projection only.
- `EdgeTask` wrapper (same protocol as `rooms9`): sample start from β(source) and goal from β(target) using the Phase 0 sampling rules; place the ant at the sampled xy (reset option if the installed version supports it, otherwise set `qpos` directly, then let the physics settle); apply the Phase 0 success criterion; terminate with failure on leaving β(edge) (the safe set) or on timeout T_e.
- T_e = κ · (planner path length in cells) · (steps per cell). Set κ in the protocol file and freeze it.
- Layouts (ASCII constants in `catalogue.py`): `umaze` (pilot, about 3×5), `medium` (about 8×8), `large` (about 12×12). Each layout carries named slots for the pockets and hazards that mutations need (see 6.1).

### 4.4 Learnability risk and fallback ladder

PPO on Ant from scratch within the paper's 80k–200k-step budgets is unlikely to certify Tier 2. **Run a one-edge pilot before building anything else on AntMaze**, then follow this ladder and record which rung was used:

1. PPO on the Ant with a per-edge budget raised to the pilot-determined level (document the budget).
2. Frozen low-level goal-reaching controller; the edge policy acts at waypoint level. The controller is part of the environment and is identical for gold and faulty.
3. PointMaze (same Gymnasium-Robotics API and same adapter) as the stand-in. Report it as a stand-in, never as AntMaze.

---

## 5. Golden specification

### 5.1 Construction (`golden/builder.py`)

Input: a layout with named start and goal cell-boxes. Output: a `GoldSpec`. Steps, in order:

1. **Source and goal.** Choose start region S and goal region G as rectangular cell-boxes (convex by default, see 6.2). Every cell in them MUST be reachable and viable.
2. **Waypoints.** Compute cut vertices between S and G. Place a waypoint vertex w at each selected cut vertex (box = `cell_rect`). Every feasible path passes through a cut vertex, so a waypoint removes no feasible trajectory.
3. **Safe sets.** β*(e) = free space minus hazard regions. Different edges MAY have different safe sets (needed for M3).
4. **Vertex labels.** β*(u) = the intended box ∩ Viab_{e_out}(u) ∩ cells reachable via e_in. If the intersection is not a box, shrink to the largest box inside it (record the shrink).
5. **Compile** with the existing `compile_spec`, and keep the SpectRL text for provenance.

### 5.2 Certification (`golden/certify.py`)

A gold spec is accepted only if all three tiers pass.

| Tier | Check | Pass rule |
| --- | --- | --- |
| 1. Feasibility | For every edge, `viability` is non-empty and covers the whole gold source region of that edge | exact, by BFS |
| 2. Learnability | Train DiRL (existing learner, large budget) on cert seeds (≥ 5). Evaluate each edge with 1000 rollouts per seed | one-sided 95% Clopper-Pearson lower bound ≥ `p_gold_min` for every edge and every seed. Defaults: `p_gold_min` = 0.8 for rooms, set from the pilot for AntMaze and then frozen |
| 3. Fixpoint | Run Algorithm 1 on gold at threshold p_gold (5.3), on **held-out** seeds | zero refinements accepted on every seed |

Operational meaning of "not over/under-specified": *not over-constrained* means gold removes no environment-feasible trajectory (Tier 1). *Not under-specified* means every gold edge learns without refinement (Tiers 2 and 3).

### 5.3 Threshold p

The paper sets p = 0.99 on purpose, so that refinement is visible. For gold-relative experiments set `p_gold = min over cert seeds and edges of the CP lower bound, minus δ` (δ = 0.05, in the protocol file), capped at 0.99. Otherwise gold itself would fail its own threshold. Also run a **sensitivity sweep** with the paper's p = 0.99 and report both.

---

## 6. Fault injection

### 6.1 Mutations (`faults/mutations.py`)

One family per operator. All are applied to a certified gold spec. Each mutation is a pure function `(GoldSpec, MutationParams, rng) -> GroundTruthRepair` and MUST be deterministic given the seed.

| ID | Kind | Mutation | Target operator | Ground truth R* |
| --- | --- | --- | --- | --- |
| M1 | `GOAL_INFLATION` | b′ = b* ∪ P, where P is a **sealed** pocket (no path from the source under the edge safe set). Layout slot: sealed cells adjacent to G | ReachRefine | b* at β(u′) |
| M2 | `HAZARD_UNLABELED` | c′ = c* ∪ H′, where H′ ⊂ H is the unlabeled part of a hazard H (gold labels H = core ∪ approach part). The hazard straddles the shortest path, so the approach part funnels into the labeled core | AvoidRefine | c* at β(e) |
| M3 | `DEAD_START` | u′ = u* ∪ P, where P is a pocket that is **reachable from upstream but not viable downstream**. Its mouth is a hazard cell in c*(e_in)-allowed but excluded from c*(e_out) | PastRefine | u* at β(u) |
| M4 | `WAYPOINT_CONTRACTION` | Replace u→w→u′ by u→u′ (drop 1 or 2 waypoints at cut vertices) | AddRefine | w* (the dropped vertex) |
| M5 | `ROUTE_PRUNING` | Remove a detour edge u→u_i that gold contains, so the primary route is blocked and the detour is the only success route | OrRefine | the removed edge |

**Why M1 and M3 differ.** M1's pocket is unreachable, so it never appears among reached states, and ReachRefine's convex hull excludes it. M3's pocket is reachable, so ReachRefine on the incoming edge cannot remove it. It is non-viable for the outgoing edge, so only PastRefine can isolate the viable start states. Do not merge them.

**Severity (3 levels, set in the protocol file; defaults)**

| Kind | Knob | low / mid / high |
| --- | --- | --- |
| M1, M3 | pocket mass ρ = \|P\| / \|region′\| | 0.10 / 0.25 / 0.50 |
| M2 | unlabeled fraction of H, and distance of H to the shortest path | 0.25 / 0.50 / 0.75; far / mid / on-path |
| M4 | path length in cells, and waypoint position along the path (fraction of arc length) | short / mid / long; 0.5 / 0.25 / 0.75 |
| M5 | primary-route success under φ_f | ≈ 0.5 / 0.2 / 0 |

**Semantic note on M5.** Gold and faulty MAY accept the same trajectories (the detour is already admitted by another path), so M5 is a *learnability fault*. Check this: sample trajectories and compute L(φ*) vs L(φ_f) with `models_trajectory`. If the faulty graph accepts strictly fewer trajectories, discard the mutant (it violates principle 2).

### 6.2 Recoverability factors (record on every mutant)

AutoSpec's geometry limits what it can recover. Treat this as an experimental factor, so that "AutoSpec failed" is never confused with "R* is not recoverable by this operator".

- `hull_recoverable` (M1): P ∩ ConvexHull(Viab ∩ b′) = ∅. Gold boxes are convex by default; include one non-convex (L-shaped) gold level.
- `halfplane_separable` (M3): the existing half-plane fit separates P from u* with zero error. Include one non-separable level (two pockets on opposite sides).
- `witness_aligned` (M4): |position − 0.5|, since AddRefine uses the temporal midpoint of successful trajectories.

### 6.3 Gates (`faults/gates.py`)

Every mutant passes three gates, in this order. Count and report every rejection (principle 7).

1. **Validity.** (a) Region containment R* ⊆ R_f at the locus (exact or rasterized). (b) Semantic: on the tiny-state-space harness used by `test_compiled_graph_agrees_with_spec_semantics`, graph and spec semantics agree for φ_f, and L(φ*) ⊆ L(φ_f). Failure here is a bug; fail loudly.
2. **Effectiveness.** With the learner and budget used in evaluation, φ_f's baseline edge success at the locus is below p_gold, while gold's is at or above it. Require non-overlapping one-sided CP intervals in ≥ 80% of paired seeds, and a mean gap ≥ Δ_min (default 0.15).
3. **Observability.** Dry-run the operator's witness extraction (Phase 0 item 3) on the baseline φ_f rollouts (using the configured number of trajectories). The witness set MUST be non-empty and intersect the fault locus, with at least m_min states or trajectories (default 20). Mutants that pass the first two gates but fail this one go into the **unobservable stratum** and are reported separately, never counted as incorrect repairs.

### 6.4 Composites and nulls

- **Composites:** combine two mutations from different families on different loci (default: M1+M2, M2+M4). Both must pass the gates independently, and the composite as a whole must pass validity. They test operator ordering (Seq → Add → Past → Or).
- **Nulls:** φ_f = φ* (gold, unmodified). At least 20% of the instance set. They measure the false-repair rate.

### 6.5 Ground-truth record and operator ceiling

`ground_truth.py` builds `GroundTruthRepair` for each accepted mutant.

- **Operator ceiling R_op.** Apply the repo's own operator to *oracle witnesses*. Oracle witnesses are synthetic trajectories generated by a planner and sampled uniformly from R* and Viab: the reached set for ReachRefine, trajectories ending outside c for AvoidRefine, successful and failing starts for PastRefine, midpoints of planner paths for AddRefine. R_op is the best repair the operator's geometry can produce with perfect data. Report IoU(R_r, R*) *and* IoU(R_op, R*); their ratio is the **attainment ratio**. It separates geometry limits from learner or witness limits.
- **Operator sets.** `expected_operators` is the a priori taxonomy (M1, M2 → SeqRefine; M3 → PastRefine; M4 → AddRefine; M5 → OrRefine). `oracle_valid_operators` is every operator that, applied with oracle witnesses, yields a candidate that passes the threshold on the faulty spec. Report operator accuracy against both (strict and lenient).

---

## 7. Metrics

### 7.1 Region measure

Use exact polygon operations where `geometry/` provides them. Otherwise use deterministic grid rasterization over the xy projection with one resolution for the whole benchmark (protocol file), and report the discretization error bound. |·| is area.

### 7.2 Tier A: correctness (`metrics/regions.py`, `metrics/success.py`)

With R_f ⊇ R* ⊇ ∅ and R_r ⊆ R_f (check; a violation is a soundness violation, see 7.5):

| Metric | Definition | Reads as |
| --- | --- | --- |
| Fault removal | 1 − \|R_r \ R*\| / \|R_f \ R*\| | under-repair (1 is best). Defined because the effectiveness gate guarantees \|R_f \ R*\| > 0 |
| Collateral retention | \|R_r ∩ R*\| / \|R*\| | over-repair, i.e. shrinking the task (1 is best) |
| IoU | \|R_r ∩ R*\| / \|R_r ∪ R*\| | combined. IoU = 0 if R_r is empty, and set `degenerate = True` |
| F1 | harmonic mean of precision \|R_r ∩ R*\| / \|R_r\| and recall (= collateral retention) | combined, with components |
| wIoU | IoU with each point weighted by occupancy density of gold-policy or planner trajectories on the same grid | importance-weighted |
| Shrink ratio | \|R_r\| / \|R*\| | values far below 1 signal trivialization |

Always report fault removal and collateral retention separately. IoU alone hides which error occurred.

**Success estimators (both reported)**

- **S_prod:** the paper's estimator, the product of edge success probabilities along the best path in φ_r, from 1000 rollouts per edge.
- **S\*** (primary): chain the final edge policies along φ_r's best path from start states drawn from the **original** η (gold source distribution), and score each full trajectory with `models_trajectory(G*, ζ)`. Goal sampling follows gold.
- **Proxy gap:** Δ = S_r − S*, where S_r is the success AutoSpec itself reports. A positive gap is proxy-vs-truth inflation.

Interpretation (state it in the report): shrinking a *source* region is exposed by S* (conditioning on a subpopulation). Shrinking a *target* region is exposed by collateral retention, not S*.

### 7.3 Tier B: structure (`metrics/semantic.py`, `metrics/structure.py`)

- **Semantic agreement.** Sample trajectories from a reference mixture: gold-policy rollouts, planner paths with noise, and perturbed or biased random walks. Use boundary-targeted stratified sampling with endpoints inside R_f \ R*, R*, and R_r, so that rare classes are populated (minimum count per class in the protocol file). For each trajectory compute y* = `models_trajectory(G*, ζ)` and y_r = `models_trajectory(G_r, ζ)`. Precision = P(y* | y_r = 1) (low means φ_r is too loose). Recall = P(y_r | y* = 1) (low means φ_r is too strict). Report F1 with CP intervals.
- **Region-aware graph edit distance.** Match vertices of G_r to G* by Hungarian assignment with cost 1 − IoU(β_r(u), β*(u)), with unmatched vertices costing 1. Then
  `GED_r = [Σ_matched (1 − IoU) + (#unmatched vertices) + |E_r Δ E*| + Σ_matched-edges (1 − IoU(safe sets))] / (|V*| + |E*|)`.
  It is not capped at 1; report it raw. Use it for M4 and M5, where region IoU does not apply. Plain GED MUST NOT be the primary metric.

### 7.4 Applicability matrix

| Kind | Region metrics | S*, gap | Semantic | GED_r | Waypoint IoU |
| --- | --- | --- | --- | --- | --- |
| M1, M2, M3 | yes | yes | yes | secondary | no |
| M4 | no | yes | yes | yes | yes (inserted vertex vs w*, Hungarian) |
| M5 | no | yes | yes | yes | no |
| NULL | collateral only (R_r vs R*) | yes | yes | yes | no |

### 7.5 Tier C: selection and soundness (`metrics/selection.py`)

- **Operator accuracy:** fraction of runs where the operator accepted at the fault locus lies in the target set (strict: `expected_operators`; lenient: `oracle_valid_operators`). Also report the confusion matrix (fault kind × accepted operator). Algorithm 1's fixed order biases this; interpret accordingly.
- **False-repair rate:** fraction of NULL runs with ≥ 1 accepted refinement, with CP interval. Also count spurious refinements on non-fault edges in fault runs.
- **Soundness audit (all runs, every operator).** Check R_r ⊆ R_f at region loci, and L(φ_r) ⊆ L(φ_f) on the sampled trajectory set via the compiled graphs. **Do not assume Theorem 1.** In particular, OrRefine adds edges u→u_i, which in general enlarges the accepted language, and the paper's proof for it is informal. Report any violation with a minimal counterexample trajectory. A violation is a finding, not a test failure.

### 7.6 Baselines (`benchmark/baselines.py`)

1. **Unrefined:** φ_f as is.
2. **Random operator:** uniformly random operator order, same acceptance rule.
3. **Oracle:** applies the correct operator with oracle witnesses (R_op). It defines the attainable ceiling.
4. **Trivializer:** a sound-by-construction adversary that shrinks target and source regions to a tiny ε-box of the easiest point inside R*. It achieves a high S_r with low collateral retention. It exists to demonstrate that success rate alone does not certify a correct repair.

---

## 8. Experimental protocol and statistics

**Instance grid** (all values in `protocol.toml`): environments {rooms9, rooms100, antmaze} × mutation families {M1–M5} × 3 severities × geometry factor (convex / hull-recoverable vs not; separable vs not) × ≥ 3 layouts per environment × 5 evaluation seeds, plus composites and nulls (≥ 20%). Report the funnel for every cell (6.3).

**Procedure per instance:** build φ_f → gates → run Algorithm 1 (existing loop, unchanged) → record run trace → compute all metrics → run baselines on the same instance and seeds.

**Statistics (`metrics/stats.py`)**

- Clopper-Pearson: one-sided 95% lower bound for gating, two-sided 95% for reporting. Implement with `scipy.stats.beta.ppf` (add `scipy` as an explicit dependency). Test values: two-sided 95% for 0/10 gives upper ≈ 0.3085; for 5/10 gives ≈ [0.187, 0.813]; one-sided 95% lower for 10/10 is 0.05^(1/10) ≈ 0.7411.
- Unit of analysis is the fault instance (layout × mutation × severity); seeds are nested inside it. Use a **cluster bootstrap** over fault instances (10,000 resamples, percentile CI), with seeds averaged within an instance.
- Comparisons against baselines are paired by instance: Wilcoxon signed-rank with Holm correction across fault kinds. Report effect sizes, not only p-values.
- Report every metric stratified by fault kind, severity, geometry factor, and observable vs unobservable.

---

## 9. Testing and acceptance

All tests are pytest and fast by default. Anything that trains a policy carries the `slow` marker.

| ID | Test |
| --- | --- |
| T1 | `viability`, `cut_vertices`, `shortest_path` match brute-force on random small mazes |
| T2 | AntMaze cell↔xy round trip for every cell (skip if the optional dependency is absent) |
| T3 | Region metric identities: R_r = R_f → removal 0, retention 1. R_r = R* → removal 1, retention 1, IoU 1. R_r empty → IoU 0 and `degenerate`. Include hand-computed numeric cases |
| T4 | Clopper-Pearson values from section 8 |
| T5 | Every mutation, over many seeds and layouts: R* ⊆ R_f and L(φ*) ⊆ L(φ_f) on the tiny state-space harness; determinism under a fixed seed |
| T6 | Mutation recoverability flags agree with a direct geometric computation (hull and half-plane) |
| T7 | Oracle baseline attains R_op exactly. Trivializer has S_r high, retention low |
| T8 | Instrumentation regression: loop output with and without trace hooks is identical for fixed seeds |
| T9 | If Phase 0 item 7 applies: injection into gold reproduces the existing 9-rooms scenarios up to region equality |
| T10 | (slow) Tier 2 and Tier 3 certification on one rooms9 gold spec with the tabular learner, then with PPO |

**Acceptance for the whole task:** T1–T9 pass in the default suite. `code-check` is clean. The 9-rooms and 100-rooms pipelines run end-to-end with the tabular learner and produce a result JSON with the full funnel. The AntMaze pilot (4.4) has been run and its outcome and chosen rung are documented.

---

## 10. Implementation order

1. Phase 0 audit → `docs/contracts.md`.
2. `mazes/` + T1. Region metrics + T3. Statistics + T4.
3. Golden builder and Tier 1 on rooms9. Mutations M1–M5 with validity gate + T5, T6.
4. Ground-truth encoder, operator ceiling, oracle and trivializer baselines + T7, T8, T9.
5. AntMaze adapter + T2, then the one-edge learnability pilot (4.4). Decide the rung.
6. Tier 2 and 3 certification, effectiveness and observability gates (slow).
7. Benchmark runner, composites and nulls, statistics report.

---

## 11. Known risks (record outcomes in the report)

- **Learnability on AntMaze:** see 4.4. A failed Tier 2 certification invalidates the gold spec, not the benchmark.
- **Goal-inflation effectiveness** depends on the Phase 0 success criterion. If M1 never passes the effectiveness gate, say so and report it as a finding.
- **AvoidRefine witness dependence:** if its witness requires the trajectory to end outside the labeled safe set, then M2 needs a partly labeled hazard (labeled core plus unlabeled approach), which is why 6.1 defines it that way. Confirm in Phase 0.
- **Operator-order bias** in Algorithm 1 can make a non-target operator accept first. This is measured by selection accuracy (7.5), not corrected.
- **Geometry limits** (convex hull, single half-plane) bound exact recovery. This is why the ceiling and recoverability factors exist.
- **OrRefine soundness** may not hold in general (7.5). M5 is judged by learnability and structure metrics.