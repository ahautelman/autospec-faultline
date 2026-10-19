# FaultLine visualization refactor: implementation plan

Goal: academic-quality figures for **any** run (full AutoSpec loop or a single scenario, one seed or many), generated from persisted data, with the room drawn dynamically and annotations chosen from the refinements that were actually applied.

Decisions (defaults agreed): paper-ready vector output (PDF + PNG), before/after panels with sampled trajectories, small multiples per refinement step plus a cumulative panel, geometry persisted as JSON (option a), paper reference numbers optional, rooms9 only (no generic env protocol yet), no new dependency (matplotlib + numpy only).

## 0. Current state (why this is needed)

| Problem | Where |
|---|---|
| Overlay assumes one bottleneck edge and a toy `Overlay` record | `experiments/overlay_figures.py`, `overlays.py` |
| Full loop reports carry an empty overlay | `autospec/records.py::to_scenario_report` |
| `RefinementStep` stores names and scores only, so no refined geometry, no rollouts | `refinement/loop.py` |
| Legend entry added on every `shade_region` / `scatter_points` call | `experiments/drawing.py` |
| Per-pixel Python predicate evaluation, 220x220 | `drawing.py::shade_region` |
| Styling is ad hoc (hard-coded colours, dpi 150, tiny legends) | `figures.py`, `overlay_figures.py` |

## 1. New package layout

```
faultline/viz/
  __init__.py        public API: render_run, render_scenario
  style.py           rcParams theme, ROLE colour map, figure sizes, save helper (pdf+png)
  geometry.py        Predicate -> drawable (polygons / half-plane / raster fallback)
  room.py            draw_room(ax, layout, graph, unsafe_region)
  trajectories.py    draw_trajectories(ax, trajs, ...)
  annotations.py     registry: operator -> annotator; draw_step(ax, step)
  legend.py          role-based legend builder (one shared legend per figure)
  captions.py        title / caption text generated from steps and metrics
  figures/
    overlay.py       per-run overlay figure (before | after | cumulative, or per-step multiples)
    curves.py        learning curves, generic over runs
    seeds.py         multi-seed spread
    reference.py     optional paper-comparison bars
  cli.py             `python -m faultline.viz <data-dir> [--scenario X] [--seed N]`
```

The old `experiments/{drawing,figures,overlay_figures,overlays}.py` are kept until the new package reaches parity, then deleted in the last step. `rooms9_report.py` and `rooms9_autospec.py` call `faultline.viz` afterwards and shrink to thin wrappers around it.

## 2. Data model changes (the foundation; do first)

### 2.1 Serialisable regions (done)
Added `faultline/geometry/serialize.py` (in `geometry`, not `spectrl`, because `ConvexHull` lives there and `geometry` already imports `spectrl`): `predicate_to_dict` / `predicate_from_dict` covering `Box, Disc, HalfPlane, ConvexHull, And, Or, Not, Universe, Empty`. This is a small tagged-dict mapping, e.g. `{"t": "box", "lo": [...], "hi": [...]}`. Round-trip tests for each type. (`spectrl/printer.py` exists, but string round-tripping of hulls is lossy and parsing is slower, so use dicts.)

### 2.2 Refinement trace (done: `refinement/trace.py`, `refinement/trajectories.py`)
Implemented as a diff of graph snapshots (vertex region changes, safe-region changes, added vertices/edges, removed edges) plus `StepSamples`, stored as `RefinementStep.trace`. The field list below was the first sketch.
Extend `RefinementStep` (or add a sibling `StepTrace` stored on `AutoSpecReport.traces`) with:

```python
operator: str | None
edge: (source, target)
region_before: dict        # serialised target / source / safe region as relevant
region_after: dict | None
removed_safe: dict | None  # AvoidRefine: area subtracted
waypoint: dict | None      # AddRefine
half_plane: {normal, offset} | None   # PastRefine
added_edge: (src, dst) | None         # OrRefine
witness_points: [[x,y],...]           # capped, e.g. 200
unsafe_tail_points: [[x,y],...]
```

`_refine_edge` already has `candidate` and `edge`, so each operator's output is available at commit time. Operators return graphs; the trace is derived by diffing `graph` and `candidate.graph` (which regions or edges changed), so operator code stays untouched. For `seq`, the diff yields both the target change (Reach) and the safe change (Avoid) and reports them as two sub-effects.

### 2.3 Graph snapshots (done: `graph/snapshot.py`)
`AutoSpecReport` additionally stores `initial_graph` and `final_graph` as `{vertices: {name: region_dict}, edges: [...], source, target, safe per edge}`. This lets the overlay draw before/after without re-running the scenario.

### 2.4 Trajectories (done)
Rollouts already reach the parent process in `TrainingRecord.estimate`, so no worker change was needed. Samples are per-edge segments of the best path (`trajectories_before` / `trajectories_after`), not end-to-end paths, because each edge trains from its own sampled starts. Capture K trajectories (default 8 success + 8 failure) for the best path in the initial graph and in the final graph, subsampled in time (every n-th state, endpoints kept). Stored as `{graph: "before"|"after", success: bool, xy: [[x,y],...]}`.
- **Check first:** whether rollouts survive the worker boundary (`executor.py` returns "light records, never the model"). `TrainingRecord` may need a `sample_trajectories` field filled in the worker, where the policy lives. If so, add it there, capped, so IPC stays small.

### 2.5 Compatibility (done)
Bump a `schema_version` on the JSON. The loader accepts old files (no traces) and the figure code degrades gracefully: room plus curves, no annotations, with a note in the caption. Overlay data for old runs needs a re-run; curves keep working.

## 3. Drawing layer

Step 4 done: `annotations.py`, `legend.py`, `captions.py`. Deviation from the sketch below: annotations are driven by the *shape of the edit* in the trace (changed region, removed safe area, new region, new edge, half-plane boundary), not by an operator-name registry, so `seq` and unknown operators need no special case; operator names are used only for titles and captions. Trajectory drawing (`trajectories.py`) moves to step 5 with the overlay figure.

Status: `style.py`, `polygons.py`, `geometry.py`, `room.py` done (step 3). `polygons.py` is a numpy-only split of the exact-polygon logic out of `geometry.py`, so it is testable without matplotlib. `draw_room` takes a `RoomsLayout`, a `GraphSnapshot` and an optional trap predicate, and returns the set of `Role`s drawn for the legend.

1. **`style.py`**: serif font (STIX/Computer Modern-like), 9 pt base, thin axes, no top/right spines on line plots, colour-blind-safe palette keyed by **role** (`start, goal, safe, unsafe, refined, waypoint, boundary, traj_ok, traj_fail, before, after`). Figure widths: 3.4 in (single column), 7.0 in (double). Saves `.pdf` and `.png` (300 dpi). Tests assert figure size and file creation, not pixels.
2. **`geometry.py`**: `region_to_artists(predicate, ax, role)`:
   - `Box` -> `Rectangle`, `ConvexHull` -> `Polygon`, `HalfPlane` -> clipped half-space polygon.
   - `And/Or/Not/Disc` -> raster contour fallback at 400x400 with vectorised evaluation (batch `evaluate` where possible).
   Artists are created *without* legend labels; roles are tracked separately (see 5).
3. **`room.py`**: `draw_room(ax, layout, graph, unsafe_region)` draws outer boundary, internal walls with door gaps (port of `draw_layout`), start/goal regions from graph source/target, intermediate vertices as waypoints, hidden trap if `unsafe_region` is set. Axis labels `x`, `y` with ticks at room boundaries (0, 8, 16, 24), so the figure is not unlabelled.
4. **`annotations.py`**: `ANNOTATORS: dict[str, Callable[[Ax, StepTrace], set[Role]]]`. Each annotator draws and returns the roles it used.
   - `reach`: faded original target + solid refined target, witness points.
   - `avoid`: hatched removed-safe area, unsafe tail points.
   - `add`: waypoint region plus arrows source -> waypoint -> target.
   - `past`: dashed half-plane line, retained-start shading, failed starts in light grey.
   - `or`: arrow for the added edge.
   `seq` = `reach` + `avoid`, so there is no special case.
   Unknown operator -> generic "region changed" diff drawing, so new operators don't crash the figures.
5. **`legend.py`**: collects roles returned by everything drawn, builds one legend below the figure (`fig.legend`, `ncol` auto) with one entry per role, fixed order, short labels (e.g. "Start", "Goal", "Unsafe", "Refined region", "Successful path", "Failed path"). No per-call labels. Operator-specific detail goes in the caption instead.
6. **`captions.py`**: from steps, build a title (`<scenario>: <ops applied>`) and a caption sentence per step (`Step 1 - ReachRefine on s0->g: target shrunk to visited hull (area 36 -> 11); edge success 0.31 -> 0.87.`). The caption is also written to a sidecar `.txt` and optionally rendered under the figure.

## 4. Figures

Step 5 done: `viz/run.py` (`RunView`, `run_view_from_report`), `viz/trajectories.py`, `viz/figures/overlay.py`. Footer (legend plus optional caption) is reserved explicitly in inches rather than left to constrained layout, which overlapped them. `draw_region` now returns `None` for a region with no area so the legend never names an invisible element. Captions are also written as a `.caption.txt` sidecar.

All take an `AutoSpecReport`-like run object (`RunView`), a thin adapter that normalises both the toy `ScenarioReport` and `AutoSpecReport`, so a single-scenario run and a full loop share one code path.

- **`overlay`**: layout chosen from the step count:
  - 0 steps: single room with trajectories (nothing was refined; the caption says so).
  - 1+ steps: `[Before | After]` panels, each with the room and trajectories; the "After" panel carries all annotations cumulatively. With 2+ steps, add an optional second row of small multiples (one panel per step, only that step's annotation) via `--per-step`.
- **`curves`**: before vs after best-path success, x = steps (k), y = success probability, threshold line, dashed paper reference marker only if present. Mean line plus faint seed lines plus min-max band when several seeds exist; single seed degrades to a single line.
- **`seeds`**: dot/strip plot of before vs after per seed with mean markers, replacing the current bars when n_seeds <= 10.
- **`reference`**: unchanged semantics, optional, skipped when `reference_*` are `None`.
- Combined multi-scenario grids (`overlays.png`, `learning_curves.png`) are built by composing the same per-scenario drawing functions into a shared grid with a single legend.

## 5. Entry points

Step 6 done: `figures/curves.py`, `seeds.py`, `reference.py`, `grid.py` (one panel per scenario under a shared footer), `footer.py` (extracted from the overlay). `RunView` now carries curves, paper references, threshold and budget, with adapters `run_view_from_report` and `run_view_from_scenario_report` (toy report: curves and a geometry-less step, no overlay annotations) and `group_by_scenario`. New roles: threshold, paper-reference marker, paper-reference bar. The seed figure became a paired dot plot (dots, pair lines, mean bar, std whisker) instead of mean-only bars.

- `faultline.viz.render_run(run_or_runs, out_dir, formats=("pdf","png"))` picks figures by data present (traces -> overlay; multiple seeds -> seeds figure; curves -> curves).
- `python -m faultline.viz reports/rooms9 [--scenario narrow_door] [--seed 0]` re-renders from JSON, no training. `rooms9_report --plot-only` and `rooms9_autospec --plot-only` delegate to it.
- Scenario config -> room needs only `SCENARIOS[name]`, or the scenario embedded in the report if it's not in the catalogue. Store `layout` (walls/doors) and `unsafe_region` in the report so even a custom scenario can be redrawn from JSON alone.

Step 7 done: `viz/load.py` (recognises files by content; catalogue order), `viz/render.py` (`RenderOptions`, `render`, `render_directory`; skips with a note what the data cannot support), `viz/cli.py` + `python -m faultline.viz`. `rooms9_report` and `rooms9_autospec` now draw into `<output>/figures` through it. Toy spread summaries become per-seed two-point views, used only for the per-seed and comparison figures. Toy `Overlay` data is not drawn (no overlay for toy runs).

## 6. Implementation order (each step lands with tests and keeps the old figures working)

1. `spectrl/serialize.py` + round-trip tests.
2. Data capture: `StepTrace`, graph snapshots, layout serialisation, trajectory capture (incl. worker-boundary check); schema version + backwards-compatible loader; tests on a stub learner (no PPO).
3. `viz/style.py`, `geometry.py`, `room.py`: render the five toy rooms; eyeball against current images.
4. `annotations.py` + `legend.py` + `captions.py`; unit tests that each operator's trace yields the expected roles and that legend roles are unique.
5. `figures/overlay.py`; render all five scenarios and a multi-refinement synthetic case.
6. `figures/curves.py`, `seeds.py`, `reference.py` on `RunView`.
7. `cli.py`, wire into `rooms9_report` / `rooms9_autospec`, `--plot-only`.
8. Delete the old drawing modules, update README, regenerate `reports/`.

Step 8 done: old `experiments/{drawing,figures,overlay_figures}.py` and the draw helpers in `overlays.py` deleted (`build_overlay` data capture kept), old-figure adapters removed from `autospec/records.py`, README has a Figures section. Toy overlays are rebuilt by `viz/replay.py`, which replays the real operator on the saved points.

## 7. Testing

- Unit: serialiser round-trips; diff -> `StepTrace` for each operator; annotator role sets; legend de-duplication; caption text.
- Smoke: `render_run` on fixtures (one single-step, one multi-step, one zero-step, one multi-seed) writes the expected files without error under Agg.
- Backward compat: loading an old `reports/rooms9/data/*.json` renders curves and a plain room without raising.
- Visual review: contact sheet of all outputs checked by hand once per iteration (pixel tests would be brittle).

## 8. Risks and open items

- **Rollout capture across the process boundary** may need a `TrainingRecord` change (see 2.4). Verify before step 2.
- **Trace by graph diff** assumes operators only change what they claim. For `seq`, separating the target change from the safe change relies on comparing `edge.target.region` and `edge.safe` before and after. Add a test for that.
- **Rasterised regions** (`And/Or/Not`) are the only pixelated case. Refined regions (box intersect hull) can be exact polygons with a small clipping routine if the raster looks poor.
- **Old data** is not redrawable as overlays; this is accepted, and the loader says so.
- **Deferred**: a generic `EnvRenderer` protocol for non-rooms9 environments. `room.py` is the only env-specific module, so it is the one seam to abstract later.
