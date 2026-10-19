# FaultLine MVP plan (RQ1 first results, RQ2 logging, draft-ready docs)

Goal: a fast, correct, documented MVP that shows AutoSpec's fault recovery on the certified rooms9 golds, logs
trajectories for RQ2, and supports a paper-style draft. RQ3 is dropped. No full grid runs: timing is verified on
single instances, and results come from a small grid over the existing certificates.

## Constraints that shape every step
- **Certificates stay valid.** `reports/faultline_v1/certificates.json` (5 certified golds, protocol hash `694b17d4`
  = current `protocol.toml`, made with the current PPO recipe incl. `ent_coef = 0.05`) is the base for all results.
  Any change to `RoomsEnv`, the PPO recipe, DiRL or `protocol.toml` invalidates it (recertification is hours). So:
  review findings are documented; code is changed only for a defect that makes results wrong, and then we recertify.
- **Speed-ups must be exact.** A cached result must be bit-identical to retraining (training is deterministic:
  measured, same edge + seed gives the same estimate). Verified by a test that compares runs with and without the cache.
- **The loop stays Algorithm 1.** Logging hooks observe; they do not change decisions.

## Step 0. Commit the certificates (5 min)
`git add -f reports/faultline_v1/certificates.json` plus the spec and plan files. `reports/` stays ignored otherwise
(add to `.gitignore`, with `reports/cache/`). `funnel.json` is regenerated deterministically and not committed.

## Step 1. RL correctness review (0.5 day, read-only) -> `docs/paper/appendix-rl-review.md`
Check against SB3 semantics, the AutoSpec paper and DiRL (Jothimurugan et al. 2021). Preliminary findings to verify:
1. `ent_coef` 0.05 vs the test's 0.02: the test is stale, the certificates use 0.05. Fix the test, document the deviation.
2. Observations are unnormalised (0-24 on rooms9, 0-120 on rooms100). Fine on certified rooms9; prime suspect for
   the rooms100 speed collapse. Document; no change (would invalidate certificates).
3. Heading action is a clipped Gaussian on [-pi, pi]: goals due west sit on the clip boundary. Document; check
   against the rooms100 failing edges.
4. Candidate scoring (`route_score`) trains replacement edges from uniform source starts, while DiRL estimates the
   graph under its induced start distribution. The commit gate and the final estimate therefore use different start
   distributions. Check the paper and original code; document as a method note.
5. Budget: 80k steps = 39 PPO updates of 2048 steps; enough for one-room hops (gold lower bounds 0.88-0.997),
   evidently not for long edges. Truncation is bootstrapped correctly by SB3 (`TimeLimit.truncated`).
6. Seeds: DiRL edge seeds and the 10,000-seed evaluation blocks do not collide; estimate seeds = train seeds.
7. Reward: success +10, violation -10 (terminal), step -0.01 + progress shaping; absorbing trap makes M2
   (hazard unlabeled) ineffective by construction (already decided, stratum reported).
Output: a short table finding / impact / action. Anything rated "results wrong" is raised before Step 3.

## Step 2. Exact edge-estimate cache (1 day) -> `faultline/learners/cache.py`
- `CachedLearner(inner, store)` wraps the edge learner. `train_edge` returns a token; if the estimate is cached it
  does not train. `estimate_reach` returns the cached `EdgeEstimate` or trains, estimates and stores.
  LSTS (`continue_edge`) bypasses the cache.
- Key (sha256): env fingerprint (layout doors, max_steps, trap regions), edge source/target/safe `repr`, DiRL start
  states (bytes digest of the wrapper's array + its seed), train seed, budget, recipe, n_rollouts, estimate seed,
  SB3/torch versions, and a `CACHE_VERSION` constant bumped whenever env or learner code changes.
- Store: one compressed `.npz` per key under `reports/cache/` (states as float32: observations are float32, so the
  round trip is exact), atomic write (tmp + rename) so pool workers can share it.
- Runner: wrap the learner in `Settings.edge_learner`; precompute gold baselines once per gold in the pool before
  dispatching instances (removes the per-worker `_GOLD_CACHE` recompute).
- Sharing this buys: gate baseline = AutoSpec initial phase = random-operator initial phase for the same seed;
  random-operator reuses AutoSpec's candidate trainings; DiRL reruns after a commit reuse upstream edges when their
  seeds line up; reruns and resumed runs are near-free.
- Tests: cached vs uncached `run_autospec` gives equal `AutoSpecRun` (tabular learner, and PPO at a tiny budget);
  key changes when any key field changes; concurrent writers do not corrupt.
- **Timing verification** (no full runs): one instance (`rooms9_mixed_star|goal_inflation|mid|0`), gates + 1 eval
  seed, (a) current code, (b) cold cache, (c) warm cache. Record wall-clock and training counts in
  `docs/paper/03-results.md`. Expected: (a) ~8 min, (b) roughly half, (c) seconds.

## Step 3. RQ2 trajectory log (1 day) -> `faultline/rq2/`
Defined before any run, so the MVP grid already produces RQ2 data.
- **Observer hook** in `run_autospec` (`observer=None`): called at each weak edge with the edge, its estimate, and
  each candidate's name and score. Pure observation.
- **Full label vector** (`--score-all-operators`): after the loop commits, the observer also scores the operators
  the loop did not try. Costs extra training (cached, shared with random-operator); off by default, on for the MVP.
- **Files** under `<out>/rq2/`:
  - `edges.jsonl`: one row per (instance, eval seed, phase, edge). Ids, `is_weak`, `p_hat`, edge geometry
    (box centres and sizes, straight-line length, walls crossed, distance to trap), labels: `fault_kind`,
    `is_locus_edge`, `expected_operator` (taxonomy), `committed_operator`, `operator_scores` {seq, add, past, or},
    `operators_clearing_threshold`.
  - `rollouts/<row_id>.npz`: per rollout the (x, y) path (horizon 30, no striding), `reached`, `safe`,
    `final_safe`, and the sampled goal point.
  - Gate baselines also write rows (unrefined faulty graph, 3 seeds), so screened instances add data without an
    AutoSpec run.
- `faultline/rq2/features.py`: offline features from the npz (failure location along the edge: start / mid / goal
  thirds, end-point spread and entropy, fraction timeouts vs violations vs wall-stuck, witness count, distance of
  failures to the target box). Recomputed without rerunning anything.
- `scripts/rq2_dataset.py` -> one table (CSV); XGBoost itself is out of scope here (optional `rq2` extra later).

## Step 4. MVP run on the certified golds (0.5 day setup, runs in background)
- `funnel` -> `screen` on 5 certified golds, instance seed 0, all severities (~1 h on 12 workers, less with cache).
- `run` on accepted + unobservable instances of goal inflation, dead start, waypoint contraction (`|n2`),
  route pruning, nulls: severity mid, instance seed 0, eval seeds 0-2. Estimated 20-30 instances, ~1-2 h wall.
  Hazard unlabeled: screened and reported as the ineffective-by-construction stratum only.
- The plan's "20% nulls" holds within this subset. Everything is labelled preliminary in the docs.

## Step 5. Analysis script (1 day) -> `faultline/benchmark/analyze.py`, `python -m faultline.benchmark analyze`
- Tables (markdown + CSV): funnel per family; per family x severity: recovery rate (CP 95%), fault removal,
  collateral retention, F1, attainment, GED, audit; AutoSpec vs random operator vs unrefined / oracle / trivializer;
  operator selected vs expected (confusion matrix); false-repair rate on nulls (weak-by-design edges excluded).
  Paired Wilcoxon AutoSpec vs random operator with Holm (`metrics/inference.py`), reported with n.
- Figures (existing `viz`): one repair overlay per family, F1 per family with CIs, operator confusion matrix,
  runtime breakdown. Written to `reports/faultline_v1/figures/` and referenced from the results doc.

## Step 6. Cleanup (0.5 day)
- Remove the superseded pipeline: `faults/{catalogue,injectors}.py`, `experiments/faultline_benchmark.py`,
  `experiments/faultline_metrics.py`, their scripts and tests; port T9 (`test_t9_old_cases.py`) to the new mutations
  first or drop it with a note.
- `faultline/envs/rooms/data/*.json` stay: they are the frozen rooms100 worlds (reproducibility), documented as such.
- Rooms100 naming: policy-node order and region names (which OR branch, which region order) made explicit in
  `world.py` / `catalogue.py` docstrings and the README.
- Document `reuse_candidate_cache` (item 9): what it is (keep the policies trained while scoring a committed
  candidate instead of rerunning DiRL), why it exists (speed), why it is off (it changes which evidence the final
  estimate rests on), ablation only. Distinguish it clearly from the new exact cache.

## Step 7. Documentation (1 day)
- **Names, not codes**: M1-M5 -> goal inflation, hazard unlabeled, dead start, waypoint contraction, route pruning
  (code identifiers already use `FaultKind`; fix docstrings, docs, protocol comments, figure labels).
- **README.md (code)**: project goal (RQ1, RQ2), install, how to run (certify / funnel / screen / run / analyze /
  rq2 dataset / viz), how to extend (new env, new gold, new mutation family, new learner), project structure, cache,
  remaining work. RQ3 removed.
- **docs/paper/** (research-paper structure):
  - `01-introduction.md`: problem (success estimate vs intended spec), RQ1 and RQ2, contributions.
  - `02-background.md`: SpectRL, abstract graphs, AutoSpec operators and Algorithm 1, DiRL (condensed from
    `AutoSpec_research_paper.md` and the README).
  - `03-method.md`: environments (rooms9, n-rooms generation), gold specs and the three-tier certification, mutation
    families and severities, gates and funnel, metrics, baselines, statistics, the cache (exactness argument), RQ2
    data format.
  - `04-results.md`: certification table, funnel, preliminary RQ1 tables and figures, runtime measurements,
    threats to validity (gold selection after pilots, small n, rooms100 deferred, M2 by construction).
  - `appendix-rl-review.md` from Step 1.
- Old plans stay in `plans/` as history; `docs/contracts.md` folds into `03-method.md`.

## Order and checkpoints
0 -> 1 -> 2 (timing checkpoint) -> 3 -> 4 (screen funnel checkpoint: families with zero accepted instances are
reported, not tuned) -> 5 -> 6 -> 7. Steps 6-7 can overlap with the Step 4 background runs.
Total: about 5-6 working days plus background compute.

## Out of scope
RQ3, AntMaze, rooms100 certification (documented as future work with the review's suspects), threshold sweep,
XGBoost training, full-protocol evaluation grid (5 instance seeds x 5 eval seeds).
