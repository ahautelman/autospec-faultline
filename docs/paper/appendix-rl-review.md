# Appendix: correctness review of the RL stack

Scope: `faultline/envs/rooms/env.py`, `faultline/learners/{ppo,dirl,start_distribution}.py`, the edge-training parts of
`faultline/refinement/loop.py`, checked against Stable-Baselines3 2.9 semantics, the AutoSpec paper and its original
code (`autospec-original-code/`), and DiRL (Jothimurugan et al., 2021).

Constraint: the rooms9 gold certificates (`reports/faultline_v1/certificates.json`, protocol hash `694b17d4`) were made
with exactly this env, recipe and DiRL. A change to any of them invalidates the certificates, so findings are fixed
only when they make results wrong; the rest are documented as limitations or future work.

| # | Finding | Impact on results | Action |
|---|---|---|---|
| 1 | `ent_coef = 0.05` in `DEFAULT_RECIPE`; the test still asserted 0.02. | None: the certificates use 0.05. | Test fixed. Deviation from the paper (no entropy term) documented. |
| 2 | Truncation at `max_steps` is reported as `truncated`, SB3's `DummyVecEnv` marks it `TimeLimit.truncated` and PPO bootstraps the value there. | Correct. | None. |
| 3 | Success = within `SUCCESS_RADIUS = 1.0` of a goal point sampled in the target region, and safe throughout. A violation (spec unsafe or physical trap) terminates the episode with -10. | Correct reach-avoid semantics; makes partly unreachable targets score low, which is what the refinement operators need. | None. |
| 4 | Progress shaping `0.1 * (d_prev - d)` uses Euclidean distance to the goal, ignoring walls. Moves that would cross a wall leave the agent in place. | Fine for one-room hops (rooms9 golds certify at 0.88-0.997 lower bounds). On edges whose straight line crosses a wall the shaping rewards approaching the wall and blocked moves cost almost nothing, a classic deceptive-shaping local optimum. Prime suspect for the rooms100 "speed collapse" (agent heads at the goal and crawls). | Documented; rooms100 future work. A wall-aware (geodesic) potential would need recertification. |
| 5 | Observations are unnormalised: `x, y` in `[0, world]` (24 on rooms9, 120 on rooms100), distance up to `1.5 * world`. No `VecNormalize`. | Fine on rooms9. On rooms100 the inputs are five times larger, which slows PPO with a 64x64 tanh MLP. Second suspect for rooms100. | Documented; future work. |
| 6 | The heading action is a Gaussian clipped to `[-pi, pi]`; west-pointing goals sit at the clip boundary where the sampled actions split across `+pi` and `-pi`. | Not observed as a failure on rooms9 (westward edges certify). The project notes found open-map edges in all four directions learn on rooms100. | Documented only. |
| 7 | Evaluation uses the deterministic (mean) action; the original code calls `policy.predict(obs)`, which is stochastic by default. | Deterministic evaluation gives higher, lower-variance estimates; the threshold p = 0.8 is applied to that estimate. Consistent across gold, faulty and repaired graphs, so comparisons are fair. | Documented as a deviation. |
| 8 | Candidate scoring (`_EdgeTrainer.route_score`) trains the replacement edges from uniform source-region starts, as the original code does for every edge. With DiRL as graph learner, the graph's estimates are taken under DiRL's induced start distribution (successful end states of the incoming edge). The commit gate and the final estimate therefore use different start distributions. | The commit decision can differ from what DiRL then measures on the committed graph: a candidate can clear p under uniform starts and score lower under DiRL (or vice versa). The final `best_path_success` is always DiRL's. | Method note in the results. Unchanged: changing it changes Algorithm 1. |
| 9 | Budget 80,000 steps = 39 PPO updates of 2048 steps. | Enough for one-room hops (certified). Not enough evidence for longer edges; this is the reason golds are one-room-hop chains. | Documented (gold design choice). |
| 10 | Seeds: DiRL edge seed `seed + 10000 * k + i`; evaluation episodes use seeds `edge_seed * 10000 + episode`. No collisions within a run; training and evaluation of an edge are independent streams. | Correct. Training is deterministic for a given edge, env and seed (measured: identical estimate on two trainings), which the exact cache relies on. | None. |
| 11 | Absorbing trap identical for gold and faulty specs: the agent is penalised by the physical trap whatever the spec says. | Hazard-unlabeled faults cannot lower the faulty spec's success: ineffective by construction. | Reported as its own stratum (project decision). |
| 12 | DiRL: Dijkstra over `-log p` costs (non-negative), stops when the target is popped, induced distribution = successful end states inside the target region. Matches Jothimurugan et al. 2021. | Correct. DiRL learns only edges on explored paths; unlearned edges are skipped by the scan, as Algorithm 1 specifies. | None. |

Verdict: no finding makes the rooms9 results wrong. Findings 4 and 5 explain why the learner does not transfer to
rooms100 and are left as future work; finding 8 is a property of combining AutoSpec with DiRL and is reported.
