# 1. Introduction

Reinforcement learning from logical specifications replaces a hand-tuned reward with a task written in a
specification language such as SpectRL (Jothimurugan et al., 2019): "reach the start region, then a waypoint, then
the goal, while staying out of the hazard". Writing such a specification is itself error prone. A goal region drawn
too large, a hazard labelled too small or a missing waypoint makes the task unlearnable, or learnable only in an
unintended way.

AutoSpec (Ambadkar, Žikić and Verma, ICLR 2026) repairs such specifications automatically. It compiles the
specification into an abstract graph whose edges are reach-avoid subtasks, trains one policy per edge, and, when an
edge's estimated success probability falls below a threshold `p`, applies one of four refinement operators that
shrink a region, add a waypoint, cut the source region or route around the edge. Every operator is *sound*: a
trajectory that satisfies the refined specification also satisfies the original one.

Soundness is a one-way guarantee. AutoSpec's own evaluation measures only end-to-end task success after refinement.
A refinement can raise that success by making the task easier than intended: shrinking a goal region until it is
trivial to hit is sound and successful, and it is still the wrong repair. **Success is a proxy for correctness, not
evidence of it.**

FaultLine measures the gap. It starts from a *gold* specification that is known to be correct and learnable,
injects a controlled fault whose ground-truth repair is known by construction, runs AutoSpec on the faulty
specification, and scores the repair against the ground truth.

## Research questions

- **RQ1 (core).** Given a specification with a known injected fault, does AutoSpec recover the intended repair, or
  only a repair that raises the success rate? Measured per fault family: fault removal, collateral retention and their
  F1, region-aware graph edit distance to gold, operator choice against the expected operator, and the false-repair
  rate on unfaulted (null) specifications, against unrefined, random-operator, oracle and trivializer baselines.
- **RQ2 (extension).** Can failure statistics of an edge's rollouts predict which refinement operator will fix it?
  A classifier (XGBoost as a first model) on features such as where along the edge failures happen, how
  concentrated their end points are and whether success depends on the start position, would replace AutoSpec's
  fixed sequential operator search. This report defines and collects the data; the model is future work.

## Contributions

1. A from-scratch, tested implementation of SpectRL, the abstract-graph compiler, the four AutoSpec operators,
   Algorithm 1, and the DiRL graph learner, checked against the paper and its original code
   ([appendix](appendix-rl-review.md)).
2. A fault-injection methodology with certified gold specifications (three-tier certification under a frozen
   protocol), five parametric mutation families with severity levels, a validity gate (faults only coarsen gold) and
   effectiveness / observability gates with a reported funnel.
3. Metrics that cannot be gamed by trivial repairs (removal and retention together), statistics over independent
   instances, and baselines that bracket AutoSpec from below (unrefined, random operator, trivializer) and above
   (oracle).
4. An exact edge-estimate cache that makes the benchmark several times cheaper without changing any result, and a
   logged RQ2 data set of rollouts with full operator labels.

The rest of this document: [background](02-background.md), [method](03-method.md),
[preliminary results](04-results.md), and the [RL stack review](appendix-rl-review.md).
