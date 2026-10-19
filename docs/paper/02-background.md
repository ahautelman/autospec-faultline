# 2. Background

Detailed notes on the AutoSpec paper, with pseudocode and hyperparameters, are in
[`AutoSpec_research_paper.md`](../../AutoSpec_research_paper.md). This section summarises what the method relies on.

## 2.1 SpectRL and the abstract graph

A SpectRL specification is built from predicates over states (regions such as boxes, with `∧`, `∨`, `¬`) and four
task operators: `achieve b` (reach region `b`), `φ ensuring b` (do `φ` while staying in `b`), `φ1 ; φ2` (sequence)
and `φ1 or φ2` (choice). It compiles into an **abstract graph** `G = (V, E, β, s, t)`: each vertex `v` carries a
region `β(v)`, each edge `u → v` is a reach-avoid subtask "from `β(u)`, reach `β(v)` while staying in the edge's safe
set", `s` is the source vertex and `t` the target. A trajectory satisfies `G` when it passes through the regions of
some `s → t` path in order and respects every edge's safe set. Compilation is exact: a trajectory satisfies the
specification if and only if it satisfies the graph (tested exhaustively on a small state space in
`tests/test_compiler.py`).

Example, the paper's fixed 100-rooms task (no safety clause, two choices):

```
achieve Start ; (achieve M1 or achieve M2) ; achieve M3 ; (achieve M4 or achieve M5) ; achieve Goal
```

compiles to the 8-edge DAG `Start → {M1, M2} → M3 → {M4, M5} → Goal`, every edge with the whole world as safe set.

## 2.2 Learning on the graph

One policy is trained per edge (PPO, a 64x64 MLP, 80k steps in this project). Its success probability `p̂(e)` is
estimated from rollouts (1000 here). The success of the whole task is the **best-path product**: the maximum over
`s → t` paths of the product of edge probabilities.

**DiRL** (Jothimurugan et al., 2021) decides which edges to learn: a Dijkstra search over edge costs `-log p̂(e)`,
learning a vertex's outgoing edges only when the vertex is popped, and training each edge from the *induced start
distribution*, the end states of successful rollouts of the best incoming edge. It is the default graph learner in
this project (as in the AutoSpec paper's main experiments).

## 2.3 AutoSpec

Algorithm 1: learn the graph, scan edges in topological order, and for every edge with `p̂(e) < p` try four operators
in a fixed order, committing the first candidate whose replacement edges, after training, clear `p`:

| Operator | Edit | Evidence it consumes | Symptom it fixes |
|---|---|---|---|
| SeqRefine = ReachRefine + AvoidRefine | shrink the target region to the hull of states successful rollouts reached; remove from the safe set the hull of the last `k` states of rollouts that ended unsafe | successful target states, unsafe tails | target partly unreachable; safe set too loose |
| AddRefine | insert a waypoint at the hull of successful rollouts' temporal midpoints | midpoints of successes | edge too long for one policy |
| PastRefine | cut the source region with the half-plane separating successful from failing start states | start states | some starts always fail |
| OrRefine | route around the edge through another parent of its target | none | edge blocked |

Every operator only removes behaviour from the specification's language, so the refined specification is *sound*: a
trajectory satisfying it satisfies the original (Theorem 1). Operators refuse (`CannotRefineError`) when they have no
evidence, for example ReachRefine with zero successful rollouts.

## 2.4 Why success is not enough

AutoSpec stops when the best path clears `p`. Its evaluation reports the resulting success rate. A repair that
shrinks a goal to a corner the agent always reaches is sound and successful but wrong: the intended task has been
replaced by an easier one. Specification gaming (Krakovna et al., 2020) is the general phenomenon; fault injection
with known ground truth (Defects4J, Just et al., 2014) is the standard way to measure repair quality in program
repair. FaultLine brings the second to the first.
