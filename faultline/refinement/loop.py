"""The Algorithm 1 refinement loop (paper section 3).

Wires the learner, the four refinement operators, and the commit gate into the
paper's train-scan-estimate-refine cycle. One run:

1. trains a policy for every edge of the input graph;
2. walks the edges in topological order and reads each edge's estimate;
3. for a weak edge, tries SeqRefine, AddRefine, PastRefine, then OrRefine in that
   fixed order, scoring each candidate by training its replacement edges;
4. commits the first candidate that clears the threshold, relearning only the
   edges the edit changed, then continues with the remaining edges;
5. reports the repaired graph, the per-edge success, and the best-path product.

Determinism follows from four fixed choices: the topological scan order, the
operator order, the seeded training calls, and the seeded estimates. The same
inputs therefore produce the same :class:`AutoSpecRun` even when a learner uses
randomness internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from faultline.graph.abstract_graph import AbstractGraph, Edge
from faultline.graph.snapshot import snapshot_graph
from faultline.graph.semantics import best_path_product
from faultline.learners.base import CannotRefineError, EdgeEstimate
from faultline.learners.ppo import DEFAULT_BUDGET, DEFAULT_ROLLOUTS
from faultline.refinement.candidates import (
    Candidate,
    add_candidate,
    or_candidate,
    past_candidate,
    seq_candidate,
)
from faultline.refinement.operators import AVOID_TAIL_LENGTH
from faultline.refinement.trace import StepTrace, build_trace
from faultline.refinement.trajectories import collect_step_samples

# The paper's fixed operator order (section 3): the smallest structural change
# first, the largest last. The loop commits the first candidate above threshold.
_CANDIDATE_BUILDERS = (
    ("seq", seq_candidate),
    ("add", add_candidate),
    ("past", past_candidate),
    ("or", or_candidate),
)


class LoopObserver(Protocol):
    """Receives what the loop sees and decides, without influencing it (RQ2 logging).

    Attributes:
        score_all_operators: When true, after the loop has decided on a weak edge, the
            operators it did not try are scored too (on the same graph and cache) and
            reported with ``by_loop=False``. This costs training but changes nothing the
            loop returns.
    """

    score_all_operators: bool

    def initial(self, graph: AbstractGraph, estimates: dict[Edge, EdgeEstimate]) -> None:
        """Called once with the first-phase estimates of every learned edge."""

    def candidate(self, edge: Edge, estimate: EdgeEstimate, operator: str, score: float | None,
                  committed: bool, by_loop: bool) -> None:
        """Called per operator on a weak edge; ``score`` is ``None`` when the operator refused."""


@dataclass(frozen=True)
class AutoSpecConfig:
    """Tunables for one Algorithm 1 run.

    Args:
        threshold: Success probability an edge or candidate must exceed to count
            as satisfied; the paper's ``p`` (section 6.2).
        budget: Training budget per edge, in environment steps.
        n_rollouts: Episodes per edge success estimate (paper section 6.2).
        avoid_tail_length: Trailing states AvoidRefine blocks (paper section 6.2).
        reuse_candidate_cache: When a graph learner is used, keep the policies
            trained while scoring an accepted candidate instead of replacing the
            whole cache with a fresh graph-learner run. Off by default.
        seed: Base RNG seed threaded into every training call and estimate, so a
            run is reproducible even when a learner trains stochastically.

    How:
        Holds the loop's hyperparameters. The defaults follow the paper's strict
        ``p = 0.99`` and 1000 evaluation rollouts. The avoid tail defaults to
        :data:`~faultline.refinement.operators.AVOID_TAIL_LENGTH` (2), a
        deliberate deviation from the paper's 10 (section 4.3).
    """

    threshold: float = 0.99
    budget: int = DEFAULT_BUDGET
    n_rollouts: int = DEFAULT_ROLLOUTS
    avoid_tail_length: int = AVOID_TAIL_LENGTH
    reuse_candidate_cache: bool = False
    seed: int = 0


@dataclass(frozen=True)
class RefinementStep:
    """One weak edge the loop examined and what it did about it.

    Args:
        source: Name of the examined edge's source vertex.
        target: Name of the examined edge's target vertex.
        operators_tried: Operator names attempted in order, ending at the
            committed one when ``operator`` is not ``None``.
        operator: Committed operator name, or ``None`` for the fallback.
        before: The edge's success estimate before refinement.
        after: The committed replacement's score, or ``before`` when nothing was
            committed.
        trace: What the committed operator changed in the graph and the samples
            it was based on, or ``None`` when nothing was committed.

    How:
        Records the decision in primitive values, so a run can be reported or
        serialised without replaying the loop.
    """

    source: str
    target: str
    operators_tried: tuple[str, ...]
    operator: str | None
    before: float
    after: float
    trace: StepTrace | None = None


@dataclass(frozen=True)
class EdgeSuccess:
    """The final success estimate for one edge of the repaired graph.

    Args:
        source: Source vertex name.
        target: Target vertex name.
        probability: Estimated reach-avoid success probability.

    How:
        Stores primitive values only, so the record serialises to JSON in a later
        phase without touching graph objects.
    """

    source: str
    target: str
    probability: float | None


@dataclass(frozen=True)
class AutoSpecRun:
    """The structured result of one Algorithm 1 run.

    Args:
        graph: The repaired abstract graph.
        config: The configuration the run used.
        steps: One record per weak edge examined, in scan order.
        initial_edge_success: Per-edge estimates before any refinement, in the
            input graph's edge order.
        edge_success: Final per-edge success estimates, in graph edge order.
        best_path_success: Whole-graph success, the best start-to-target path's
            probability product (paper section 6.2).

    How:
        Keeps the repaired graph and the numbers a caller needs for reporting, so
        the scenario runner in a later phase does not re-run training.
    """

    graph: AbstractGraph
    config: AutoSpecConfig
    steps: tuple[RefinementStep, ...]
    initial_edge_success: tuple[EdgeSuccess, ...]
    edge_success: tuple[EdgeSuccess, ...]
    best_path_success: float


@dataclass(frozen=True)
class _Training:
    """A trained policy and its success estimate for one edge.

    Args:
        policy: The backend-specific policy handle from ``train_edge``.
        estimate: The reach-avoid estimate from ``estimate_reach``.

    How:
        Pairs the two values the loop keeps per edge: the handle to hand back to
        the learner and the estimate to score and report.
    """

    policy: Any
    estimate: EdgeEstimate


class _EdgeTrainer:
    """Trains and estimates edge policies, reusing unchanged edges.

    Args:
        env: The environment handed to the learner.
        learner: The spec-guided RL backend.
        config: The run configuration supplying budget, rollouts, and seed.

    How:
        Wraps the learner so the loop can train an edge, store the result under
        the edge's structural key, and reuse that result whenever a graph edit
        left the edge unchanged. That reuse is the warm start the paper describes
        in section 6.3, and it is what keeps "relearn affected edges" cheap.
    """

    def __init__(self, env: Any, learner: Any, config: AutoSpecConfig) -> None:
        self.env = env
        self.learner = learner
        self.config = config

    def train(self, edge: Edge) -> _Training:
        """Train and estimate one edge.

        Args:
            edge: The edge to learn.

        How:
            Calls the learner's ``train_edge`` for the configured budget and
            seed, then ``estimate_reach`` for the configured rollouts and the
            same seed. Passing the seed to training is what makes a stochastic
            learner reproducible across two runs of the loop.

        Returns:
            The policy handle and its estimate.
        """
        policy = self.learner.train_edge(
            self.env, edge, self.config.budget, seed=self.config.seed
        )
        estimate = self.learner.estimate_reach(
            self.env, edge, policy, self.config.n_rollouts, self.config.seed
        )
        return _Training(policy, estimate)

    def align(self, graph: AbstractGraph, cache: dict) -> dict:
        """Return a training cache covering exactly ``graph``'s edges.

        Args:
            graph: The graph the cache must cover.
            cache: Existing trainings keyed by :func:`_edge_key`.

        How:
            Walks the graph's edges in insertion order and reuses a cached
            training when the edge's key is present, otherwise trains the edge.
            Only edges an edit changed or added are therefore trained.

        Returns:
            A new cache keyed by :func:`_edge_key`, one entry per graph edge.
        """
        aligned = {}
        for edge in graph.edges:
            key = _edge_key(edge)
            training = cache.get(key)
            aligned[key] = training if training is not None else self.train(edge)
        return aligned

    def route_score(self, candidate: Candidate, cache: dict) -> tuple[float, dict]:
        """Score a candidate by its best replacement route, training as needed.

        Args:
            candidate: The candidate to score.
            cache: Existing trainings keyed by :func:`_edge_key`.

        How:
            For each route, trains any edge missing from the cache and multiplies
            the edge success probabilities; the candidate's score is the largest
            route product. This is the paper's ``LEARNPOLICY`` gate: a candidate's
            replacement edges are trained before the candidate can be committed.

        Returns:
            The best route product and an extended cache holding any newly
            trained edges.
        """
        working = dict(cache)
        best = 0.0
        for route in candidate.routes:
            product = 1.0
            for edge in route:
                key = _edge_key(edge)
                if key not in working:
                    working[key] = self.train(edge)
                product *= working[key].estimate.success_probability
            best = max(best, product)
        return best, working

    def from_graph_learner(self, graph: AbstractGraph, graph_learner: Any) -> dict:
        """Run DiRL or LSTS and return the edges it learned as a training cache.

        Args:
            graph: The graph to learn over.
            graph_learner: A graph-level learner with a ``learn_graph`` method.

        Returns:
            A cache keyed by :func:`_edge_key`. Edges the learner skipped are
            absent.
        """
        results = graph_learner.learn_graph(
            graph=graph,
            env=self.env,
            edge_learner=self.learner,
            budget=self.config.budget,
            n_rollouts=self.config.n_rollouts,
            seed=self.config.seed,
        )

        return {
            _edge_key(result.edge): _Training(policy=result.policy, estimate=result.estimate)
            for result in results
        }


def run_autospec(
    graph: AbstractGraph,
    env: Any,
    learner: Any,
    config: AutoSpecConfig | None = None,
    graph_learner: Any | None = None,
    observer: LoopObserver | None = None,
) -> AutoSpecRun:
    """Run the Algorithm 1 refinement loop once.

    Args:
        graph: The compiled abstract graph, with source and target set.
        env: The environment, which implements ``set_abstract_states`` and the
            gymnasium interface.
        learner: The spec-guided RL backend that trains and estimates edges.
        config: Run configuration; defaults to :class:`AutoSpecConfig`.
        graph_learner: Optional graph-level strategy, :class:`~faultline.learners.DiRL`
            or :class:`~faultline.learners.LSTS`, that chooses which edges to
            learn. ``None`` trains every edge. Edges it leaves unlearned have no
            estimate (``None`` in the initial record, ``0.0`` in the final one) and
            are skipped by the refinement scan, as in Algorithm 1.
        observer: Optional :class:`LoopObserver` that logs the first-phase estimates and
            every operator score (RQ2). It never changes the run.

    How:
        Trains every edge, then scans the edges in topological order. A weak edge
        goes to :func:`_refine_edge`, which scores the fixed operator sequence and
        commits the first candidate above the threshold, relearning only the edges
        the edit changed. Edges already at or above the threshold are skipped.
        Finally the best-path product is computed over the repaired graph.

    Returns:
        An :class:`AutoSpecRun` with the repaired graph, the decision record, the
        final per-edge success, and the best-path product.

    Raises:
        ValueError: If the graph's source or target has not been set.
    """
    if graph.source is None or graph.target is None:
        raise ValueError("Graph source and target must be set before running AutoSpec")
    active_config = config if config is not None else AutoSpecConfig()
    trainer = _EdgeTrainer(env, learner, active_config)
    working = graph.copy()

    if graph_learner is None:
        cache = trainer.align(working, {})
    else:
        cache = trainer.from_graph_learner(working, graph_learner)
    initial_edge_success = tuple(
        EdgeSuccess(
            edge.source.name,
            edge.target.name,
            (
                cache[_edge_key(edge)].estimate.success_probability
                if _edge_key(edge) in cache
                else None
            ),
        )
        for edge in working.edges
    )
    if observer is not None:
        observer.initial(working, {
            edge: cache[_edge_key(edge)].estimate for edge in working.edges if _edge_key(edge) in cache
        })
    steps = []
    for template in _ordered_edges(working):
        edge = _locate(
            working,
            template.source.name,
            template.target.name,
        )

        if edge is None:
            continue

        key = _edge_key(edge)

        # Algorithm 1: π_e is Null when the underlying spec-guided
        # learner did not learn this edge.
        if key not in cache:
            continue

        estimate = cache[key].estimate

        if estimate.success_probability >= active_config.threshold:
            continue

        working, cache, step = _refine_edge(
            working,
            edge,
            estimate,
            cache,
            trainer,
            align=graph_learner is None,
            observer=observer,
        )
        steps.append(step)

        # The graph learner decides which edges of the refined graph to learn,
        # so rerun it instead of training every new edge.
        if step.operator is not None and graph_learner is not None:
            learned_cache = trainer.from_graph_learner(working, graph_learner)
            if active_config.reuse_candidate_cache:
                # Keep the policies trained while scoring the accepted candidate:
                # they are the evidence the commit decision was based on. Only
                # edges with no existing policy take the freshly learned one.
                cache = {
                    key: cache[key] if key in cache else learned_cache[key]
                    for key in (_edge_key(e) for e in working.edges)
                    if key in cache or key in learned_cache
                }
            else:
                cache = learned_cache

    probabilities = {
        edge: (
            cache[_edge_key(edge)].estimate.success_probability
            if _edge_key(edge) in cache
            else 0.0
        )
        for edge in working.edges
    }
    edge_success = tuple(
        EdgeSuccess(edge.source.name, edge.target.name, probabilities[edge])
        for edge in working.edges
    )
    return AutoSpecRun(
        graph=working,
        config=active_config,
        steps=tuple(steps),
        initial_edge_success=initial_edge_success,
        edge_success=edge_success,
        best_path_success=best_path_product(working, probabilities),
    )


def _refine_edge(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    cache: dict,
    trainer: _EdgeTrainer,
    align: bool = True,
    observer: LoopObserver | None = None,
) -> tuple[AbstractGraph, dict, RefinementStep]:
    """Try the operators in order and commit the first that clears the threshold.

    Args:
        graph: The current graph.
        edge: The weak edge to refine.
        estimate: The edge's current estimate, supplying the rollouts.
        cache: Trainings keyed by :func:`_edge_key`.
        trainer: The trainer that scores each candidate.
        align: Whether to train every edge the committed graph adds. A graph
            learner passes ``False``, since it relearns the refined graph itself.
        observer: Optional logger of every operator's score.

    How:
        Builds candidates in the paper's fixed order. An operator that refuses for
        lack of data is skipped. The first candidate whose best route product
        exceeds the configured threshold is committed after aligning the cache to
        its graph. When none clears, the graph and cache are returned unchanged
        with a fallback record.

    Returns:
        A ``(graph, cache, step)`` triple: the graph after the attempt, its
        aligned training cache, and the decision record.
    """
    before = estimate.success_probability
    tried = []
    for index, (name, build) in enumerate(_CANDIDATE_BUILDERS):
        tried.append(name)
        try:
            candidate = build(graph, edge, estimate, trainer.config.avoid_tail_length)
        except CannotRefineError:
            if observer is not None:
                observer.candidate(edge, estimate, name, None, committed=False, by_loop=True)
            continue
        score, candidate_cache = trainer.route_score(candidate, cache)
        committed = score > trainer.config.threshold
        if observer is not None:
            observer.candidate(edge, estimate, name, score, committed=committed, by_loop=True)
        if committed:
            _score_remaining(graph, edge, estimate, cache, trainer, observer, _CANDIDATE_BUILDERS[index + 1:])
            trace = build_trace(
                snapshot_graph(graph),
                snapshot_graph(candidate.graph),
                collect_step_samples(edge, estimate, trainer.config.avoid_tail_length),
            )
            step = RefinementStep(
                edge.source.name, edge.target.name, tuple(tried), name, before, score, trace
            )
            if align:
                candidate_cache = trainer.align(candidate.graph, candidate_cache)
            return candidate.graph, candidate_cache, step
    step = RefinementStep(edge.source.name, edge.target.name, tuple(tried), None, before, before)
    return graph, cache, step


def _score_remaining(
    graph: AbstractGraph,
    edge: Edge,
    estimate: EdgeEstimate,
    cache: dict,
    trainer: _EdgeTrainer,
    observer: LoopObserver | None,
    builders: tuple,
) -> None:
    """Score the operators the loop skipped after committing, for the observer only.

    How:
        Builds and scores each remaining candidate against the pre-commit graph and
        cache, exactly as the loop would have, and discards the trainings. Nothing the
        loop returns depends on this.
    """
    if observer is None or not observer.score_all_operators:
        return
    for name, build in builders:
        try:
            candidate = build(graph, edge, estimate, trainer.config.avoid_tail_length)
        except CannotRefineError:
            observer.candidate(edge, estimate, name, None, committed=False, by_loop=False)
            continue
        score, _ = trainer.route_score(candidate, cache)
        observer.candidate(edge, estimate, name, score, committed=False, by_loop=False)


def _ordered_edges(graph: AbstractGraph) -> tuple[Edge, ...]:
    """Return the graph's edges in topological scan order.

    Args:
        graph: The graph to order.

    How:
        Lists each vertex in a topological order, then that vertex's outgoing
        edges, so every edge is scanned after the edges that can reach its source.

    Returns:
        The edges in topological scan order.
    """
    return tuple(edge for vertex in graph.topological_order() for edge in graph.successors(vertex))


def _locate(graph: AbstractGraph, source_name: str, target_name: str) -> Edge | None:
    """Find an edge of the current graph by its endpoint names.

    Args:
        graph: The graph to search.
        source_name: Source vertex name of the edge wanted.
        target_name: Target vertex name of the edge wanted.

    How:
        Returns the first edge whose endpoint names match. The scan holds
        pre-refinement edges, and a refinement preserves every other edge's
        endpoints, so a name match re-locates an edge after a graph edit.

    Returns:
        The matching edge, or ``None`` when it is no longer in the graph.
    """
    for edge in graph.edges:
        if edge.source.name == source_name and edge.target.name == target_name:
            return edge
    return None


def _edge_key(edge: Edge) -> tuple:
    """Return a structural key for an edge, used to reuse unchanged estimates.

    Args:
        edge: The edge to key.

    How:
        Combines the endpoint names with the endpoint regions and the safe
        region. Predicates implement equality and hashing, so a deep-copied but
        otherwise unchanged edge keys the same and its estimate can be reused.

    Returns:
        A hashable tuple identifying the edge's learned behaviour.
    """
    return (
        edge.source.name,
        edge.source.region,
        edge.target.name,
        edge.target.region,
        edge.safe,
    )
