from faultline.envs import Rooms9, default_regions
from faultline.graph.abstract_graph import AbstractGraph
from faultline.learners import PPOEdgeLearner


regions = default_regions()

graph = AbstractGraph()

start = graph.add_vertex(
    "start",
    regions["start"],
)

goal = graph.add_vertex(
    "goal",
    regions["goal"],
)

edge = graph.add_edge(start, goal)

graph.source = start
graph.target = goal

env = Rooms9()

ppo = PPOEdgeLearner(device="cpu")

policy = ppo.train_edge(
    env,
    edge,
    budget=100_000,
    seed=0,
)

estimate = ppo.estimate_reach(
    env,
    edge,
    policy,
    n_rollouts=50,
    seed=0,
)

print(estimate.success_probability)