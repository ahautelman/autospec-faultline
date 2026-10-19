from faultline.learners import LSTS, PPOEdgeLearner
from faultline.scenarios.catalogue import blocked_two_path

# THIS ONE TOO HARD PROBABLY
scenario = blocked_two_path()

graph = scenario.build_graph()
env = scenario.make_env()


ppo = PPOEdgeLearner(
    device="cpu",
)

# lsts = LSTS(
#     epsilon=0.3,
#     alpha=0.2,
#
#     # Use an intentionally low threshold for a tiny smoke test.
#     threshold=0.5,
#
#     tolerance=0.10,
#
#     # Small chunks so we can actually observe task switching.
#     chunk_size=5_000,
#
#     # Total training effort for all tasks.
#     max_interactions=30_000,
# )
#
# results = lsts.learn_graph(
#     graph=graph,
#     env=env,
#     edge_learner=ppo,
#     budget=30_000,
#     n_rollouts=30,
#     seed=0,
# )
#
# print("\nLSTS trained edges:")
#
# for result in results:
#     edge = result.edge
#
#     print(
#         f"{edge.source.name:10s} -> "
#         f"{edge.target.name:10s} "
#         f"p={result.estimate.success_probability:.2f}"
#     )

edge = graph.successors(graph.source)[0]

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