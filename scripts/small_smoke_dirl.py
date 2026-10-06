from faultline.learners import DiRL, TabularLearner
from faultline.scenarios.catalogue import blocked_two_path


scenario = blocked_two_path()

graph = scenario.build_graph()
env = scenario.make_env()

dirl = DiRL()
learner = TabularLearner()

results = dirl.learn_graph(
    graph=graph,
    env=env,
    edge_learner=learner,
    budget=0,
    n_rollouts=20,
    seed=0,
)

for result in results:
    edge = result.edge

    print(
        edge.source.name,
        "->",
        edge.target.name,
        result.estimate.success_probability,
    )