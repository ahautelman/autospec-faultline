"""Regression checks for independent evaluation seed blocks."""


def _episode_seed(base_seed: int, episode: int) -> int:
    # Mirrors PPOEdgeLearner and TabularLearner.
    return int(base_seed) * 10_000 + episode


def test_adjacent_experiment_seeds_do_not_share_rollout_seeds():
    first = {_episode_seed(0, episode) for episode in range(1000)}
    second = {_episode_seed(1, episode) for episode in range(1000)}

    assert first.isdisjoint(second)


def test_episode_seed_is_reproducible():
    assert _episode_seed(4, 37) == 40_037
