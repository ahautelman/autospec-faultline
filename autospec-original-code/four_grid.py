
# import random
# random.seed(0)
# import numpy as np
# np.random.seed(0)
# import torch
# torch.random.manual_seed(0)

from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
from env.dirl_grid import RoomsEnv
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM

start_region = Goal(1, 1, 6, 6)
mid_region_1 = Goal(25, 1, 12, 6)
mid_region_2 = Goal(1, 25, 12, 6)
goal_region = Goal(25, 25, 6, 6)

start_node = Node(start_region, False, False, "start")
mid_node_1 = Node(mid_region_1, True, False, "mid_1")
mid_node_2 = Node(mid_region_2, True, False, "mid_2")
goal_node = Node(goal_region, False, True, "goal")

start_node.add_child(mid_node_1)
mid_node_1.add_child(mid_node_2)
mid_node_2.add_child(goal_node)
mid_node_1.add_child(goal_node)



env = RoomsEnv(GRID_PARAMS_LIST[1], START_ROOM[1], FINAL_ROOM[1], max_timesteps=MAX_TIMESTEPS[1])


def run_4grid(minimum_reach: float = 0.9, n_episodes: int = 3000):
    depth_first_traversal(start_node, env, minimum_reach, n_episodes)