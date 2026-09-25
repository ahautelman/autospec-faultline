from refinement.graph import Node, depth_first_traversal
from refinement.goal import Goal
from env.dirl_grid import RoomsEnv
from env.rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM
import matplotlib.pyplot as plt
from env.gridworldenv import ContinuousGridworld

start_region = Goal(1, 1, 6, 6)
mid_region = Goal(17, 1, 12, 6)
goal_region = Goal(17, 17, 6, 6)

start_node = Node(start_region, False, False, "start")
mid_node = Node(mid_region, True, False, "mid")
goal_node = Node(goal_region, False, True, "goal")

start_node.add_child(mid_node)
mid_node.add_child(goal_node)


custom_doors={
    ((0, 0), (1, 0)): 7,
    ((0, 0), (0, 1)): 7,
    # ((1, 0), (1, 1)): 6,
    ((1, 0), (2, 0)): 7,
    ((2, 1), (2, 2)): 7,
    ((0, 1), (1, 1)): 7,
    ((1, 1), (2, 1)): 7,
}

env = ContinuousGridworld(
    custom_doors = custom_doors,
    render=False

)
def run_3grid(minimum_reach: float = 0.9, n_episodes: int = 3000, n_episodes_test: int = 3000, path: str = ""):
    depth_first_traversal(start_node, env, minimum_reach, n_episodes, n_episodes_test, path)
  