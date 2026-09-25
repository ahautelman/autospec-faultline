from env.dirl_grid import GridParams

GRID_PARAMS_LIST = []
MAX_TIMESTEPS = []
START_ROOM = []
FINAL_ROOM = []

# parameters for a 3-by-3 grid
size1 = (3, 3)
edges1 = [((0, 0), (0, 1)), ((0, 1), (0, 2)),
          ((1, 1), (1, 2)), ((0, 1), (1, 1)),
          ((1, 2), (2, 2))]
room_size1 = (8, 8)
wall_size1 = (5, 5)
vertical_door1 = (2, 6)
horizontal_door1 = (2, 6)

GRID_PARAMS_LIST.append(GridParams(size1, edges1, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(300)
START_ROOM.append((0, 0))
FINAL_ROOM.append((2, 2))

# parameters for a 4-by-4 grid
size1 = (4, 4)
edges1 = [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 2), (0, 3)), ((0, 1), (1, 1)), ((0, 2), (1, 2)),
          ((1, 2), (1, 3)),((1, 3), (2, 3)),
          ((2, 3), (2, 2)), ((2, 2), (2, 1)), ((2, 1), (2, 0)), ((2, 1), (3, 1)),
          ((3, 0), (3, 1)), ((3, 1), (3, 2)), ((3, 2), (3, 3)),
          ]
room_size1 = (8, 8)
wall_size1 = (1, 1)
vertical_door1 = (2, 6)
horizontal_door1 = (2, 6)

GRID_PARAMS_LIST.append(GridParams(size1, edges1, room_size1, wall_size1,
                                   vertical_door1, horizontal_door1))
MAX_TIMESTEPS.append(300)
START_ROOM.append((0, 0))
FINAL_ROOM.append((3, 3))
