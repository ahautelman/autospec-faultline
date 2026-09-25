import numpy as np
import gymnasium as gym

class GridWorldEnv(gym.Env):
    
    metadata = {'render.modes': ['human']}
    def __init__(self, n_rows = 3, n_cols = 3, room_size = 4, doors = [(0, 1), (1, 2), (1, 4), (4, 5), (5, 8)], current_pos = (2, 2)):    
        super(GridWorldEnv, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.room_size = room_size
        self.door_size = room_size//2
        self.wall_thickness = room_size//16
        self.doors = doors
        self.current_pos = current_pos

    def step(self, action):

        direction = action[0] # radians
        velocity = action[1] # units (always positive)

        velocity = np.clip(velocity, 0, 1)
        direction = (direction) * np.pi/2

        x_displacement = velocity*np.cos(direction)
        y_displacement = velocity*np.sin(direction)


    def new_location(self, x_displacement, y_displacement):

        current_room = int(self.current_pos[0]/(self.room_size))*(self.room_size - 1) + int(self.current_pos[1]/self.room_size)

        





