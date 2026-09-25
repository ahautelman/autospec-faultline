import pygame   
import sys
from pygame.locals import *
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from func_timeout import func_timeout, FunctionTimedOut
import sys

class ContinuousGridworld(gym.Env):
    def __init__(self, num_rooms = 3, room_size = 8, screen_size=600, custom_doors = {}, start_room = (0, 0), goal_room = (1, 0), n_steps = 100, render = True, render_mode = "human"):
        super(ContinuousGridworld, self).__init__()
        
        self.grid_size = num_rooms * room_size
        self.screen_size = screen_size
        self.room_size = room_size
        self.num_rooms = num_rooms
        self.current_steps = 0
        self.custom_doors = custom_doors
        self.start_room = start_room
        self.goal_room = goal_room
        self.n_steps = n_steps
        
        self.observation_space = spaces.Box(low=np.array([0, 0, -1, 0]), high=np.array([self.grid_size, self.grid_size, 1, self.grid_size*2]), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        
        
        self.agent_pos = np.array([np.random.uniform(low = (self.start_room[0])*room_size+1, high = (self.start_room[0]+1)*room_size-1), 
                                   np.random.uniform(low = (self.start_room[1])*room_size+1, high = (self.start_room[1]+1)*room_size-1)])
        
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Continuous Gridworld')
        
        self.clock = pygame.time.Clock()
        self.agent_color = (255, 0, 0)
        self.wall_color = (0, 0, 0)
        self.background_color = (255, 255, 255) 
        self.trajectories = []
        self.goal_node = None
        self.start_node = None
        self.avoid = None
        
        self.create_walls(custom_doors)
    
    def create_walls(self, custom_doors = {}):
        self.walls = {}
        for room_row in range(3):
            for room_col in range(3):
                room_x = room_row * self.room_size
                room_y = room_col * self.room_size
                
                wall_bottom = ((room_x, room_y), (room_x + self.room_size, room_y))
                wall_left = ((room_x, room_y), (room_x, room_y + self.room_size))
                wall_right = ((room_x + self.room_size, room_y), (room_x + self.room_size, room_y + self.room_size))
                wall_top = ((room_x, room_y + self.room_size), (room_x + self.room_size, room_y + self.room_size))
                
                if wall_right not in self.walls:
                    self.walls[wall_right] = None
                    if ((room_row, room_col), (room_row + 1, room_col)) in custom_doors:
                        size = custom_doors[((room_row, room_col), (room_row + 1, room_col))]/2
                        self.walls[wall_right] = ((room_x + self.room_size, room_y + self.room_size//2 - size), (room_x + self.room_size, room_y + self.room_size//2 + size))
                if wall_top not in self.walls:
                    self.walls[wall_top] = None
                    if ((room_row, room_col), (room_row, room_col + 1)) in custom_doors:
                        size = custom_doors[((room_row, room_col), (room_row, room_col + 1))]/2
                        self.walls[wall_top] = ((room_x + self.room_size//2 - size, room_y + self.room_size), (room_x + self.room_size//2 + size, room_y + self.room_size))
                if wall_bottom not in self.walls:
                    self.walls[wall_bottom] = None
                if wall_left not in self.walls:
                    self.walls[wall_left] = None
        
    def step(self, action):
        # action = action[0]
        velocity = abs(action[0])*self.room_size
        direction_deg = (action[1])*np.pi
        
        
        dx = velocity * np.cos(direction_deg)
        dy = velocity * np.sin(direction_deg)

        if self.goal_node is not None:
            angle_rad, distance = self.angle_and_distance(self.agent_pos, self.goal_node.goal.current_goal)
        else:
            angle_rad, distance = self.angle_and_distance(self.agent_pos, np.array([self.goal_room[0]*self.room_size/2, self.goal_room[1]*self.room_size/2]))
        
        new_pos = self.agent_pos.copy() + np.array([dx, dy])       
        # new_pos = np.clip(new_pos, a_min = 0.001, a_max = self.grid_size-0.001)
        is_success = False
        if self.check_intersection(self.agent_pos, new_pos):
            reward = -5
            done = True
        
        elif self.avoid is not None and self.avoid.check_region(self.agent_pos, new_pos):
            reward = -10
            done = False
        else:
            
            if self.goal_node is None:
                done = self.check_in_goal(new_pos)
                is_success = done
                reward = 100 if done else 0.5
            else:
                done = self.goal_node.goal.predicate(new_pos)
                is_success = done
                reward = self.goal_node.goal.reward(np.append(new_pos, [angle_rad, distance]))
                if is_success:
                    reward+=self.n_steps/(self.current_steps+1)
            
            self.trajectories[-1][0].append(new_pos)
            self.agent_pos = new_pos
        
        if done and is_success:
            self.trajectories[-1][1] = True
        
        self.current_steps+=1    
        if self.goal_node is not None:
            angle_rad, distance = self.angle_and_distance(self.agent_pos, self.goal_node.goal.current_goal)
        else:
            angle_rad, distance = self.angle_and_distance(self.agent_pos, np.array([self.goal_room[0]*self.room_size/2, self.goal_room[1]*self.room_size/2]))
        
        return np.append(self.agent_pos, [angle_rad, distance]), reward, done, self.current_steps >= self.n_steps, {"is_success": is_success}
    
    def check_in_goal(self, new_pos):
        return self.goal_room[0]*self.room_size <= new_pos[0] <= (self.goal_room[0]+1)*self.room_size \
                and self.goal_room[1]*self.room_size + 1 <= new_pos[1] <= (self.goal_room[1]+1)*self.room_size
    
    def check_intersection(self, old_pos, new_pos):
        
        pass_through = []
        for wall, door in self.walls.items():
            if self.intersect(old_pos, new_pos, wall[0], wall[1]):
                pass_through.append(True)
                if door is not None:
                    if self.intersect(old_pos, new_pos, door[0], door[1]):
                          # Intersection with door, agent didn't pass 
                          pass_through[-1] = False
            else:
                pass_through.append(False)
        return any(pass_through)  # No intersection found

    def intersect(self, A, B, C, D):
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)) or \
            (ccw(A, C, D) != ccw(B, C, D)) and (ccw(A, B, C) != ccw(A, B, D)) or \
            (ccw(C, A, B) != ccw(D, A, B)) and (ccw(C, D, A) != ccw(C, D, B)) or \
            (ccw(C, A, B) != ccw(D, A, B)) and (ccw(C, D, A) != ccw(C, D, B))




    def reset(self, seed = None):
        if self.start_node is None:
            self.agent_pos = np.array([np.random.uniform(low = (self.start_room[0])*self.room_size+1, high = (self.start_room[0]+1)*self.room_size-1), 
                                   np.random.uniform(low = (self.start_room[1])*self.room_size+1, high = (self.start_room[1]+1)*self.room_size-1)])
            angle_rad, distance = self.angle_and_distance(self.agent_pos, np.array([self.goal_room[0]*self.room_size/2, self.goal_room[1]*self.room_size/2]))
        else:
            try:
                func_timeout(10, self.goal_node.goal.reset)
                self.agent_pos = func_timeout(10, self.start_node.sample_state)
            except FunctionTimedOut as e:
                print(e)
                print("Function timed out")
                sys.exit(1)
            angle_rad, distance = self.angle_and_distance(self.agent_pos, self.goal_node.goal.current_goal)
        self.current_steps = 0
        self.trajectories.append([[self.agent_pos], False, self.goal_node.goal.current_goal])
        
        
        return np.append(self.agent_pos, [angle_rad, distance]), {")is_success": False}
    
    def angle_and_distance(self, point1, point2):
        # Calculate the difference between the points
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        
        # Calculate the distance between the points
        distance = np.sqrt(delta_x**2 + delta_y**2)
        
        # Calculate the angle between the line connecting the points and the x-axis
        angle_rad = np.arctan2(delta_y, delta_x)
        # angle_deg = np.degrees(angle_rad)
        
        return angle_rad/np.pi, distance

    
    
    def render(self, mode='human'):
        self.screen.fill(self.background_color)
        cell_size = self.screen_size / self.grid_size

        # Draw walls
        for wall in self.walls:
            pygame.draw.line(self.screen, self.wall_color, (wall[0][0] * cell_size, self.screen_size - wall[0][1] * cell_size),
                            (wall[1][0] * cell_size, self.screen_size - wall[1][1] * cell_size), 3)

            if self.walls[wall] is not None:
                door = self.walls[wall]
                pygame.draw.line(self.screen, self.background_color, (door[0][0] * cell_size, self.screen_size - door[0][1] * cell_size),
                                (door[1][0] * cell_size, self.screen_size - door[1][1] * cell_size), 3)

        # Draw agent
        
        for trajectory in self.trajectories[-1:]:
            color = (0, 255, 0) if trajectory[1] else (255, 0, 0)
            for idx in range(len(trajectory[0])-1):
                
                pygame.draw.line(self.screen, color,
                                (int(trajectory[0][idx][0]) * cell_size, self.screen_size - int(trajectory[0][idx][1]) * cell_size),
                                (int(trajectory[0][idx + 1][0]) * cell_size, self.screen_size - int(trajectory[0][idx + 1][1]) * cell_size),
                                3)
            # if trajectory[1]:
            #     pygame.draw.rect(self.screen, color, (trajectory[2][0] * cell_size, self.screen_size - trajectory[2][1] * cell_size, cell_size//2, cell_size//2))
            

        
        pygame.draw.rect(self.screen, self.agent_color, (self.agent_pos[0] * cell_size, self.screen_size - self.agent_pos[1] * cell_size, cell_size//2, cell_size//2))

        pygame.display.flip()
        self.clock.tick(60)  # Limit to 30 frames per second

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
    
    def set_abstract_states(self, start_node, goal_node, avoid = None):
        
        self.start_node = start_node
        self.goal_node = goal_node    
        self.avoid = avoid
