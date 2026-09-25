import pygame
import sys
from rooms_envs import GRID_PARAMS_LIST, MAX_TIMESTEPS, START_ROOM, FINAL_ROOM

# Constants
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 480
ROOM_COLOR = (200, 200, 200)
WALL_COLOR = (0, 0, 0)
DOOR_COLOR = (100, 100, 100)
SMALL_ROOM_COLOR = (150, 150, 150)
ROOM_SIZE = 240
SMALL_ROOM_SIZE = 80
DOOR_START = 20
DOOR_END = 60

class PygameRoomVisualizer:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Room Visualization")

    def draw_large_room(self):
        large_room_rect = pygame.Rect((SCREEN_WIDTH - ROOM_SIZE) // 2, (SCREEN_HEIGHT - ROOM_SIZE) // 2, ROOM_SIZE, ROOM_SIZE)
        pygame.draw.rect(self.screen, ROOM_COLOR, large_room_rect)

    def draw_small_rooms(self):
        small_room_size = SMALL_ROOM_SIZE
        for i in range(3):
            for j in range(3):
                small_room_rect = pygame.Rect((SCREEN_WIDTH - ROOM_SIZE) // 2 + i * small_room_size,
                                              (SCREEN_HEIGHT - ROOM_SIZE) // 2 + j * small_room_size,
                                              small_room_size, small_room_size)
                pygame.draw.rect(self.screen, SMALL_ROOM_COLOR, small_room_rect)

    def draw_doors(self, edges):
        small_room_size = SMALL_ROOM_SIZE
        for (room1, room2) in edges:
            # Draw door in the vertical wall between rooms
            if room1[1] == room2[1]:
                x = (SCREEN_WIDTH - ROOM_SIZE) // 2 + room1[1] * small_room_size
                door_start = (SCREEN_HEIGHT - ROOM_SIZE) // 2 + min(room1[0], room2[0]) * small_room_size + DOOR_START
                door_end = (SCREEN_HEIGHT - ROOM_SIZE) // 2 + max(room1[0], room2[0]) * small_room_size - DOOR_END
                pygame.draw.line(self.screen, DOOR_COLOR, (x, door_start), (x, door_end), 2)

            # Draw door in the horizontal wall between rooms
            elif room1[0] == room2[0]:
                y = (SCREEN_HEIGHT - ROOM_SIZE) // 2 + room1[0] * small_room_size
                door_start = (SCREEN_WIDTH - ROOM_SIZE) // 2 + min(room1[1], room2[1]) * small_room_size + DOOR_START
                door_end = (SCREEN_WIDTH - ROOM_SIZE) // 2 + max(room1[1], room2[1]) * small_room_size - DOOR_END
                pygame.draw.line(self.screen, DOOR_COLOR, (door_start, y), (door_end, y), 2)

    def run(self, edges):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill((255, 255, 255))  # Fill the screen with a white background
            self.draw_large_room()  # Draw the large room
            self.draw_small_rooms()  # Draw the small rooms
            self.draw_doors(edges)  # Draw doors based on edges
            pygame.display.flip()  # Update the display to show the changes

            clock.tick(30)  # Control the frame rate

        pygame.quit()
        sys.exit()

# Example usage:
if __name__ == "__main__":
    print(GRID_PARAMS_LIST[0].edges)
    visualizer = PygameRoomVisualizer()
    visualizer.run(GRID_PARAMS_LIST[0].edges)
