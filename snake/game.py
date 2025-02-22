import pygame, gym, torch, colorsys
import random as rnd
from pygame.math import Vector2
import numpy as np
from gym import spaces
import torch.nn as nn 
import torch.optim as optim
from model import DQN
from a_star import * 

class Game(gym.Env): 
    
    def __init__(self, ai_mode=False, model_path='dqn_final_model.pth'):
        super(Game, self).__init__()
        # Initialize Pygame
        pygame.init()

        # Set up screen
        self.screen_width, self.screen_height = 800, 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)
        self.grid_size = 20 # grid cell in pixels

        # Snake properties
        self.snake_radius = 10
        self.initial_snake_length = 5
        # Create surfaces for the snake's head and body 
        self.head_surface = pygame.Surface((self.grid_size, self.grid_size))
        self.head_surface.fill((255,255,255))
        self.body_surface = pygame.Surface((self.grid_size, self.grid_size))
        self.body_surface.fill((0, 255, 0))
        # move one grid cell at a time
        self.snake_speed = self.grid_size
        self.direction = Vector2(1, 0)  # Start moving to the right
        self.growing = False

        # Walls
        self.walls = []
        self.wall_spawn_interval = 5000 # time in ms
        self.last_wall_spawn_time = pygame.time.get_ticks()
        self.wall_size = self.grid_size
        self.max_walls=20

# Food properties
        self.food_radius = 10
        self.food_position = None

        # Game state
        self.score = 0
        self.game_over = False

        # Define action space (3 actions: move forward, turn left or right)
        self.action_space = spaces.Discrete(3)

        # Define observation space (snake head (x,y), direction (x,y), food (x,y), walls)
        wall_obs_size = self.max_walls * 2
        self.observation_space = spaces.Box(
                low=0, 
                high=800, 
                shape=(6 + wall_obs_size,), 
                dtype=np.float32
        )
        
        #self.MOVE_EVENT = pygame.USEREVENT + 1.

        # AI mode flag 
        self.ai_mode = ai_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN().to(self.device)

        if self.ai_mode:
            # Initialize and load the trained model 
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except FileNotFoundError:
                print(f"Model file '{model_path}' not found.")
                exit()
            self.model.eval()

        # Start the game
        self.reset_game()

    def eval(env, model, episodes=5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_rewards = []
        for _ in range(episodes):
            state = env.reset()
            done = False 
            episode_reward = 0 

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = q_values.argmax().item()

                state, reward, done, _ = env.step(action)
                episode_reward += reward
                env.visTraining()

            total_rewards.append(episode_reward)
        average_reward = sum(total_rewards) / episodes 
        return average_reward

    def get_ai_action(self):
        path = self.get_next_move()
        if path:
            next_cell = path[0]
            current_cell = (int(self.snake_body[0].x // self.grid_size), int(self.snake_body[0].y // self.grid_size))
            dx = next_cell[0] - current_cell[0]
            dy = next_cell[1] - current_cell[1]
            # Map the movement to an action (0: Forward, 1: Left, 2: Right)
            action = self.map_direction_to_action(dx, dy)
            return action
        else:
            # No path found, take default action 
            return 0 # move forward
    
    def get_next_move(self):
        grid = self.create_grid()
        start = (int(self.snake_body[0].x // self.grid_size), int(self.snake_body[0].y // self.grid_size))
        goal = (int(self.food_position.x // self.grid_size), int(self.food_position.y // self.grid_size))
        path = a_star_search(start, goal, grid)
        return path

    def map_direction_to_action(self, dx, dy):
        """
        Map the desired movement direction (dx, dy) to an action 
        (0: Forward, 1: Left, 2: Right), based on the snake's 
        current direction 
        """
        # Get the current direction as a tuple
        current_direction = (int(self.direction.x), int(self.direction.y))
        optimal_direction = (dx, dy)

        # If the optimal direction is the same as the current_direction, move forward
        if optimal_direction == current_direction:
            return 0 
        # Define the possible directions in order 
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left
        # Get indices of current and best direction 
        current_idx = directions.index(current_direction)
        optimal_idx = directions.index(optimal_direction)

        # Calculate the difference in indices 
        diff = (optimal_idx - current_idx) % 4
        if diff == 1:
            return 2 # turn right 
        elif diff == 3:
            return 1 # turn left 
        else: 
            # the snake cannot reverse direction (diff == 2), so return forward 
            return 0

    def update_direction(self, action):
        """Update the snake's direction based on the action."""
        # Define possible directions
        direction_mappings = {
            (0, -1): {'left': Vector2(-1, 0), 'right': Vector2(1, 0)},  # Up
            (0, 1): {'left': Vector2(1, 0), 'right': Vector2(-1, 0)},   # Down
            (-1, 0): {'left': Vector2(0, 1), 'right': Vector2(0, -1)},  # Left
            (1, 0): {'left': Vector2(0, -1), 'right': Vector2(0, 1)},   # Right
        }
    
        # Get the current direction as a tuple
        current_direction = (int(self.direction.x), int(self.direction.y))
    
        # Determine new direction based on the action 
        if action == 0:  # Move forward
            new_direction = self.direction
        elif action == 1:  # Turn left
            new_direction = direction_mappings[current_direction]['left']
        elif action == 2:  # Turn right
            new_direction = direction_mappings[current_direction]['right']
        else:
            new_direction = self.direction  # Default to moving forward if action is invalid
    
        self.direction = new_direction

    def step(self, action):
        """Execute one step in the environment"""
        # Update the snake's direction based on the action 
        self.update_direction(action)

        # Previous distance to food using Euclidean distance
        prev_distance = self.snake_body[0].distance_to(self.food_position)
        # New distance to food using Euclidean distance
        new_distance = self.snake_body[0].distance_to(self.food_position)

        self.move_snake()
        done = False 
        reward = 0 

        # check for collisions and reward accordingly
        if self.check_food_collision():
            reward += 2 
        if self.check_collision():
            reward -= 10
            done = True
            self.game_over = True
        else:
            # small penalty to encourage faster completion
            reward -= 0.05
            # small reward for moving closer to the food 
            if new_distance < prev_distance:
                reward += 0.1
            else:
                reward -= 0.1
        # Return the current state, reward, done flag, and info dict 
        observation = self.get_observation()
        return observation, reward, done, {}

    def get_observation(self):
        """Get the current state of the game"""
        head = self.snake_body[0]
        food = self.food_position
        walls = []
        for wall in self.walls:
            walls.extend([wall.x, wall.y])
        # Pad walls to maximum number
        while len(walls) < self.max_walls * 2:
            walls.extend([0, 0])  # Use zeros or a specific value to indicate no wall

        observation = np.array(
            [head.x, head.y, self.direction.x, self.direction.y, food.x, food.y] + walls,
            dtype=np.float32
        )
        return observation

    def reset_game(self):
        """Reset game to initial state"""
         # Align starting position to the grid
        self.start_pos = Vector2(self.screen_width // 2, self.screen_height // 2)
        self.start_pos.x = (self.start_pos.x // self.grid_size) * self.grid_size + self.grid_size // 2
        self.start_pos.y = (self.start_pos.y // self.grid_size) * self.grid_size + self.grid_size // 2

        self.snake_body = [self.start_pos - Vector2(i * self.snake_speed * self.direction.x, i * self.snake_speed * self.direction.y) for i in range(self.initial_snake_length)]
        self.direction = Vector2(1, 0)
        self.food_position = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.growing = False
        self.walls = []

        pygame.time.set_timer(pygame.USEREVENT + 1, 150) # starting speed 

    def reset(self):
        """Reset envrionment to start a new episode"""
        self.reset_game()
        return self.get_observation()

    def spawn_food(self):
        """Randomly spawn food within the screen bounds."""
        while True:
            position = Vector2(
                rnd.randint(self.food_radius, self.screen_width - self.food_radius),
                rnd.randint(self.food_radius, self.screen_height - self.food_radius)
            )
            if all(segment.distance_to(position) > (self.snake_radius + self.food_radius) for segment in self.snake_body):
                return position

    def move_snake(self):
        next_position = self.snake_body[0] + self.direction * self.snake_speed

        # Boundary collision detection
        if (next_position.x < 0 or next_position.x > self.screen_width or
            next_position.y < 0 or next_position.y > self.screen_height):
            self.game_over = True
            return

        # Insert the new head
        self.snake_body.insert(0, next_position)

        # Remove the last segment if not growing
        if not self.growing:
            self.snake_body.pop()
        else:
            self.growing = False

    def update_speed(self):
        """Adjust the snake's movement speed based on the score."""
        base_interval = 150  # Starting speed
        minimum_interval = 50
        # Decrease interval by 5ms evrey 10 points scored
        interval_decrease = (self.score // 10) * 5
        new_interval = max(base_interval - interval_decrease, minimum_interval)
        return new_interval

    def check_collision(self):
        head = self.snake_body[0]
        # Collision with self
        for segment in self.snake_body[1:]:
            if head == segment: return True
        # Collision with walls
        for wall in self.walls:
            if head == wall: return True
        # Collision with boundaries (if applicable)
        if (head.x < 0 or head.x >= self.screen_width or
            head.y < 0 or head.y >= self.screen_height): return True
        return False

    def check_food_collision(self):
        if self.snake_body[0].distance_to(self.food_position) < self.snake_radius + self.food_radius:
            self.score += 1
            self.growing = True
            new_interval = self.update_speed()
            pygame.time.set_timer(pygame.USEREVENT + 1, new_interval)
            return True
        return False

    def handle_input(self):
        keys = pygame.key.get_pressed()
        new_direction = self.direction
        if keys[pygame.K_w]:
            new_direction = Vector2(0, -1)  # Up
        elif keys[pygame.K_s]:
            new_direction = Vector2(0, 1)   # Down
        elif keys[pygame.K_a]:
            new_direction = Vector2(-1, 0)  # Left
        elif keys[pygame.K_d]:
            new_direction = Vector2(1, 0)   # Right

        # Prevent reversing direction
        if (new_direction + self.direction) != Vector2(0, 0):
            self.direction = new_direction
    
    def draw_grid(self):
        """Draw grid lines on the game window"""
        # Set the color for the grid lines 
        grid_color = (40, 40, 40) # Dark gray color 
        for x in range(0, self.screen_width, self.grid_size):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, self.grid_size):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.screen_width, y))

    def create_grid(self):
        """Helper for A* algorithm"""
        grid_width = self.screen_width // self.grid_size
        grid_height = self.screen_height // self.grid_size
        grid = [[0 for _ in range(grid_height)] for _ in range(grid_width)]

        # mark walls as obstacles
        for wall in self.walls:
            x = int(wall.x // self.grid_size)
            y = int(wall.y // self.grid_size)
            grid[x][y] = 1 # 1 = obstacle 

        # mark the snakes body as obstacles 
        for segment in self.snake_body:
            x = int(segment.x // self.grid_size)
            y = int(segment.y // self.grid_size)
            grid[x][y] = 1

        return grid 

    def get_gradient_colors(self, num_segments, time_step):
        """Generate a list of gradient colors for the snake's body"""
        colors = []
        for i in range(num_segments):
            # Calculate a hue value that changes over time and position 
            hue = (270 + (i * 10 + time_step)) % 360 # speed multiplier
            saturation = 1.0
            brightness = 1.0
            # Normalize hue to [0, 1]
            hue_norm = hue / 360 
            # Convert HSV to RGB
            color = self.hsv_to_rgb(hue_norm, saturation, brightness)
            colors.append(color)
        return colors

    def hsv_to_rgb(self, h, s, v):
        """Convert HSV color to RGB"""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)

    def draw_snake(self):

        num_segments = len(self.snake_body)
        time_step = pygame.time.get_ticks() / 100 # adjust to control speed
        colors = self.get_gradient_colors(num_segments, time_step)

        for idx, segment in enumerate(self.snake_body):
            pos = (
                int(segment.x - self.grid_size // 2), 
                int(segment.y - self.grid_size // 2)
            )
            # get the color for this segment 
            color = colors[idx]
            pygame.draw.rect(self.screen, color, 
                             pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))
        
    def spawn_food(self):
        """Randomly spawn food aligned to the grid"""
        max_x = (self.screen_width - self.grid_size) // self.grid_size 
        max_y = (self.screen_height - self.grid_size) // self.grid_size 
        while True:
            x = rnd.randint(0, max_x) * self.grid_size + self.grid_size // 2 
            y = rnd.randint(0, max_y) * self.grid_size + self.grid_size // 2 
            position = Vector2(x, y)
            if all(segment.distance_to(position) > (self.snake_radius + self.food_radius) for segment in self.snake_body):
                return position

    def draw_food(self):
        pos = (int(self.food_position.x - self.grid_size // 2), 
               int(self.food_position.y - self.grid_size // 2))
        pygame.draw.rect(self.screen, (255, 0, 0), 
                         pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))

    def spawn_walls(self):
        """Spawn a wall at a random position not occupied by the snake or food"""
        if len(self.walls) >= self.max_walls:
            return
        max_x = (self.screen_width - self.grid_size) // self.grid_size
        max_y = (self.screen_height - self.grid_size) // self.grid_size
        while True:
            x = rnd.randint(0, max_x) * self.grid_size + self.grid_size // 2
            y = rnd.randint(0, max_y) * self.grid_size + self.grid_size // 2
            position = Vector2(x, y)

            # Ensure the wall doesn't spawn on the snake or food
            occupied = any(segment == position for segment in self.snake_body)
            occupied = occupied or position == self.food_position
            occupied = occupied or any(wall == position for wall in self.walls)
            if not occupied:
                self.walls.append(position)
                break

    def draw_walls(self):
        wall_color = (128, 128, 128)  # Gray color for walls
        for wall in self.walls:
            pos = (int(wall.x - self.grid_size // 2), int(wall.y - self.grid_size // 2))
            pygame.draw.rect(self.screen, wall_color, pygame.Rect(pos[0], pos[1], self.grid_size, self.grid_size))

    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def draw_game_over(self):
        game_over_text = self.font.render("Game Over! Press 'R' to Restart", True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(game_over_text, text_rect)

    def draw_ai_mode(self):
        mode_text = "AI Mode" if self.ai_mode else "Manual Mode"
        mode_surface = self.font.render(mode_text, True, (255, 255, 0))
        self.screen.blit(mode_surface, (10, 50))

    def run(self):
        #reset_game()
        running = True
        while running:
            current_time = pygame.time.get_ticks()
            self.screen.fill((0, 0, 0))  # Clear screen with black
            self.draw_grid()
            self.draw_walls()

            if current_time - self.last_wall_spawn_time > self.wall_spawn_interval:
                self.spawn_walls()
                self.last_wall_spawn_time = current_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == (pygame.USEREVENT + 1) and not self.game_over:
                    if self.ai_mode:
                        action = self.get_ai_action()
                        # Use the step fn to process the action 
                        obseravtion, reward, done, _ = self.step(action)
                        if self.check_collision():
                            self.game_over = True 
                        if self.check_food_collision():
                            self.food_position = self.spawn_food()
                    else:
                        self.handle_input()  
                        # Handle user input
                        self.move_snake()
                        if self.check_collision():
                            self.game_over = True
                        if self.check_food_collision():
                            self.food_position = self.spawn_food()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()
                    # toggle ai mode with 'i' key 
                    if event.key == pygame.K_i:
                        self.ai_mode = not self.ai_mode 
                        print(f"AI Mode {'Enabled' if self.ai_mode else 'Disabled'}")

            if not self.game_over:
                self.handle_input()
                self.draw_snake()
                self.draw_food()
                self.draw_score()
                self.draw_ai_mode()
            else:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(100)  # Control the frame rate

        pygame.quit()

    def visTraining(self):
        """Visualise the AI training"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((0,0,0))
        self.draw_grid()
        self.draw_walls()
        self.draw_snake()
        self.draw_food()
        self.draw_score()

        pygame.display.flip()
        self.clock.tick(100)

if __name__ == '__main__':

    # to run the file/game in ai mode run 
    # $ python game.py ai

    import sys 

    ai_mode = False 
    if len(sys.argv) > 1:
        if sys.argv[1] == 'ai':
            ai_mode = True

    game = Game(ai_mode=ai_mode)
    game.run()

