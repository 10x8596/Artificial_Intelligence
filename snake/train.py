import pygame
import random
from pygame.math import Vector2
from scipy.interpolate import splprep, splev
import numpy as np
import gym 
from gym import spaces

class Game(gym.Env):
    def __init__(self):
        super(Game, self).__init__()
        # Initialize Pygame
        pygame.init()

        # Set up screen
        self.screen_width, self.screen_height = 800, 800
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 48)

        # Snake properties
        self.snake_radius = 10
        self.initial_snake_length = 5
        self.snake_speed = self.snake_radius * 2.5
        self.direction = Vector2(1, 0)  # Start moving to the right
        self.growing = False

        # Food properties
        self.food_radius = 10
        self.food_position = None

        # Game state
        self.score = 0
        self.game_over = False

        # Define action space (4 directions: up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Define observation space (snake head (x,y), direction (x,y), food (x,y))
        self.observation_space = spaces.Box(low=0, high=800, shape=(6,), dtype=np.float32)

        # Start the game
        self.reset_game()

    def reset_game(self):
        """Reset game to initial state"""
        self.start_pos = Vector2(360, 200)
        self.snake_body = [self.start_pos - Vector2(i * self.snake_speed, 0) for i in range(self.initial_snake_length)]
        self.direction = Vector2(1, 0)
        self.food_position = self.spawn_food()
        self.score = 0
        self.game_over = False
        self.growing = False

        pygame.time.set_timer(self.MOVE_EVENT, 150)  # Starting speed

    def reset(self):
        """Reset envrionment to start a new episode"""
        self.reset_game()
        return self.get_observation()

    def spawn_food(self):
        """Randomly spawn food within the screen bounds."""
        while True:
            position = Vector2(
                random.randint(self.food_radius, self.screen_width - self.food_radius),
                random.randint(self.food_radius, self.screen_height - self.food_radius)
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

    def step(self, action):
        """Execute one step in the environment"""
        if action == 0: # up 
            self.direction = Vector2(0, -1)
        elif action == 1: # Down 
            self.direction = Vector2(0, 1)
        elif action == 2: # Left 
            self.direction = Vector2(-1, 0)
        elif action == 3: # Right 
            self.direction = Vector2(1, 0)

        self.move_snake()
        done = False 
        reward = 0 

        # check for collisions and reward accordingly
        if self.check_food_collision():
            reward = 1 
        if self.check_collision():
            reward = -1.5
            done = True 

        # Return the current state, reward, done flag, and info dict 
        observation = self.get_observation()
        return observation, reward, done, {}

    def get_observation(self):
        """Get the current state of the game"""
        head = self.snake_body[0]
        food = self.food_position
        return np.array([head.x, head.y, self.direction.x, self.direction.y, food.x, food.y], dtype=np.float32)

    def update_speed(self):
        """Adjust the snake's movement speed based on the score."""
        base_interval = 150  # Starting speed
        minimum_interval = 50
        interval_decrease = (self.score // 10) * 5
        new_interval = max(base_interval - interval_decrease, minimum_interval)
        return new_interval

    def check_collision(self):
        head = self.snake_body[0]
        for segment in self.snake_body[1:]:
            if head.distance_to(segment) < self.snake_radius:
                return True
        return False

    def check_food_collision(self):
        if self.snake_body[0].distance_to(self.food_position) < self.snake_radius + self.food_radius:
            self.score += 1
            self.growing = True
            new_interval = self.update_speed()
            pygame.time.set_timer(self.MOVE_EVENT, new_interval)
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

    def draw_snake(self):
        head_color = (255, 255, 255)  # White
        body_color = (0, 255, 0)      # Green

        x = [segment.x for segment in self.snake_body][::-1]
        y = [segment.y for segment in self.snake_body][::-1]

        if len(self.snake_body) > 3:
            tck, _ = splprep([x, y], s=0, k=3)
            unew = np.linspace(0, 1.0, num=200)
            out = splev(unew, tck)
            spline_points = list(zip(out[0], out[1]))

            body_thickness = self.snake_radius
            for point in spline_points:
                pos = (int(point[0]), int(point[1]))
                pygame.draw.circle(self.screen, body_color, pos, body_thickness)
        else:
            for segment in self.snake_body:
                pos = (int(segment.x), int(segment.y))
                pygame.draw.circle(self.screen, body_color, pos, self.snake_radius * 2)

        # Draw the head
        head_pos = (int(self.snake_body[0].x), int(self.snake_body[0].y))
        pygame.draw.circle(self.screen, head_color, head_pos, self.snake_radius + 2)

    def draw_food(self):
        pos = (int(self.food_position.x), int(self.food_position.y))
        pygame.draw.circle(self.screen, (255, 0, 0), pos, self.food_radius)

    def draw_score(self):
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def draw_game_over(self):
        game_over_text = self.font.render("Game Over! Press 'R' to Restart", True, (255, 0, 0))
        text_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
        self.screen.blit(game_over_text, text_rect)

    def run(self):
        running = True
        while running:
            self.screen.fill((0, 0, 0))  # Clear screen with black

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == self.MOVE_EVENT and not self.game_over:
                    self.move_snake()
                    if self.check_collision():
                        self.game_over = True
                    if self.check_food_collision():
                        self.food_position = self.spawn_food()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r and self.game_over:
                        self.reset_game()

            if not self.game_over:
                self.handle_input()
                self.draw_snake()
                self.draw_food()
                self.draw_score()
            else:
                self.draw_game_over()

            pygame.display.flip()
            self.clock.tick(120)  # Control the frame rate

        pygame.quit()

if __name__ == '__main__':
    game = Game()
    game.run()

