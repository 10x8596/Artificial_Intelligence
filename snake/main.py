import pygame
import random
from pygame.math import Vector2
from scipy.interpolate import splprep, splev
import numpy as np

pygame.init()

# Set up screen
screen_width, screen_height = 1200, 1200
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 48)

# Snake properties
snake_radius = 10
initial_snake_length = 5
# match snake speed with segment spacing
snake_speed = snake_radius * 3.5
direction = Vector2(1, 0)  # Start moving to the right
growing = False

# Food properties
food_radius = 10
food_position = None

# Game state
score = 0
game_over = False

def reset_game():
    global snake_body, direction, food_position, score, game_over

    start_pos = Vector2(360, 200)
    # make sure to space out the segments to prevent immediate self-collision
    snake_body = [start_pos - Vector2(i * snake_speed, 0) for i in range(initial_snake_length)]  
    direction = Vector2(1, 0)  # Reset direction to the right
    food_position = spawn_food()  # Reset food position
    score = 0  # Reset score
    game_over = False 
    growing = False

    # start movement timer with base interval 
    pygame.time.set_timer(MOVE_EVENT, 150) # Starting speed 

def spawn_food():
    """Randomly spawn food within the screen bounds."""
    while True:
        position = Vector2(
            random.randint(food_radius, screen_width - food_radius),
            random.randint(food_radius, screen_height - food_radius)
        )
        # Check if the position is not on the snake
        if all(segment.distance_to(position) > (snake_radius + food_radius) for segment in snake_body):
            return position 

def move_snake():
    global direction, game_over, growing

    # Get the new head position by moving the snake in the current direction
    next_position = snake_body[0] + direction * snake_speed
    
    # Boundary collision detection
    if (next_position.x < 0 or next_position.x > screen_width or \
        next_position.y < 0 or next_position.y > screen_width):
        game_over = True 
        return

    # Insert the new head 
    snake_body.insert(0, next_position)

    # Remove the last segment if not growing
    if not growing:
        snake_body.pop()
    else:
        growing = False

def update_speed():
    """Adjust the snake's movement speed based on the score."""
    # Base interval in milliseconds
    base_interval = 150  # Starting speed
    # Minimum interval to cap the speed increase
    minimum_interval = 50
    # Decrease interval by 5 ms every 10 points scored
    interval_decrease = (score // 10) * 5
    # Calculate new interval
    new_interval = max(base_interval - interval_decrease, minimum_interval)
    return new_interval

def check_collision():
    # Check if the snake's head collides with its body
    head = snake_body[0]
    for segment in snake_body[1:]:
        if head.distance_to(segment) < snake_radius:  # Collided with itself
            return True
    return False

def check_food_collision():
    global score, growing
    # Check if the snake's head collides with the food
    if snake_body[0].distance_to(food_position) < snake_radius + food_radius:
        score += 1
        growing = True
        # Update the movement interval 
        new_interval = update_speed()
        pygame.time.set_timer(MOVE_EVENT, new_interval)
        return True
    return False

def grow_snake():
    # Add a new segment at the tail to make the snake longer
    tail = snake_body[-1]
    snake_body.append(tail)

def handle_input():
    global direction

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        new_direction = Vector2(0, -1) # up
    elif keys[pygame.K_s]:
        new_direction = Vector2(0, 1)  # down
    elif keys[pygame.K_a]:
        new_direction = Vector2(-1, 0) # left
    elif keys[pygame.K_d]:
        new_direction = Vector2(1, 0)  # right
    else: 
        new_direction = direction

    # Prevent reversing direction
    if (new_direction + direction) != Vector2(0, 0):
        direction = new_direction 

def draw_snake():
    head_color = (255, 255, 255) # whiite
    body_color = (0, 255, 0) # greeen
    
    # Extract x and y coordinates from snake body
    x = [segment.x for segment in snake_body]
    y = [segment.y for segment in snake_body]

    # Reverse the lists to start drawing from head 
    x = x[::-1]
    y = y[::-1]

    if len(snake_body) > 3:
        # Create spline representation
        tck, u = splprep([x, y], s=0, k=3)
        # Generate new points for smooth curve
        unew = np.linspace(0, 1.0, num=200)
        out = splev(unew, tck)
        spline_points = list(zip(out[0], out[1])) #list(zip(map(int, out[0]), map(int, out[1])))

        # draw the snake body by drawing circles along the spline 
        body_thickness = snake_radius * 1 
        for point in spline_points:
            pos = (int(point[0]), int(point[1]))
            pygame.draw.circle(screen, body_color, pos, body_thickness)
    else:
        # If not enough points for spline, draw straight lines
        snake_points = [(int(segment.x), int(segment.y)) for segment in snake_body]
        for pos in snake_points:
            pygame.draw.circle(screen, body_color, pos, snake_radius * 2)

    # Draw the head 
    head_pos = (int(snake_body[0].x), int(snake_body[0].y))
    pygame.draw.circle(screen, head_color, head_pos, body_thickness + 2)

def draw_food():
    pygame.draw.circle(screen, (255, 0, 0), (int(food_position.x), int(food_position.y)), food_radius)

def draw_score():
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

def draw_game_over():
    game_over_text = font.render("Game Over! Press 'R' to Restart", True, (255, 0, 0))
    text_rect = game_over_text.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(game_over_text, text_rect)

# Start the game
MOVE_EVENT = pygame.USEREVENT + 1
reset_game()

# Game loop
running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == MOVE_EVENT and not game_over:
            move_snake()
            # Check for self-collision
            if check_collision():
                game_over = True
            # Check for food collision
            if check_food_collision():
                # grow_snake()
                food_position = spawn_food()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r and game_over:
                reset_game()

    if not game_over:
        handle_input()
        # Draw everything
        draw_snake()
        draw_food()
        draw_score()
    else:
        draw_game_over()

    pygame.display.flip()
    clock.tick(60)  # Control the speed of the game

pygame.quit()

