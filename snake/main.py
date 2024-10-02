import pygame
from pygame.math import Vector2

pygame.init()

# Set up screen
screen_width, screen_height = 600, 400
screen = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Snake properties
snake_radius = 10
snake_body = [Vector2(360, 200), Vector2(340, 200), Vector2(320, 200), Vector2(300, 200), Vector2(280, 200), Vector2(260, 200), Vector2(240, 200), Vector2(220, 200), Vector2(200, 200)]  # A longer snake
snake_speed = 5
direction = Vector2(1, 0)  # Start moving to the right

# List of edge movement directions (right, down, left, up)
edge_directions = [
    Vector2(1, 0),  # Right
    Vector2(0, 1),  # Down
    Vector2(-1, 0), # Left
    Vector2(0, -1)  # Up
]

current_direction_index = 0  # Start with the right direction
direction = edge_directions[current_direction_index]

def move_snake():
    global direction, current_direction_index

    # Get the new head position by moving the snake in the current direction
    next_position = snake_body[0] + direction * snake_speed

    # Change direction if the snake reaches the edge of the screen
    if next_position.x >= screen_width - snake_radius:  # Reached right edge
        current_direction_index = 1  # Move down
    elif next_position.y >= screen_height - snake_radius:  # Reached bottom edge
        current_direction_index = 2  # Move left
    elif next_position.x <= snake_radius:  # Reached left edge
        current_direction_index = 3  # Move up
    elif next_position.y <= snake_radius:  # Reached top edge
        current_direction_index = 0  # Move right

    direction = edge_directions[current_direction_index]  # Update direction

    # Insert the new head and pop the last segment to move the snake
    snake_body.insert(0, next_position)
    snake_body.pop()

# Game loop
running = True
while running:
    screen.fill((0, 0, 0))  # Clear screen with black

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update snake movement
    move_snake()

    # Draw snake
    for segment in snake_body:
        pygame.draw.circle(screen, (0, 255, 0), (int(segment.x), int(segment.y)), snake_radius)

    pygame.display.flip()
    clock.tick(30)  # Control the speed of the game (30 FPS)

pygame.quit()

