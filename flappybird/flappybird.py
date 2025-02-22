import pygame
import random
import sys
from constants import *
import pickle
import os
import neat

class Ball:

    def __init__(self):
        self.x = 350
        self.y = WINDOW_HEIGHT // 2
        self.radius = 20 
        self.velocity = 0

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (self.x, int(self.y)), self.radius)

    def jump(self, strength=1):
        self.velocity = JUMP_STRENGTH * strength

    def apply_gravity(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def collision(self):
        return pygame.Rect(self.x-self.radius, self.y-self.radius, self.radius * 2, self.radius * 2)

class Pipe:

    def __init__(self, speed, previous_pipe=None):
        self.x = WINDOW_WIDTH
        self.width = PIPE_WIDTH
        self.speed = speed
        self.passed = False
        self.gap_size = PIPE_GAP

        if previous_pipe:
            # Constrain the vertical position of the new gap based on the previous pipe
            previous_gap_y = (previous_pipe.top_height + previous_pipe.bottom_height) / 2
            new_gap_y = previous_gap_y + random.randint(-MAX_GAP_CHANGE, MAX_GAP_CHANGE)
            new_gap_y = max(self.gap_size, min(WINDOW_HEIGHT - self.gap_size, new_gap_y))
        else:
            # If there's no previous pipe, generate the gap randomly within bounds
            new_gap_y = random.randint(self.gap_size, WINDOW_HEIGHT - self.gap_size)

        # Calculate top and bottom pipe heights
        self.top_height = new_gap_y - self.gap_size // 2
        self.bottom_height = WINDOW_HEIGHT - (new_gap_y + self.gap_size // 2)

    def move(self):
        self.x -= self.speed

    def draw(self, surface):
        # top pipe 
        pygame.draw.rect(surface, WHITE, (self.x, 0, self.width, self.top_height))
        # bottom pipe (with gap)
        pygame.draw.rect(surface, WHITE, (self.x, WINDOW_HEIGHT - self.bottom_height, self.width, self.bottom_height))

    def get_rects(self):
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        bottom_pipe_rect = pygame.Rect(self.x, WINDOW_HEIGHT - self.bottom_height, self.width, self.bottom_height)
        return top_pipe_rect, bottom_pipe_rect

def test_ai(config_path):

    # Load the best genome from pickle 
    with open("bestgenome.pickle", "rb") as f:
        best_genome = pickle.load(f)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # Create a neural network from the best genome
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    
    return net, config 

class Game:
    def __init__(self, ai_net=None):
        pygame.init()

        window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("AI plays Flappy Bird")

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()

        self.ball = Ball()
        self.pipes = [Pipe(PIPE_SPEED)]
        self.score = 0
        self.high_score = self.load_high_score()
        self.game_active = True
        self.pipe_speed = PIPE_SPEED
        self.ai_mode = False

        # AI neural network 
        self.ai_net = ai_net

    def save_high_score(self):
        with open("high_score.txt", "w") as f:
            f.write(str(self.high_score))

    def load_high_score(self):
        if os.path.exists("high_score.txt"):
            with open("high_score.txt", "r") as f:
                return int(f.read().strip())
        return 0
    
    def restart_game(self):
        self.ball = Ball()
        self.pipes = [Pipe(PIPE_SPEED)]
        self.score = 0
        self.game_active = True
        self.pipe_speed = PIPE_SPEED

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and self.game_active and not self.ai_mode:
                    self.ball.jump()
                if event.key == pygame.K_r and not self.game_active:
                    # restart
                    self.restart_game()
                if event.key == pygame.K_q:
                    # quit 
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_a:
                    # toggle ai mode 
                    self.ai_mode = not self.ai_mode

    def update(self):
        if self.game_active:

            self.ball.apply_gravity()
#---------------------------------------------------------------------------------------
            
            # AI control 
            if self.ai_mode and self.ai_net:
                # Get the closest pipe 
                if len(self.pipes) > 0:
                    pipe = self.pipes[0]
                    if pipe.x + pipe.width < self.ball.x and len(self.pipes) > 1:
                        pipe = self.pipes[1]  

                # AI decision making based on nn output
                # Get next pipe
                top_rect, bottom_rect = pipe.get_rects()

                # Calculate the position of the next pipe's gap 
                center_of_gap = ((top_rect.y + bottom_rect.y) / 2) / WINDOW_HEIGHT
                distance_to_next_gap = abs(self.ball.y - center_of_gap) / WINDOW_HEIGHT

                # Normalize input values
                ball_y_norm = self.ball.y / WINDOW_HEIGHT
                top_pipe_norm = abs(self.ball.y - top_rect.y) / WINDOW_HEIGHT
                bottom_pipe_norm = abs(self.ball.y - bottom_rect.y) / WINDOW_HEIGHT
                horizontal_dist = abs(self.ball.x - pipe.x) / WINDOW_WIDTH
                vertical_dist = abs(self.ball.y - center_of_gap) / WINDOW_HEIGHT 
                gap_size_norm = PIPE_GAP / WINDOW_HEIGHT
                if self.ball.velocity != 0:
                    time_to_top_collision = abs(self.ball.y - top_rect.y) / abs(self.ball.velocity)
                    time_to_bot_collision = abs(self.ball.y - bottom_rect.y) / abs(self.ball.velocity)
                else:
                    time_to_top_collision = float('inf')
                    time_to_bot_collision = float('inf')
                safety_margin = min(abs(self.ball.y - top_rect.y), abs(self.ball.y - bottom_rect.y)) / WINDOW_HEIGHT
                output = self.ai_net.activate((ball_y_norm, top_pipe_norm, bottom_pipe_norm, horizontal_dist, gap_size_norm, time_to_top_collision, time_to_bot_collision, safety_margin, distance_to_next_gap, vertical_dist))
                if output[0] > 0.5:
                    jump_strength = min(max(output[0], 0.1), 1)
                    #if distance_to_center > 0.5:
                    #    jump_strength = min(max(output[0], 0.1), 2.0)
                    self.ball.jump(jump_strength)
 

#---------------------------------------------------------------------------------------

            # pipe movement and adding new pipes 
            if self.pipes[-1].x < WINDOW_WIDTH // 2:
                self.pipes.append(Pipe(self.pipe_speed, self.pipes[-1]))
            self.pipes = [pipe for pipe in self.pipes if pipe.x > -PIPE_WIDTH]

            # update pipes and check for collision
            for pipe in self.pipes:
                pipe.move()
                top_rect, bottom_rect = pipe.get_rects()
                if self.ball.collision().colliderect(top_rect) or self.ball.collision().colliderect(bottom_rect):
                    # ball hit pipe
                    self.game_active = False

            # Check if ball hits the ground or flies off screen 
            if self.ball.y >= WINDOW_HEIGHT or self.ball.y <= 0:
                self.game_active = False

            # Increment score 
            for pipe in self.pipes:
                if pipe.x + pipe.width < self.ball.x and not pipe.passed:
                    pipe.passed = True
                    self.score += 1

            # Increase game speed every 30 points
            if self.score > 0 and self.score % SPEED_INCREASE_THRESHOLD == 0:
                self.pipe_speed = min(self.pipe_speed + 1, MAX_PIPE_SPEED)


    def draw(self):
        # Draw game 
        self.screen.fill(BLACK)
        self.ball.draw(self.screen)

        for pipe in self.pipes:
            pipe.draw(self.screen)

        font = pygame.font.SysFont(None, 48) 

        # display score 
        score_surface = font.render(f'Score: {self.score}', True, 'red')
        self.screen.blit(score_surface, (10, 10)) # top left corner

        # display highest score 
        high_score_surface = font.render(f'High score: {self.high_score}', True, 'pink')
        self.screen.blit(high_score_surface, (10, 50))

        # restart message
        if not self.game_active: 
            text_surface = font.render('Press R to restart', True, 'red')
            self.screen.blit(text_surface, (WINDOW_WIDTH // 4, WINDOW_HEIGHT // 2))

        # Display ai mode status
        mode_surface = font.render(f'AI Mode: {"On" if self.ai_mode else "Off"}', True, 'red')
        self.screen.blit(mode_surface, (WINDOW_WIDTH - 200, 10)) # top right corner

        pygame.display.flip()

    def run(self):
        # Main game loop
        while True:
            self.handle_events()
            self.update()
            self.draw()
            # Check for new high score 
            if not self.game_active and self.score > self.high_score:
                self.high_score = self.score 
                self.save_high_score()
            # Frame rate 
            self.clock.tick(FPS)

if __name__ == '__main__':
    
    # r to restart, q to quit, space to jump, a to switch on ai mode 

    # Load the AI from the best genome and the config file
    config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
    net, config = test_ai(config_path)

    # Pass the AI neural network to the game
    game = Game(ai_net=net) 
    game.run()
