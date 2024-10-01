import pygame
import random
import sys
from constants import *
import pickle

class Ball:

    def __init__(self):
        self.x = 100
        self.y = WINDOW_HEIGHT // 2
        self.radius = 20 
        self.velocity = 0

    def draw(self, surface):
        pygame.draw.circle(surface, WHITE, (self.x, int(self.y)), self.radius)

    def jump(self):
        self.velocity = FLAP_STRENGTH

    def apply_gravity(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def collision(self):
        return pygame.Rect(self.x-self.radius, self.y-self.radius, self.radius * 2, self.radius * 2)

class Pipe:

    def __init__(self, speed):
        self.x = WINDOW_WIDTH
        self.height = random.randint(100, 400)
        self.width = PIPE_WIDTH
        self.speed = speed

    def move(self):
        self.x -= self.speed

    def draw(self, surface):
        # top pipe 
        pygame.draw.rect(surface, WHITE, (self.x, 0, self.width, self.height))
        # bottom pipe (with gap)
        pygame.draw.rect(surface, WHITE, (self.x, self.height + PIPE_GAP, self.width, WINDOW_HEIGHT - self.height - PIPE_GAP))

    def get_rects(self):
        top_pipe_rect = pygame.Rect(self.x, 0, self.width, self.height)
        bottom_pipe_rect = pygame.Rect(self.x, self.height + PIPE_GAP, self.width, WINDOW_HEIGHT - self.height - PIPE_GAP)
        return top_pipe_rect, bottom_pipe_rect

def AI(ball, pipes):
    
    # Calculate the gradient from the ball to the gap.
    # If gradient is negative, need to fall
    # If gradient is positive, need to jump 
    # else maintain a gradient of 0 if at 0 
    pass


class Game:
    def __init__(self):
        pygame.init()

        window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("AI plays Flappy Bird")

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()

        self.ball = Ball()
        self.pipes = [Pipe(PIPE_SPEED)]
        self.score = 0
        self.game_active = True
        self.pipe_speed = PIPE_SPEED
        self.ai_mode = False
    
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

            # AI control 
            if self.ai_mode:
                if AI(self.ball, self.pipes):
                    self.ball.jump()

            # pipe movement and adding new pipes 
            if self.pipes[-1].x < WINDOW_WIDTH // 2:
                self.pipes.append(Pipe(self.pipe_speed))
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
                if pipe.x + pipe.width < self.ball.x and not hasattr(pipe, 'passed'):
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
            # Frame rate 
            self.clock.tick(FPS)

if __name__ == '__main__':
    game = Game()
    game.run()
