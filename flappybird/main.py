import pygame
import random
import sys
import os
from constants import *
import pickle
import neat
from flappybird import Ball, Pipe

# Inputs = ball.y, top pipe, bottom pipe (3 neurons)
# Output = jump or fall (1 neuron)
# Activation func: tanh (if > 0.5 ? jump : fall)
# fitness function: game() from flappybird.py (bird with greatest score will make it to next gen)
# max generations = 30

pygame.init()

window_size = (WINDOW_WIDTH, WINDOW_HEIGHT)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("AI plays Flappy Bird")

# Clock for controlling frame rate
clock = pygame.time.Clock()

# Create a func called game to init the game as well as test and train ai 
def eval_genomes(genomes, config):
    
    nets = []
    ge = []
    balls = []
    pipes = [Pipe(PIPE_SPEED)]
    score = 0
    game_active = True
    pipe_speed = PIPE_SPEED
    ai_mode = False
    model = None

    # setup neural networks for genomes 
    for g in genomes:
        net = neat.nn.FeedForwardNetwork(g, config)
        nets.append(net)
        balls.append(Ball()) # Ball(230, 350)
        g.fitness = 0
        # initial fitness is 0
        ge.append(g)

    # Game loop
    while game_active:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_active = False
        
        for ball in balls:
            ball.apply_gravity()

        # pipe movement and adding new pipes 
        if pipes[-1].x < WINDOW_WIDTH // 2:
            pipes.append(Pipe(pipe_speed))
        pipes = [pipe for pipe in pipes if pipe.x > -PIPE_WIDTH]

        # update pipes and check for collision
        for pipe in pipes:
            pipe.move()
            top_rect, bottom_rect = pipe.get_rects()
            for x, ball in enumerate(balls):
                # get rid of birds the hit pipe
                if ball.collision().colliderect(top_rect) or ball.collision().colliderect(bottom_rect):
                    # decrease fitness score for that bird
                    ge[x].fitness -= 1
                    
        # Check if ball hits the ground or flies off screen
        for ball in balls:
            if ball.y >= WINDOW_HEIGHT or ball.y <= 0:
                pass

        # Increment score 
        for pipe in pipes:
            for ball in balls:
                if pipe.x + pipe.width < ball.x and not hasattr(pipe, 'passed'):
                    pipe.passed = True
                    score += 1

        # Increase game speed every 30 points
        if score > 0 and score % SPEED_INCREASE_THRESHOLD == 0:
            pipe_speed = min(pipe_speed + 1, MAX_PIPE_SPEED)

        # Draw game 
        screen.fill(BLACK)
        for ball in balls:
            ball.draw(screen)

        for pipe in pipes:
            pipe.draw(screen)

        font = pygame.font.SysFont(None, 48) 

        # display score 
        score_surface = font.render(f'Score: {score}', True, 'red')
        screen.blit(score_surface, (10, 10)) # top left corner

        pygame.display.flip()

        # Frame rate 
        clock.tick(FPS) 

    def test_ai(self, genome):
        pass 

    def train_ai(self, genomes):
        pass 

    def calculate_fitness():
        pass 

def run_neat(config):
    # population = neat.Checkpointer.restore_checkpoint('./checkpoints/neat-checkpoint-')
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))
    
    # evaluate genomes up to 100 times
    best_genome = population.run(eval_genomes, 100) # change to 1 when restoring checkpoint

    # Save the neural network using pickle
    with open("bestgenome.pickle", "wb") as f:
        pickle.dump(best_genome, f)

def test_ai(config):
    with open("bestgenome.pickle", "rb") as f:
        best_genome = pickle.load(f)
    flappy_bird = FlappyBird()
    flappy_bird.test_ai(best_genome)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # train AI
    #run_neat(config)
    # Load AI from pickle to test
    #test_ai(config)
    eval_genomes()

