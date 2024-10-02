import pygame
import random
import sys
import os
from constants import *
import pickle
import neat
from flappybird import Ball 
from flappybird import Pipe

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

    # setup neural networks for genomes 
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        balls.append(Ball()) # Ball(230, 350)
        g.fitness = 0
        # initial fitness is 0
        ge.append(g)
    
    # Debugging
    print(f"Balls: {len(balls)}, Pipes: {len(pipes)}") 

    # Game loop
    while game_active:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Find the closest pipe
        if len(balls) > 0 and len(pipes) > 0:
            pipe_idx = 0
            # find the closest pipe by checking the x-position
            for i, pipe in enumerate(pipes):
                # pipe ahead of the ball
                if pipe.x + pipe.width > balls[0].x:
                    pipe_idx = i 
                    break
        else:
            game_active = False
            break
        
        # move and update balls 
        removal_indices = []
        for x, ball in enumerate(balls):
            ball.apply_gravity()

            top_rect, bottom_rect = pipes[pipe_idx].get_rects()

            # neural network output 
            ge[x].fitness += 0.1
            output = nets[x].activate((ball.y, abs(ball.y - top_rect.y), abs(ball.y - bottom_rect.y)))
            if output[0] > 0.5:
                ball.jump()

            # check for collision 
            if ball.collision().colliderect(top_rect) or ball.collision().colliderect(bottom_rect) or ball.y >= WINDOW_HEIGHT or ball.y <= 0:
                ge[x].fitness -= 1 
                removal_indices.append(x)

        # remove collided balls 
        for i in sorted(removal_indices, reverse=True):
            balls.pop(i)
            nets.pop(i)
            ge.pop(i)

        # pipe movement and adding new pipes 
        if pipes[-1].x < WINDOW_WIDTH // 2:
            pipes.append(Pipe(pipe_speed))
        for pipe in pipes:
            pipe.move()
        pipes = [pipe for pipe in pipes if pipe.x > -PIPE_WIDTH]

        # Increment score 
        for pipe in pipes:
            for x, ball in enumerate(balls):
                if pipe.x + pipe.width < ball.x and not pipe.passed:
                    pipe.passed = True
                    score += 1
                    # increase fitness score for ball's that passed 
                    ge[x].fitness += 5 
                    

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

# Training 
def run_neat(config):
    # population = neat.Checkpointer.restore_checkpoint('./checkpoints/neat-checkpoint-')
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    # evaluate genomes up to 100 times
    best_genome = population.run(eval_genomes, 300)

    # Save the neural network using pickle
    with open("bestgenome.pickle", "wb") as f:
        pickle.dump(best_genome, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    # train AI
    run_neat(config)

