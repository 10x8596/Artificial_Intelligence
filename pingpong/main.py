import pygame
from pong import Game
import neat
import os
import pickle

class PingPong:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def test_ai(self, genome, config):
        """
        Test the AI against a human player by passing a NEAT neural network
        """
        
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome, config)
        
        runGame = True
        clock = pygame.time.Clock()

        while runGame:
            clock.tick(120) # set 120 fps
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            
            # Player controls
            #keys = pygame.key.get_pressed()
            #if keys[pygame.K_w]:
            #    self.game.move_paddle(left=True, up=True)
            #if keys[pygame.K_s]:
            #    self.game.move_paddle(left=True, up=False)

            output = net.activate((self.right_paddle.y, self.ball.y, 
                                     abs(self.right_paddle.x - self.ball.x)))
            decision = output.index(max(output))
            output2 = net2.activate((self.left_paddle.y, self.ball.y,
                                     abs(self.left_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            if decision2 == 0:
                pass 
            elif decision2 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            game_info = self.game.loop()
            #print(game_info.left_score, game_info.right_score)
            self.game.draw()
            pygame.display.update()

        pygame.quit() 

    def train_ai(self, genome1, genome2, config):
        """
        Train the AI by passing two NEAT neural networks and the NEAt config object.
        These AI's will play against eachother to determine their fitness.
        """
        net1 = neat.nn.FeedForwardNetwork.create(genome1, config)
        net2 = neat.nn.FeedForwardNetwork.create(genome2, config)
        runGame = True
        while runGame:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    quit()
            
            output1 = net1.activate((self.left_paddle.y, self.ball.y, 
                                     abs(self.left_paddle.x - self.ball.x)))
            # 0: stay still, 1: move up, 2: move down
            decision1 = output1.index(max(output1))
            
            if decision1 == 0:
                pass
            elif decision1 == 1:
                self.game.move_paddle(left=True, up=True)
            else:
                self.game.move_paddle(left=True, up=False)

            output2 = net2.activate((self.right_paddle.y, self.ball.y, 
                                     abs(self.right_paddle.x - self.ball.x)))
            decision2 = output2.index(max(output2))

            if decision2 == 0:
                pass
            elif decision2 == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False) 

            print(output1, output2)

            game_info = self.game.loop()
            self.game.draw(False, True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                self.calculate_fitness(genome1, genome2, game_info)
                break

    def calculate_fitness(self, genome1, genome2, game_info):
       genome1.fitness += game_info.left_hits
       genome2.fitness += game_info.right_hits

def eval_genomes(genomes, config):
    width, height = 1500, 800
    window = pygame.display.set_mode((width, height))
    
    # Run each genome against every other genome one time
    for i, (genome_id1, genome1) in enumerate(genomes):
        if i == len(genomes) - 1:
            break
        genome1.fitness = 0
        # ensure same genomes doesn't play against each other multiple times
        for genome_id2, genome2 in genomes[i+1:]:
            genome2.fitness = 0 if genome2.fitness == None else genome2.fitness
            game = PingPong(window, width, height)
            game.train_ai(genome1, genome2, config)

def run_neat(config):
    # population = neat.Checkpointer.restore_checkpoint('./checkpoints/neat-checkpoint-')
    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    population.add_reporter(neat.Checkpointer(1))
    
    # evaluate genomes up to 50 times
    best_genome = population.run(eval_genomes, 100) # change to 1 when restoring checkpoint

    # Save the neural network using pickle
    with open("bestgenome.pickle", "wb") as f:
        pickle.dump(best_genome, f)

def test_ai(config):
    width, height = 1500, 800
    window = pygame.display.set_mode((width, height))

    with open("bestgenome.pickle", "rb") as f:
        best_genome = pickle.load(f)

    game = PingPong(window, width, height)
    game.test_ai(best_genome, config)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    
    # load configuration file
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    # train AI
    # run_neat(config)
    # Load AI from pickle to test (play against it)
    test_ai(config)
