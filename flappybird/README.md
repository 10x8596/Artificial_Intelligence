# AI-Flappy-Bird

This project implements a Flappy Bird game where the bird is controlled by a Neuroevolution of Augmenting Topologies (NEAT) AI.  The AI learns to play the game through a genetic algorithm, evolving neural networks that determine when the bird should jump.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Usage](#usage)
- [Files](#files)
- [AI Training](#ai-training)
- [Controls](#controls)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)

## Introduction

This project combines the classic Flappy Bird gameplay with the power of NEAT.  The AI learns to play the game by evolving a population of neural networks.  Each network controls the bird in a separate game instance, and their performance (score) determines their fitness.  The fittest networks are then used to create the next generation, with mutations and crossovers introducing new variations. This process continues until a sufficiently skilled AI is evolved or a maximum number of generations is reached.

## Features

- **NEAT AI:**  Uses the NEAT algorithm to evolve neural networks for controlling the bird.
- **Genetic Algorithm:** Employs a genetic algorithm for training the AI.
- **High Score:** Tracks and saves the highest score achieved.
- **AI Toggle:** Allows switching between manual control and AI control during gameplay.
- **Speed Increase:**  Game speed increases as the score rises, making the challenge progressively harder.
- **Adaptive Gap Placement:** Pipe gap positions are dynamically adjusted to provide a more engaging experience.

## Usage

1. Run the game
   ```bash
   python flappybird.py
   ```
2. Train the AI (separately):
   ```bash
   python train.py
   ```
   This will generate a bestgenome.pickle file, which the main game uses to load the trained AI.

## Files

- flappybird.py: Contains the main game logic and rendering.
- train.py: Handles the NEAT AI training process.
- constants.py: Defines game constants (window size, colors, gravity, etc.).
- config.txt: Configuration file for the NEAT algorithm.
- bestgenome.pickle: Stores the best performing genome after training.
- high_score.txt: Stores the highest score achieved.

## AI Training

The neat_trainer.py script uses the NEAT library to train the AI.  It initializes a population of neural networks and evaluates their fitness based on their performance in the game.  The script saves the best performing genome to bestgenome.pickle, which can be loaded by flappybird.py to play the game with the trained AI.  The training script should be run separately before running the main game if you wish to train a new AI or use a different training configuration.

## Controls

    Space: Jump (manual control).
    R: Restart the game.
    Q: Quit the game.
    A: Toggle AI control (switches between manual and AI).

## Dependencies

    Python 3
    Pygame
    NEAT-python

## Future Improvements
- Visualizations: Add visualizations of the neural network and the training process.
- Improved Fitness Function: Experiment with different fitness functions to encourage more diverse and efficient learning.
- Level Design: Implement more complex level designs with varying obstacles.
