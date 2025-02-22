# Snake Game AI with Deep Q-Learning and A* Pathfinding

This project implements a classic Snake game with an AI agent trained using Deep Q-Learning (DQN).  It also incorporates A* pathfinding to guide the AI agent towards the food.  The game can be played manually or in AI mode.

## Table of Contents

- [Features](#features)
- [How to Play](#how-to-play)
- [AI Training](#ai-training)
- [Dependencies](#dependencies)
- [Model Details](#model-details)
- [A* Pathfinding](#a-pathfinding)
- [Future Improvements](#future-improvements)

## Features

- **Deep Q-Learning AI:** The AI agent learns to play the game using a DQN model.
- **A* Pathfinding:**  Improves AI performance by calculating the shortest path to the food.
- **Manual Control:** Play the game yourself using keyboard controls.
- **Wall Obstacles:** Adds walls to the game, increasing the challenge.
- **Score Tracking:** Keeps track of the player's score.
- **Gradient Snake:** Snake body segments have a color gradient effect.
- **Speed Increase:** The snake's speed increases as the score increases.
- **Model Saving/Loading:**  Trained models can be saved and loaded.
- **Visual Training:** Option to visualize the AI training process.

## How to Play
### Manual Mode
1. Run the game:
   ```bash
   python game.py
   ```
2. Use the following keys to control the snake:
   w: Up
   s: Down
   a: Left
   d: Right
   r: Restart the game after game over.
   i: Toggle AI mode.
### AI Mode
1. Run the game in AI mode:
   ```bash
   python game.py ai
   ```
2. The AI will control the snake.  Press i to toggle back to manual mode.

## AI Training
The train.py script trains the DQN model.
1. Run the training script:
   ```bash
   python train.py
   ```
2. The script will train the model for a specified number of episodes, save checkpoints periodically, and save the final trained model as dqn_final_model.pth.  Adjust the episodes variable in the train.py script to change the training length.
3. The visTraining=True argument in the train() function allows you to visualize the training process.

## Dependencies
* Python 3
* Pygame
* Gym
* PyTorch
* NumPy

## Model Details
The DQN model consists of three fully connected layers with ReLU activation functions.  The input layer size is determined by the observation space (snake head position, direction, food position, and wall positions). The output layer has three neurons, representing the Q-values for the three possible actions (forward, left, right).

## A* Pathfinding
The a_star.py file implements the A* pathfinding algorithm. It's used by the AI to find the shortest path to the food, improving its efficiency and reducing random movements. The Manhattan distance is used as the heuristic function.

## Future Improvements
- More Complex Environments: Add more obstacles or different types of food to make the game more challenging.
- Prioritized Experience Replay: Implement prioritized experience replay to improve the training efficiency.
- Double DQN: Consider using Double DQN to address the overestimation bias in Q-learning.
- GUI Enhancements: Improve the game's visuals and add a menu system.
- Modularization: Refactor the code to improve modularity and readability.
