import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random as rnd

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.max_walls = 20
        input_size = 6 + self.max_walls * 2 
        self.fc1 = nn.Linear(input_size, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.fc1(x)), negative_slope=0.01)
        x = F.leaky_relu(self.ln2(self.fc2(x)), negative_slope=0.01)
        return self.fc3(x)

# Hyperparameter tune and optimise model 

# Learning rate controls how much the model weights are updated during training
# use a scheduler to adjust the lr during training using schedulers
# Larger batch sizes will provide smoother updates (more memory though)
# Gamma determines the importance of future rewards (b/w 0.9 and 0.99)
# Higher gamma will value future rewards more than immediate rewards. 
# good for encouraging longer-term strategies.
# Epsilon controls the trade-off between exploration and exploitation
# replay buffer size is the number of past experiences stored for training.
# larger buffer provides more varied experiences but consumes more memory 
# batch normalization normalizes the input to each layer to improve training speed and stability
# dropouts randomly zeroes some of the elements of the input tensor with probability p during training,
# which helps prevent overfitting. Allows wider networks with 512 or 1024 neurons.
# weight initialization can improve convergence speed and model performance such as using
# methods like Xavier initialization for linear layers
# Gradient clipping prevents exploding gradients by capping the gradient values 
# torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=1.0)
# Include this line after loss.backward() and before optimizer.step()

