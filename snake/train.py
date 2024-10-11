import numpy as np 
from collections import deque
from model import DQN
from game import Game
import torch 
import torch.nn as nn 
import torch.optim as optim
import random as rnd

def train(episodes, visTraining=True):
    env = Game()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dqn = DQN().to(device)
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)
    gamma = 0.99
    epsilon = 0.1 
    batch_size = 64

    for episode in range(episodes):
        # retrieve game state
        state = env.reset()
        done = False
        total_reward = 0 

        while not done:
            state = torch.tensor(state, dtype=torch.float32).to(device)
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = dqn(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # store the transition in the replay buffer
            replay_buffer.append((state.cpu().numpy(), action, reward, next_state, done))

            if len(replay_buffer) >= batch_size:
                mini_batch = rnd.sample(replay_buffer, batch_size)
                for s, a, r, ns, d in mini_batch:
                    s = torch.tensor(s, dtype=torch.float32).to(device)
                    ns = torch.tensor(ns, dtype=torch.float32).to(device)
                    r = torch.tensor([r], dtype=torch.float32).to(device)
                    d = torch.tensor([d], dtype=torch.float32).to(device)

                    q_values = dqn(s)
                    next_q_values = dqn(ns)

                    target_q = r + (gamma * next_q_values.max().item() * (1 - d.item()))
                    q_values[a] = target_q

                    optimizer.zero_grad()
                    loss = loss_fn(q_values, torch.tensor(target_q, dtype=torch.float32))
                    loss.backward()
                    optimizer.step()

            state = next_state
            
            # visualise the AI playing in rendering enabled
            if visTraining:
                env.visTraining()

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")
    env.close()

# Train the model 
train(1000, visTraining=True)

