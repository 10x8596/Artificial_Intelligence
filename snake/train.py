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
    # target network that updates less frequently to provide stable targets for Q-value updates
    target_dqn = DQN().to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = optim.Adam(dqn.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    loss_fn = nn.SmoothL1Loss()

    replay_buffer = deque(maxlen=10000)
    gamma = 0.99
    # epsilon decay to reduce exploration over time 
    epsilon_start = 1.0
    epsilon_end = 0.01 
    epsilon_decay = 0.995
    epsilon = epsilon_start
    batch_size = 64
    total_steps = 0 
    target_update_frequency = 1000 

    for episode in range(episodes):
        # retrieve game state
        state = env.reset()
        done = False
        total_reward = 0 

        while not done:
            
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = dqn(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            total_steps += 1

            # store the transition in the replay buffer
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state # keep state as a numpy array

            if len(replay_buffer) >= batch_size:
                mini_batch = rnd.sample(replay_buffer, batch_size)

                batch = list(zip(*mini_batch))

                # Prepare batch tensors
                batch_states = torch.tensor(batch[0], dtype=torch.float32).to(device)
                batch_actions = torch.tensor(batch[1], dtype=torch.long).unsqueeze(1).to(device)
                batch_rewards = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)
                batch_next_states = torch.tensor(batch[3], dtype=torch.float32).to(device)
                batch_dones = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)

                # Compute Q-values and target Q-values
                q_values = dqn(batch_states)
                next_q_values = target_dqn(batch_next_states)

                max_next_q_values = next_q_values.max(1)[0].unsqueeze(1)
                target_q_values = batch_rewards + (gamma * max_next_q_values * (1 - batch_dones))

                # Gather the Q-values for the actions taken
                current_q_values = q_values.gather(1, batch_actions)

                # Compute loss
                loss = loss_fn(current_q_values, target_q_values.detach())

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the target network periodically
                if total_steps % target_update_frequency == 0:
                    target_dqn.load_state_dict(dqn.state_dict())

            #state = next_state -- remove
            
            # visualise the AI playing in rendering enabled
            if visTraining and total_steps % 10 == 0:
                env.visTraining()

        # decay epsilon 
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        scheduler.step()
        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

        # save the model every 100 episodes
        if (episode + 1) % 100 == 0:
            checkpoint_path = f"dqn_checkpoint_episode_{episode + 1}.pth"
            torch.save(dqn.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")

    # Save the final model 
    torch.save(dqn.state_dict(), 'dqn_final_model.pth')
    print("Final model saved")

    env.close()

# Train the model 
train(800, visTraining=True)



